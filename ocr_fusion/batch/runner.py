"""Running a folder of documents without losing work.

The shape is deliberately plain: one document at a time, each one's result
written the moment it finishes, the checkpoint flushed on every state change.
No broker, no worker pool, no scheduler.

That is not a shortcut, it is the measurement. On this hardware a single page
occupies every core for minutes, so a second concurrent document does not add
throughput - it halves the speed of both and doubles the memory, which on a
16 GB machine is how a batch starts swapping and stops finishing at all. The
place concurrency belongs here is overlapping the *cheap* stages with the
expensive one, and the cheap stages already cost under a second a document.

What the plain shape does buy is the property that actually matters over
weeks of work: a crash costs one document.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ocr_fusion.batch.jobs import (
    BatchState,
    DocumentJob,
    JobState,
    available_memory_mb,
    wait_for_memory,
)
from ocr_fusion.config.schema import AppSettings, OutputFormat
from ocr_fusion.documents import load_document
from ocr_fusion.documents.errors import DocumentError

logger = logging.getLogger(__name__)

#: Rough resident size of a loaded 3B model plus its context. Starting a
#: document with less than this free is how a run ends up in swap.
DEFAULT_MEMORY_FLOOR_MB = 3500.0


def memory_floor_for(settings: AppSettings) -> float:
    """How much free memory this configuration actually needs.

    A run that reads text layers and nothing else holds no model and should
    never be held back for memory it will not use - refusing to extract text
    because a model would not have fitted is a guard protecting against
    nothing.
    """
    from ocr_fusion.config.schema import UnlimitedBackend

    local_model_backends = {UnlimitedBackend.OLLAMA, UnlimitedBackend.TRANSFORMERS}
    needs_model = settings.qwen.enabled or (
        settings.unlimited_ocr.enabled
        and settings.unlimited_ocr.backend in local_model_backends
    )
    return DEFAULT_MEMORY_FLOOR_MB if needs_model else 0.0


class BatchCancelled(Exception):
    """Raised when a caller's cancel check asks the batch to stop."""


class BatchRunner:
    """Processes a list of documents, checkpointing as it goes."""

    def __init__(
        self,
        settings: AppSettings,
        output_directory: Path,
        *,
        output_format: OutputFormat = OutputFormat.JSON,
        max_attempts: int = 2,
        memory_floor_mb: float | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.settings = settings
        self.output_directory = Path(output_directory)
        self.output_format = output_format
        self.max_attempts = max(1, max_attempts)
        self.memory_floor_mb = (
            memory_floor_for(settings) if memory_floor_mb is None else memory_floor_mb
        )
        self.on_event = on_event
        self.should_cancel = should_cancel

    # -- reporting --------------------------------------------------------

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception:  # noqa: BLE001 - a broken listener must not end
                # a batch that may already be hours old.
                logger.debug("Batch listener raised", exc_info=True)

    # -- execution --------------------------------------------------------

    def run(self, state: BatchState) -> BatchState:
        """Work through every unfinished document in ``state``.

        Safe to call again on the same state: finished documents are skipped,
        which is what makes resuming and "the same folder plus today's new
        files" the same operation.
        """
        self.output_directory.mkdir(parents=True, exist_ok=True)
        state.save()

        pending = state.pending
        self._emit("batch_started", documents=len(pending), total=len(state.jobs))

        for job in pending:
            if self.should_cancel and self.should_cancel():
                state.mark(job, JobState.CANCELLED, error="cancelled before it started")
                self._emit("cancelled", source=job.source)
                break

            if self.memory_floor_mb and not wait_for_memory(self.memory_floor_mb):
                # Refusing to start is the right answer: a document begun into
                # a full machine swaps, and a swapping run is slower than no
                # run. Leave it queued so a resume can pick it up.
                available = available_memory_mb()
                self._emit(
                    "paused_for_memory",
                    source=job.source,
                    available_mb=available,
                    needed_mb=self.memory_floor_mb,
                )
                break

            self._run_one(state, job)

        state.save()
        self._emit("batch_finished", progress=state.progress())
        return state

    def _run_one(self, state: BatchState, job: DocumentJob) -> None:
        """Process one document, retrying a bounded number of times."""
        from ocr_fusion.export import export_bytes
        from ocr_fusion.pipeline import build_pipeline

        source = Path(job.source)
        while job.attempts < self.max_attempts:
            attempt = job.attempts + 1
            state.mark(
                job,
                JobState.PROCESSING if attempt == 1 else JobState.RETRYING,
                attempts=attempt,
            )
            self._emit("document_started", source=job.source, attempt=attempt)

            started = time.perf_counter()
            try:
                document = load_document(source, self.settings.documents)
            except DocumentError as exc:
                # A corrupt or unreadable file will be just as corrupt next
                # time; retrying it only delays the rest of the batch.
                state.mark(
                    job,
                    JobState.FAILED,
                    error=exc.user_message,
                    seconds=time.perf_counter() - started,
                )
                self._emit("document_failed", source=job.source, error=exc.user_message)
                return
            except Exception as exc:  # noqa: BLE001 - one bad file, not a dead batch
                state.mark(
                    job,
                    JobState.FAILED,
                    error=f"could not be read: {exc}",
                    seconds=time.perf_counter() - started,
                )
                self._emit("document_failed", source=job.source, error=str(exc))
                return

            try:
                result = build_pipeline(self.settings).execute(document)
            except Exception as exc:  # noqa: BLE001 - retryable: a timeout or a
                # runtime that went away can succeed on a second attempt.
                logger.exception("Batch document %s raised", job.source)
                job.seconds += time.perf_counter() - started
                if job.attempts >= self.max_attempts:
                    state.mark(
                        job,
                        JobState.FAILED,
                        error=f"failed after {job.attempts} attempt(s): {exc}",
                    )
                    self._emit("document_failed", source=job.source, error=str(exc))
                    return
                state.save()
                continue

            elapsed = time.perf_counter() - started
            output = self._write_result(source, result, export_bytes)
            flagged = sum(1 for c in result.confidence if c.needs_second_opinion)

            state.mark(
                job,
                JobState.REVIEW_REQUIRED if flagged else JobState.COMPLETED,
                pages=document.page_count,
                pages_via_model=(
                    len(result.routing.ocr_pages)
                    if result.routing
                    else document.page_count
                ),
                pages_flagged=flagged,
                seconds=elapsed,
                output=str(output) if output else None,
                error=None,
            )
            self._emit(
                "document_finished",
                source=job.source,
                pages=document.page_count,
                seconds=round(elapsed, 2),
                flagged=flagged,
                progress=state.progress(),
            )
            return

    def _write_result(self, source: Path, result: Any, export_bytes: Any) -> Path | None:
        """Persist one document's result immediately.

        Immediately, rather than at the end of the batch, so that a run
        stopped after nine hours has nine hours of usable output rather than
        none - and so a person can start reviewing while the rest runs.
        """
        try:
            payload = export_bytes(result, self.settings, self.output_format)
        except Exception:  # noqa: BLE001 - an export bug must not lose the run
            logger.exception("Could not export %s", source.name)
            return None

        destination = self.output_directory / f"{source.stem}.{self.output_format.value}"
        # Two source folders can hold the same filename; keeping both matters
        # more than a tidy name.
        counter = 1
        while destination.exists():
            destination = (
                self.output_directory
                / f"{source.stem}_{counter}.{self.output_format.value}"
            )
            counter += 1
        try:
            destination.write_bytes(payload)
        except OSError:
            logger.exception("Could not write %s", destination)
            return None
        return destination


def collect_documents(target: Path, pattern: str = "*") -> list[Path]:
    """Every supported document under ``target``, sorted for a stable order."""
    from ocr_fusion.documents.loaders import SUPPORTED_EXTENSIONS

    if target.is_file():
        return [target]
    return sorted(
        path
        for path in target.rglob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


__all__ = [
    "DEFAULT_MEMORY_FLOOR_MB",
    "BatchCancelled",
    "BatchRunner",
    "collect_documents",
    "memory_floor_for",
]
