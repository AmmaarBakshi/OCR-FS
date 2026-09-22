"""Job state for a batch, and the checkpoint that survives a crash.

A batch of ten thousand documents is not one job. At the rate this hardware
manages it is weeks of work, so the thing that matters is not raw speed but
that no minute of it is ever spent twice: a crash on document 4,812 must cost
document 4,812 and nothing else.

So the batch is a list of independent documents, each with its own state, and
the state is written to disk as it changes. Restarting reads the checkpoint
and picks up where it stopped. A document that fails is recorded with the
reason, retried a bounded number of times, and then set aside for a person -
never retried forever, which turns one broken file into an infinite loop.

The checkpoint is JSON and holds no document text: filenames, states, counts
and timings only. The transcriptions live in their own result files, written
per document as each one finishes.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JobState(str, Enum):
    """Where one document has got to."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    """Exhausted its retries. A person needs to look at the file itself."""

    RETRYING = "retrying"
    REVIEW_REQUIRED = "review_required"
    """Transcribed, but the confidence check flagged pages. Usable, with a
    caveat - and far cheaper than re-reading everything to be sure."""

    CANCELLED = "cancelled"

    @property
    def is_finished(self) -> bool:
        """Whether a resumed run should leave this document alone."""
        return self in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.REVIEW_REQUIRED,
            JobState.CANCELLED,
        )


@dataclass
class DocumentJob:
    """One document's place in the batch."""

    source: str
    state: JobState = JobState.QUEUED
    pages: int = 0
    pages_via_model: int = 0
    pages_flagged: int = 0
    seconds: float = 0.0
    attempts: int = 0
    error: str | None = None
    output: str | None = None
    """Where this document's result was written, once it has been."""

    finished_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentJob:
        data = dict(data)
        data["state"] = JobState(data.get("state", "queued"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class BatchState:
    """The whole batch, and the file it is mirrored to.

    Every state change is flushed, because the failure this exists for is the
    process dying without warning.
    """

    checkpoint: Path
    jobs: list[DocumentJob] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    label: str = "batch"

    # -- progress ---------------------------------------------------------

    def counts(self) -> dict[str, int]:
        tally = {state.value: 0 for state in JobState}
        for job in self.jobs:
            tally[job.state.value] += 1
        return tally

    @property
    def pending(self) -> list[DocumentJob]:
        return [job for job in self.jobs if not job.state.is_finished]

    @property
    def completed_pages(self) -> int:
        return sum(job.pages for job in self.jobs if job.state.is_finished)

    @property
    def elapsed_seconds(self) -> float:
        return sum(job.seconds for job in self.jobs)

    @property
    def pages_per_minute(self) -> float | None:
        """Measured throughput, or ``None`` before anything has finished."""
        if self.elapsed_seconds <= 0 or self.completed_pages == 0:
            return None
        return self.completed_pages / self.elapsed_seconds * 60

    def estimated_remaining_seconds(self) -> float | None:
        """Projection from the rate actually observed, not from a guess.

        ``None`` until enough has finished to have a rate at all - an ETA
        invented from no data is worse than no ETA.
        """
        done = [job for job in self.jobs if job.state.is_finished and job.pages]
        if not done or not self.pending:
            return None
        seconds_per_document = sum(j.seconds for j in done) / len(done)
        return seconds_per_document * len(self.pending)

    def progress(self) -> dict[str, Any]:
        """What the UI or the console shows while a batch is running."""
        finished = sum(1 for job in self.jobs if job.state.is_finished)
        return {
            "documents_total": len(self.jobs),
            "documents_completed": finished,
            "pages_completed": self.completed_pages,
            "counts": self.counts(),
            "pages_per_minute": (
                round(self.pages_per_minute, 3) if self.pages_per_minute else None
            ),
            "estimated_remaining_seconds": (
                round(self.estimated_remaining_seconds() or 0.0, 0)
                if self.estimated_remaining_seconds()
                else None
            ),
        }

    # -- persistence ------------------------------------------------------

    def save(self) -> None:
        """Write the checkpoint atomically.

        Atomically because the whole point is surviving an abrupt stop, and a
        half-written checkpoint would lose the batch it exists to protect.
        """
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self.label,
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "progress": self.progress(),
            "jobs": [job.as_dict() for job in self.jobs],
        }
        handle, temporary = tempfile.mkstemp(
            dir=str(self.checkpoint.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, indent=1)
            os.replace(temporary, self.checkpoint)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, checkpoint: Path) -> BatchState | None:
        """Read a checkpoint, or ``None`` when there is nothing to resume."""
        if not checkpoint.exists():
            return None
        try:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        state = cls(
            checkpoint=checkpoint,
            label=payload.get("label", "batch"),
            started_at=payload.get("started_at", ""),
        )
        state.jobs = [DocumentJob.from_dict(job) for job in payload.get("jobs", [])]
        return state

    @classmethod
    def for_documents(
        cls, paths: list[Path], checkpoint: Path, *, label: str = "batch"
    ) -> BatchState:
        """Start a batch, reusing any progress already recorded for these files.

        Documents in the checkpoint keep their state; documents new to this
        run are queued; documents no longer present are dropped. That makes
        "run it again over the same folder, plus the twelve that arrived this
        morning" the ordinary case rather than a special one.
        """
        existing = cls.load(checkpoint)
        known = {job.source: job for job in existing.jobs} if existing else {}

        state = cls(checkpoint=checkpoint, label=label)
        state.jobs = [known.get(str(path)) or DocumentJob(source=str(path)) for path in paths]
        if existing:
            state.started_at = existing.started_at
        return state

    def mark(self, job: DocumentJob, state: JobState, **fields: Any) -> None:
        """Move a job to a new state and flush the checkpoint."""
        job.state = state
        for key, value in fields.items():
            setattr(job, key, value)
        if state.is_finished:
            job.finished_at = datetime.now(timezone.utc).isoformat()
        self.save()


def elapsed_text(seconds: float | None) -> str:
    """Human duration, or ``n/a`` when there is nothing to report."""
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def available_memory_mb() -> float | None:
    """Free physical memory, or ``None`` when it cannot be observed."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:  # noqa: BLE001 - a metric, not a feature
        return None


def wait_for_memory(
    minimum_mb: float, *, timeout: float = 120.0, poll: float = 5.0
) -> bool:
    """Hold until enough memory is free, or give up and say so.

    A 3B model that does not fit swaps, and a swapping run is slower than no
    run at all - so starting the next document into a machine that has no room
    for it is worse than waiting a moment. Returns True when there is room, or
    when memory cannot be measured at all: a missing metric must not become a
    deadlock.
    """
    deadline = time.monotonic() + timeout
    while True:
        available = available_memory_mb()
        if available is None or available >= minimum_mb:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


__all__ = [
    "BatchState",
    "DocumentJob",
    "JobState",
    "available_memory_mb",
    "elapsed_text",
    "wait_for_memory",
]
