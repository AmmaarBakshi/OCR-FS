"""The OCR pipeline executor.

Runs the configured stages in order:

    Upload -> Qwen2.5-VL -> Unlimited-OCR -> Comparison -> Fusion -> Final result

The pipeline knows nothing about any specific engine. It is handed
:class:`~ocr_fusion.ocr.interface.OCRProvider` instances (resolved from the
registry) and drives them through the interface alone, which is what allows a
new engine to join a run without changing this file (spec s11).

Isolation is the other guarantee: a provider that fails, times out or raises
unexpectedly is recorded as a failed stage while the run continues, so the user
can still inspect whatever did succeed (spec s13).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocr_fusion.config.schema import AppSettings, EngineMode, ProcessingLocation
from ocr_fusion.documents.loaders import apply_page_limit
from ocr_fusion.documents.models import Document
from ocr_fusion.ocr.interface import (
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
)
from ocr_fusion.pipeline.comparison import compare_texts
from ocr_fusion.pipeline.confidence import PageConfidence, score_page
from ocr_fusion.pipeline.events import EventLog, LogEvent, StageStatus
from ocr_fusion.pipeline.fusion import fuse, fuse_pagewise
from ocr_fusion.pipeline.result import PipelineResult
from ocr_fusion.pipeline.routing import PageClass, RoutingPlan, route_document, subset

logger = logging.getLogger(__name__)

#: Stage keys used by both the executor and the UI visualisation.
STAGE_UPLOAD = "upload"
STAGE_ROUTING = "routing"
STAGE_VERIFICATION = "verification"
STAGE_COMPARISON = "comparison"
STAGE_FUSION = "fusion"


@dataclass
class RunProgress:
    """How far a run has got, and how much longer it is likely to take.

    A nine-page run takes tens of minutes on CPU, and a progress bar with no
    estimate behind it is only slightly better than a spinner. The estimate
    comes from the pages this run has already finished, never from a constant:
    page cost varies by a factor of ten with how much text is on the page, so
    a figure derived from this document is the only one worth showing - and
    before any page has finished there is no figure, and it says so.
    """

    engine: str = ""
    page: int = 0
    """Page number within the current engine's share of the work."""

    pages_in_engine: int = 0
    pages_done: int = 0
    """Pages finished across the whole run."""

    pages_total: int = 0
    """Pages that will reach an engine, known from routing before work starts."""

    durations: list[float] = field(default_factory=list)

    @property
    def seconds_per_page(self) -> float | None:
        if not self.durations:
            return None
        return sum(self.durations) / len(self.durations)

    @property
    def estimated_remaining_seconds(self) -> float | None:
        rate = self.seconds_per_page
        if rate is None or self.pages_total <= self.pages_done:
            return None
        return rate * (self.pages_total - self.pages_done)

    def describe(self) -> str:
        """One line for the UI. Says "working it out" rather than guessing."""
        if not self.pages_total:
            return ""
        position = f"Page {self.pages_done + 1} of {self.pages_total}"
        remaining = self.estimated_remaining_seconds
        if remaining is None:
            return f"{position} - timing the first page before estimating"
        minutes = remaining / 60
        if minutes < 1:
            return f"{position} - under a minute left"
        return f"{position} - about {minutes:.0f} min left"


class OCRPipeline:
    """Executes OCR providers in sequence and reconciles their output."""

    def __init__(
        self,
        settings: AppSettings,
        providers: Sequence[OCRProvider] | None = None,
        *,
        log: EventLog | None = None,
    ) -> None:
        self.settings = settings
        self.providers: list[OCRProvider] = list(providers or [])
        self.log = log or EventLog(
            redact_text=settings.privacy.redact_text_in_logs,
            preview_chars=settings.privacy.log_preview_chars,
        )
        #: Per-page confidence from the last run, populated by verification.
        self.confidence: list[PageConfidence] = []
        #: Live progress, readable by a UI while execute() is running.
        self.progress = RunProgress()

    # -- composition -------------------------------------------------------

    def add_provider(self, provider: OCRProvider) -> OCRPipeline:
        """Append an engine. Returns self so calls can be chained."""
        self.providers.append(provider)
        return self

    def subscribe(self, callback: Callable[[LogEvent], None]) -> OCRPipeline:
        """Receive every log event as it happens, for live UI updates."""
        self.log.subscribe(callback)
        return self

    def health_report(self) -> dict[str, Any]:
        """Readiness of every engine, for the Settings and status panels."""
        report: dict[str, Any] = {}
        for provider in self.providers:
            status = provider.health_check()
            report[provider.provider_id] = {
                "name": provider.provider_name,
                "available": status.available,
                "message": status.message,
                "remedy": status.remedy,
                "details": status.details,
            }
        return report

    # -- execution ---------------------------------------------------------

    def _prepare_stages(self) -> None:
        """Register every stage up front so the UI can draw the full pipeline
        before anything has run (spec s4: stages start as Waiting)."""
        self.log.add_stage(STAGE_UPLOAD, "Document")
        if self._routing_enabled:
            self.log.add_stage(STAGE_ROUTING, "Routing")
        for provider in self.providers:
            self.log.add_stage(provider.provider_id, provider.provider_name)
        if self.settings.pipeline.run_comparison:
            self.log.add_stage(STAGE_COMPARISON, "Comparison")
        if self.settings.pipeline.run_fusion:
            self.log.add_stage(STAGE_FUSION, "Fusion")

    def execute(self, document: Document) -> PipelineResult:
        """Run the full pipeline over ``document``."""
        started = time.perf_counter()
        self.confidence = []
        self.progress = RunProgress()
        self._prepare_stages()

        result = PipelineResult(
            document=document,
            log=self.log,
            started_at=datetime.now(timezone.utc),
            settings_snapshot=self.settings.redacted(),
        )

        self._stage_document(document)
        plan = self._stage_routing(document)
        result.routing = plan
        result.engine_results = self._run_providers(document, plan)
        if plan is not None:
            routed = self._fill_routed_pages(document, plan, result.engine_results)
            if routed is not None:
                result.engine_results.append(routed)
        result.confidence = list(self.confidence)
        result.comparison = self._stage_comparison(result.engine_results)
        result.fusion = self._stage_fusion(result.engine_results, plan)

        result.total_duration_seconds = time.perf_counter() - started
        result.stages = self.log.ordered_stages()

        self.log.info(
            f"Run finished in {result.total_duration_seconds:.2f}s - "
            f"{len(result.successful_engines)} of {len(result.engine_results)} "
            "engines produced text."
        )
        return result

    # -- stages ------------------------------------------------------------

    def _stage_document(self, document: Document) -> None:
        stage = self.log.start_stage(
            STAGE_UPLOAD, f"Document loaded: {document.filename}"
        )
        limit = self.settings.pipeline.max_pages
        if limit:
            apply_page_limit(document, limit)

        if document.kind.value == "pdf":
            self.log.info(
                f"PDF rendered to {document.page_count} "
                f"image{'s' if document.page_count != 1 else ''} at "
                f"{self.settings.documents.pdf_render_dpi} DPI"
            )
            if document.has_text_layer:
                self.log.info(
                    "The PDF already contains a text layer; OCR will run anyway "
                    "so the engines can be compared."
                )
        for warning in document.warnings:
            self.log.warning(warning)

        stage.detail = (
            f"{document.page_count} page{'s' if document.page_count != 1 else ''}"
        )
        self.log.finish_stage(
            STAGE_UPLOAD,
            StageStatus.COMPLETED,
            detail=stage.detail,
            message=f"Document ready: {document.page_count} page(s)",
        )

    @property
    def _routing_enabled(self) -> bool:
        return self.settings.routing.enabled

    def _stage_routing(self, document: Document) -> RoutingPlan | None:
        """Work out what each page needs before any engine is asked to read it."""
        if not self._routing_enabled:
            return None

        self.log.start_stage(STAGE_ROUTING, "Working out what each page needs")
        plan = route_document(document, self.settings.routing)

        self.log.info(plan.headline(), stage=STAGE_ROUTING)
        for route in plan.routes:
            if route.page_class is not PageClass.NEEDS_OCR:
                self.log.info(
                    f"Page {route.page_number}: {route.reason}.", stage=STAGE_ROUTING
                )

        avoided = plan.pages_avoided
        detail = (
            f"{len(plan.ocr_pages)} of {len(plan.routes)} page(s) need an engine"
            if plan.routes
            else "nothing to route"
        )
        self.log.finish_stage(
            STAGE_ROUTING,
            StageStatus.COMPLETED,
            detail=detail,
            message=(
                f"Routing complete in {plan.duration_seconds:.2f}s - {detail}"
                + (f", {avoided} answered without one." if avoided else ".")
            ),
        )
        return plan

    def _run_one(self, provider: OCRProvider, document: Document) -> OCRResult:
        """Run one engine over one (possibly partial) document.

        Failure is contained here rather than at the call site so that every
        way of driving the engines - all-engines or cascade - inherits the
        same guarantee: a provider that raises becomes a failed result, not a
        lost run (spec s13).
        """
        key = provider.provider_id
        name = provider.provider_name
        total_pages = document.page_count

        pages = document.pages
        clock = {"page_started": time.perf_counter()}

        def report_progress(current: int, total: int) -> None:
            now = time.perf_counter()
            if current > 1:
                # The previous page has just finished; its cost is the best
                # evidence available for what the next one will take.
                self.progress.durations.append(now - clock["page_started"])
                self.progress.pages_done += 1
            clock["page_started"] = now
            self.progress.engine = name
            self.progress.page = current
            self.progress.pages_in_engine = total

            stage = self.log.stage(key)
            stage.detail = f"Processing page {current} of {total}..."
            # An info event as well as the stage detail: a nine-page run sits
            # inside this call for minutes and silence reads as a hang.
            estimate = self.progress.describe()
            self.log.info(
                f"{name} processing page {current} of {total}"
                + (f" - {estimate}" if estimate else "..."),
                stage=key,
            )

            # The engine has finished with the previous page, and a rendered
            # page is ~200 KB. Holding all of them costs a 500-page scan about
            # 100 MB for pixels nothing will look at again - and on a machine
            # with under a gigabyte free, that is the difference between
            # running and swapping. The page can still redraw itself if the
            # review queue asks for it later.
            if self.settings.documents.release_pages_after_reading and current > 1:
                pages[current - 2].release_image()

        if total_pages == 0:
            return OCRResult.skipped(key, name, "no pages were routed to this engine")

        try:
            return provider.process(document, on_progress=report_progress)
        except Exception as exc:  # noqa: BLE001 - a provider bug must not take
            # down a run the user has already waited minutes for.
            logger.exception("Provider %s raised", key)
            return OCRResult.failure(
                key, name, f"The engine failed unexpectedly: {exc}"
            )

    def _run_providers(
        self, document: Document, plan: RoutingPlan | None
    ) -> list[OCRResult]:
        """Drive the engines according to the configured engine mode."""
        if plan is None or self.settings.pipeline.engine_mode is EngineMode.ALL_ENGINES:
            return self._run_all_engines(document)
        return self._run_cascade(document, plan)

    def _run_all_engines(self, document: Document) -> list[OCRResult]:
        """Every enabled engine reads every page - the original behaviour.

        Kept because the side-by-side comparison needs two engines to have
        read the same page. It costs one full pass per engine.
        """
        results: list[OCRResult] = []
        for provider in self.providers:
            key = provider.provider_id
            self.log.start_stage(key, f"{provider.provider_name} started")
            result = self._run_one(provider, document)
            results.append(result)
            self._record_provider_outcome(key, result)

            if result.status is OCRStatus.FAILED and not (
                self.settings.pipeline.continue_on_provider_error
            ):
                self.log.warning(
                    "Stopping after a failed engine because "
                    "'continue on provider error' is switched off."
                )
                break
        return results

    def _run_cascade(self, document: Document, plan: RoutingPlan) -> list[OCRResult]:
        """Each engine handles only what the engines before it could not.

        The ordering is the whole optimisation. Text-layer extraction answers
        most pages at no cost; the first OCR engine reads only what is left;
        a second engine is spent only on the pages the confidence check
        flagged. A page therefore reaches a model once, or not at all.
        """
        results: list[OCRResult] = []
        text_layer_pages = [
            r.page_number for r in plan.routes if r.page_class is PageClass.TEXT_LAYER
        ]
        ocr_pages = plan.ocr_pages

        # Known before any work starts, which is what makes an estimate
        # possible rather than a guess that improves as it goes.
        self.progress.pages_total = len(ocr_pages)
        ocr_providers = [p for p in self.providers if p.provider_id != "text_layer"]
        extractor = next(
            (p for p in self.providers if p.provider_id == "text_layer"), None
        )

        if extractor is not None:
            key = extractor.provider_id
            if text_layer_pages:
                self.log.start_stage(key, "Reading the text the PDF already carries")
                result = self._run_one(extractor, subset(document, text_layer_pages))
                results.append(result)
                self._record_provider_outcome(key, result)
            else:
                self.log.skip_stage(key, "no page carries a usable text layer")

        if not ocr_providers:
            if ocr_pages:
                # Silence here would hand back a document with pages quietly
                # missing from it, which is the one outcome worse than slow.
                self.log.warning(
                    f"{len(ocr_pages)} page(s) need an OCR engine and none is "
                    "enabled, so those pages are missing from the result: "
                    + ", ".join(str(n) for n in ocr_pages)
                )
            return results

        primary, *fallbacks = ocr_providers
        if not ocr_pages:
            for provider in ocr_providers:
                self.log.skip_stage(
                    provider.provider_id,
                    "every page was answered without an engine",
                )
            return results

        self.log.start_stage(
            primary.provider_id,
            f"{primary.provider_name} reading {len(ocr_pages)} page(s)",
        )
        primary_result = self._run_one(primary, subset(document, ocr_pages))
        results.append(primary_result)
        self._record_provider_outcome(primary.provider_id, primary_result)

        flagged = self._stage_verification(primary_result, plan)
        for provider in fallbacks:
            if not flagged:
                self.log.skip_stage(
                    provider.provider_id,
                    "every page the primary engine read looks complete",
                )
                continue
            self.log.start_stage(
                provider.provider_id,
                f"{provider.provider_name} re-reading {len(flagged)} flagged page(s)",
            )
            result = self._run_one(provider, subset(document, flagged))
            results.append(result)
            self._record_provider_outcome(provider.provider_id, result)

        return results

    def _stage_verification(
        self, primary: OCRResult, plan: RoutingPlan
    ) -> list[int]:
        """Score the primary engine's pages and pick the ones worth re-reading.

        Returns the page numbers to send to a fallback engine, capped by
        ``max_fallback_page_share``: a document the primary engine handled
        badly throughout is a configuration problem, and silently paying twice
        for every page would hide it.
        """
        if not self.settings.pipeline.fallback_enabled:
            return []

        self.log.add_stage(STAGE_VERIFICATION, "Verification")
        self.log.start_stage(STAGE_VERIFICATION, "Checking the transcription")

        scores = [
            score_page(page, self.settings.confidence, plan.route(page.page_number))
            for page in primary.pages
        ]
        self.confidence = scores
        flagged = [s for s in scores if s.needs_second_opinion]

        for score in flagged:
            self.log.warning(
                f"Page {score.page_number} needs a second look: {score.reason}.",
                stage=STAGE_VERIFICATION,
            )

        cap = max(1, int(len(scores) * self.settings.pipeline.max_fallback_page_share))
        selected = [s.page_number for s in flagged][:cap]
        if len(flagged) > len(selected):
            self.log.warning(
                f"{len(flagged)} of {len(scores)} page(s) were flagged, which is "
                "more than the fallback is allowed to re-read. Only the first "
                f"{len(selected)} will be re-read - check the engine and the "
                "render DPI rather than paying twice for every page.",
                stage=STAGE_VERIFICATION,
            )

        detail = (
            f"{len(selected)} of {len(scores)} page(s) flagged"
            if selected
            else f"all {len(scores)} page(s) look complete"
        )
        self.log.finish_stage(
            STAGE_VERIFICATION,
            StageStatus.COMPLETED,
            detail=detail,
            message=f"Verification complete - {detail}.",
        )
        return selected

    def _fill_routed_pages(
        self, document: Document, plan: RoutingPlan, results: list[OCRResult]
    ) -> OCRResult | None:
        """Account for the pages routing answered without an engine.

        Blank pages have no text by definition. A duplicate takes the text
        already produced for the page it duplicates, which is why this runs
        after the engines: the original's transcription has to exist first.
        """
        handled = [
            r
            for r in plan.routes
            if r.page_class in (PageClass.BLANK, PageClass.DUPLICATE)
        ]
        if not handled:
            return None

        by_page: dict[int, str] = {}
        for result in results:
            for page in result.pages:
                if page.text and page.page_number not in by_page:
                    by_page[page.page_number] = page.text

        pages: list[PageResult] = []
        for route in handled:
            text = ""
            if route.page_class is PageClass.DUPLICATE and route.duplicate_of:
                text = by_page.get(route.duplicate_of, "")
            pages.append(
                PageResult(
                    page_number=route.page_number,
                    text=text,
                    status=OCRStatus.SUCCESS,
                    duration_seconds=0.0,
                    confidence=1.0 if route.page_class is PageClass.BLANK else None,
                    raw_response={"routed": route.page_class.value, "reason": route.reason},
                )
            )

        return OCRResult(
            provider_id="routing",
            provider_name="Routing",
            status=OCRStatus.SUCCESS,
            pages=pages,
            # No model ran, and no tokens were spent. Reporting either would
            # be a fabricated measurement (spec s8).
            model_name=None,
            backend="routing",
            processing_location=ProcessingLocation.LOCAL,
            metadata={"pages": [r.as_dict() for r in handled]},
        )

    def _record_provider_outcome(self, key: str, result: OCRResult) -> None:
        """Turn a provider result into a stage status and log lines."""
        model = result.model_name or "unknown model"
        if result.status is OCRStatus.FAILED:
            self.log.finish_stage(
                key,
                StageStatus.FAILED,
                detail=result.error or "Failed",
                message=f"{result.provider_name} failed: {result.error}",
            )
            if result.remedy:
                self.log.warning(f"How to fix: {result.remedy}", stage=key)
            return

        summary = self.log.text_summary(result.text)
        status = (
            StageStatus.COMPLETED
            if result.status is OCRStatus.SUCCESS
            else StageStatus.COMPLETED
        )
        detail = model
        if result.status is OCRStatus.PARTIAL:
            failed = sum(1 for p in result.pages if p.status is OCRStatus.FAILED)
            detail = f"{model} - {failed} page(s) failed"
            self.log.warning(
                f"{result.provider_name} could not read {failed} page(s).", stage=key
            )

        self.log.finish_stage(
            key,
            status,
            detail=detail,
            message=f"{result.provider_name} completed - {summary}",
        )

    def _stage_comparison(self, results: list[OCRResult]):
        if not self.settings.pipeline.run_comparison:
            return None
        if STAGE_COMPARISON not in self.log.stages:
            return None

        # Routing is not an engine: its pages are blanks and copies of another
        # engine's work, so comparing against it would compare output with
        # itself and report a meaningless agreement figure.
        usable = [
            r
            for r in results
            if r.succeeded and r.text.strip() and r.provider_id != "routing"
        ]
        if len(usable) < 2:
            reason = (
                "fewer than two engines produced text"
                if usable
                else "no engine produced text"
            )
            self.log.skip_stage(STAGE_COMPARISON, reason)
            return None

        # The pair with the most pages in common, not simply the first two.
        # Under cascade the engines cover different pages: text extraction
        # might hold sixteen, the primary engine three and a fallback one, so
        # taking the first two would compare two disjoint sets and conclude
        # there was nothing to compare - exactly when the fallback has given
        # us the one genuine disagreement worth showing.
        first, second, shared = _best_comparison_pair(usable)
        if first is None or second is None or not shared:
            # Expected in cascade mode: engines divided the pages, so no page
            # has two readings to disagree about. That is the saving working,
            # not a failure.
            self.log.skip_stage(
                STAGE_COMPARISON,
                "no page was read by two engines, so there is nothing to compare",
            )
            return None

        self.log.start_stage(STAGE_COMPARISON, "Comparing engine outputs")
        comparison = compare_texts(
            _text_for_pages(first, shared),
            _text_for_pages(second, shared),
            engine_a=first.provider_name,
            engine_b=second.provider_name,
        )
        if len(shared) < max(len(first.pages), len(second.pages)):
            self.log.info(
                f"Comparing the {len(shared)} page(s) both engines read.",
                stage=STAGE_COMPARISON,
            )
        detail = f"{comparison.agreement_percent}% agreement"
        if comparison.numeric_conflicts:
            count = len(comparison.numeric_conflicts)
            detail += f", {count} numeric conflict{'s' if count != 1 else ''}"
            self.log.warning(
                f"The engines disagree on figures in {count} line(s); check the "
                "comparison view before relying on the numbers."
            )
        self.log.finish_stage(
            STAGE_COMPARISON,
            StageStatus.COMPLETED,
            detail=detail,
            message=(
                f"Comparison complete - {comparison.agreement_percent}% agreement "
                f"across {comparison.total_lines} lines"
            ),
        )
        return comparison

    def _stage_fusion(self, results: list[OCRResult], plan: RoutingPlan | None = None):
        if not self.settings.pipeline.run_fusion:
            return None
        if STAGE_FUSION not in self.log.stages:
            return None

        usable = [r for r in results if r.succeeded and r.text.strip()]
        if not usable:
            self.log.skip_stage(STAGE_FUSION, "no engine produced text")
            return None

        self.log.start_stage(STAGE_FUSION, "Generating final result")
        started = time.perf_counter()
        try:
            # When engines divide the pages between them, the unit of
            # reconciliation has to be the page: comparing one engine's three
            # pages against another's sixteen would be meaningless.
            fusion = (
                fuse_pagewise(results, self.settings)
                if plan is not None
                and self.settings.pipeline.engine_mode is EngineMode.CASCADE
                else fuse(results, self.settings)
            )
        except Exception as exc:  # noqa: BLE001 - fusion must never lose the
            # engine output the user already waited for.
            logger.exception("Fusion raised")
            self.log.finish_stage(
                STAGE_FUSION,
                StageStatus.FAILED,
                detail=str(exc),
                message=f"Fusion failed: {exc}. The raw engine results are still available.",
            )
            return None

        fusion.duration_seconds = time.perf_counter() - started
        for warning in fusion.warnings:
            self.log.warning(warning, stage=STAGE_FUSION)

        self.log.finish_stage(
            STAGE_FUSION,
            StageStatus.COMPLETED,
            detail=f"{fusion.strategy} - {self.log.text_summary(fusion.text)}",
            message=f"Final result generated ({fusion.strategy}) - {fusion.detail}",
        )
        return fusion


def _best_comparison_pair(
    results: list[OCRResult],
) -> tuple[OCRResult | None, OCRResult | None, list[int]]:
    """The two engines that read the most pages in common.

    Ties break towards the earlier pair in pipeline order, so the choice is
    deterministic and a run twice over the same document compares the same
    two engines.
    """
    best: tuple[OCRResult | None, OCRResult | None, list[int]] = (None, None, [])
    for index, first in enumerate(results):
        pages_a = {p.page_number for p in first.pages if p.text.strip()}
        for second in results[index + 1 :]:
            pages_b = {p.page_number for p in second.pages if p.text.strip()}
            shared = sorted(pages_a & pages_b)
            if len(shared) > len(best[2]):
                best = (first, second, shared)
    return best


def _text_for_pages(result: OCRResult, pages: list[int]) -> str:
    """Just the named pages of an engine's output, in page order."""
    wanted = set(pages)
    return "\n\n".join(
        page.text
        for page in sorted(result.pages, key=lambda p: p.page_number)
        if page.page_number in wanted and page.text
    )


def build_pipeline(
    settings: AppSettings,
    provider_ids: Sequence[str] | None = None,
    *,
    log: EventLog | None = None,
) -> OCRPipeline:
    """Build a pipeline from the provider registry.

    With no ``provider_ids``, the enabled engines run in the order declared by
    :data:`~ocr_fusion.ocr.providers.PIPELINE_ORDER`.
    """
    from ocr_fusion.ocr.providers import PIPELINE_ORDER
    from ocr_fusion.ocr.registry import default_registry

    if provider_ids is None:
        enabled = {spec.provider_id for spec in default_registry.enabled_specs(settings)}
        provider_ids = [pid for pid in PIPELINE_ORDER if pid in enabled]

    providers = [
        default_registry.create(pid, settings)
        for pid in provider_ids
        if default_registry.has(pid)
    ]

    if settings.cache.enabled:
        from ocr_fusion.ocr.cache import PageCache, wrap_with_cache

        cache = PageCache(Path(settings.cache.directory) / "pages.sqlite3")
        providers = [wrap_with_cache(p, cache) for p in providers]

    return OCRPipeline(settings, providers, log=log)


__all__ = [
    "RunProgress",
    "STAGE_COMPARISON",
    "STAGE_FUSION",
    "STAGE_ROUTING",
    "STAGE_UPLOAD",
    "STAGE_VERIFICATION",
    "OCRPipeline",
    "build_pipeline",
]
