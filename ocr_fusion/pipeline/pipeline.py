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
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from ocr_fusion.config.schema import AppSettings
from ocr_fusion.documents.loaders import apply_page_limit
from ocr_fusion.documents.models import Document
from ocr_fusion.ocr.interface import OCRProvider, OCRResult, OCRStatus
from ocr_fusion.pipeline.comparison import compare_texts
from ocr_fusion.pipeline.events import EventLog, LogEvent, StageStatus
from ocr_fusion.pipeline.fusion import fuse
from ocr_fusion.pipeline.result import PipelineResult

logger = logging.getLogger(__name__)

#: Stage keys used by both the executor and the UI visualisation.
STAGE_UPLOAD = "upload"
STAGE_COMPARISON = "comparison"
STAGE_FUSION = "fusion"


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
        for provider in self.providers:
            self.log.add_stage(provider.provider_id, provider.provider_name)
        if self.settings.pipeline.run_comparison:
            self.log.add_stage(STAGE_COMPARISON, "Comparison")
        if self.settings.pipeline.run_fusion:
            self.log.add_stage(STAGE_FUSION, "Fusion")

    def execute(self, document: Document) -> PipelineResult:
        """Run the full pipeline over ``document``."""
        import time

        started = time.perf_counter()
        self._prepare_stages()

        result = PipelineResult(
            document=document,
            log=self.log,
            started_at=datetime.now(timezone.utc),
            settings_snapshot=self.settings.redacted(),
        )

        self._stage_document(document)
        result.engine_results = self._run_providers(document)
        result.comparison = self._stage_comparison(result.engine_results)
        result.fusion = self._stage_fusion(result.engine_results)

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

    def _run_providers(self, document: Document) -> list[OCRResult]:
        """Run each engine, isolating failures so one cannot end the run."""
        results: list[OCRResult] = []
        for provider in self.providers:
            key = provider.provider_id
            self.log.start_stage(key, f"{provider.provider_name} started")
            try:
                result = provider.process(document)
            except Exception as exc:  # noqa: BLE001 - a provider bug must not
                # take down a run the user has already waited minutes for.
                logger.exception("Provider %s raised", key)
                result = OCRResult.failure(
                    provider.provider_id,
                    provider.provider_name,
                    f"The engine failed unexpectedly: {exc}",
                )

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

        usable = [r for r in results if r.succeeded and r.text.strip()]
        if len(usable) < 2:
            reason = (
                "fewer than two engines produced text"
                if usable
                else "no engine produced text"
            )
            self.log.skip_stage(STAGE_COMPARISON, reason)
            return None

        self.log.start_stage(STAGE_COMPARISON, "Comparing engine outputs")
        first, second = usable[0], usable[1]
        comparison = compare_texts(
            first.text,
            second.text,
            engine_a=first.provider_name,
            engine_b=second.provider_name,
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

    def _stage_fusion(self, results: list[OCRResult]):
        if not self.settings.pipeline.run_fusion:
            return None
        if STAGE_FUSION not in self.log.stages:
            return None

        usable = [r for r in results if r.succeeded and r.text.strip()]
        if not usable:
            self.log.skip_stage(STAGE_FUSION, "no engine produced text")
            return None

        import time

        self.log.start_stage(STAGE_FUSION, "Generating final result")
        started = time.perf_counter()
        try:
            fusion = fuse(results, self.settings)
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
    return OCRPipeline(settings, providers, log=log)


__all__ = [
    "STAGE_COMPARISON",
    "STAGE_FUSION",
    "STAGE_UPLOAD",
    "OCRPipeline",
    "build_pipeline",
]
