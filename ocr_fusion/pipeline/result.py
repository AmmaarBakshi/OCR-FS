"""The result of one pipeline run.

:class:`PipelineResult` is the single object the UI renders and the exporters
serialise. Its :meth:`PipelineResult.as_dict` shape is the documented, stable
JSON schema other projects consume (spec s10), so the top-level keys are fixed:
``document``, ``pipeline``, ``engines``, ``comparison``, ``final_result`` and
``metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ocr_fusion.documents.models import Document
from ocr_fusion.ocr.interface import OCRResult, OCRStatus
from ocr_fusion.pipeline.comparison import ComparisonResult
from ocr_fusion.pipeline.events import EventLog, Stage
from ocr_fusion.pipeline.fusion import FusionResult

#: Version of the export schema. Bump on any breaking shape change.
SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class PipelineResult:
    """Everything one run produced."""

    document: Document
    engine_results: list[OCRResult] = field(default_factory=list)
    comparison: ComparisonResult | None = None
    fusion: FusionResult | None = None
    stages: list[Stage] = field(default_factory=list)
    log: EventLog | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_duration_seconds: float = 0.0
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    """Redacted configuration used for this run (spec s16)."""

    @property
    def final_text(self) -> str:
        """The text the user is shown as the final result.

        Falls back to the first successful engine when fusion did not run, so
        there is always something to display and export.
        """
        if self.fusion and self.fusion.text.strip():
            return self.fusion.text
        for result in self.engine_results:
            if result.succeeded and result.text.strip():
                return result.text
        return ""

    @property
    def succeeded(self) -> bool:
        """True when at least one engine produced text."""
        return any(r.succeeded and r.text.strip() for r in self.engine_results)

    @property
    def failed_engines(self) -> list[OCRResult]:
        return [r for r in self.engine_results if r.status is OCRStatus.FAILED]

    @property
    def successful_engines(self) -> list[OCRResult]:
        return [r for r in self.engine_results if r.succeeded]

    def engine(self, provider_id: str) -> OCRResult | None:
        return next(
            (r for r in self.engine_results if r.provider_id == provider_id), None
        )

    # -- export ------------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        """Aggregate run metrics. Unknown values stay None so the UI shows N/A."""
        input_tokens: int | None = None
        output_tokens: int | None = None
        for result in self.engine_results:
            usage = result.tokens
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens

        pages = self.document.page_count
        total_tokens = (
            None
            if input_tokens is None and output_tokens is None
            else (input_tokens or 0) + (output_tokens or 0)
        )
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "pages_processed": pages,
            "average_seconds_per_page": (
                self.total_duration_seconds / pages if pages else None
            ),
            "engines_run": len(self.engine_results),
            "engines_succeeded": len(self.successful_engines),
            "engines_failed": len(self.failed_engines),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "final_character_count": len(self.final_text),
            "final_word_count": len(self.final_text.split()),
        }

    def as_dict(self, *, include_logs: bool = True) -> dict[str, Any]:
        """The stable JSON export shape (spec s10)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.started_at.isoformat(),
            "document": self.document.metadata(),
            "pipeline": {
                "stages": [stage.as_dict() for stage in self.stages],
                "total_duration_seconds": self.total_duration_seconds,
                "succeeded": self.succeeded,
            },
            "engines": [
                {
                    **result.summary(),
                    "text": result.text,
                    "pages": [
                        {
                            "page": page.page_number,
                            "status": page.status.value,
                            "text": page.text,
                            "error": page.error,
                            "duration_seconds": page.duration_seconds,
                            "character_count": page.character_count,
                            "word_count": page.word_count,
                            "tokens": page.tokens.as_dict(),
                            "confidence": page.confidence,
                        }
                        for page in result.pages
                    ],
                }
                for result in self.engine_results
            ],
            "comparison": self.comparison.as_dict() if self.comparison else None,
            "final_result": {
                "text": self.final_text,
                "character_count": len(self.final_text),
                "word_count": len(self.final_text.split()),
                **(self.fusion.as_dict() if self.fusion else {}),
            },
            "metrics": self.metrics(),
            "configuration": self.settings_snapshot,
            "logs": (
                self.log.as_dicts() if (include_logs and self.log is not None) else []
            ),
        }


__all__ = ["SCHEMA_VERSION", "PipelineResult"]
