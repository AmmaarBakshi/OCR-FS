"""Display-ready metrics.

Turns a :class:`~ocr_fusion.pipeline.result.PipelineResult` into rows the UI can
render directly. Formatting lives here rather than in the UI so that the "never
fabricate a metric" rule (spec s8) is enforced in one place: a value the engine
did not report becomes the string ``"N/A"``, never ``0``.

The Output settings decide which rows exist at all, so a client demo can hide
token counts entirely while a developer session shows everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ocr_fusion.config.schema import AppSettings
from ocr_fusion.ocr.interface import OCRResult
from ocr_fusion.pipeline.result import PipelineResult

NOT_AVAILABLE = "N/A"


@dataclass(slots=True)
class Metric:
    """One label/value pair for display."""

    label: str
    value: str
    help_text: str = ""

    @property
    def is_available(self) -> bool:
        return self.value != NOT_AVAILABLE


def format_duration(seconds: float | None) -> str:
    """Human-readable duration. Sub-minute values keep two decimals."""
    if seconds is None:
        return NOT_AVAILABLE
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.0f}s"


def format_int(value: int | None) -> str:
    """Thousands-separated integer, or N/A when the engine reported nothing."""
    return NOT_AVAILABLE if value is None else f"{value:,}"


def format_rate(value: float | None, unit: str = "tok/s") -> str:
    return NOT_AVAILABLE if value is None else f"{value:.1f} {unit}"


def format_percent(value: float | None) -> str:
    return NOT_AVAILABLE if value is None else f"{value:.1f}%"


def engine_metrics(result: OCRResult, settings: AppSettings) -> list[Metric]:
    """Per-engine metrics, filtered by the Output settings (spec s8)."""
    output = settings.output
    metrics: list[Metric] = []

    if output.show_engine_name:
        metrics.append(Metric("Engine", result.provider_name))
    if output.show_model_name:
        metrics.append(Metric("Model", result.model_name or NOT_AVAILABLE))
        if result.backend:
            metrics.append(
                Metric(
                    "Backend",
                    result.backend,
                    "How this engine was executed.",
                )
            )
    metrics.append(Metric("Status", result.status.value.title()))
    metrics.append(
        Metric(
            "Processing",
            result.processing_location.value.title(),
            "Whether document pages left this machine.",
        )
    )

    if output.show_processing_time:
        metrics.append(Metric("Total time", format_duration(result.duration_seconds)))
        load = _sum_optional(p.load_duration_seconds for p in result.pages)
        inference = _sum_optional(p.inference_duration_seconds for p in result.pages)
        metrics.append(
            Metric(
                "Model load",
                format_duration(load),
                "Time spent loading the model into memory, when reported.",
            )
        )
        metrics.append(Metric("Inference", format_duration(inference)))
        if result.pages:
            metrics.append(
                Metric(
                    "Per page",
                    format_duration(result.duration_seconds / len(result.pages)),
                )
            )

    if output.show_token_usage:
        usage = result.tokens
        metrics.append(Metric("Input tokens", format_int(usage.input_tokens)))
        metrics.append(Metric("Output tokens", format_int(usage.output_tokens)))
        metrics.append(Metric("Total tokens", format_int(usage.total_tokens)))
        metrics.append(Metric("Throughput", format_rate(result.tokens_per_second)))

    metrics.append(Metric("Pages", format_int(len(result.pages))))
    if output.show_character_count:
        metrics.append(Metric("Characters", format_int(result.character_count)))
    if output.show_word_count:
        metrics.append(Metric("Words", format_int(result.word_count)))

    if output.show_confidence:
        confidence = _average_optional(p.confidence for p in result.pages)
        metrics.append(
            Metric(
                "Confidence",
                NOT_AVAILABLE if confidence is None else format_percent(confidence * 100),
                "Vision-language models do not report a confidence score.",
            )
        )

    return metrics


def total_metrics(result: PipelineResult, settings: AppSettings) -> list[Metric]:
    """Run-level totals (spec s8, 'Total Processing')."""
    output = settings.output
    data = result.metrics()
    metrics: list[Metric] = [
        Metric("Total time", format_duration(data["total_duration_seconds"])),
        Metric("Pages processed", format_int(data["pages_processed"])),
        Metric(
            "Average per page",
            format_duration(data["average_seconds_per_page"]),
        ),
        Metric(
            "OCR engines",
            f"{data['engines_succeeded']} of {data['engines_run']} succeeded",
        ),
    ]
    if output.show_token_usage:
        metrics.append(Metric("Total tokens", format_int(data["total_tokens"])))
        metrics.append(Metric("Input tokens", format_int(data["input_tokens"])))
        metrics.append(Metric("Output tokens", format_int(data["output_tokens"])))
    if output.show_character_count:
        metrics.append(Metric("Characters", format_int(data["final_character_count"])))
    if output.show_word_count:
        metrics.append(Metric("Words", format_int(data["final_word_count"])))
    if result.comparison is not None:
        metrics.append(
            Metric("Engine agreement", format_percent(result.comparison.agreement_percent))
        )
    return metrics


def stage_timings(result: PipelineResult) -> list[dict[str, Any]]:
    """Per-stage timings for the pipeline visualisation (spec s4)."""
    return [
        {
            "label": stage.label,
            "status": stage.status.value,
            "symbol": stage.status.symbol,
            "duration": format_duration(stage.duration_seconds),
            "detail": stage.detail,
        }
        for stage in result.stages
    ]


def _sum_optional(values) -> float | None:
    """Sum values, returning None when none of them were reported."""
    collected = [value for value in values if value is not None]
    return sum(collected) if collected else None


def _average_optional(values) -> float | None:
    collected = [value for value in values if value is not None]
    return sum(collected) / len(collected) if collected else None


__all__ = [
    "NOT_AVAILABLE",
    "Metric",
    "engine_metrics",
    "format_duration",
    "format_int",
    "format_percent",
    "format_rate",
    "stage_timings",
    "total_metrics",
]
