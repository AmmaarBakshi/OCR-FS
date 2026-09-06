"""Display-ready metrics for the UI and reports."""

from ocr_fusion.metrics.collector import (
    NOT_AVAILABLE,
    Metric,
    engine_metrics,
    format_duration,
    format_int,
    format_percent,
    format_rate,
    stage_timings,
    total_metrics,
)

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
