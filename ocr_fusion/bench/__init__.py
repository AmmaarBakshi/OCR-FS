"""Measurement tools for the pipeline.

Kept inside the framework rather than in a scripts folder because a
performance claim that cannot be re-run is not a claim. Every optimisation in
this project is supposed to be answerable with a before-and-after number from
here, on the operator's own documents.

Nothing in this package is imported by the pipeline or the UI; it depends on
them, never the other way round.
"""

from __future__ import annotations

from ocr_fusion.bench.accuracy import (
    GoldPage,
    GoldSet,
    PageScore,
    build_gold_set,
    character_error_rate,
    figure_recall,
    score_page,
    word_error_rate,
    word_recall,
)
from ocr_fusion.bench.runner import (
    BenchmarkReport,
    DocumentTiming,
    PageTiming,
    benchmark_accuracy,
    benchmark_documents,
    settings_summary,
)

__all__ = [
    "BenchmarkReport",
    "DocumentTiming",
    "GoldPage",
    "GoldSet",
    "PageScore",
    "PageTiming",
    "benchmark_accuracy",
    "benchmark_documents",
    "build_gold_set",
    "character_error_rate",
    "figure_recall",
    "score_page",
    "settings_summary",
    "word_error_rate",
    "word_recall",
]
