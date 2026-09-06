"""Metric formatting - above all, that nothing is ever fabricated."""

from __future__ import annotations

import pytest

from ocr_fusion.metrics import (
    NOT_AVAILABLE,
    engine_metrics,
    format_duration,
    format_int,
    format_percent,
    format_rate,
    stage_timings,
    total_metrics,
)
from ocr_fusion.ocr.interface import OCRResult, PageResult, TokenUsage
from ocr_fusion.pipeline.events import Stage, StageStatus
from ocr_fusion.pipeline.result import PipelineResult


@pytest.fixture
def vlm_result() -> OCRResult:
    return OCRResult(
        provider_id="qwen_vl",
        provider_name="Qwen2.5-VL",
        model_name="qwen2.5vl:3b",
        backend="ollama",
        duration_seconds=72.73,
        pages=[
            PageResult(
                1,
                "some text here",
                tokens=TokenUsage(3037, 378),
                load_duration_seconds=0.3,
                inference_duration_seconds=67.6,
            )
        ],
    )


@pytest.fixture
def classical_result() -> OCRResult:
    """An engine with no notion of tokens, load time or inference time."""
    return OCRResult(
        provider_id="tesseract",
        provider_name="Tesseract",
        model_name="tesseract:eng",
        backend="cli",
        duration_seconds=1.9,
        pages=[PageResult(1, "some text here")],
    )


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds,expected",
        [(0.5, "0.50s"), (1.9, "1.90s"), (59.9, "59.90s"), (72.7, "1m 13s"), (None, NOT_AVAILABLE)],
    )
    def test_durations(self, seconds, expected):
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize("value,expected", [(0, "0"), (3037, "3,037"), (None, NOT_AVAILABLE)])
    def test_integers(self, value, expected):
        assert format_int(value) == expected

    def test_zero_is_not_treated_as_missing(self):
        # A real zero and an unreported metric are different facts.
        assert format_int(0) == "0"
        assert format_int(None) == NOT_AVAILABLE

    def test_rates_and_percentages(self):
        assert format_rate(5.234) == "5.2 tok/s"
        assert format_rate(None) == NOT_AVAILABLE
        assert format_percent(86.25) == "86.2%"
        assert format_percent(None) == NOT_AVAILABLE


class TestEngineMetrics:
    def test_reports_real_values(self, vlm_result, settings):
        values = {m.label: m.value for m in engine_metrics(vlm_result, settings)}
        assert values["Model"] == "qwen2.5vl:3b"
        assert values["Input tokens"] == "3,037"
        assert values["Output tokens"] == "378"
        assert values["Total tokens"] == "3,415"

    def test_unreported_metrics_are_na(self, classical_result, settings):
        values = {m.label: m.value for m in engine_metrics(classical_result, settings)}
        for label in ("Input tokens", "Output tokens", "Total tokens", "Throughput", "Model load"):
            assert values[label] == NOT_AVAILABLE

    def test_no_metric_is_silently_zero(self, classical_result, settings):
        # The failure this guards: rendering 0 tokens for an engine that simply
        # does not count tokens.
        for metric in engine_metrics(classical_result, settings):
            if metric.label.endswith("tokens"):
                assert metric.value == NOT_AVAILABLE

    def test_confidence_is_na_for_vision_models(self, vlm_result, settings):
        values = {m.label: m.value for m in engine_metrics(vlm_result, settings)}
        assert values["Confidence"] == NOT_AVAILABLE

    def test_is_available_flag_matches_the_value(self, classical_result, settings):
        for metric in engine_metrics(classical_result, settings):
            assert metric.is_available == (metric.value != NOT_AVAILABLE)

    @pytest.mark.parametrize(
        "flag,hidden",
        [
            ("show_token_usage", "Input tokens"),
            ("show_processing_time", "Total time"),
            ("show_character_count", "Characters"),
            ("show_word_count", "Words"),
            ("show_confidence", "Confidence"),
        ],
    )
    def test_output_toggles_remove_rows(self, vlm_result, settings, flag, hidden):
        setattr(settings.output, flag, False)
        assert hidden not in {m.label for m in engine_metrics(vlm_result, settings)}


class TestTotals:
    def test_aggregates_across_engines(self, vlm_result, classical_result, settings, single_page_document):
        result = PipelineResult(
            document=single_page_document,
            engine_results=[vlm_result, classical_result],
            total_duration_seconds=74.8,
        )
        values = {m.label: m.value for m in total_metrics(result, settings)}
        assert values["OCR engines"] == "2 of 2 succeeded"
        assert values["Total tokens"] == "3,415"

    def test_totals_are_na_when_no_engine_reports_tokens(
        self, classical_result, settings, single_page_document
    ):
        result = PipelineResult(
            document=single_page_document,
            engine_results=[classical_result],
            total_duration_seconds=1.9,
        )
        values = {m.label: m.value for m in total_metrics(result, settings)}
        assert values["Total tokens"] == NOT_AVAILABLE

    def test_average_per_page(self, vlm_result, settings, multi_page_document):
        result = PipelineResult(
            document=multi_page_document,
            engine_results=[vlm_result],
            total_duration_seconds=90.0,
        )
        values = {m.label: m.value for m in total_metrics(result, settings)}
        assert values["Average per page"] == "30.00s"

    def test_failed_engines_are_counted(self, settings, single_page_document):
        result = PipelineResult(
            document=single_page_document,
            engine_results=[OCRResult.failure("a", "A", "down")],
            total_duration_seconds=0.2,
        )
        values = {m.label: m.value for m in total_metrics(result, settings)}
        assert values["OCR engines"] == "0 of 1 succeeded"


class TestStageTimings:
    def test_maps_stages_for_the_visualisation(self, single_page_document):
        stages = [
            Stage("upload", "Document", StageStatus.COMPLETED, 0.12),
            Stage("qwen_vl", "Qwen2.5-VL", StageStatus.COMPLETED, 72.7),
            Stage("unlimited_ocr", "Unlimited-OCR", StageStatus.FAILED, 1.1),
            Stage("comparison", "Comparison", StageStatus.SKIPPED),
        ]
        rows = stage_timings(
            PipelineResult(document=single_page_document, stages=stages)
        )
        assert [r["duration"] for r in rows] == ["0.12s", "1m 13s", "1.10s", NOT_AVAILABLE]
        assert rows[2]["status"] == "failed"
        assert all(r["symbol"] for r in rows)
