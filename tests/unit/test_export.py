"""Export formats and the stable JSON schema."""

from __future__ import annotations

import csv
import io
import json

import pytest

from ocr_fusion.config.schema import OutputFormat
from ocr_fusion.export import MIME_TYPES, export, export_filename
from ocr_fusion.metrics import NOT_AVAILABLE
from ocr_fusion.ocr.interface import OCRResult, OCRStatus, PageResult, TokenUsage
from ocr_fusion.pipeline.comparison import compare_texts
from ocr_fusion.pipeline.fusion import FusionResult
from ocr_fusion.pipeline.result import SCHEMA_VERSION, PipelineResult
from tests.conftest import INVOICE_A, INVOICE_B


@pytest.fixture
def run_result(multi_page_document) -> PipelineResult:
    """A run with one metric-rich engine and one that reports no tokens."""
    rich = OCRResult(
        provider_id="qwen_vl",
        provider_name="Qwen2.5-VL (qwen2.5vl:3b)",
        model_name="qwen2.5vl:3b",
        backend="ollama",
        duration_seconds=72.7,
        pages=[
            PageResult(1, INVOICE_A, tokens=TokenUsage(3037, 378), duration_seconds=72.7),
            PageResult(2, "page two text", tokens=TokenUsage(2900, 210), duration_seconds=60.1),
            PageResult(3, "", status=OCRStatus.FAILED, error="no text on this page"),
        ],
    )
    rich.status = OCRStatus.PARTIAL

    bare = OCRResult(
        provider_id="tesseract",
        provider_name="Tesseract (eng)",
        model_name="tesseract:eng",
        backend="cli",
        duration_seconds=1.9,
        pages=[PageResult(1, INVOICE_B, duration_seconds=1.9)],
    )

    return PipelineResult(
        document=multi_page_document,
        engine_results=[rich, bare],
        comparison=compare_texts(
            INVOICE_A, INVOICE_B, engine_a="Qwen2.5-VL", engine_b="Tesseract (eng)"
        ),
        fusion=FusionResult(
            text=INVOICE_A + "\nBank: HDFC Bank",
            strategy="line_vote",
            sources=["Qwen2.5-VL", "Tesseract (eng)"],
            detail="4 agreed, 2 reconciled",
        ),
        total_duration_seconds=134.7,
        settings_snapshot={"qwen": {"model": "qwen2.5vl:3b"}},
    )


class TestAllFormats:
    @pytest.mark.parametrize("output_format", list(OutputFormat))
    def test_produces_non_empty_output(self, run_result, settings, output_format):
        assert export(run_result, settings, output_format).strip()

    @pytest.mark.parametrize("output_format", list(OutputFormat))
    def test_has_a_mime_type(self, output_format):
        assert MIME_TYPES[output_format]

    @pytest.mark.parametrize("output_format", list(OutputFormat))
    def test_filename_uses_the_document_and_extension(self, run_result, output_format):
        name = export_filename(run_result, output_format)
        assert name.endswith(f".{output_format.value}")
        assert "report" in name

    def test_filename_is_filesystem_safe(self, run_result, settings):
        run_result.document.filename = 'in/valid:name*?.pdf'
        name = export_filename(run_result, OutputFormat.TXT)
        assert not set(name) & set('/\\:*?"<>|')

    @pytest.mark.parametrize("output_format", list(OutputFormat))
    def test_export_survives_a_failed_run(self, settings, single_page_document, output_format):
        empty = PipelineResult(
            document=single_page_document,
            engine_results=[OCRResult.failure("a", "A", "engine down")],
            total_duration_seconds=0.4,
        )
        assert export(empty, settings, output_format) is not None


class TestTxt:
    def test_contains_the_final_text(self, run_result, settings):
        assert "ACME LOGISTICS LTD" in export(run_result, settings, OutputFormat.TXT)

    def test_header_can_be_switched_off(self, run_result, settings):
        settings.output.show_engine_name = False
        settings.output.show_model_name = False
        settings.output.show_processing_time = False
        settings.output.show_character_count = False
        settings.output.show_word_count = False
        text = export(run_result, settings, OutputFormat.TXT)
        assert "Engines:" not in text

    def test_page_numbers_appear_for_multi_page_documents(self, run_result, settings):
        assert "--- Page 1 ---" in export(run_result, settings, OutputFormat.TXT)


class TestMarkdown:
    def test_has_sections(self, run_result, settings):
        text = export(run_result, settings, OutputFormat.MARKDOWN)
        assert "# OCR Result" in text
        assert "## Final result" in text
        assert "## Engine outputs" in text

    def test_reports_na_for_missing_metrics(self, run_result, settings):
        text = export(run_result, settings, OutputFormat.MARKDOWN)
        assert NOT_AVAILABLE in text  # the token-less engine

    def test_failed_engine_is_explained(self, settings, single_page_document):
        failed = PipelineResult(
            document=single_page_document,
            engine_results=[OCRResult.failure("a", "A", "Ollama is not running")],
        )
        assert "Ollama is not running" in export(failed, settings, OutputFormat.MARKDOWN)

    def test_pipe_characters_are_escaped_in_tables(self, settings, single_page_document):
        a = OCRResult("a", "A", pages=[PageResult(1, "| x | 1 |")])
        b = OCRResult("b", "B", pages=[PageResult(1, "| x | 2 |")])
        run = PipelineResult(
            document=single_page_document,
            engine_results=[a, b],
            comparison=compare_texts("| x | 1 |", "| x | 2 |"),
        )
        text = export(run, settings, OutputFormat.MARKDOWN)
        assert "\\|" in text

    def test_comparison_can_be_hidden(self, run_result, settings):
        settings.output.show_comparison = False
        assert "## Comparison" not in export(run_result, settings, OutputFormat.MARKDOWN)


class TestJson:
    def test_is_valid_json(self, run_result, settings):
        json.loads(export(run_result, settings, OutputFormat.JSON))

    def test_schema_keys_are_stable(self, run_result, settings):
        # Other projects consume this shape, so the top level is a contract.
        payload = json.loads(export(run_result, settings, OutputFormat.JSON))
        for key in (
            "schema_version",
            "document",
            "pipeline",
            "engines",
            "comparison",
            "final_result",
            "metrics",
        ):
            assert key in payload
        assert payload["schema_version"] == SCHEMA_VERSION

    def test_unreported_metrics_are_null_not_zero(self, run_result, settings):
        payload = json.loads(export(run_result, settings, OutputFormat.JSON))
        tesseract = next(e for e in payload["engines"] if e["provider_id"] == "tesseract")
        assert tesseract["tokens"]["total_tokens"] is None
        assert tesseract["pages"][0]["tokens"]["input_tokens"] is None

    def test_reported_metrics_are_present(self, run_result, settings):
        payload = json.loads(export(run_result, settings, OutputFormat.JSON))
        qwen = next(e for e in payload["engines"] if e["provider_id"] == "qwen_vl")
        assert qwen["tokens"]["input_tokens"] == 5937

    def test_page_numbers_are_preserved(self, run_result, settings):
        payload = json.loads(export(run_result, settings, OutputFormat.JSON))
        qwen = next(e for e in payload["engines"] if e["provider_id"] == "qwen_vl")
        assert [p["page"] for p in qwen["pages"]] == [1, 2, 3]

    def test_raw_results_can_be_excluded(self, run_result, settings):
        settings.output.show_raw_results = False
        payload = json.loads(export(run_result, settings, OutputFormat.JSON))
        assert "text" not in payload["engines"][0]

    def test_logs_excluded_when_switched_off(self, run_result, settings):
        settings.output.show_processing_logs = False
        assert json.loads(export(run_result, settings, OutputFormat.JSON))["logs"] == []

    def test_configuration_snapshot_is_included(self, run_result, settings):
        payload = json.loads(export(run_result, settings, OutputFormat.JSON))
        assert payload["configuration"]["qwen"]["model"] == "qwen2.5vl:3b"


class TestCsv:
    def _rows(self, run_result, settings):
        return list(csv.DictReader(io.StringIO(export(run_result, settings, OutputFormat.CSV))))

    def test_one_row_per_page_per_engine_plus_the_final(self, run_result, settings):
        rows = self._rows(run_result, settings)
        assert len(rows) == 5  # 3 qwen pages + 1 tesseract page + final

    def test_multi_line_text_stays_in_one_cell(self, run_result, settings):
        rows = self._rows(run_result, settings)
        assert "\n" in rows[0]["text"]

    def test_missing_tokens_render_as_na(self, run_result, settings):
        rows = self._rows(run_result, settings)
        tesseract = next(r for r in rows if r["engine"] == "Tesseract (eng)")
        assert tesseract["input_tokens"] == NOT_AVAILABLE

    def test_page_numbers_are_present(self, run_result, settings):
        rows = self._rows(run_result, settings)
        assert [r["page"] for r in rows[:3]] == ["1", "2", "3"]

    def test_failed_page_carries_its_error(self, run_result, settings):
        rows = self._rows(run_result, settings)
        failed = next(r for r in rows if r["status"] == "failed")
        assert "no text" in failed["error"]

    def test_columns_follow_output_settings(self, run_result, settings):
        settings.output.show_token_usage = False
        rows = self._rows(run_result, settings)
        assert "input_tokens" not in rows[0]
