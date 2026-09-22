"""The measurement tools themselves.

A harness that reports a number nobody checked is worse than no harness, so
the metrics are tested against cases with known answers, and the reporting is
tested for the one property that matters most: it must say "n/a" rather than
zero for anything it could not observe.
"""

from __future__ import annotations

import pytest

from ocr_fusion.bench import (
    BenchmarkReport,
    DocumentTiming,
    PageTiming,
    build_gold_set,
    character_error_rate,
    figure_recall,
    score_page,
    settings_summary,
    word_error_rate,
    word_recall,
)
from ocr_fusion.bench.accuracy import figures, normalise
from ocr_fusion.config.schema import AppSettings

STATEMENT = "Deposits and other additions 05/18/21 Online transfer 2,200.00 Total 38,085.10"


class TestMetrics:
    def test_a_perfect_transcription_scores_zero_error(self):
        assert word_error_rate(STATEMENT, STATEMENT) == 0.0
        assert character_error_rate(STATEMENT, STATEMENT) == 0.0
        assert word_recall(STATEMENT, STATEMENT) == 1.0
        assert figure_recall(STATEMENT, STATEMENT) == 1.0

    def test_nothing_transcribed_scores_total_failure(self):
        assert word_error_rate(STATEMENT, "") == 1.0
        assert word_recall(STATEMENT, "") == 0.0
        assert figure_recall(STATEMENT, "") == 0.0

    def test_punctuation_and_case_are_not_counted_as_errors(self):
        # Engines legitimately differ on dashes and capitals; marking that
        # wrong would bury the errors that matter.
        assert word_error_rate("Total Due: 38,085.10", "total due 38,085.10") == 0.0

    def test_a_mistranscribed_figure_is_caught(self):
        # O for 0 is the classic OCR failure, and on a statement it is the one
        # that costs money.
        damaged = STATEMENT.replace("2,200.00", "2,2OO.OO")
        assert figure_recall(STATEMENT, damaged) < 1.0

    def test_reordering_is_told_apart_from_misreading(self):
        shuffled = " ".join(reversed(STATEMENT.split()))
        assert word_recall(STATEMENT, shuffled) == 1.0
        assert word_error_rate(STATEMENT, shuffled) > 0.5

    def test_figures_keeps_separators(self):
        assert "38,085.10" in figures(STATEMENT)

    def test_normalise_keeps_what_carries_meaning(self):
        folded = normalise("Invoice #INV-1024 — $1,325.50 (due 05/18/21)")
        assert "1,325.50" in folded and "05/18/21" in folded

    def test_a_very_long_page_reports_no_cer_rather_than_stalling(self):
        # Character alignment is O(n*m); refusing is better than hanging.
        huge = "word " * 6000
        assert character_error_rate(huge, huge) is None

    def test_score_page_reports_both_sides(self):
        score = score_page(1, STATEMENT, STATEMENT)
        assert score.reference_words == score.hypothesis_words
        assert score.as_dict()["wer"] == 0.0


class TestGoldSet:
    def test_pages_with_a_text_layer_become_ground_truth(self, tmp_path):
        import fitz

        document = fitz.open()
        page = document.new_page()
        for line in range(14):
            page.insert_text((72, 90 + line * 16), f"Line {line}: total 1,250.00 due 05/18/21", fontsize=10)
        path = tmp_path / "statement.pdf"
        path.write_bytes(document.tobytes())
        document.close()

        gold = build_gold_set([path], min_words=10)
        assert len(gold) == 1
        assert "1,250.00" in gold.pages[0].reference

    def test_a_scan_is_counted_as_having_no_answer_key(self, tmp_path):
        import fitz

        document = fitz.open()
        document.new_page()  # a blank page has no text layer
        path = tmp_path / "scan.pdf"
        path.write_bytes(document.tobytes())
        document.close()

        gold = build_gold_set([path])
        assert len(gold) == 0
        assert gold.skipped_without_truth == 1

    def test_one_long_document_cannot_dominate_the_set(self, tmp_path):
        import fitz

        document = fitz.open()
        for _ in range(10):
            page = document.new_page()
            for line in range(14):
                page.insert_text((72, 90 + line * 16), f"Line {line} total 1,250.00", fontsize=10)
        path = tmp_path / "long.pdf"
        path.write_bytes(document.tobytes())
        document.close()

        gold = build_gold_set([path], min_words=10, max_pages_per_document=3)
        assert len(gold) == 3

    def test_a_corrupt_file_is_skipped_not_fatal(self, tmp_path):
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 nonsense")
        assert len(build_gold_set([broken])) == 0


class TestReport:
    def _report(self) -> BenchmarkReport:
        report = BenchmarkReport(label="test", settings_summary={})
        report.documents.append(
            DocumentTiming(
                filename="a.pdf",
                pages=4,
                seconds=120.0,
                load_seconds=0.5,
                pages_needing_ocr=1,
                pages_avoided=3,
                page_timings=[
                    PageTiming(1, "qwen_vl", 100.0, 300),
                    PageTiming(2, "text_layer", 0.001, 400),
                    PageTiming(3, "text_layer", None, 400),
                ],
            )
        )
        return report

    def test_totals_add_up(self):
        report = self._report()
        assert report.total_pages == 4
        assert report.total_ocr_pages == 1
        assert report.pages_per_minute == pytest.approx(2.0)

    def test_percentiles_expose_the_slow_page(self):
        # One difficult page dominates a document, so a mean alone hides the
        # thing worth optimising.
        spread = self._report().percentiles()
        assert spread["max"] == 100.0
        assert spread["median"] < spread["max"]

    def test_an_unmeasured_page_is_not_counted_as_zero(self):
        report = self._report()
        # Three page timings, but one had no duration to report.
        assert len(report.page_seconds()) == 2

    def test_percentiles_are_none_when_nothing_was_timed(self):
        empty = BenchmarkReport(label="test", settings_summary={})
        assert empty.percentiles()["mean"] is None

    def test_peak_memory_is_none_rather_than_zero_when_unobserved(self):
        report = self._report()
        assert report.as_dict()["totals"]["peak_rss_mb"] is None

    def test_the_report_records_what_moved_the_number(self):
        summary = settings_summary(AppSettings())
        for key in ("engine_mode", "routing_enabled", "pdf_render_dpi", "fallback_enabled"):
            assert key in summary
