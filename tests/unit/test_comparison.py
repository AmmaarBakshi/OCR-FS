"""Comparison: alignment, agreement scoring and numeric conflict detection."""

from __future__ import annotations

import time

import pytest

from ocr_fusion.pipeline.comparison import (
    DiffKind,
    compare_texts,
    extract_numbers,
    normalise_line,
    similarity_ratio,
)
from tests.conftest import INVOICE_A, INVOICE_B


class TestAgreement:
    def test_identical_text_is_full_agreement(self):
        result = compare_texts("alpha\nbeta\ngamma", "alpha\nbeta\ngamma")
        assert result.agreement_percent == 100.0
        assert result.equal_lines == 3
        assert result.differing_lines == 0

    def test_completely_different_text_scores_zero(self):
        assert compare_texts("alpha\nbeta", "zulu\nxray").agreement_percent == 0.0

    def test_partial_overlap_scores_between(self):
        result = compare_texts(INVOICE_A, INVOICE_B)
        assert 0.0 < result.similarity < 1.0

    def test_table_padding_is_not_a_disagreement(self):
        # One engine pads table cells, the other does not. That is formatting.
        result = compare_texts("| a | b |", "|  a  |   b   |")
        assert result.equal_lines == 1
        assert result.differing_lines == 0

    def test_blank_lines_are_ignored(self):
        assert compare_texts("a\n\n\nb", "a\nb").equal_lines == 2


class TestAlignment:
    def test_matches_lines_across_an_inserted_line(self):
        # The bug this guards: an extra line on one side used to shift every
        # later line, so matching rows were reported as unrelated.
        a = "header\nrow one\nrow two"
        b = "header\nEXTRA LINE\nrow one\nrow two"
        result = compare_texts(a, b)
        assert result.only_b_lines == 1
        assert result.equal_lines == 3

    def test_near_identical_lines_are_paired_as_changed(self):
        result = compare_texts(
            "| Insurance premium | 1 | 1,325.50 |",
            "| Insurance premium | 1 | 1,325.60 |",
        )
        assert result.changed_lines == 1
        assert result.lines[0].kind is DiffKind.CHANGED
        assert result.lines[0].similarity > 0.9

    def test_unrelated_lines_are_kept_separate(self):
        result = compare_texts("completely different text here", "nothing alike at all")
        assert result.changed_lines == 0
        assert result.only_a_lines == 1
        assert result.only_b_lines == 1

    def test_one_sided_input_is_reported(self):
        result = compare_texts("hello", "")
        assert result.note
        assert "nothing to compare" in result.note

    def test_two_empty_inputs(self):
        assert "Neither engine" in compare_texts("", "").note

    def test_disagreements_excludes_equal_rows(self):
        result = compare_texts("same\ndiffer a", "same\ndiffer b")
        assert all(row.kind is not DiffKind.EQUAL for row in result.disagreements())


class TestNumericConflicts:
    def test_detects_a_changed_figure(self):
        # The error that actually costs money on an invoice.
        result = compare_texts(
            "| Insurance premium | 1,325.50 |", "| Insurance premium | 1,325.60 |"
        )
        assert len(result.numeric_conflicts) == 1
        assert "1325.50" in result.numeric_conflicts[0]["numbers_a"]
        assert "1325.60" in result.numeric_conflicts[0]["numbers_b"]

    def test_thousands_separators_are_not_a_conflict(self):
        assert extract_numbers("1,250.00") == extract_numbers("1250.00")

    def test_identical_figures_produce_no_conflict(self):
        result = compare_texts("Total 38,085.10 due", "Total: 38,085.10 due")
        assert result.numeric_conflicts == []

    def test_conflicts_found_in_the_realistic_pair(self):
        result = compare_texts(INVOICE_A, INVOICE_B)
        flattened = str(result.numeric_conflicts)
        assert "1325.60" in flattened


class TestPerformance:
    @pytest.mark.parametrize("rows", [200, 800])
    def test_large_documents_compare_quickly(self, rows):
        # Guards the regression where whole-document SequenceMatcher made this
        # O(n*m) on characters and took about a minute.
        a = "\n".join(f"row {i} value {i * 7},00 description text" for i in range(rows))
        b = "\n".join(f"row {i} value {i * 7},01 description text" for i in range(rows))
        started = time.perf_counter()
        result = compare_texts(a, b)
        assert time.perf_counter() - started < 5.0
        assert result.changed_lines == rows

    def test_similarity_stays_meaningful_at_scale(self):
        a = "\n".join(f"identical row {i}" for i in range(300))
        assert compare_texts(a, a).agreement_percent == 100.0


class TestHelpers:
    def test_normalise_line_collapses_whitespace(self):
        assert normalise_line("  a   b  ") == "a b"

    def test_similarity_bounds(self):
        assert similarity_ratio("abc", "abc") == 1.0
        assert similarity_ratio("abc", "") == 0.0
        assert similarity_ratio("", "") == 1.0

    def test_extract_numbers_finds_amounts_and_ids(self):
        numbers = extract_numbers("Invoice INV-1024 total 38,085.10 on 14/03/2025")
        assert "38085.10" in numbers
        assert "1024" in numbers


class TestSerialisation:
    def test_as_dict_is_json_safe(self):
        import json

        json.dumps(compare_texts(INVOICE_A, INVOICE_B).as_dict())

    def test_as_dict_carries_the_counts(self):
        data = compare_texts(INVOICE_A, INVOICE_B).as_dict()
        for key in ("agreement_percent", "equal_lines", "changed_lines", "numeric_conflicts"):
            assert key in data
