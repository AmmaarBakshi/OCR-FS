"""Fusion strategies and the guarantee that fusion never invents content."""

from __future__ import annotations

import pytest

from ocr_fusion.config.schema import FusionStrategy
from ocr_fusion.ocr.interface import OCRResult, OCRStatus, PageResult
from ocr_fusion.pipeline.comparison import extract_numbers
from ocr_fusion.pipeline.fusion import (
    FUSION_STRATEGIES,
    _verify_against_sources,
    fuse,
    fuse_line_vote,
    fuse_prefer_longest,
    fuse_prefer_primary,
)
from tests.conftest import INVOICE_A, INVOICE_B


def result_of(provider_id: str, name: str, text: str, *, failed: bool = False) -> OCRResult:
    result = OCRResult(provider_id, name, pages=[PageResult(1, text)])
    if failed:
        result.status = OCRStatus.FAILED
    return result


@pytest.fixture
def engine_a() -> OCRResult:
    return result_of("qwen_vl", "Qwen2.5-VL", INVOICE_A)


@pytest.fixture
def engine_b() -> OCRResult:
    return result_of("unlimited_ocr", "Unlimited-OCR", INVOICE_B)


class TestStrategySelection:
    def test_every_strategy_is_registered(self):
        assert set(FUSION_STRATEGIES) == set(FusionStrategy)

    @pytest.mark.parametrize("strategy", [s for s in FusionStrategy if s is not FusionStrategy.LLM])
    def test_deterministic_strategies_produce_text(self, settings, engine_a, engine_b, strategy):
        settings.pipeline.fusion_strategy = strategy
        assert fuse([engine_a, engine_b], settings).text.strip()

    def test_settings_select_the_strategy(self, settings, engine_a, engine_b):
        settings.pipeline.fusion_strategy = FusionStrategy.PREFER_LONGEST
        assert fuse([engine_a, engine_b], settings).strategy == "prefer_longest"


class TestPreferPrimary:
    def test_takes_the_first_engine_verbatim(self, settings, engine_a, engine_b):
        assert fuse_prefer_primary([engine_a, engine_b], settings).text == INVOICE_A

    def test_skips_a_failed_first_engine(self, settings, engine_b):
        failed = result_of("qwen_vl", "Qwen2.5-VL", "", failed=True)
        assert fuse_prefer_primary([failed, engine_b], settings).text == INVOICE_B


class TestPreferLongest:
    def test_takes_the_most_complete_output(self, settings, engine_a, engine_b):
        result = fuse_prefer_longest([engine_a, engine_b], settings)
        assert result.text == INVOICE_B  # the longer of the two
        assert "most complete" in result.detail


class TestLineVote:
    def test_keeps_agreed_lines(self, settings, engine_a, engine_b):
        result = fuse_line_vote([engine_a, engine_b], settings)
        assert "17 Harbour Road, Mumbai 400001" in result.text

    def test_recovers_a_line_only_one_engine_found(self, settings, engine_a, engine_b):
        # This is the reason to run two engines at all.
        assert "Bank: HDFC" in fuse_line_vote([engine_a, engine_b], settings).text

    def test_keeps_every_figure_the_engines_agree_on(self, settings, engine_a, engine_b):
        fused = set(extract_numbers(fuse_line_vote([engine_a, engine_b], settings).text))
        agreed = set(extract_numbers(INVOICE_A)) & set(extract_numbers(INVOICE_B))
        assert agreed - fused == set()

    def test_keeps_figures_only_one_engine_found(self, settings, engine_a, engine_b):
        # The bank account number appears only in the second engine's output.
        fused = set(extract_numbers(fuse_line_vote([engine_a, engine_b], settings).text))
        assert "50200012345678" in fused

    def test_a_contested_figure_resolves_to_one_reading(self, settings, engine_a, engine_b):
        # The engines read the same line as 1,325.50 and 1,325.60. Fusion has to
        # choose - keeping both would put two different totals in one result.
        # The disagreement is not hidden: comparison reports it as a numeric
        # conflict for the user to check.
        from ocr_fusion.pipeline.comparison import compare_texts

        fused = set(extract_numbers(fuse_line_vote([engine_a, engine_b], settings).text))
        assert len({"1325.50", "1325.60"} & fused) == 1
        assert compare_texts(INVOICE_A, INVOICE_B).numeric_conflicts

    def test_does_not_duplicate_repositioned_content(self, settings):
        # A real run produced this: one engine put the invoice header at the
        # end, the other at the top, and both copies were emitted.
        a = result_of("a", "A", "body line\nInvoice Number: INV-1024\nDate: 14 March 2025")
        b = result_of("b", "B", "Invoice Number: INV-1024\nDate: 14 March 2025\nbody line")
        text = fuse_line_vote([a, b], settings).text
        assert text.count("Invoice Number: INV-1024") == 1
        assert text.count("Date: 14 March 2025") == 1

    def test_keeps_a_line_unique_to_one_engine(self, settings):
        a = result_of("a", "A", "shared\nonly in A")
        b = result_of("b", "B", "shared")
        assert "only in A" in fuse_line_vote([a, b], settings).text

    def test_longer_variant_wins_a_disagreement(self, settings):
        # OCR truncates far more often than it adds words.
        a = result_of("a", "A", "Invoice No: INV-1024")
        b = result_of("b", "B", "Invoice Number: INV-1024")
        assert "Invoice Number: INV-1024" in fuse_line_vote([a, b], settings).text

    def test_is_deterministic(self, settings, engine_a, engine_b):
        first = fuse_line_vote([engine_a, engine_b], settings).text
        second = fuse_line_vote([engine_a, engine_b], settings).text
        assert first == second

    def test_detail_explains_what_happened(self, settings, engine_a, engine_b):
        detail = fuse_line_vote([engine_a, engine_b], settings).detail
        assert "agreed" in detail and "reconciled" in detail


class TestDegradedInput:
    def test_single_engine_is_passed_through(self, settings, engine_a):
        result = fuse_line_vote([engine_a], settings)
        assert result.text == INVOICE_A
        assert "Only" in result.detail

    def test_failed_engine_is_ignored(self, settings, engine_a):
        failed = result_of("b", "B", "", failed=True)
        assert fuse_line_vote([engine_a, failed], settings).text == INVOICE_A

    def test_no_engines_yields_empty_text_not_an_error(self, settings):
        result = fuse_line_vote([], settings)
        assert result.text == ""
        assert "nothing to reconcile" in result.detail

    def test_all_engines_empty(self, settings):
        empty = result_of("a", "A", "   ")
        assert fuse_line_vote([empty], settings).text == ""


class TestAntiHallucinationGuard:
    def test_accepts_a_faithful_merge(self):
        fused = INVOICE_B + "\nTOTAL DUE 38,085.10"
        assert _verify_against_sources(fused, INVOICE_A, INVOICE_B) is None

    def test_accepts_case_and_punctuation_normalisation(self):
        # Exactly what a reconciler is allowed to change.
        fused = INVOICE_A.upper().replace(":", " :")
        assert _verify_against_sources(fused, INVOICE_A, INVOICE_B) is None

    def test_rejects_a_summary(self):
        warning = _verify_against_sources("An invoice from ACME.", INVOICE_A, INVOICE_B)
        assert warning is not None

    def test_rejects_invented_content(self):
        fused = INVOICE_A + (
            "\nThe customer negotiated a twenty percent discount with the regional "
            "sales manager during a meeting last Tuesday afternoon in Mumbai."
        ) * 3
        assert _verify_against_sources(fused, INVOICE_A, INVOICE_B) is not None

    def test_rejects_unrelated_text(self):
        warning = _verify_against_sources(
            "The weather in Paris is mild today.", INVOICE_A, INVOICE_B
        )
        assert "came from the engines" in warning

    def test_rejects_empty_output(self):
        assert _verify_against_sources("", INVOICE_A, INVOICE_B) is not None

    def test_is_fast_on_full_pages(self):
        import time

        big_a, big_b = INVOICE_A * 200, INVOICE_B * 200
        started = time.perf_counter()
        _verify_against_sources(big_a, big_a, big_b)
        assert time.perf_counter() - started < 1.0


class TestLlmFusion:
    def test_falls_back_when_the_model_is_unreachable(self, settings, engine_a, engine_b, monkeypatch):
        from ocr_fusion.ocr.ollama_client import OllamaUnavailableError

        def unreachable(*args, **kwargs):
            raise OllamaUnavailableError("no server", "start ollama")

        monkeypatch.setattr("ocr_fusion.ocr.ollama_client.OllamaClient.generate", unreachable)
        settings.pipeline.fusion_strategy = FusionStrategy.LLM

        result = fuse([engine_a, engine_b], settings)
        assert result.fallback_used
        assert result.text.strip()
        assert any("unavailable" in w for w in result.warnings)

    def test_discards_a_drifting_model_and_merges_instead(
        self, settings, engine_a, engine_b, monkeypatch
    ):
        from ocr_fusion.ocr.ollama_client import GenerateResponse

        def hallucinate(*args, **kwargs):
            return GenerateResponse(text="Here is a poem about the sea.", model="m")

        monkeypatch.setattr("ocr_fusion.ocr.ollama_client.OllamaClient.generate", hallucinate)
        settings.pipeline.fusion_strategy = FusionStrategy.LLM

        result = fuse([engine_a, engine_b], settings)
        assert result.fallback_used
        assert "poem" not in result.text
        assert "INV-1024" in result.text

    def test_accepts_a_faithful_model_output(self, settings, engine_a, engine_b, monkeypatch):
        from ocr_fusion.ocr.ollama_client import GenerateResponse

        good = INVOICE_B + "\nTOTAL DUE 38,085.10"

        def reconcile(*args, **kwargs):
            return GenerateResponse(
                text=good, model="qwen2.5:1.5b", prompt_tokens=900, completion_tokens=120
            )

        monkeypatch.setattr("ocr_fusion.ocr.ollama_client.OllamaClient.generate", reconcile)
        settings.pipeline.fusion_strategy = FusionStrategy.LLM

        result = fuse([engine_a, engine_b], settings)
        assert result.fallback_used is False
        assert result.model_name == "qwen2.5:1.5b"
        assert result.input_tokens == 900


class TestSerialisation:
    def test_as_dict_is_json_safe(self, settings, engine_a, engine_b):
        import json

        json.dumps(fuse_line_vote([engine_a, engine_b], settings).as_dict())
