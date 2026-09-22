"""The page cache: never transcribe the same page twice."""

from __future__ import annotations

import pytest

from ocr_fusion.documents.models import Document, DocumentKind, DocumentPage
from ocr_fusion.ocr.cache import CachingProvider, PageCache, page_cache_key, wrap_with_cache
from ocr_fusion.ocr.interface import OCRStatus
from tests.conftest import FakeProvider


@pytest.fixture
def cache(tmp_path) -> PageCache:
    return PageCache(tmp_path / "pages.sqlite3")


def _document(*numbers: int) -> Document:
    return Document(
        filename="batch.pdf",
        kind=DocumentKind.PDF,
        pages=[
            DocumentPage(number=n, image_bytes=f"page-{n}".encode(), width=100, height=100)
            for n in numbers
        ],
    )


class TestKey:
    def test_same_page_same_configuration_is_the_same_key(self):
        a = page_cache_key(b"pixels", "qwen_vl", "qwen2.5vl:3b", {"temperature": 0.0})
        b = page_cache_key(b"pixels", "qwen_vl", "qwen2.5vl:3b", {"temperature": 0.0})
        assert a == b

    def test_a_different_model_is_a_different_key(self):
        a = page_cache_key(b"pixels", "qwen_vl", "qwen2.5vl:3b", {})
        b = page_cache_key(b"pixels", "qwen_vl", "qwen2.5vl:7b", {})
        assert a != b

    def test_a_different_prompt_is_a_different_key(self):
        # Returning yesterday's prompt's answer would be worse than no cache.
        a = page_cache_key(b"pixels", "qwen_vl", "m", {"prompt": "Transcribe."})
        b = page_cache_key(b"pixels", "qwen_vl", "m", {"prompt": "Extract the total."})
        assert a != b

    def test_a_different_page_is_a_different_key(self):
        assert page_cache_key(b"one", "q", "m", {}) != page_cache_key(b"two", "q", "m", {})


class TestCaching:
    def test_a_second_run_does_not_reach_the_engine(self, cache):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "Invoice INV-1024")
        wrapped = CachingProvider(engine, cache)

        first = wrapped.process(_document(1, 2))
        assert engine.pages_seen == [1, 2]
        assert wrapped.stats.hits == 0 and wrapped.stats.misses == 2

        engine.pages_seen.clear()
        second = wrapped.process(_document(1, 2))
        assert engine.pages_seen == []
        assert wrapped.stats.hits == 2
        assert second.text == first.text

    def test_only_the_unseen_pages_reach_the_engine(self, cache):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "Invoice INV-1024")
        wrapped = CachingProvider(engine, cache)
        wrapped.process(_document(1))

        engine.pages_seen.clear()
        result = wrapped.process(_document(1, 2, 3))
        # Page 1 was known; only 2 and 3 cost anything.
        assert engine.pages_seen == [2, 3]
        assert wrapped.stats.hits == 1 and wrapped.stats.misses == 2
        assert [p.page_number for p in result.pages] == [1, 2, 3]

    def test_a_failed_page_is_not_remembered(self, cache):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "text", failing_pages=(2,))
        wrapped = CachingProvider(engine, cache)
        wrapped.process(_document(1, 2))

        engine.pages_seen.clear()
        wrapped.process(_document(1, 2))
        # Page 2 failed, so it is tried again rather than cached as a failure.
        assert engine.pages_seen == [2]

    def test_a_cache_hit_does_not_claim_time_this_run_did_not_spend(self, cache):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "text", duration=9.0)
        wrapped = CachingProvider(engine, cache)
        wrapped.process(_document(1))
        result = wrapped.process(_document(1))
        assert result.pages[0].duration_seconds == 0.0
        assert wrapped.stats.seconds_saved == pytest.approx(9.0)

    def test_the_wrapper_reports_the_real_engine(self, cache):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "text")
        wrapped = CachingProvider(engine, cache)
        assert wrapped.provider_id == "qwen_vl"
        assert wrapped.provider_name == "Qwen2.5-VL"
        result = wrapped.process(_document(1))
        assert result.model_name == "qwen_vl:test"

    def test_pages_stay_in_order_when_some_are_cached(self, cache):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "text")
        wrapped = CachingProvider(engine, cache)
        wrapped.process(_document(2))
        result = wrapped.process(_document(1, 2, 3))
        assert [p.page_number for p in result.pages] == [1, 2, 3]
        assert result.status is OCRStatus.SUCCESS

    def test_free_engines_are_not_wrapped(self, cache):
        from ocr_fusion.config.schema import AppSettings
        from ocr_fusion.ocr.providers.text_layer import TextLayerProvider

        free = TextLayerProvider(AppSettings())
        assert wrap_with_cache(free, cache) is free

    def test_survives_a_restart(self, tmp_path):
        engine = FakeProvider("qwen_vl", "Qwen2.5-VL", "text")
        first = PageCache(tmp_path / "pages.sqlite3")
        CachingProvider(engine, first).process(_document(1))
        first.close()

        engine.pages_seen.clear()
        reopened = PageCache(tmp_path / "pages.sqlite3")
        wrapped = CachingProvider(engine, reopened)
        wrapped.process(_document(1))
        assert engine.pages_seen == []
        assert reopened.count() == 1
