"""Tests that need a live engine runtime.

Excluded from the default suite - run them with::

    pytest -m integration

Each test skips itself when its runtime is unavailable, so the file is safe to
run anywhere; it simply reports nothing to do on a machine without Ollama.
Model inference is slow on CPU, so the transcription tests are also marked
``slow``.
"""

from __future__ import annotations

import pytest

from ocr_fusion.config import AppSettings
from ocr_fusion.config.schema import UnlimitedBackend
from ocr_fusion.documents import load_document_bytes
from ocr_fusion.ocr.ollama_client import OllamaClient
from ocr_fusion.ocr.providers.qwen_vl import QwenVLProvider
from ocr_fusion.ocr.providers.unlimited import UnlimitedOCRProvider
from ocr_fusion.pipeline import OCRPipeline

pytestmark = pytest.mark.integration


def _client(settings: AppSettings) -> OllamaClient:
    return OllamaClient(settings.ollama.host)


@pytest.fixture
def live_settings() -> AppSettings:
    settings = AppSettings()
    # Keep a live run demo-sized rather than thorough.
    settings.documents.pdf_render_dpi = 110
    settings.qwen.max_tokens = 1024
    settings.unlimited_ocr.max_tokens = 1024
    return settings


@pytest.fixture
def invoice_pdf() -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((60, 80), "ACME LOGISTICS LTD", fontsize=18, fontname="hebo")
    page.insert_text((60, 120), "Invoice Number: INV-1024", fontsize=12)
    page.insert_text((60, 145), "Date: 14 March 2025", fontsize=12)
    page.insert_text((60, 185), "Freight forwarding    12    1,250.00    15,000.00", fontsize=11)
    page.insert_text((60, 215), "TOTAL DUE    38,085.10", fontsize=13, fontname="hebo")
    data = document.tobytes()
    document.close()
    return data


class TestOllamaTransport:
    def test_server_is_reachable(self, live_settings):
        if not _client(live_settings).is_reachable():
            pytest.skip("Ollama is not running")
        assert _client(live_settings).list_models()

    def test_missing_model_is_distinguished_from_a_missing_server(self, live_settings):
        client = _client(live_settings)
        if not client.is_reachable():
            pytest.skip("Ollama is not running")
        assert client.has_model("definitely-not-a-real-model:1b") is False


class TestQwenLive:
    @pytest.mark.slow
    def test_transcribes_a_real_invoice(self, live_settings, invoice_pdf):
        provider = QwenVLProvider(live_settings)
        status = provider.health_check()
        if not status.available:
            pytest.skip(f"Qwen unavailable: {status.message}")

        document = load_document_bytes(invoice_pdf, "invoice.pdf", live_settings.documents)
        result = provider.process(document)

        assert result.succeeded, result.error
        assert "INV-1024" in result.text.replace(" ", "")
        # A real run must report real metrics, not placeholders.
        assert result.tokens.input_tokens and result.tokens.input_tokens > 0
        assert result.duration_seconds > 0


class TestUnlimitedLive:
    def test_unavailable_backends_explain_themselves(self, live_settings):
        # On a machine without a GPU these must fail with actionable guidance
        # rather than a stack trace.
        for backend in (UnlimitedBackend.HTTP, UnlimitedBackend.TRANSFORMERS, UnlimitedBackend.CLI):
            live_settings.unlimited_ocr.backend = backend
            status = UnlimitedOCRProvider(live_settings).health_check()
            if not status.available:
                assert status.message
                assert status.remedy

    @pytest.mark.slow
    def test_substitute_backend_transcribes_and_labels_itself(self, live_settings, invoice_pdf):
        live_settings.unlimited_ocr.backend = UnlimitedBackend.OLLAMA
        provider = UnlimitedOCRProvider(live_settings)
        status = provider.health_check()
        if not status.available:
            pytest.skip(f"Substitute engine unavailable: {status.message}")

        document = load_document_bytes(invoice_pdf, "invoice.pdf", live_settings.documents)
        result = provider.process(document)

        assert result.succeeded, result.error
        assert result.text.strip()
        # It must never present itself as upstream Unlimited-OCR.
        assert "substitute" in result.provider_name.lower()
        assert result.model_name == live_settings.unlimited_ocr.ollama_model


class TestFullPipelineLive:
    @pytest.mark.slow
    def test_two_engines_compare_and_fuse(self, live_settings, invoice_pdf):
        qwen = QwenVLProvider(live_settings)
        unlimited = UnlimitedOCRProvider(live_settings)
        if not (qwen.health_check().available and unlimited.health_check().available):
            pytest.skip("Both engines are required for this test")

        document = load_document_bytes(invoice_pdf, "invoice.pdf", live_settings.documents)
        result = OCRPipeline(live_settings, [qwen, unlimited]).execute(document)

        assert result.succeeded
        assert len(result.successful_engines) == 2
        assert result.comparison is not None
        assert result.fusion is not None
        assert result.final_text.strip()
        # Both raw outputs stay available alongside the fused result (spec s5).
        assert all(engine.text for engine in result.successful_engines)
