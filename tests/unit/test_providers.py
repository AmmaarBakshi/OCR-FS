"""The provider contract, the registry, and each concrete engine.

No test here needs Ollama, a GPU or a model: the Ollama transport is replaced
with a stub, which is exactly what the provider abstraction is for.
"""

from __future__ import annotations

import pytest

from ocr_fusion.config.schema import AppSettings, ProcessingLocation, UnlimitedBackend
from ocr_fusion.ocr.interface import (
    HealthStatus,
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
    TokenUsage,
)
from ocr_fusion.ocr.ollama_client import (
    GenerateResponse,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from ocr_fusion.ocr.postprocess import clean_ocr_text
from ocr_fusion.ocr.providers.qwen_vl import QwenVLProvider
from ocr_fusion.ocr.providers.unlimited import UnlimitedOCRProvider
from ocr_fusion.ocr.providers.unlimited.base import BackendError, PageOutput
from ocr_fusion.ocr.registry import ProviderRegistry, ProviderSpec, default_registry


class StubOllama:
    """Stands in for :class:`OllamaClient` without touching the network."""

    def __init__(self, *, models=("qwen2.5vl:3b",), response=None, error=None):
        self.host = "http://stub:11434"
        self._models = list(models)
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def list_models(self):
        if isinstance(self._error, OllamaUnavailableError):
            raise self._error
        return self._models

    def has_model(self, model):
        return model in self._models or f"{model}:latest" in self._models

    def show_model(self, model):
        return {"details": {"family": "test", "parameter_size": "3B"}}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response or GenerateResponse(
            text="Invoice Number: INV-1024",
            model=kwargs.get("model", "stub"),
            prompt_tokens=120,
            completion_tokens=30,
            eval_duration_seconds=1.5,
        )


class TestResultTypes:
    def test_unknown_tokens_stay_none(self):
        # The UI renders None as N/A; a zero would be a fabricated metric.
        usage = TokenUsage()
        assert usage.total_tokens is None
        assert usage.as_dict()["total_tokens"] is None

    def test_token_addition_preserves_none(self):
        assert (TokenUsage() + TokenUsage()).total_tokens is None
        assert (TokenUsage(10, 5) + TokenUsage()).total_tokens == 15

    def test_throughput_is_none_without_tokens(self):
        result = OCRResult("x", "X", duration_seconds=2.0, pages=[PageResult(1, "hi")])
        assert result.tokens_per_second is None

    def test_throughput_computed_when_available(self):
        result = OCRResult(
            "x", "X", duration_seconds=2.0, pages=[PageResult(1, "hi", tokens=TokenUsage(10, 20))]
        )
        assert result.tokens_per_second == pytest.approx(10.0)

    def test_text_is_joined_in_page_order(self):
        result = OCRResult("x", "X", pages=[PageResult(2, "second"), PageResult(1, "first")])
        assert result.text == "first\n\nsecond"

    def test_failure_helper_marks_the_result_failed(self):
        result = OCRResult.failure("x", "X", "boom", remedy="fix it")
        assert result.status is OCRStatus.FAILED
        assert result.succeeded is False
        assert result.remedy == "fix it"

    def test_summary_is_json_safe(self):
        import json

        json.dumps(OCRResult("x", "X", pages=[PageResult(1, "t")]).summary())


class TestRegistry:
    def test_builtin_engines_are_registered(self):
        import ocr_fusion.ocr.providers  # noqa: F401

        for provider_id in ("qwen_vl", "unlimited_ocr", "tesseract"):
            assert default_registry.has(provider_id)

    def test_registration_is_idempotent(self):
        # Streamlit re-executes modules on every rerun.
        import importlib

        import ocr_fusion.ocr.providers as providers

        before = set(default_registry.ids())
        importlib.reload(providers)
        assert set(default_registry.ids()) == before

    def test_duplicate_id_raises(self):
        registry = ProviderRegistry()
        registry.register(ProviderSpec("a", "A", lambda s: None))
        with pytest.raises(ValueError):
            registry.register(ProviderSpec("a", "Another", lambda s: None))

    def test_replace_allows_overriding(self):
        registry = ProviderRegistry()
        registry.register(ProviderSpec("a", "A", lambda s: None))
        registry.register(ProviderSpec("a", "A2", lambda s: None), replace=True)
        assert registry.get("a").display_name == "A2"

    def test_unknown_id_lists_what_is_available(self):
        registry = ProviderRegistry()
        registry.register(ProviderSpec("a", "A", lambda s: None))
        with pytest.raises(KeyError) as excinfo:
            registry.get("missing")
        assert "a" in str(excinfo.value)

    def test_enabled_specs_follow_settings(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        settings.qwen.enabled = False
        enabled = {spec.provider_id for spec in default_registry.enabled_specs(settings)}
        assert "qwen_vl" not in enabled
        assert "unlimited_ocr" in enabled

    def test_factories_are_lazy(self):
        # Registering must never construct a provider or import torch.
        calls = []
        registry = ProviderRegistry()
        registry.register(ProviderSpec("a", "A", lambda s: calls.append(1)))
        assert calls == []


class TestQwenProvider:
    def test_health_reports_missing_server(self, settings):
        client = StubOllama(error=OllamaUnavailableError("down", "start ollama"))
        status = QwenVLProvider(settings, client).health_check()
        assert status.available is False
        assert status.remedy == "start ollama"

    def test_health_reports_missing_model_with_pull_command(self, settings):
        provider = QwenVLProvider(settings, StubOllama(models=["other:1b"]))
        status = provider.health_check()
        assert status.available is False
        assert "ollama pull qwen2.5vl:3b" in status.remedy

    def test_health_ok_when_model_present(self, settings):
        assert QwenVLProvider(settings, StubOllama()).health_check().available

    def test_transcribes_pages(self, settings, single_page_document):
        provider = QwenVLProvider(settings, StubOllama())
        result = provider.process(single_page_document)
        assert result.status is OCRStatus.SUCCESS
        assert "INV-1024" in result.text
        assert result.model_name == "qwen2.5vl:3b"
        assert result.processing_location is ProcessingLocation.LOCAL

    def test_reports_real_token_metrics(self, settings, single_page_document):
        result = QwenVLProvider(settings, StubOllama()).process(single_page_document)
        assert result.tokens.input_tokens == 120
        assert result.tokens.output_tokens == 30

    def test_sends_configured_prompts_and_sampling(self, settings, single_page_document):
        settings.prompts.qwen_system = "SYSTEM-MARKER"
        settings.qwen.temperature = 0.3
        client = StubOllama()
        QwenVLProvider(settings, client).process(single_page_document)
        assert client.calls[0]["system"] == "SYSTEM-MARKER"
        assert client.calls[0]["temperature"] == 0.3

    def test_sends_the_page_image(self, settings, single_page_document):
        client = StubOllama()
        QwenVLProvider(settings, client).process(single_page_document)
        assert len(client.calls[0]["images"]) == 1

    def test_unavailable_server_fails_without_raising(self, settings, single_page_document):
        client = StubOllama(error=OllamaUnavailableError("down", "start it"))
        result = QwenVLProvider(settings, client).process(single_page_document)
        assert result.status is OCRStatus.FAILED
        assert result.remedy == "start it"

    def test_empty_output_is_reported_as_failure(self, settings, single_page_document):
        # A blank page and a refusing model must be distinguishable.
        client = StubOllama(response=GenerateResponse(text="   ", model="m"))
        result = QwenVLProvider(settings, client).process(single_page_document)
        assert result.status is OCRStatus.FAILED
        assert "no text" in result.error.lower()

    def test_truncation_is_surfaced_as_a_warning(self, settings, single_page_document):
        client = StubOllama(
            response=GenerateResponse(text="partial text", model="m", done_reason="length")
        )
        result = QwenVLProvider(settings, client).process(single_page_document)
        assert result.status is OCRStatus.SUCCESS
        assert "token limit" in result.page(1).error

    def test_retries_a_timeout(self, settings, single_page_document, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda *_: None)
        settings.qwen.retry_count = 2
        client = StubOllama(error=OllamaTimeoutError("too slow", "raise the timeout"))
        QwenVLProvider(settings, client).process(single_page_document)
        assert len(client.calls) == 3  # initial attempt plus two retries

    def test_does_not_retry_a_missing_model(self, settings, single_page_document):
        # Pulling a model mid-run will not happen, so retrying only wastes time.
        settings.qwen.retry_count = 3
        client = StubOllama(models=["qwen2.5vl:3b"], error=OllamaModelMissingError("gone", "pull"))
        QwenVLProvider(settings, client).process(single_page_document)
        assert len(client.calls) == 1

    def test_partial_status_when_one_page_fails(self, settings, multi_page_document):
        class Flaky(StubOllama):
            def generate(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 2:
                    return GenerateResponse(text="", model="m")
                return GenerateResponse(text="page text", model="m")

        result = QwenVLProvider(settings, Flaky()).process(multi_page_document)
        assert result.status is OCRStatus.PARTIAL
        assert result.page(2).status is OCRStatus.FAILED

    def test_metadata_describes_the_run(self, settings):
        metadata = QwenVLProvider(settings, StubOllama()).get_metadata()
        assert metadata["model"] == "qwen2.5vl:3b"
        assert metadata["runtime"] == "Ollama"


class TestUnlimitedProvider:
    @pytest.mark.parametrize("backend", list(UnlimitedBackend))
    def test_every_backend_can_be_constructed(self, settings, backend):
        # The Settings UI lists all backends, including unavailable ones, so
        # constructing one must never import torch or open a connection.
        settings.unlimited_ocr.backend = backend
        provider = UnlimitedOCRProvider(settings)
        assert provider.backend.backend_id == backend.value

    def test_substitute_backend_is_labelled_honestly(self, settings):
        settings.unlimited_ocr.backend = UnlimitedBackend.OLLAMA
        provider = UnlimitedOCRProvider(settings)
        assert provider.backend.is_substitute
        assert "substitute" in provider.provider_display_name.lower()
        assert "deepseek-ocr:3b" in provider.provider_display_name

    def test_real_backend_is_not_labelled_substitute(self, settings):
        settings.unlimited_ocr.backend = UnlimitedBackend.HTTP
        provider = UnlimitedOCRProvider(settings)
        assert provider.backend.is_substitute is False
        assert "substitute" not in provider.provider_display_name.lower()

    def test_remote_endpoint_is_reported_as_cloud(self, settings):
        # Sending pages off the machine is a privacy fact the UI must show.
        settings.unlimited_ocr.backend = UnlimitedBackend.HTTP
        settings.unlimited_ocr.endpoint = "https://gpu.example.com/v1"
        assert UnlimitedOCRProvider(settings).processing_location is ProcessingLocation.CLOUD

    def test_local_endpoint_is_reported_as_local(self, settings):
        settings.unlimited_ocr.backend = UnlimitedBackend.HTTP
        settings.unlimited_ocr.endpoint = "http://localhost:8000/v1"
        assert UnlimitedOCRProvider(settings).processing_location is ProcessingLocation.LOCAL

    def test_cli_backend_without_a_command_is_unavailable(self, settings):
        settings.unlimited_ocr.backend = UnlimitedBackend.CLI
        settings.unlimited_ocr.executable = ""
        status = UnlimitedOCRProvider(settings).health_check()
        assert status.available is False
        assert "Settings" in status.remedy

    def test_unhealthy_backend_fails_the_provider_cleanly(self, settings, single_page_document):
        settings.unlimited_ocr.backend = UnlimitedBackend.CLI
        settings.unlimited_ocr.executable = ""
        result = UnlimitedOCRProvider(settings).process(single_page_document)
        assert result.status is OCRStatus.FAILED
        assert result.remedy

    def test_backend_exception_becomes_a_failed_page(self, settings, single_page_document):
        class Exploding:
            backend_id = "boom"
            display_name = "Boom"
            is_substitute = False
            processing_location = ProcessingLocation.LOCAL
            model_label = "boom"

            def health_check(self):
                return HealthStatus.ok()

            def run_page(self, page, prompt):
                raise ZeroDivisionError("backend bug")

            def describe(self):
                return {"backend": "boom"}

        result = UnlimitedOCRProvider(settings, Exploding()).process(single_page_document)
        assert result.status is OCRStatus.FAILED
        assert "Unexpected backend error" in result.page(1).error

    def test_non_retryable_backend_error_is_attempted_once(self, settings, single_page_document):
        class Fussy:
            backend_id = "f"
            display_name = "F"
            is_substitute = False
            processing_location = ProcessingLocation.LOCAL
            model_label = "f"

            def __init__(self):
                self.attempts = 0

            def health_check(self):
                return HealthStatus.ok()

            def run_page(self, page, prompt):
                self.attempts += 1
                raise BackendError("not fixable", retryable=False)

            def describe(self):
                return {}

        settings.unlimited_ocr.retry_count = 3
        backend = Fussy()
        UnlimitedOCRProvider(settings, backend).process(single_page_document)
        assert backend.attempts == 1

    def test_successful_backend_output_is_cleaned(self, settings, single_page_document):
        class Fenced:
            backend_id = "f"
            display_name = "F"
            is_substitute = False
            processing_location = ProcessingLocation.LOCAL
            model_label = "f"

            def health_check(self):
                return HealthStatus.ok()

            def run_page(self, page, prompt):
                return PageOutput(text="```\nInvoice Number: INV-1024\n```")

            def describe(self):
                return {}

        result = UnlimitedOCRProvider(settings, Fenced()).process(single_page_document)
        assert result.text == "Invoice Number: INV-1024"


class TestPostprocess:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("```markdown\nHello\n```", "Hello"),
            ("Here is the text from the image:\n\nHello", "Hello"),
            ("Sure! Here's the transcription:\n```\nHello\n```", "Hello"),
            ("Hello\n\nLet me know if you need anything else!", "Hello"),
            ("A\n\n\n\n\nB   \n", "A\n\nB"),
            ("", ""),
        ],
    )
    def test_strips_model_packaging(self, raw, expected):
        assert clean_ocr_text(raw) == expected

    def test_leaves_inline_fences_alone(self):
        # A fence inside the page is probably real document content.
        text = "Run ```pip install x``` first"
        assert clean_ocr_text(text) == text

    def test_does_not_alter_real_content(self):
        text = "Invoice Number: INV-1024\nTotal: 38,085.10"
        assert clean_ocr_text(text) == text


class TestContract:
    def test_every_registered_provider_implements_the_interface(self, settings):
        import ocr_fusion.ocr.providers  # noqa: F401

        for spec in default_registry.specs():
            provider = default_registry.create(spec.provider_id, settings)
            assert isinstance(provider, OCRProvider)
            assert callable(provider.process)
            assert isinstance(provider.health_check(), HealthStatus)
            assert isinstance(provider.get_metadata(), dict)

    def test_health_check_never_raises(self, settings):
        # The Settings page probes engines that are known to be unavailable.
        import ocr_fusion.ocr.providers  # noqa: F401

        for spec in default_registry.specs():
            status = default_registry.create(spec.provider_id, settings).health_check()
            assert isinstance(status.available, bool)
            if not status.available:
                assert status.message
