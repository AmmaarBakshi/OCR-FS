"""Stage 1 - Qwen2.5-VL served by Ollama.

The model name, prompts, sampling parameters, timeout and retry policy all come
from :class:`~ocr_fusion.config.schema.AppSettings`; nothing here is hardcoded
(spec s1, s12).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ocr_fusion.config.schema import AppSettings, ProcessingLocation
from ocr_fusion.documents.models import Document, DocumentPage
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
    OllamaClient,
    OllamaError,
    OllamaModelMissingError,
    OllamaUnavailableError,
)
from ocr_fusion.ocr.postprocess import clean_ocr_text

logger = logging.getLogger(__name__)


class QwenVLProvider(OCRProvider):
    """Transcribes pages with a Qwen vision-language model through Ollama."""

    provider_id = "qwen_vl"
    provider_name = "Qwen2.5-VL"
    processing_location = ProcessingLocation.LOCAL

    def __init__(self, settings: AppSettings, client: OllamaClient | None = None) -> None:
        self.settings = settings
        self.config = settings.qwen
        # The client is injectable so tests can drive the provider without a
        # live server.
        self.client = client or OllamaClient(
            settings.ollama.host,
            connect_timeout=settings.ollama.connect_timeout_seconds,
        )

    @property
    def provider_display_name(self) -> str:
        """Name including the configured model, e.g. 'Qwen2.5-VL (qwen2.5vl:3b)'."""
        return f"{self.provider_name} ({self.config.model})"

    # -- readiness ---------------------------------------------------------

    def health_check(self) -> HealthStatus:
        """Confirm Ollama is up and the configured model has been pulled."""
        started = time.perf_counter()
        try:
            models = self.client.list_models()
        except OllamaUnavailableError as exc:
            return HealthStatus.unavailable(
                exc.message,
                exc.remedy,
                host=self.client.host,
                checked=self.config.model,
            )

        if not self.client.has_model(self.config.model):
            preview = ", ".join(sorted(models)[:8]) or "none"
            return HealthStatus.unavailable(
                f"Ollama is running, but the model '{self.config.model}' is not installed.",
                f"Run: ollama pull {self.config.model}",
                host=self.client.host,
                available_models=preview,
            )

        details = self.client.show_model(self.config.model).get("details", {})
        status = HealthStatus.ok(
            f"{self.config.model} is ready.",
            host=self.client.host,
            model=self.config.model,
            family=details.get("family"),
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
        )
        status.checked_in_seconds = time.perf_counter() - started
        return status

    def get_metadata(self) -> dict[str, Any]:
        """Effective configuration for this run, shown under Developer Mode."""
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "engine": "Qwen2.5-VL",
            "runtime": "Ollama",
            "backend": "ollama",
            "model": self.config.model,
            "host": self.client.host,
            "processing_location": self.processing_location.value,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "timeout_seconds": self.config.timeout_seconds,
            "retry_count": self.config.retry_count,
            "keep_alive": self.config.keep_alive,
        }

    # -- inference ---------------------------------------------------------

    def process(self, document: Document) -> OCRResult:
        """Transcribe every page, keeping going when individual pages fail."""
        started = self._timer()
        result = OCRResult(
            provider_id=self.provider_id,
            provider_name=self.provider_display_name,
            model_name=self.config.model,
            backend="ollama",
            processing_location=self.processing_location,
            metadata=self.get_metadata(),
        )

        # One upfront readiness check: failing 40 pages one at a time against a
        # server that is down wastes minutes and produces 40 identical errors.
        health = self.health_check()
        if not health.available:
            return OCRResult.failure(
                self.provider_id,
                self.provider_display_name,
                health.message,
                remedy=health.remedy,
                duration_seconds=self._elapsed(started),
                metadata=self.get_metadata(),
            )

        for page in document.pages:
            result.pages.append(self._process_page(page))

        result.duration_seconds = self._elapsed(started)
        result.status = _aggregate_status(result.pages)
        if result.status is OCRStatus.FAILED:
            result.error = _first_error(result.pages) or "No text could be extracted."
        return result

    def _process_page(self, page: DocumentPage) -> PageResult:
        """Run one page, retrying transient failures per the configured policy."""
        attempts = self.config.retry_count + 1
        last_error: OllamaError | None = None
        page_started = self._timer()

        for attempt in range(1, attempts + 1):
            try:
                response = self._generate(page)
            except OllamaModelMissingError as exc:
                # Pulling a model mid-run will not happen; do not retry.
                last_error = exc
                break
            except OllamaError as exc:
                last_error = exc
                if attempt < attempts:
                    logger.warning(
                        "Qwen page %s attempt %s/%s failed: %s",
                        page.number,
                        attempt,
                        attempts,
                        exc.message,
                    )
                    time.sleep(self.config.retry_backoff_seconds)
                    continue
                break
            else:
                return self._page_result(page, response, self._elapsed(page_started))

        message = last_error.message if last_error else "The page could not be processed."
        return PageResult(
            page_number=page.number,
            status=OCRStatus.FAILED,
            error=message,
            duration_seconds=self._elapsed(page_started),
        )

    def _generate(self, page: DocumentPage) -> GenerateResponse:
        return self.client.generate(
            model=self.config.model,
            prompt=self.settings.prompts.qwen_user,
            system=self.settings.prompts.qwen_system,
            images=[page.image_base64()],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
            keep_alive=self.config.keep_alive,
        )

    def _page_result(
        self, page: DocumentPage, response: GenerateResponse, duration: float
    ) -> PageResult:
        text = clean_ocr_text(response.text)
        status = OCRStatus.SUCCESS
        error = None

        # An empty transcription is a real outcome worth surfacing, not a crash:
        # a blank page and a refusing model look identical without this.
        if not text.strip():
            status = OCRStatus.FAILED
            error = "The model returned no text for this page."
        elif response.truncated:
            error = (
                "Output stopped at the token limit; the transcription may be "
                "incomplete. Raise Max tokens in Settings."
            )

        return PageResult(
            page_number=page.number,
            text=text,
            status=status,
            error=error,
            duration_seconds=duration,
            tokens=TokenUsage(
                input_tokens=response.prompt_tokens,
                output_tokens=response.completion_tokens,
            ),
            load_duration_seconds=response.load_duration_seconds,
            inference_duration_seconds=response.eval_duration_seconds,
            raw_response={
                "model": response.model,
                "done_reason": response.done_reason,
                "total_duration_seconds": response.total_duration_seconds,
                "tokens_per_second": response.tokens_per_second,
            },
        )


def _aggregate_status(pages: list[PageResult]) -> OCRStatus:
    """Roll page outcomes up into one provider status."""
    if not pages:
        return OCRStatus.FAILED
    succeeded = sum(1 for p in pages if p.status is OCRStatus.SUCCESS)
    if succeeded == len(pages):
        return OCRStatus.SUCCESS
    if succeeded == 0:
        return OCRStatus.FAILED
    return OCRStatus.PARTIAL


def _first_error(pages: list[PageResult]) -> str | None:
    return next((p.error for p in pages if p.error), None)


def build_qwen_provider(settings: AppSettings) -> QwenVLProvider:
    """Registry factory."""
    return QwenVLProvider(settings)


__all__ = ["QwenVLProvider", "build_qwen_provider"]
