"""Ollama backend - a locally pulled OCR VLM standing in for Unlimited-OCR.

Upstream Unlimited-OCR needs a CUDA GPU. On a CPU-only machine none of the
other backends can run, which would leave Stage 2 permanently unavailable and
the two-engine comparison undemonstrable.

This backend keeps the pipeline honest instead of empty: it runs a real second
OCR model that is genuinely available locally (``deepseek-ocr:3b`` by default -
Unlimited-OCR is itself DeepSeek-OCR-derived) and reports the true model name
with ``is_substitute`` set, so the UI states plainly that this is a stand-in and
not upstream Unlimited-OCR. No output is ever fabricated.
"""

from __future__ import annotations

import logging
from typing import Any

from ocr_fusion.config.schema import (
    OllamaSettings,
    ProcessingLocation,
    UnlimitedOCRSettings,
)
from ocr_fusion.documents.models import DocumentPage
from ocr_fusion.ocr.interface import HealthStatus
from ocr_fusion.ocr.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelMissingError,
    OllamaUnavailableError,
)
from ocr_fusion.ocr.providers.unlimited.base import (
    BackendError,
    PageOutput,
    UnlimitedBackendBase,
)

logger = logging.getLogger(__name__)


class OllamaBackend(UnlimitedBackendBase):
    """Runs a second OCR-capable vision model through Ollama."""

    backend_id = "ollama"
    display_name = "Unlimited-OCR (local substitute via Ollama)"
    processing_location = ProcessingLocation.LOCAL
    is_substitute = True

    def __init__(
        self,
        config: UnlimitedOCRSettings,
        ollama: OllamaSettings | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        super().__init__(config)
        ollama = ollama or OllamaSettings()
        self.client = client or OllamaClient(
            ollama.host, connect_timeout=ollama.connect_timeout_seconds
        )

    @property
    def model_label(self) -> str:
        return self.config.ollama_model

    def health_check(self) -> HealthStatus:
        try:
            models = self.client.list_models()
        except OllamaUnavailableError as exc:
            return HealthStatus.unavailable(exc.message, exc.remedy, host=self.client.host)

        if not self.client.has_model(self.config.ollama_model):
            return HealthStatus.unavailable(
                f"The model '{self.config.ollama_model}' is not installed.",
                f"Run: ollama pull {self.config.ollama_model}",
                host=self.client.host,
                available_models=sorted(models)[:8],
            )

        details = self.client.show_model(self.config.ollama_model).get("details", {})
        return HealthStatus.ok(
            f"{self.config.ollama_model} is ready (local substitute for Unlimited-OCR).",
            host=self.client.host,
            model=self.config.ollama_model,
            substitute=True,
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
        )

    def run_page(self, page: DocumentPage, prompt: str) -> PageOutput:
        try:
            response = self.client.generate(
                model=self.config.ollama_model,
                prompt=_strip_image_token(prompt),
                images=[page.image_base64()],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds,
                keep_alive="5m",
            )
        except OllamaModelMissingError as exc:
            raise BackendError(exc.message, exc.remedy, retryable=False) from exc
        except OllamaError as exc:
            raise BackendError(exc.message, exc.remedy) from exc

        return PageOutput(
            text=response.text,
            input_tokens=response.prompt_tokens,
            output_tokens=response.completion_tokens,
            load_duration_seconds=response.load_duration_seconds,
            inference_duration_seconds=response.eval_duration_seconds,
            truncated=response.truncated,
            raw={
                "model": response.model,
                "done_reason": response.done_reason,
                "tokens_per_second": response.tokens_per_second,
                "substitute_for": "baidu/Unlimited-OCR",
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "engine": "Unlimited-OCR (substitute)",
            "runtime": "Ollama",
            "model": self.config.ollama_model,
            "host": self.client.host,
            "is_substitute": True,
            "substitute_note": (
                f"Running {self.config.ollama_model} locally. This is not "
                "baidu/Unlimited-OCR, which requires a CUDA GPU."
            ),
            "timeout_seconds": self.config.timeout_seconds,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "processing_location": self.processing_location.value,
        }


def _strip_image_token(prompt: str) -> str:
    """Remove Unlimited-OCR's ``<image>`` marker.

    Upstream prompts embed the image position inline; Ollama takes images as a
    separate field, so the literal token would otherwise appear as text.
    """
    return prompt.replace("<image>", "").strip()


__all__ = ["OllamaBackend"]
