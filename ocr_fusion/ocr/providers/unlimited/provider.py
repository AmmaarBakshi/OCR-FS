"""Stage 2 - the Unlimited-OCR provider.

Owns policy: backend selection, the page loop, retries, status roll-up and
metrics. Mechanism lives in the backends, so a new deployment shape never
touches this file.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ocr_fusion.config.schema import (
    AppSettings,
    ProcessingLocation,
    UnlimitedBackend,
)
from ocr_fusion.documents.models import Document, DocumentPage
from ocr_fusion.ocr.interface import (
    HealthStatus,
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
    TokenUsage,
)
from ocr_fusion.ocr.postprocess import clean_ocr_text
from ocr_fusion.ocr.providers.unlimited.base import BackendError, UnlimitedBackendBase
from ocr_fusion.ocr.providers.unlimited.cli_backend import CliBackend
from ocr_fusion.ocr.providers.unlimited.http_backend import HttpBackend
from ocr_fusion.ocr.providers.unlimited.ollama_backend import OllamaBackend
from ocr_fusion.ocr.providers.unlimited.transformers_backend import TransformersBackend

logger = logging.getLogger(__name__)

#: Backend registry. Add an entry to support a new deployment shape.
BACKENDS: dict[UnlimitedBackend, type[UnlimitedBackendBase]] = {
    UnlimitedBackend.HTTP: HttpBackend,
    UnlimitedBackend.TRANSFORMERS: TransformersBackend,
    UnlimitedBackend.CLI: CliBackend,
    UnlimitedBackend.OLLAMA: OllamaBackend,
}


def build_backend(settings: AppSettings) -> UnlimitedBackendBase:
    """Instantiate the backend named in settings."""
    backend_class = BACKENDS.get(settings.unlimited_ocr.backend)
    if backend_class is None:  # pragma: no cover - enum keeps this unreachable
        raise ValueError(f"Unknown backend: {settings.unlimited_ocr.backend}")
    if backend_class is OllamaBackend:
        return OllamaBackend(settings.unlimited_ocr, settings.ollama)
    return backend_class(settings.unlimited_ocr)


class UnlimitedOCRProvider(OCRProvider):
    """Second OCR engine, executed through a configurable backend."""

    provider_id = "unlimited_ocr"
    provider_name = "Unlimited-OCR"

    def __init__(
        self, settings: AppSettings, backend: UnlimitedBackendBase | None = None
    ) -> None:
        self.settings = settings
        self.config = settings.unlimited_ocr
        self.backend = backend or build_backend(settings)

    @property
    def processing_location(self) -> ProcessingLocation:  # type: ignore[override]
        return self.backend.processing_location

    @property
    def provider_display_name(self) -> str:
        """Name that states what really ran, so a substitute is never mistaken
        for upstream Unlimited-OCR (spec s16)."""
        if self.backend.is_substitute:
            return f"Unlimited-OCR (substitute: {self.backend.model_label})"
        return f"{self.provider_name} ({self.backend.model_label})"

    # -- readiness ---------------------------------------------------------

    def health_check(self) -> HealthStatus:
        started = time.perf_counter()
        status = self.backend.health_check()
        status.details.setdefault("backend", self.backend.backend_id)
        status.details.setdefault("is_substitute", self.backend.is_substitute)
        status.checked_in_seconds = time.perf_counter() - started
        return status

    def get_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "enabled": self.config.enabled,
            "retry_count": self.config.retry_count,
            "is_substitute": self.backend.is_substitute,
        }
        metadata.update(self.backend.describe())
        return metadata

    # -- inference ---------------------------------------------------------

    def process(self, document: Document) -> OCRResult:
        started = self._timer()
        result = OCRResult(
            provider_id=self.provider_id,
            provider_name=self.provider_display_name,
            model_name=self.backend.model_label,
            backend=self.backend.backend_id,
            processing_location=self.processing_location,
            metadata=self.get_metadata(),
        )

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

        prompt = self.settings.prompts.unlimited_ocr_task
        for page in document.pages:
            result.pages.append(self._process_page(page, prompt))

        result.duration_seconds = self._elapsed(started)
        result.status = _aggregate_status(result.pages)
        if result.status is OCRStatus.FAILED:
            result.error = _first_error(result.pages) or "No text could be extracted."
            result.remedy = _first_remedy(result.pages)
        return result

    def _process_page(self, page: DocumentPage, prompt: str) -> PageResult:
        attempts = self.config.retry_count + 1
        last: BackendError | None = None
        page_started = self._timer()

        for attempt in range(1, attempts + 1):
            try:
                output = self.backend.run_page(page, prompt)
            except BackendError as exc:
                last = exc
                if exc.retryable and attempt < attempts:
                    logger.warning(
                        "Unlimited-OCR page %s attempt %s/%s failed: %s",
                        page.number,
                        attempt,
                        attempts,
                        exc.message,
                    )
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                # A backend bug must degrade to a failed page, never crash a run.
                logger.exception("Unlimited-OCR backend raised on page %s", page.number)
                last = BackendError(f"Unexpected backend error: {exc}", retryable=False)
                break
            else:
                text = clean_ocr_text(output.text)
                status = OCRStatus.SUCCESS
                error = None
                if not text.strip():
                    status = OCRStatus.FAILED
                    error = "The engine returned no text for this page."
                elif output.truncated:
                    error = (
                        "Output stopped at the token limit; the transcription may "
                        "be incomplete. Raise Max tokens in Settings."
                    )
                return PageResult(
                    page_number=page.number,
                    text=text,
                    status=status,
                    error=error,
                    duration_seconds=self._elapsed(page_started),
                    tokens=TokenUsage(output.input_tokens, output.output_tokens),
                    load_duration_seconds=output.load_duration_seconds,
                    inference_duration_seconds=output.inference_duration_seconds,
                    raw_response=output.raw,
                )

        return PageResult(
            page_number=page.number,
            status=OCRStatus.FAILED,
            error=last.message if last else "The page could not be processed.",
            duration_seconds=self._elapsed(page_started),
            raw_response={"remedy": last.remedy} if last and last.remedy else {},
        )


def _aggregate_status(pages: list[PageResult]) -> OCRStatus:
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


def _first_remedy(pages: list[PageResult]) -> str:
    for page in pages:
        remedy = page.raw_response.get("remedy")
        if remedy:
            return str(remedy)
    return ""


def build_unlimited_provider(settings: AppSettings) -> UnlimitedOCRProvider:
    """Registry factory."""
    return UnlimitedOCRProvider(settings)


__all__ = [
    "BACKENDS",
    "UnlimitedOCRProvider",
    "build_backend",
    "build_unlimited_provider",
]
