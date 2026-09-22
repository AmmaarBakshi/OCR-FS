"""Stage 0 - the text a PDF already contains.

This is an OCR provider that performs no recognition. It reads the text layer
the PDF was authored with and hands it back, which on a born-digital page is
not an approximation of the content but the content itself.

It is a provider rather than a special case inside the pipeline for one
reason: it satisfies the same contract as every other engine, so the router,
the comparison view, fusion and the exporters treat it exactly as they treat a
vision model. The only thing that marks it out is honesty about what it is -
no model ran, so ``model_name`` stays unset and no token counts are invented.

A page whose layer does not pass :func:`~ocr_fusion.documents.textlayer.assess_text_layer`
is returned as a failed page with a plain-language reason, which is the signal
the router uses to send that page to a real OCR engine instead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ocr_fusion.config.schema import AppSettings, ProcessingLocation
from ocr_fusion.documents.models import Document, DocumentPage
from ocr_fusion.documents.textlayer import (
    TextLayerAssessment,
    assess_text_layer,
    clean_text_layer,
)
from ocr_fusion.ocr.interface import (
    HealthStatus,
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
)

logger = logging.getLogger(__name__)


class TextLayerProvider(OCRProvider):
    """Returns a PDF page's embedded text, when that text is trustworthy."""

    provider_id = "text_layer"
    provider_name = "PDF text layer"
    processing_location = ProcessingLocation.LOCAL

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.config = settings.text_layer

    # -- readiness ---------------------------------------------------------

    def health_check(self) -> HealthStatus:
        """Always ready: extraction needs no runtime, model or network.

        Whether it will *produce* anything depends on the document, which is a
        per-page question the router answers, not a readiness question.
        """
        return HealthStatus.ok(
            "Ready. Applies to PDF pages that carry their own text.",
            min_words=self.config.min_words,
            min_quality=self.config.min_quality,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "engine": "PDF text layer extraction",
            "runtime": "PyMuPDF",
            "backend": "text_layer",
            "model": None,
            "performs_recognition": False,
            "processing_location": self.processing_location.value,
            "min_words": self.config.min_words,
            "min_quality": self.config.min_quality,
        }

    # -- extraction --------------------------------------------------------

    def assess(self, page: DocumentPage) -> TextLayerAssessment:
        """Judge one page's layer. Used by the router before a run starts."""
        return assess_text_layer(
            page.embedded_text,
            min_words=self.config.min_words,
            min_quality=self.config.min_quality,
        )

    def process(
        self,
        document: Document,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> OCRResult:
        started = self._timer()
        result = OCRResult(
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            # No model ran. Reporting one would be a fabricated measurement.
            model_name=None,
            backend="text_layer",
            processing_location=self.processing_location,
            metadata=self.get_metadata(),
        )

        total = len(document.pages)
        for index, page in enumerate(document.pages, 1):
            if on_progress:
                on_progress(index, total)
            result.pages.append(self._process_page(page))

        result.duration_seconds = self._elapsed(started)
        result.status = _aggregate_status(result.pages)
        if result.status is OCRStatus.FAILED:
            result.error = "No page carried a usable text layer."
            result.remedy = (
                "This document is a scan. Leave an OCR engine enabled so the "
                "pages can be read."
            )
        return result

    def _process_page(self, page: DocumentPage) -> PageResult:
        page_started = self._timer()
        assessment = self.assess(page)

        if not assessment.usable:
            return PageResult(
                page_number=page.number,
                status=OCRStatus.FAILED,
                error=f"No usable text layer: {assessment.reason}.",
                duration_seconds=self._elapsed(page_started),
            )

        return PageResult(
            page_number=page.number,
            text=clean_text_layer(page.embedded_text or ""),
            status=OCRStatus.SUCCESS,
            duration_seconds=self._elapsed(page_started),
            # A measured property of the layer, not a model's self-report.
            confidence=assessment.quality,
            raw_response={"text_layer": assessment.as_dict()},
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


def build_text_layer_provider(settings: AppSettings) -> TextLayerProvider:
    """Registry factory."""
    return TextLayerProvider(settings)


__all__ = ["TextLayerProvider", "build_text_layer_provider"]
