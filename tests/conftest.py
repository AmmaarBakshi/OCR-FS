"""Shared fixtures.

The whole suite runs without Ollama, without a GPU and without any model:
engines are represented by :class:`FakeProvider`, which implements the same
:class:`~ocr_fusion.ocr.interface.OCRProvider` contract as the real ones. That
is the point of the provider abstraction, and it keeps the tests fast enough to
run on every change (spec s18).
"""

from __future__ import annotations

import io

import pytest

from ocr_fusion.config.schema import AppSettings, ProcessingLocation
from ocr_fusion.documents.models import Document, DocumentKind, DocumentPage
from ocr_fusion.ocr.interface import (
    HealthStatus,
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
    TokenUsage,
)

# Two transcriptions of the same invoice, differing the way real engines do:
# markdown style, table padding, one line only the second engine found, and one
# genuine numeric disagreement.
INVOICE_A = """ACME LOGISTICS LTD
17 Harbour Road, Mumbai 400001
Invoice Number: INV-1024
| Freight forwarding | 12 | 1,250.00 | 15,000.00 |
| Insurance premium | 1 | 1,325.50 | 1,325.50 |
TOTAL DUE 38,085.10"""

INVOICE_B = """## ACME LOGISTICS LTD
17 Harbour Road, Mumbai 400001
Invoice No: INV-1024
| Freight forwarding | 12 | 1,250.00 | 15,000.00 |
| Insurance premium | 1 | 1,325.50 | 1,325.60 |
TOTAL DUE 38,085.10
Bank: HDFC Bank, A/C 50200012345678"""


class FakeProvider(OCRProvider):
    """A provider that behaves however a test needs it to.

    Covers the outcomes the pipeline must survive: success, whole-provider
    failure, a single bad page, an unexpected exception, and an engine that
    reports no token metrics at all.
    """

    def __init__(
        self,
        provider_id: str = "fake",
        provider_name: str = "Fake Engine",
        text: str = "some text",
        *,
        healthy: bool = True,
        fail: bool = False,
        raises: bool = False,
        report_tokens: bool = True,
        failing_pages: tuple[int, ...] = (),
        duration: float = 0.1,
    ) -> None:
        self.provider_id = provider_id
        self.provider_name = provider_name
        self.processing_location = ProcessingLocation.LOCAL
        self.text = text
        self.healthy = healthy
        self.fail = fail
        self.raises = raises
        self.report_tokens = report_tokens
        self.failing_pages = failing_pages
        self.duration = duration
        self.process_calls = 0

    def health_check(self) -> HealthStatus:
        if self.healthy:
            return HealthStatus.ok(f"{self.provider_name} ready")
        return HealthStatus.unavailable(
            f"{self.provider_name} is not available", "Start the engine"
        )

    def get_metadata(self) -> dict:
        return {"engine": self.provider_name, "backend": "fake"}

    def process(self, document: Document) -> OCRResult:
        self.process_calls += 1
        if self.raises:
            raise RuntimeError("engine crashed")
        if self.fail:
            return OCRResult.failure(
                self.provider_id,
                self.provider_name,
                f"{self.provider_name} is not available",
                remedy="Start the engine",
            )

        pages: list[PageResult] = []
        for page in document.pages:
            if page.number in self.failing_pages:
                pages.append(
                    PageResult(
                        page_number=page.number,
                        status=OCRStatus.FAILED,
                        error="This page could not be read.",
                    )
                )
                continue
            pages.append(
                PageResult(
                    page_number=page.number,
                    text=self.text,
                    duration_seconds=self.duration,
                    tokens=TokenUsage(100, 40) if self.report_tokens else TokenUsage(),
                )
            )

        succeeded = sum(1 for p in pages if p.status is OCRStatus.SUCCESS)
        status = (
            OCRStatus.SUCCESS
            if succeeded == len(pages)
            else OCRStatus.FAILED
            if succeeded == 0
            else OCRStatus.PARTIAL
        )
        return OCRResult(
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            status=status,
            pages=pages,
            model_name=f"{self.provider_id}:test",
            backend="fake",
            duration_seconds=self.duration * max(1, len(pages)),
            processing_location=self.processing_location,
            metadata=self.get_metadata(),
        )


@pytest.fixture
def settings() -> AppSettings:
    """Default settings, isolated per test."""
    return AppSettings()


@pytest.fixture
def single_page_document() -> Document:
    return Document(
        filename="invoice.pdf",
        kind=DocumentKind.PDF,
        pages=[DocumentPage(number=1, image_bytes=b"png", width=1240, height=1755)],
        source_bytes_size=7395,
        checksum="a" * 64,
    )


@pytest.fixture
def multi_page_document() -> Document:
    return Document(
        filename="report.pdf",
        kind=DocumentKind.PDF,
        pages=[
            DocumentPage(number=n, image_bytes=b"png", width=1240, height=1755)
            for n in (1, 2, 3)
        ],
        source_bytes_size=20000,
    )


@pytest.fixture
def qwen_like() -> FakeProvider:
    return FakeProvider("qwen_vl", "Qwen2.5-VL", INVOICE_A)


@pytest.fixture
def unlimited_like() -> FakeProvider:
    return FakeProvider("unlimited_ocr", "Unlimited-OCR", INVOICE_B)


@pytest.fixture
def png_bytes() -> bytes:
    """A real single-page PNG, for exercising the loaders."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (600, 200), "white")
    ImageDraw.Draw(image).text((20, 20), "Invoice Number: INV-1024", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def pdf_bytes() -> bytes:
    """A real two-page PDF with a text layer."""
    import fitz

    document = fitz.open()
    for index in range(2):
        page = document.new_page()
        page.insert_text((72, 100), f"Invoice page {index + 1}", fontsize=14)
        page.insert_text((72, 130), "Invoice Number: INV-1024", fontsize=11)
    data = document.tobytes()
    document.close()
    return data
