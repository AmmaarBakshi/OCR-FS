"""Document loading: format detection, PDF rendering, errors and limits."""

from __future__ import annotations

import io

import pytest

from ocr_fusion.config.schema import DocumentSettings
from ocr_fusion.documents import (
    DocumentKind,
    apply_page_limit,
    load_document,
    load_document_bytes,
    sniff_kind,
)
from ocr_fusion.documents.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from ocr_fusion.documents.models import Document, DocumentPage, checksum_bytes


class TestFormatDetection:
    def test_detects_pdf_by_magic_number(self, pdf_bytes):
        assert sniff_kind(pdf_bytes, "x.pdf") is DocumentKind.PDF

    def test_detects_png_by_magic_number(self, png_bytes):
        assert sniff_kind(png_bytes, "x.png") is DocumentKind.IMAGE

    def test_content_beats_a_wrong_extension(self, pdf_bytes):
        # A PDF saved as .png must still be treated as a PDF.
        assert sniff_kind(pdf_bytes, "actually.png") is DocumentKind.PDF

    def test_extension_used_when_content_is_unrecognised(self):
        assert sniff_kind(b"\x00\x00\x00", "photo.jpg") is DocumentKind.IMAGE

    def test_unknown_content_and_extension(self):
        assert sniff_kind(b"\x00\x00\x00", "thing.xyz") is DocumentKind.UNKNOWN


class TestImageLoading:
    def test_loads_a_png(self, png_bytes):
        document = load_document_bytes(png_bytes, "scan.png")
        assert document.kind is DocumentKind.IMAGE
        assert document.page_count == 1
        assert document.page(1).image_bytes.startswith(b"\x89PNG")
        assert document.checksum == checksum_bytes(png_bytes)

    def test_image_has_no_text_layer(self, png_bytes):
        assert load_document_bytes(png_bytes, "scan.png").has_text_layer is False

    def test_oversized_pages_are_downscaled(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (5000, 1000), "white").save(buffer, "PNG")
        document = load_document_bytes(
            buffer.getvalue(), "big.png", DocumentSettings(max_image_dimension=1024)
        )
        assert max(document.page(1).width, document.page(1).height) == 1024

    def test_multi_frame_tiff_becomes_multiple_pages(self):
        from PIL import Image

        frames = [Image.new("RGB", (200, 100), shade) for shade in ("white", "gray")]
        buffer = io.BytesIO()
        frames[0].save(buffer, "TIFF", save_all=True, append_images=frames[1:])
        document = load_document_bytes(buffer.getvalue(), "multi.tiff")
        assert document.page_count == 2
        assert [p.number for p in document.pages] == [1, 2]


class TestPdfLoading:
    def test_renders_every_page(self, pdf_bytes):
        document = load_document_bytes(pdf_bytes, "invoice.pdf")
        assert document.kind is DocumentKind.PDF
        assert document.page_count == 2
        assert all(page.image_bytes for page in document.pages)

    def test_page_numbers_start_at_one(self, pdf_bytes):
        assert [p.number for p in load_document_bytes(pdf_bytes, "x.pdf").pages] == [1, 2]

    def test_text_layer_is_detected_and_kept_separate(self, pdf_bytes):
        document = load_document_bytes(pdf_bytes, "invoice.pdf")
        assert document.has_text_layer
        assert "INV-1024" in document.page(1).embedded_text
        # The text layer is never presented as OCR output.
        assert document.page(1).embedded_text != ""

    def test_text_layer_detection_can_be_switched_off(self, pdf_bytes):
        document = load_document_bytes(
            pdf_bytes, "x.pdf", DocumentSettings(detect_text_layer=False)
        )
        assert document.has_text_layer is False

    def test_render_dpi_is_honoured(self, pdf_bytes):
        low = load_document_bytes(pdf_bytes, "x.pdf", DocumentSettings(pdf_render_dpi=72))
        high = load_document_bytes(pdf_bytes, "x.pdf", DocumentSettings(pdf_render_dpi=200))
        assert high.page(1).width > low.page(1).width
        assert low.page(1).dpi == 72


class TestErrors:
    def test_empty_file(self):
        with pytest.raises(EmptyDocumentError):
            load_document_bytes(b"", "empty.png")

    def test_unsupported_format(self):
        with pytest.raises(UnsupportedFormatError) as excinfo:
            load_document_bytes(b"\x00\x01\x02rubbish", "notes.docx")
        assert ".docx" in excinfo.value.user_message

    def test_file_too_large(self, png_bytes):
        with pytest.raises(DocumentTooLargeError) as excinfo:
            load_document_bytes(
                b"x" * 3_000_000, "big.png", DocumentSettings(max_file_size_mb=1)
            )
        assert "MB" in excinfo.value.user_message

    def test_corrupt_pdf(self):
        with pytest.raises(CorruptDocumentError):
            load_document_bytes(b"%PDF-1.4 truncated garbage", "broken.pdf")

    def test_corrupt_image(self):
        with pytest.raises(CorruptDocumentError):
            load_document_bytes(b"\x89PNG\r\n\x1a\n corrupted", "broken.png")

    def test_errors_carry_a_readable_message(self):
        # The UI shows user_message directly, so it must never be blank.
        with pytest.raises(UnsupportedFormatError) as excinfo:
            load_document_bytes(b"\x00rubbish", "x.zip")
        assert len(excinfo.value.user_message) > 20

    def test_missing_path(self, tmp_path):
        with pytest.raises(CorruptDocumentError):
            load_document(tmp_path / "nope.png")


class TestPageLimit:
    def test_trims_and_warns(self, pdf_bytes):
        document = apply_page_limit(load_document_bytes(pdf_bytes, "x.pdf"), 1)
        assert document.page_count == 1
        assert document.warnings and "page limit" in document.warnings[0]

    def test_zero_means_no_limit(self, pdf_bytes):
        document = apply_page_limit(load_document_bytes(pdf_bytes, "x.pdf"), 0)
        assert document.page_count == 2
        assert document.warnings == []

    def test_limit_above_page_count_is_a_no_op(self, pdf_bytes):
        document = apply_page_limit(load_document_bytes(pdf_bytes, "x.pdf"), 99)
        assert document.page_count == 2
        assert document.warnings == []


class TestMetadata:
    def test_metadata_contains_no_document_text(self, pdf_bytes):
        # Metadata goes into logs and exports, so content must stay out of it.
        metadata = load_document_bytes(pdf_bytes, "invoice.pdf").metadata()
        assert "INV-1024" not in str(metadata)
        assert metadata["page_count"] == 2
        assert metadata["filename"] == "invoice.pdf"

    def test_page_summary_reports_shape_only(self):
        page = DocumentPage(1, b"x" * 100, 800, 600, embedded_text="secret text")
        summary = page.summary()
        assert summary["embedded_text_chars"] == 11
        assert "secret" not in str(summary)

    def test_lookup_by_page_number(self):
        document = Document(
            filename="x.pdf",
            kind=DocumentKind.PDF,
            pages=[DocumentPage(n, b"x", 10, 10) for n in (1, 2)],
        )
        assert document.page(2).number == 2
        with pytest.raises(KeyError):
            document.page(7)

    def test_base64_is_valid(self, png_bytes):
        import base64

        page = load_document_bytes(png_bytes, "x.png").page(1)
        assert base64.b64decode(page.image_base64()) == page.image_bytes
