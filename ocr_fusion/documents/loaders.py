"""Turn uploaded bytes into a :class:`~ocr_fusion.documents.models.Document`.

The loader registry is keyed by :class:`DocumentKind`, so adding DOCX support
later means writing one function and registering it - no pipeline or UI change
(spec s2: "the architecture should allow future support for DOCX").
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

from ocr_fusion.config.schema import DocumentSettings
from ocr_fusion.documents.errors import (
    CorruptDocumentError,
    DocumentTooLargeError,
    EmptyDocumentError,
    PdfRenderError,
    UnsupportedFormatError,
)
from ocr_fusion.documents.models import (
    Document,
    DocumentKind,
    DocumentPage,
    checksum_bytes,
)

logger = logging.getLogger(__name__)

#: Extensions accepted by the UI file picker.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

#: Magic-number prefixes, checked before trusting a file extension.
_MAGIC: tuple[tuple[bytes, DocumentKind], ...] = (
    (b"%PDF-", DocumentKind.PDF),
    (b"\x89PNG\r\n\x1a\n", DocumentKind.IMAGE),
    (b"\xff\xd8\xff", DocumentKind.IMAGE),  # JPEG
    (b"GIF87a", DocumentKind.IMAGE),
    (b"GIF89a", DocumentKind.IMAGE),
    (b"BM", DocumentKind.IMAGE),  # BMP
    (b"II*\x00", DocumentKind.IMAGE),  # TIFF little-endian
    (b"MM\x00*", DocumentKind.IMAGE),  # TIFF big-endian
)

_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
}


def sniff_kind(data: bytes, filename: str = "") -> DocumentKind:
    """Identify a document from its magic number, falling back to its extension.

    Content is checked first so a mislabelled ``.png`` that is really a PDF is
    still handled correctly.
    """
    for prefix, kind in _MAGIC:
        if data.startswith(prefix):
            return kind
    # RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return DocumentKind.IMAGE

    suffix = Path(filename).suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return DocumentKind.PDF
    if suffix in IMAGE_EXTENSIONS:
        return DocumentKind.IMAGE
    return DocumentKind.UNKNOWN


def _downscale(image, max_dimension: int):
    """Shrink an image so its longest edge fits ``max_dimension``.

    Vision models tile large images; sending a 6000px scan multiplies latency
    without improving transcription, so pages are capped before inference.
    """
    from PIL import Image

    longest = max(image.width, image.height)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def _encode_png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def load_image_document(
    data: bytes, filename: str, settings: DocumentSettings
) -> Document:
    """Load a single- or multi-frame image (TIFF may carry several pages)."""
    try:
        from PIL import Image, ImageSequence
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
        raise CorruptDocumentError(
            "Image support is unavailable because Pillow is not installed."
        ) from exc

    try:
        source = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise CorruptDocumentError(
            "The image could not be opened. It may be damaged or in an "
            "unsupported encoding."
        ) from exc

    pages: list[DocumentPage] = []
    try:
        for index, frame in enumerate(ImageSequence.Iterator(source), start=1):
            # Flatten to RGB: alpha channels and palettes confuse some engines,
            # and PNG round-tripping of P-mode images loses fidelity.
            converted = frame.convert("RGB")
            converted = _downscale(converted, settings.max_image_dimension)
            pages.append(
                DocumentPage(
                    number=index,
                    image_bytes=_encode_png(converted),
                    width=converted.width,
                    height=converted.height,
                )
            )
    except Exception as exc:
        raise CorruptDocumentError(
            "The image could not be decoded past page "
            f"{len(pages) + 1}. The file may be truncated."
        ) from exc

    if not pages:
        raise EmptyDocumentError()

    return Document(
        filename=filename,
        kind=DocumentKind.IMAGE,
        pages=pages,
        source_bytes_size=len(data),
        mime_type=_MIME_BY_EXTENSION.get(Path(filename).suffix.lower(), "image/png"),
        checksum=checksum_bytes(data),
    )


def _render_pdf_page(data: bytes, index: int, settings: DocumentSettings) -> bytes:
    """Rasterise one page of an in-memory PDF to PNG bytes.

    Re-opening the document per page costs about 4ms against the ~149ms the
    render itself takes, which is a small price for not holding an open
    MuPDF handle for the lifetime of a browser session.
    """
    import fitz

    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        zoom = settings.pdf_render_dpi / 72.0
        page = pdf.load_page(index)
        try:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        except Exception as exc:
            raise PdfRenderError(
                f"Page {index + 1} could not be converted to an image."
            ) from exc

        image_bytes = pixmap.tobytes("png")
        if max(pixmap.width, pixmap.height) > settings.max_image_dimension:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = _downscale(image, settings.max_image_dimension)
            image_bytes = _encode_png(image)
        return image_bytes
    finally:
        pdf.close()


def _rendered_size(width_pt: float, height_pt: float, settings: DocumentSettings) -> tuple[int, int]:
    """Pixel size a page will have once rendered, computed without rendering."""
    zoom = settings.pdf_render_dpi / 72.0
    width, height = round(width_pt * zoom), round(height_pt * zoom)
    longest = max(width, height)
    if longest > settings.max_image_dimension:
        scale = settings.max_image_dimension / longest
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    return width, height


def load_pdf_document(data: bytes, filename: str, settings: DocumentSettings) -> Document:
    """Read a PDF's text layer and prepare its pages for rendering on demand.

    Pages are *not* rasterised here. Most pages of a born-digital PDF are
    answered from their own text and never need pixels, and rendering one
    costs roughly thirty times what reading its text costs - so an 83-page
    tax return used to spend about twelve seconds producing images that
    nothing would ever look at. A page that does need an image renders the
    moment something asks for it (spec s2).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise PdfRenderError(
            "PDF support is unavailable because PyMuPDF is not installed. "
            "Install it with: pip install pymupdf"
        ) from exc

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise CorruptDocumentError(
            "The PDF could not be opened. It may be damaged or password protected."
        ) from exc

    warnings: list[str] = []
    pages: list[DocumentPage] = []
    try:
        if pdf.needs_pass:
            raise CorruptDocumentError(
                "The PDF is password protected, so its pages cannot be read."
            )

        total = pdf.page_count
        if total == 0:
            raise EmptyDocumentError()

        for index in range(total):
            page = pdf.load_page(index)
            width, height = _rendered_size(
                page.rect.width, page.rect.height, settings
            )

            embedded = None
            if settings.detect_text_layer:
                try:
                    extracted = page.get_text("text") or ""
                    embedded = extracted if extracted.strip() else None
                except Exception:  # pragma: no cover - text layer is best effort
                    logger.debug("Text layer extraction failed on page %s", index + 1)

            pages.append(
                DocumentPage(
                    number=index + 1,
                    width=width,
                    height=height,
                    embedded_text=embedded,
                    dpi=settings.pdf_render_dpi,
                    # Bound at definition time: a late-binding closure over the
                    # loop variable would render whatever page came last.
                    render=partial(_render_pdf_page, data, index, settings),
                )
            )
    finally:
        pdf.close()

    return Document(
        filename=filename,
        kind=DocumentKind.PDF,
        pages=pages,
        source_bytes_size=len(data),
        mime_type="application/pdf",
        checksum=checksum_bytes(data),
        warnings=warnings,
    )


#: Loader registry. Register a new kind here to add a document format.
LOADERS: dict[DocumentKind, Callable[[bytes, str, DocumentSettings], Document]] = {
    DocumentKind.IMAGE: load_image_document,
    DocumentKind.PDF: load_pdf_document,
}


def load_document_bytes(
    data: bytes,
    filename: str,
    settings: DocumentSettings | None = None,
) -> Document:
    """Load a document from raw bytes.

    Raises a :class:`~ocr_fusion.documents.errors.DocumentError` subclass whose
    ``user_message`` is safe to show directly in the UI.
    """
    settings = settings or DocumentSettings()

    if not data:
        raise EmptyDocumentError("The uploaded file is empty.")

    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise DocumentTooLargeError(
            f"The file is {size_mb:.1f} MB, above the "
            f"{settings.max_file_size_mb:.0f} MB limit."
        )

    kind = sniff_kind(data, filename)
    loader = LOADERS.get(kind)
    if loader is None:
        suffix = Path(filename).suffix or "(no extension)"
        raise UnsupportedFormatError(
            f"{suffix} files are not supported. Supported types: PNG, JPG, "
            "WEBP, TIFF and PDF."
        )
    return loader(data, filename or f"document.{kind.value}", settings)


def load_document(
    path: str | Path, settings: DocumentSettings | None = None
) -> Document:
    """Load a document from a filesystem path."""
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise CorruptDocumentError(f"Could not read {file_path.name}: {exc}") from exc
    return load_document_bytes(data, file_path.name, settings)


def apply_page_limit(document: Document, max_pages: int) -> Document:
    """Trim a document to ``max_pages``, recording a warning when pages are cut.

    Returns the document unchanged when ``max_pages`` is 0 (no limit).
    """
    if max_pages <= 0 or document.page_count <= max_pages:
        return document
    dropped = document.page_count - max_pages
    document.pages = document.pages[:max_pages]
    document.warnings.append(
        f"Only the first {max_pages} of {max_pages + dropped} pages were "
        "processed because of the configured page limit."
    )
    return document


__all__ = [
    "IMAGE_EXTENSIONS",
    "LOADERS",
    "PDF_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "apply_page_limit",
    "load_document",
    "load_document_bytes",
    "load_image_document",
    "load_pdf_document",
    "sniff_kind",
]
