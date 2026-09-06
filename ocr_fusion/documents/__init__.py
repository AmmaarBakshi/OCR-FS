"""Document input layer: loading, page rendering and metadata."""

from ocr_fusion.documents.errors import (
    CorruptDocumentError,
    DocumentError,
    DocumentTooLargeError,
    EmptyDocumentError,
    PdfRenderError,
    UnsupportedFormatError,
)
from ocr_fusion.documents.loaders import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    apply_page_limit,
    load_document,
    load_document_bytes,
    sniff_kind,
)
from ocr_fusion.documents.models import (
    Document,
    DocumentKind,
    DocumentPage,
    checksum_bytes,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "PDF_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
    "CorruptDocumentError",
    "Document",
    "DocumentError",
    "DocumentKind",
    "DocumentPage",
    "DocumentTooLargeError",
    "EmptyDocumentError",
    "PdfRenderError",
    "UnsupportedFormatError",
    "apply_page_limit",
    "checksum_bytes",
    "load_document",
    "load_document_bytes",
    "sniff_kind",
]
