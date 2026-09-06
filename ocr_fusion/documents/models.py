"""Document and page abstractions.

A :class:`Document` is the single input type the pipeline understands. Loaders
turn PDFs, images and (in future) other formats into this shape, so providers
never need to know what the user originally uploaded - they only ever see
:class:`DocumentPage` objects carrying render-ready image bytes.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DocumentKind(str, Enum):
    """The class of source file, after sniffing."""

    IMAGE = "image"
    PDF = "pdf"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class DocumentPage:
    """One renderable page.

    ``image_bytes`` is always a PNG encoding of the page, sized according to
    :class:`~ocr_fusion.config.schema.DocumentSettings`. Providers that need a
    file on disk (the CLI backend) materialise it themselves; providers that
    speak HTTP use :meth:`image_base64`.
    """

    number: int
    """1-based page number, preserved through the whole pipeline (spec s2)."""

    image_bytes: bytes
    width: int
    height: int
    embedded_text: str | None = None
    """Text extracted from a PDF text layer, when one exists. Never OCR output."""

    dpi: int | None = None

    @property
    def has_text_layer(self) -> bool:
        """True when the source page already carried selectable text."""
        return bool(self.embedded_text and self.embedded_text.strip())

    def image_base64(self) -> str:
        """Base64 PNG, the form both Ollama and OpenAI-compatible APIs accept."""
        return base64.b64encode(self.image_bytes).decode("ascii")

    def summary(self) -> dict[str, Any]:
        """Non-sensitive description of the page, safe for logs and exports."""
        return {
            "page": self.number,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "size_bytes": len(self.image_bytes),
            "has_text_layer": self.has_text_layer,
            "embedded_text_chars": len(self.embedded_text or ""),
        }


@dataclass(slots=True)
class Document:
    """A loaded document, ready for OCR."""

    filename: str
    kind: DocumentKind
    pages: list[DocumentPage] = field(default_factory=list)
    source_bytes_size: int = 0
    mime_type: str = "application/octet-stream"
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""
    """SHA-256 of the source bytes. Identifies a document without storing it."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal loader notes, surfaced in the UI (e.g. page limit applied)."""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def has_text_layer(self) -> bool:
        """True when any page carried selectable text (spec s2)."""
        return any(page.has_text_layer for page in self.pages)

    @property
    def embedded_text(self) -> str:
        """Concatenated PDF text layer, if present."""
        return "\n\n".join(p.embedded_text or "" for p in self.pages if p.has_text_layer)

    def page(self, number: int) -> DocumentPage:
        """Look up a page by its 1-based number."""
        for candidate in self.pages:
            if candidate.number == number:
                return candidate
        raise KeyError(f"Document has no page {number}")

    def metadata(self) -> dict[str, Any]:
        """Document metadata for the UI, exports and logs. Contains no content."""
        return {
            "filename": self.filename,
            "kind": self.kind.value,
            "mime_type": self.mime_type,
            "page_count": self.page_count,
            "size_bytes": self.source_bytes_size,
            "checksum_sha256": self.checksum,
            "loaded_at": self.loaded_at.isoformat(),
            "has_text_layer": self.has_text_layer,
            "pages": [page.summary() for page in self.pages],
            "warnings": list(self.warnings),
        }


def checksum_bytes(data: bytes) -> str:
    """SHA-256 hex digest, used to identify a document without persisting it."""
    return hashlib.sha256(data).hexdigest()


__all__ = ["Document", "DocumentKind", "DocumentPage", "checksum_bytes"]
