"""Deciding, per page, how much computation the page is worth.

The pipeline's original shape ran every enabled engine over every page. That
is the most expensive policy available, and on the reference corpus it is
almost entirely waste: 84.9% of pages already carry their own text, some pages
are blank, and a batch repeats pages verbatim. Measured on this machine a
single 150 DPI page costs qwen2.5vl:3b about 491 seconds, so each page that
never reaches a model is worth roughly eight minutes.

This module answers the question "what does this page actually need?" using
only cheap, deterministic signals - the PDF's text layer, an exact content
hash, and the proportion of dark pixels. No model is consulted to decide
whether to consult a model, which would defeat the purpose.

The routing rules are ordered by what each one costs to evaluate:

1. **Text layer** - the PDF states its own text and the layer passes
   :func:`~ocr_fusion.documents.textlayer.assess_text_layer`. Reading it
   needs no pixels, so a page settled here is never even rendered.
2. **Blank** - almost no ink and no text. Nothing to read.
3. **Duplicate** - byte-identical to a page already routed in this run.
4. **OCR** - everything else, which is the only class that costs a model.

Rules 2-4 need pixels, and asking a page for its pixels renders it. Putting
the free rule first is what keeps an 83-page born-digital PDF from
rasterising a single page: routing it takes 0.27s in total.

Duplicate detection is an exact hash of the rendered bytes, never a
perceptual one. On financial documents two pages can be visually near
identical and differ only in the digits that matter, and a perceptual hash
would happily merge them. Exactness is the whole point here.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from ocr_fusion.config.schema import RoutingSettings
from ocr_fusion.documents.models import Document, DocumentPage
from ocr_fusion.documents.textlayer import TextLayerAssessment, assess_text_layer

logger = logging.getLogger(__name__)


class PageClass(str, Enum):
    """What a page needs, from cheapest to most expensive."""

    BLANK = "blank"
    """Effectively no ink and no text. Skipped entirely."""

    DUPLICATE = "duplicate"
    """Byte-identical to an earlier page in this run; its result is reused."""

    TEXT_LAYER = "text_layer"
    """The PDF carries usable text of its own. No recognition needed."""

    NEEDS_OCR = "needs_ocr"
    """A scan, a photograph, or a layer too poor to trust. Costs a model."""


@dataclass(slots=True)
class PageRoute:
    """The decision made about one page, and why."""

    page_number: int
    page_class: PageClass
    reason: str
    fingerprint: str
    """SHA-256 of the rendered page bytes. Identifies the page without storing it."""

    duplicate_of: int | None = None
    ink_ratio: float | None = None
    """Share of pixels darker than the ink threshold, or ``None`` if not measured."""

    text_layer: TextLayerAssessment | None = None

    @property
    def needs_model(self) -> bool:
        return self.page_class is PageClass.NEEDS_OCR

    def as_dict(self) -> dict[str, object]:
        return {
            "page": self.page_number,
            "class": self.page_class.value,
            "reason": self.reason,
            "fingerprint": self.fingerprint[:16],
            "duplicate_of": self.duplicate_of,
            "ink_ratio": None if self.ink_ratio is None else round(self.ink_ratio, 5),
            "text_layer": self.text_layer.as_dict() if self.text_layer else None,
        }


@dataclass(slots=True)
class RoutingPlan:
    """Every page's decision, plus the arithmetic that justifies it."""

    routes: list[PageRoute] = field(default_factory=list)
    duration_seconds: float = 0.0

    def route(self, page_number: int) -> PageRoute | None:
        return next((r for r in self.routes if r.page_number == page_number), None)

    @property
    def ocr_pages(self) -> list[int]:
        """Page numbers that must reach a model, in order."""
        return [r.page_number for r in self.routes if r.needs_model]

    @property
    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {c.value: 0 for c in PageClass}
        for route in self.routes:
            tally[route.page_class.value] += 1
        return tally

    @property
    def pages_avoided(self) -> int:
        """Pages that will not reach a model because of routing."""
        return len(self.routes) - len(self.ocr_pages)

    def summary(self) -> dict[str, object]:
        return {
            "pages": len(self.routes),
            "pages_needing_ocr": len(self.ocr_pages),
            "pages_avoided": self.pages_avoided,
            "counts": self.counts,
            "duration_seconds": self.duration_seconds,
            "routes": [r.as_dict() for r in self.routes],
        }

    def headline(self) -> str:
        """One line for the run log."""
        total = len(self.routes)
        ocr = len(self.ocr_pages)
        if total == 0:
            return "No pages to route."
        counts = self.counts
        parts = []
        if counts[PageClass.TEXT_LAYER.value]:
            parts.append(f"{counts[PageClass.TEXT_LAYER.value]} from the text layer")
        if counts[PageClass.DUPLICATE.value]:
            parts.append(f"{counts[PageClass.DUPLICATE.value]} duplicate")
        if counts[PageClass.BLANK.value]:
            parts.append(f"{counts[PageClass.BLANK.value]} blank")
        avoided = ", ".join(parts)
        if not avoided:
            return f"All {total} page(s) need OCR."
        return (
            f"{ocr} of {total} page(s) need OCR; the rest are covered without a "
            f"model ({avoided})."
        )


def page_fingerprint(page: DocumentPage) -> str:
    """Content address of a page: SHA-256 over its rendered bytes."""
    return hashlib.sha256(page.image_bytes).hexdigest()


def measure_ink(page: DocumentPage, *, thumbnail: int = 200, threshold: int = 200) -> float | None:
    """Share of dark pixels on a small thumbnail of the page.

    Decoding and shrinking a page costs single-digit milliseconds against the
    several minutes a model would spend on it, so this is worth doing before
    every routing decision. Returns ``None`` when the image cannot be decoded,
    which routes the page to OCR rather than guessing it is blank.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(page.image_bytes)) as image:
            grey = image.convert("L")
            # Shrinking first makes the pixel scan cheap and, because the
            # reducing filter averages, a thin stroke still darkens its cell.
            grey.thumbnail((thumbnail, thumbnail), Image.BILINEAR)
            histogram = grey.histogram()
    except Exception:  # noqa: BLE001 - an undecodable page is an OCR problem,
        # not a routing problem; say "unknown" and let the model look at it.
        logger.debug("Ink measurement failed on page %s", page.number, exc_info=True)
        return None

    total = sum(histogram)
    if not total:
        return None
    dark = sum(histogram[: threshold + 1])
    return dark / total


def route_document(document: Document, settings: RoutingSettings) -> RoutingPlan:
    """Decide what each page of ``document`` needs.

    Nothing is transcribed here and the document is not modified. Pages the
    text layer cannot answer are rendered, because the later rules need
    pixels - but those are exactly the pages an engine was going to render
    anyway, so the work is moved rather than added.
    """
    started = time.perf_counter()
    plan = RoutingPlan()
    seen: dict[str, int] = {}

    for page in document.pages:
        assessment = assess_text_layer(
            page.embedded_text,
            min_words=settings.text_layer_min_words,
            min_quality=settings.text_layer_min_quality,
        )

        # The text layer is free to read; the fingerprint and the ink ratio
        # both need pixels, and asking for pixels renders the page. So a page
        # the text layer already answers is settled here and never rendered -
        # which is the whole reason rendering is deferred in the first place.
        if settings.use_text_layer and assessment.usable:
            plan.routes.append(
                PageRoute(
                    page_number=page.number,
                    page_class=PageClass.TEXT_LAYER,
                    reason=assessment.reason,
                    fingerprint="",
                    text_layer=assessment,
                )
            )
            continue

        fingerprint = page_fingerprint(page)
        ink = (
            measure_ink(page, threshold=settings.ink_threshold)
            if settings.skip_blank_pages
            else None
        )
        plan.routes.append(
            _classify(page, fingerprint, assessment, ink, seen, settings)
        )
        seen.setdefault(fingerprint, page.number)

    plan.duration_seconds = time.perf_counter() - started
    return plan


def _classify(
    page: DocumentPage,
    fingerprint: str,
    assessment: TextLayerAssessment,
    ink: float | None,
    seen: dict[str, int],
    settings: RoutingSettings,
) -> PageRoute:
    """Classify a page the text layer could not answer.

    Reached only once the page has been rendered, so the fingerprint and the
    ink ratio are available.
    """
    common = {
        "page_number": page.number,
        "fingerprint": fingerprint,
        "ink_ratio": ink,
        "text_layer": assessment,
    }

    if (
        settings.skip_blank_pages
        and not assessment.usable
        and ink is not None
        and ink < settings.blank_ink_ratio
    ):
        return PageRoute(
            page_class=PageClass.BLANK,
            reason=f"the page is blank ({ink:.3%} of it carries ink)",
            **common,
        )

    if settings.detect_duplicates and fingerprint in seen:
        original = seen[fingerprint]
        return PageRoute(
            page_class=PageClass.DUPLICATE,
            reason=f"the page is identical to page {original}",
            duplicate_of=original,
            **common,
        )

    return PageRoute(
        page_class=PageClass.NEEDS_OCR,
        reason=assessment.reason if assessment.word_count else "the page must be read by an engine",
        **common,
    )


def subset(document: Document, page_numbers: list[int]) -> Document:
    """A view of ``document`` holding only the named pages.

    This is how routing reaches the providers without any provider knowing
    routing exists: a provider is simply handed a document with fewer pages.
    Page objects are shared rather than copied, so the image bytes are not
    duplicated in memory.
    """
    wanted = set(page_numbers)
    return Document(
        filename=document.filename,
        kind=document.kind,
        pages=[p for p in document.pages if p.number in wanted],
        source_bytes_size=document.source_bytes_size,
        mime_type=document.mime_type,
        loaded_at=document.loaded_at,
        checksum=document.checksum,
        warnings=list(document.warnings),
    )


__all__ = [
    "PageClass",
    "PageRoute",
    "RoutingPlan",
    "measure_ink",
    "page_fingerprint",
    "route_document",
    "subset",
]
