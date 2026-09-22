"""Measuring whether a faster configuration is still a correct one.

Speed work is only credible next to an accuracy number, and an accuracy
number is only credible next to a ground truth. This corpus supplies one for
free: a born-digital PDF page carries the exact text it was authored with, so
rendering that page to an image and transcribing the image produces a
prediction whose answer key is already in the file.

That gives a gold set of any size at no labelling cost, drawn from the real
documents rather than from a benchmark someone else assembled. The pages are
genuine forms, statements and tax schedules, with the tables, column layouts
and dense figures that are exactly where OCR fails.

The one thing it cannot measure is a true scan, which has no answer key. Those
still need a human, and :func:`build_gold_set` says how many pages it had to
skip for that reason.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Comparison is case-insensitive and ignores the punctuation engines
#: legitimately disagree about, so that a transcription is not marked wrong
#: for writing a dash differently. Digits, separators and currency survive,
#: because those are the characters that carry the meaning on these documents.
_KEEP = re.compile(r"[^a-z0-9.,%$/:()\-\s]")
_SPACE = re.compile(r"\s+")

#: Above this many characters, character-level alignment gets expensive
#: (difflib is O(n*m)). A page is well under it; a whole document is not, so
#: scoring is always per page.
_CER_CHAR_LIMIT = 20_000


#: Punctuation that carries meaning inside a token but not at its edges.
#: "1,325.50" and "10:30" must survive intact; "(due" and "Total:" must fold
#: to "due" and "total", or every engine that punctuates differently is
#: scored as having misread the word.
_EDGE_PUNCTUATION = ".,:;()-/$%"


def normalise(text: str) -> str:
    """Fold text to the form the metrics compare."""
    folded = _SPACE.sub(" ", _KEEP.sub(" ", text.lower())).strip()
    tokens = [token.strip(_EDGE_PUNCTUATION) for token in folded.split()]
    return " ".join(token for token in tokens if token)


def _ratio(reference: list, hypothesis: list) -> float:
    """Share of the reference that the hypothesis fails to match."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return max(0.0, 1.0 - matched / len(reference))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Share of reference words not aligned to the same word in the output."""
    return _ratio(normalise(reference).split(), normalise(hypothesis).split())


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    """Character-level error rate, or ``None`` when the page is too long to align."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if len(ref) > _CER_CHAR_LIMIT or len(hyp) > _CER_CHAR_LIMIT:
        return None
    return _ratio(list(ref), list(hyp))


def word_recall(reference: str, hypothesis: str) -> float:
    """Share of reference words that appear anywhere in the output.

    Order-insensitive, which is the point: an engine that reads every figure
    correctly but emits a table in a different order has not lost anything a
    person reading the result would care about. Reported alongside the word
    error rate so the two failure modes - misreading and re-ordering - do not
    look alike.
    """
    ref = normalise(reference).split()
    if not ref:
        return 1.0
    hyp = set(normalise(hypothesis).split())
    return sum(1 for word in ref if word in hyp) / len(ref)


def figures(text: str) -> list[str]:
    """Every number in the text, in order.

    A mistranscribed figure on a statement is the error that actually costs
    something, so it gets its own metric rather than being averaged into the
    word rate.
    """
    return re.findall(r"\d[\d,]*(?:\.\d+)?", text)


def figure_recall(reference: str, hypothesis: str) -> float:
    """Share of the reference's figures that survive into the output."""
    ref = figures(reference)
    if not ref:
        return 1.0
    from collections import Counter

    available = Counter(figures(hypothesis))
    found = 0
    for value in ref:
        if available[value] > 0:
            available[value] -= 1
            found += 1
    return found / len(ref)


@dataclass(slots=True)
class PageScore:
    """How one transcription compared with its ground truth."""

    page_number: int
    reference_words: int
    hypothesis_words: int
    wer: float
    cer: float | None
    recall: float
    figure_recall: float

    def as_dict(self) -> dict[str, object]:
        return {
            "page": self.page_number,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
            "wer": round(self.wer, 4),
            "cer": None if self.cer is None else round(self.cer, 4),
            "recall": round(self.recall, 4),
            "figure_recall": round(self.figure_recall, 4),
        }


def score_page(page_number: int, reference: str, hypothesis: str) -> PageScore:
    """Score one transcription against its ground truth."""
    return PageScore(
        page_number=page_number,
        reference_words=len(normalise(reference).split()),
        hypothesis_words=len(normalise(hypothesis).split()),
        wer=word_error_rate(reference, hypothesis),
        cer=character_error_rate(reference, hypothesis),
        recall=word_recall(reference, hypothesis),
        figure_recall=figure_recall(reference, hypothesis),
    )


@dataclass(slots=True)
class GoldPage:
    """One page whose correct transcription is known."""

    source: str
    page_number: int
    reference: str

    @property
    def word_count(self) -> int:
        return len(self.reference.split())


@dataclass(slots=True)
class GoldSet:
    """A regression set drawn from documents that state their own text."""

    pages: list[GoldPage] = field(default_factory=list)
    skipped_without_truth: int = 0
    """Scanned pages, which have no answer key and cannot be scored here."""

    def __len__(self) -> int:
        return len(self.pages)

    def summary(self) -> dict[str, object]:
        return {
            "pages": len(self.pages),
            "documents": len({p.source for p in self.pages}),
            "words": sum(p.word_count for p in self.pages),
            "skipped_without_truth": self.skipped_without_truth,
        }


def build_gold_set(
    paths: list[Path],
    *,
    min_words: int = 60,
    max_pages_per_document: int = 3,
    limit: int = 0,
) -> GoldSet:
    """Collect pages whose text layer can serve as ground truth.

    ``max_pages_per_document`` spreads the set across documents instead of
    letting one 83-page tax return dominate it, and ``min_words`` keeps out
    cover sheets that would score perfectly and measure nothing.
    """
    import fitz

    from ocr_fusion.documents.textlayer import assess_text_layer, clean_text_layer

    gold = GoldSet()
    for path in paths:
        if gold.pages and limit and len(gold.pages) >= limit:
            break
        try:
            pdf = fitz.open(path)
        except Exception:  # noqa: BLE001 - a corrupt file is not a gold page
            continue
        taken = 0
        try:
            for index in range(pdf.page_count):
                if taken >= max_pages_per_document:
                    break
                if limit and len(gold.pages) >= limit:
                    break
                text = pdf.load_page(index).get_text("text") or ""
                assessment = assess_text_layer(text)
                if not assessment.usable:
                    gold.skipped_without_truth += 1
                    continue
                cleaned = clean_text_layer(text)
                if len(cleaned.split()) < min_words:
                    continue
                gold.pages.append(GoldPage(str(path), index + 1, cleaned))
                taken += 1
        finally:
            pdf.close()
    return gold


__all__ = [
    "GoldPage",
    "GoldSet",
    "PageScore",
    "build_gold_set",
    "character_error_rate",
    "figure_recall",
    "figures",
    "normalise",
    "score_page",
    "word_error_rate",
    "word_recall",
]
