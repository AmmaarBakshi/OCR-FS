"""Deciding whether a page's transcription is worth a second opinion.

Running two vision models over every page doubles the most expensive thing
the system does in order to find disagreement on the small minority of pages
where disagreement exists. The cheaper policy is to read each page once and
spend the second model only where the first result looks unsafe.

That needs a confidence signal, and vision-language models do not report one.
So this module derives one from the transcription itself, using properties
that are checkable without another model:

* **Nothing came back.** An empty or failed page is the clearest signal there
  is.
* **Output stopped at the token limit.** The page is cut off mid-transcription
  and the tail is missing.
* **The model started repeating itself.** A small VLM on a hard page falls
  into a loop, emitting the same line until it runs out of budget. Real
  documents do repeat - a column of identical amounts, a row of leader dots -
  so the test is a long *consecutive* run, not mere duplication.
* **The page has ink the text does not account for.** A dense scan that
  transcribed to two lines was not read.
* **The text is mostly not language.** A collapse into punctuation or symbol
  soup.

None of these prove the transcription is wrong; they identify pages where
another look is worth eight minutes of CPU. The scoring is deliberately
conservative in the expensive direction: a page is only flagged when a named
signal fires, so a clean run flags nothing and costs nothing extra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ocr_fusion.config.schema import ConfidenceSettings
from ocr_fusion.ocr.interface import OCRStatus, PageResult
from ocr_fusion.pipeline.routing import PageRoute

#: Tokens that count as language: a word or a figure, including the internal
#: separators that hold "INV-1024" and "1,325.50" together.
_WORD_RE = re.compile(r"[A-Za-z0-9][\w,.\-/']*")


class ConfidenceFlag(str, Enum):
    """A named reason a page looks unsafe. Shown to the user verbatim."""

    EMPTY = "empty"
    TRUNCATED = "truncated"
    REPETITION = "repetition"
    SPARSE_FOR_INK = "sparse_for_ink"
    NOT_LANGUAGE = "not_language"


#: What each flag means, in words a non-engineer can act on.
FLAG_REASONS: dict[ConfidenceFlag, str] = {
    ConfidenceFlag.EMPTY: "the engine returned nothing for this page",
    ConfidenceFlag.TRUNCATED: "the transcription stopped at the token limit, so the end of the page is missing",
    ConfidenceFlag.REPETITION: "the engine repeated the same line many times, which means it lost its place",
    ConfidenceFlag.SPARSE_FOR_INK: "the page carries far more ink than the transcription accounts for",
    ConfidenceFlag.NOT_LANGUAGE: "most of what came back does not read as words or figures",
}


@dataclass(slots=True)
class PageConfidence:
    """How much to trust one page of one engine's output."""

    page_number: int
    score: float
    """0.0-1.0. Derived from the transcription, never self-reported by a model."""

    flags: list[ConfidenceFlag] = field(default_factory=list)

    @property
    def needs_second_opinion(self) -> bool:
        return bool(self.flags)

    @property
    def reason(self) -> str:
        """Why this page was flagged, as one sentence."""
        if not self.flags:
            return "the transcription looks complete"
        return "; ".join(FLAG_REASONS[flag] for flag in self.flags)

    def as_dict(self) -> dict[str, object]:
        return {
            "page": self.page_number,
            "score": round(self.score, 3),
            "flags": [f.value for f in self.flags],
            "reason": self.reason,
        }


def _longest_repeat_run(text: str) -> int:
    """Length of the longest run of identical consecutive non-blank lines."""
    longest = current = 0
    previous: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            previous = stripped
    return longest


def _language_ratio(text: str) -> float:
    """Share of whitespace-separated tokens that are words or figures."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _WORD_RE.match(t)) / len(tokens)


def score_page(
    page: PageResult,
    settings: ConfidenceSettings,
    route: PageRoute | None = None,
) -> PageConfidence:
    """Judge one page's transcription.

    ``route`` supplies the ink measurement when routing has already taken it,
    which is what makes the "dense page, thin transcription" test possible
    without decoding the image a second time.
    """
    flags: list[ConfidenceFlag] = []
    text = page.text or ""
    words = len(text.split())

    if page.status is OCRStatus.FAILED or not text.strip():
        return PageConfidence(page.page_number, 0.0, [ConfidenceFlag.EMPTY])

    # Ollama reports this as done_reason="length"; the provider turns it into
    # an error string while keeping the page successful.
    if page.raw_response.get("done_reason") == "length":
        flags.append(ConfidenceFlag.TRUNCATED)

    if _longest_repeat_run(text) >= settings.max_repeated_lines:
        flags.append(ConfidenceFlag.REPETITION)

    ink = route.ink_ratio if route else None
    if (
        ink is not None
        and ink >= settings.dense_ink_ratio
        and words < settings.min_words_for_dense_page
    ):
        flags.append(ConfidenceFlag.SPARSE_FOR_INK)

    ratio = _language_ratio(text)
    if ratio < settings.min_language_ratio:
        flags.append(ConfidenceFlag.NOT_LANGUAGE)

    # One clean signal is worth more than an arithmetic blend of five: the
    # score exists to order the review queue, and the flags are what explain
    # the decision.
    score = max(0.0, 1.0 - 0.3 * len(flags))
    return PageConfidence(page.page_number, score, flags)


__all__ = [
    "FLAG_REASONS",
    "ConfidenceFlag",
    "PageConfidence",
    "score_page",
]
