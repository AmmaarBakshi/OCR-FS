"""Judging whether a PDF's own text layer can be trusted.

A born-digital PDF already contains its text, exactly, as the authoring
program wrote it. Re-deriving that text with a vision model is the most
expensive way imaginable of learning something the file already states: on
this project's own corpus a 150 DPI page costs a 3B VLM around eight minutes
of CPU, against about two milliseconds to read the text layer - and the text
layer is the *more* accurate of the two, because it is not a transcription at
all.

So the text layer is worth using wherever it is real. The catch is that "has a
text layer" and "has a usable text layer" are different questions. A PDF can
carry:

* a complete, correct layer (a tax package, a bank e-statement);
* a layer from someone else's OCR pass, which may be wrong;
* a broken layer, where a missing or corrupt font encoding turns every glyph
  into a replacement character or an unrelated codepoint;
* a token layer - a page number and a footer stamped over a scan, and nothing
  else.

:func:`assess_text_layer` separates the first case from the rest using cheap,
deterministic signals. It is deliberately conservative: when a layer looks
doubtful the page is sent to OCR, because a slow correct answer beats a fast
wrong one. Nothing here rewrites text - :func:`clean_text_layer` only repairs
encoding damage and whitespace.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Characters a PDF viewer shows when it cannot map a glyph. A layer full of
#: these has a broken font encoding and is worse than no layer at all.
_REPLACEMENT_CHARS = "�￾￿"

#: Words that carry a vowel, the cheapest available signal that a run of
#: letters is language rather than the output of a scrambled CID map.
_VOWEL_RE = re.compile(r"[aeiouy]", re.IGNORECASE)

#: A "word" for counting purposes: any run of non-space characters, which
#: keeps reference numbers and amounts together as single tokens.
_WORD_RE = re.compile(r"\S+")

#: Runs of spaces or tabs that PDF text extraction leaves between columns.
_HSPACE_RE = re.compile(r"[ \t]{2,}")

#: Three or more blank lines, which extraction produces around figures.
_BLANKS_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class TextLayerAssessment:
    """What a page's embedded text looks like, and whether to trust it."""

    usable: bool
    """True when the layer is complete enough to stand in for OCR."""

    quality: float
    """0.0-1.0. The share of words that read as real language or figures."""

    word_count: int
    reason: str
    """Plain-language explanation, shown in the run log when a page is routed."""

    def as_dict(self) -> dict[str, object]:
        return {
            "usable": self.usable,
            "quality": round(self.quality, 3),
            "word_count": self.word_count,
            "reason": self.reason,
        }


def clean_text_layer(text: str) -> str:
    """Repair extraction artefacts without touching the wording.

    Three things only: normalise the Unicode form so visually identical
    characters compare equal, collapse the column padding extraction leaves
    behind, and trim trailing whitespace. Words, figures, spelling and reading
    order are left exactly as the document has them.
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _HSPACE_RE.sub(" ", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = _BLANKS_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _looks_like_language(word: str) -> bool:
    """Whether one token reads as something a person wrote.

    A figure, a reference number or a date counts: on a financial document
    those are the tokens that matter most. A long run of consonants with no
    vowel does not - that is the signature of a broken glyph mapping.
    """
    if any(ch.isdigit() for ch in word):
        return True
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        # Punctuation-only tokens (bullets, leader dots, rule characters) are
        # neutral rather than evidence of damage.
        return True
    if len(letters) <= 2:
        return True
    return bool(_VOWEL_RE.search(word))


def assess_text_layer(
    text: str | None,
    *,
    min_words: int = 25,
    min_quality: float = 0.7,
) -> TextLayerAssessment:
    """Decide whether ``text`` can stand in for an OCR pass on the page.

    ``min_words`` guards against a scan with only a page number stamped on it;
    ``min_quality`` guards against a layer whose font encoding is broken. Both
    are configurable because the right threshold depends on the corpus - a
    page of a form can legitimately be sparse.
    """
    raw = text or ""
    if not raw.strip():
        return TextLayerAssessment(False, 0.0, 0, "the page carries no text layer")

    cleaned = clean_text_layer(raw)
    words = _WORD_RE.findall(cleaned)
    count = len(words)

    if count < min_words:
        return TextLayerAssessment(
            False,
            0.0,
            count,
            f"the text layer holds only {count} word(s), too few to be the page",
        )

    damaged = sum(1 for ch in cleaned if ch in _REPLACEMENT_CHARS)
    if damaged > len(cleaned) * 0.01:
        return TextLayerAssessment(
            False,
            0.0,
            count,
            "the text layer is full of replacement characters, so its font "
            "encoding is broken",
        )

    plausible = sum(1 for word in words if _looks_like_language(word))
    quality = plausible / count

    if quality < min_quality:
        return TextLayerAssessment(
            False,
            quality,
            count,
            f"only {quality:.0%} of the text layer reads as language, so it is "
            "probably a damaged encoding",
        )

    return TextLayerAssessment(
        True,
        quality,
        count,
        f"the page carries a usable text layer of {count} words",
    )


__all__ = [
    "TextLayerAssessment",
    "assess_text_layer",
    "clean_text_layer",
]
