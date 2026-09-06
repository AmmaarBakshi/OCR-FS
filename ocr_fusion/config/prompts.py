"""Default prompt templates.

Prompts are data, not logic. They live here (and are overridable at runtime via
the settings store) so that tuning model behaviour never requires touching
provider or pipeline code.

Every prompt supports ``str.format``-style placeholders. Unknown placeholders
are left untouched by :func:`render`, so a user editing a prompt in Settings
cannot crash a run by typing a stray brace.
"""

from __future__ import annotations

import string

QWEN_SYSTEM_PROMPT = """\
You are a precise optical character recognition engine.

Your only task is to transcribe the text that is visually present in the image.

Rules:
- Reproduce the text exactly as it appears, preserving the original wording,
  spelling, casing, punctuation and numbering.
- Preserve the reading order and the visual structure of the page. Keep
  headings, paragraphs, lists and line breaks where they occur.
- Render tables as GitHub-flavoured Markdown tables when a table is present.
- If a region is illegible, write [illegible] instead of guessing.
- Never translate, summarise, explain, correct or complete the content.
- Never add commentary, preamble or closing remarks.

Output the transcription only.\
"""

QWEN_USER_PROMPT = """\
Transcribe every piece of text visible in this image, working from the top of
the page down to the bottom.

Include the header, any company name or logo text, addresses, reference and
document numbers, dates, the body or table content, totals, and any footer or
small print. Do not stop after the main table.

Return plain text with Markdown used only where it reflects the document's own
structure (headings, lists, tables). Do not wrap the whole answer in a code
fence.\
"""

UNLIMITED_OCR_PROMPT = """<image>\nFree OCR."""
"""Task prompt sent to Unlimited-OCR.

The upstream model is prompt-driven and understands several task verbs, e.g.
``<image>\nFree OCR.`` for plain transcription and
``<image>\ndocument parsing.`` for layout-aware structured extraction.
"""

UNLIMITED_OCR_LAYOUT_PROMPT = """<image>\ndocument parsing."""

FUSION_SYSTEM_PROMPT = """\
You are a careful OCR reconciliation engine.

You receive two independent transcriptions of the SAME page produced by two
different OCR engines. Your task is to produce a single corrected transcription.

Rules:
- Use only content that appears in at least one of the two transcriptions.
- Never invent, infer, complete or hallucinate text that is in neither input.
- Where the two inputs agree, keep the agreed text verbatim.
- Where they disagree, choose the reading that is more likely to be a correct
  transcription of a real document: prefer complete words over fragments,
  prefer plausible number and date formats, and prefer the variant whose
  surrounding context is intact.
- Preserve the document's structure, reading order and line breaks.
- If one input is empty or is an error message, return the other verbatim.
- Do not add commentary, notes, confidence scores or explanations.

Output the reconciled transcription only.\
"""

FUSION_USER_PROMPT = """\
Reconcile these two OCR transcriptions of the same page into one accurate text.

--- TRANSCRIPTION A ({engine_a}) ---
{text_a}

--- TRANSCRIPTION B ({engine_b}) ---
{text_b}

--- END ---

Return the reconciled transcription only.\
"""


class _SafeFormatter(string.Formatter):
    """Formatter that leaves unknown placeholders in place instead of raising."""

    def get_value(self, key, args, kwargs):  # type: ignore[override]
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)


_FORMATTER = _SafeFormatter()


def render(template: str, **values: object) -> str:
    """Fill ``template`` with ``values``, tolerating unknown/stray placeholders.

    User-editable prompts must never be able to raise ``KeyError`` or
    ``ValueError`` at run time, so malformed braces degrade to literal text.
    """
    try:
        return _FORMATTER.vformat(template, (), dict(values))
    except (ValueError, IndexError):
        # Unbalanced braces typed by a user - fall back to the raw template.
        return template


DEFAULT_PROMPTS: dict[str, str] = {
    "qwen_system": QWEN_SYSTEM_PROMPT,
    "qwen_user": QWEN_USER_PROMPT,
    "unlimited_ocr_task": UNLIMITED_OCR_PROMPT,
    "fusion_system": FUSION_SYSTEM_PROMPT,
    "fusion_user": FUSION_USER_PROMPT,
}

__all__ = [
    "QWEN_SYSTEM_PROMPT",
    "QWEN_USER_PROMPT",
    "UNLIMITED_OCR_PROMPT",
    "UNLIMITED_OCR_LAYOUT_PROMPT",
    "FUSION_SYSTEM_PROMPT",
    "FUSION_USER_PROMPT",
    "DEFAULT_PROMPTS",
    "render",
]
