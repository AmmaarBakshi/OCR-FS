"""Normalisation applied to raw engine output.

Vision-language models reliably add a little packaging around a transcription
even when the prompt forbids it: a ```` ``` ```` fence, or a lead-in line such
as "Here is the text from the image:". Stripping that is *formatting* cleanup,
not content invention - nothing here rewrites, corrects or completes the text
the model actually produced.

The raw response is always preserved on :class:`~ocr_fusion.ocr.interface.PageResult`
so Developer Mode can show exactly what the engine returned.
"""

from __future__ import annotations

import re

#: Whole-output code fence: ```markdown\n...\n```
_FENCE_RE = re.compile(
    r"\A\s*```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?[ \t]*```\s*\Z",
    re.DOTALL,
)

#: Conversational lead-ins a VLM adds before the transcription.
_PREAMBLE_RE = re.compile(
    r"\A[ \t]*(?:"
    r"here(?:'s| is| are)[^\n:]{0,80}:"
    r"|the (?:extracted |transcribed |visible )?text[^\n:]{0,80}:"
    r"|sure[,!][^\n]{0,80}:"
    r"|certainly[,!][^\n]{0,80}:"
    r"|of course[,!][^\n]{0,80}:"
    r"|transcription[^\n:]{0,40}:"
    r"|ocr (?:result|output)[^\n:]{0,40}:"
    r")[ \t]*\r?\n+",
    re.IGNORECASE,
)

#: Trailing offers of further help.
_EPILOGUE_RE = re.compile(
    r"\r?\n+[ \t]*(?:"
    r"(?:let me know|feel free)[^\n]{0,120}"
    r"|(?:would you like|do you want)[^\n]{0,120}"
    r"|if you (?:need|have)[^\n]{0,120}"
    r")[ \t]*\Z",
    re.IGNORECASE,
)


def strip_code_fence(text: str) -> str:
    """Remove a fence that wraps the entire output.

    A fence around only part of the text is left alone - that is probably real
    document content, such as a printed code sample.
    """
    match = _FENCE_RE.match(text)
    return match.group("body") if match else text


def strip_preamble(text: str) -> str:
    """Remove a single conversational lead-in line."""
    return _PREAMBLE_RE.sub("", text, count=1)


def strip_epilogue(text: str) -> str:
    """Remove a trailing offer of further assistance."""
    return _EPILOGUE_RE.sub("", text, count=1)


def normalise_whitespace(text: str) -> str:
    """Normalise line endings and trim trailing spaces, preserving structure.

    Blank lines are meaningful (paragraph breaks), so runs of them are capped at
    two rather than collapsed away.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ocr_text(text: str) -> str:
    """Apply the full cleanup chain to one engine response."""
    if not text:
        return ""
    cleaned = strip_code_fence(text)
    cleaned = strip_preamble(cleaned)
    cleaned = strip_epilogue(cleaned)
    # A model may fence its output *and* introduce it; run the fence pass again
    # now that the preamble is gone.
    cleaned = strip_code_fence(cleaned)
    return normalise_whitespace(cleaned)


__all__ = [
    "clean_ocr_text",
    "normalise_whitespace",
    "strip_code_fence",
    "strip_epilogue",
    "strip_preamble",
]
