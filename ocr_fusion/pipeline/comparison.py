"""Comparison of two engine outputs.

Neither raw result is ever discarded (spec s5). This module measures how far
apart the engines are and produces an aligned, line-level diff the UI renders
side by side - the view that makes a two-engine pipeline worth watching:

    Qwen:          "Invoice Number: INV-1024"
    Unlimited-OCR: "Invoice No: INV-1024"

Everything here is pure text analysis on the standard library's
:mod:`difflib`; no model is involved, so comparison is effectively free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any


class DiffKind(str, Enum):
    """Relationship between the two engines on one aligned line."""

    EQUAL = "equal"
    """Both engines produced the same line."""

    CHANGED = "changed"
    """Both produced a line, but they differ."""

    ONLY_A = "only_a"
    """Only the first engine produced this line."""

    ONLY_B = "only_b"
    """Only the second engine produced this line."""


@dataclass(slots=True)
class DiffLine:
    """One aligned row of the comparison view."""

    kind: DiffKind
    text_a: str = ""
    text_b: str = ""
    similarity: float = 1.0
    """0.0-1.0 for a changed line; 1.0 for an equal one."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "similarity": round(self.similarity, 4),
        }


@dataclass(slots=True)
class ComparisonResult:
    """The full comparison between two engine outputs."""

    engine_a: str
    engine_b: str
    similarity: float = 0.0
    """Overall agreement, 0.0-1.0, aggregated from the aligned rows so it
    reflects exactly what the comparison view shows."""

    lines: list[DiffLine] = field(default_factory=list)
    equal_lines: int = 0
    changed_lines: int = 0
    only_a_lines: int = 0
    only_b_lines: int = 0
    character_count_a: int = 0
    character_count_b: int = 0
    word_count_a: int = 0
    word_count_b: int = 0
    numeric_conflicts: list[dict[str, Any]] = field(default_factory=list)
    """Lines where the engines disagree on digits - the errors that matter most
    on invoices and forms, so they are called out separately."""

    note: str = ""
    """Set when a comparison could not be made, e.g. only one engine succeeded."""

    @property
    def agreement_percent(self) -> float:
        return round(self.similarity * 100, 1)

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    @property
    def differing_lines(self) -> int:
        return self.changed_lines + self.only_a_lines + self.only_b_lines

    def disagreements(self, limit: int = 0) -> list[DiffLine]:
        """Only the rows where the engines differ."""
        rows = [line for line in self.lines if line.kind is not DiffKind.EQUAL]
        return rows[:limit] if limit > 0 else rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "similarity": round(self.similarity, 4),
            "agreement_percent": self.agreement_percent,
            "total_lines": self.total_lines,
            "equal_lines": self.equal_lines,
            "changed_lines": self.changed_lines,
            "only_a_lines": self.only_a_lines,
            "only_b_lines": self.only_b_lines,
            "character_count_a": self.character_count_a,
            "character_count_b": self.character_count_b,
            "word_count_a": self.word_count_a,
            "word_count_b": self.word_count_b,
            "numeric_conflicts": self.numeric_conflicts,
            "note": self.note,
            "lines": [line.as_dict() for line in self.lines],
        }


_NUMBER_RE = re.compile(r"\d[\d,.\-/]*")
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_line(line: str) -> str:
    """Collapse whitespace for alignment purposes.

    Engines format tables differently - one pads cells, another does not - and
    that is presentation, not a transcription disagreement.
    """
    return _WHITESPACE_RE.sub(" ", line).strip()


def similarity_ratio(a: str, b: str) -> float:
    """Character-level similarity between two strings, 0.0-1.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def extract_numbers(text: str) -> list[str]:
    """Digit sequences, with thousands separators stripped for comparison.

    ``1,250.00`` and ``1250.00`` are the same amount transcribed two ways, and
    flagging that as a conflict would bury the real ones.
    """
    return [match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(text)]


def _meaningful_lines(text: str) -> list[str]:
    return [normalise_line(line) for line in text.splitlines() if normalise_line(line)]


def compare_texts(
    text_a: str,
    text_b: str,
    *,
    engine_a: str = "Engine A",
    engine_b: str = "Engine B",
    changed_threshold: float = 0.55,
) -> ComparisonResult:
    """Align two transcriptions and describe how they differ.

    Lines are matched with :class:`difflib.SequenceMatcher`. A replaced block
    goes through :func:`_align_block`, which pairs lines to maximise total
    similarity; a pair clearing ``changed_threshold`` is reported as one line
    transcribed two ways rather than as two unrelated lines.
    """
    result = ComparisonResult(
        engine_a=engine_a,
        engine_b=engine_b,
        character_count_a=len(text_a),
        character_count_b=len(text_b),
        word_count_a=len(text_a.split()),
        word_count_b=len(text_b.split()),
    )

    lines_a = _meaningful_lines(text_a)
    lines_b = _meaningful_lines(text_b)

    if not lines_a and not lines_b:
        result.note = "Neither engine produced any text to compare."
        return result
    if not lines_a or not lines_b:
        present = engine_b if not lines_a else engine_a
        result.note = f"Only {present} produced text, so there is nothing to compare."
        kind = DiffKind.ONLY_B if not lines_a else DiffKind.ONLY_A
        for line in lines_b or lines_a:
            if kind is DiffKind.ONLY_A:
                result.lines.append(DiffLine(kind, text_a=line, similarity=0.0))
                result.only_a_lines += 1
            else:
                result.lines.append(DiffLine(kind, text_b=line, similarity=0.0))
                result.only_b_lines += 1
        return result

    matcher = SequenceMatcher(None, lines_a, lines_b, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                result.lines.append(
                    DiffLine(DiffKind.EQUAL, lines_a[i1 + offset], lines_b[j1 + offset])
                )
                result.equal_lines += 1
        elif tag == "replace":
            for line in _align_block(
                lines_a[i1:i2], lines_b[j1:j2], changed_threshold
            ):
                result.lines.append(line)
                if line.kind is DiffKind.CHANGED:
                    result.changed_lines += 1
                    _record_numeric_conflict(result, line.text_a, line.text_b)
                elif line.kind is DiffKind.ONLY_A:
                    result.only_a_lines += 1
                else:
                    result.only_b_lines += 1
        elif tag == "delete":
            for line in lines_a[i1:i2]:
                result.lines.append(DiffLine(DiffKind.ONLY_A, text_a=line, similarity=0.0))
                result.only_a_lines += 1
        elif tag == "insert":
            for line in lines_b[j1:j2]:
                result.lines.append(DiffLine(DiffKind.ONLY_B, text_b=line, similarity=0.0))
                result.only_b_lines += 1

    result.similarity = _aggregate_similarity(result.lines)
    return result


def _aggregate_similarity(lines: list[DiffLine]) -> float:
    """Overall agreement, derived from the alignment rather than raw characters.

    Running SequenceMatcher across two whole documents is O(n*m) on characters -
    close to a minute for a pair of 14k-character transcriptions, longer than
    the OCR that produced them. Aggregating the per-row results is linear, and
    it has the better property of reporting exactly the agreement the user can
    see in the comparison view.

    Rows are weighted by length, so a disagreement on a long table row counts
    for more than one on a two-word heading.
    """
    matched = 0.0
    total = 0
    for line in lines:
        weight = len(line.text_a) + len(line.text_b)
        total += weight
        if line.kind is DiffKind.EQUAL:
            matched += weight
        elif line.kind is DiffKind.CHANGED:
            matched += line.similarity * weight
    return matched / total if total else 0.0


#: Largest replace block aligned with the quadratic matcher. Beyond this the
#: similarity matrix costs more than the improved alignment is worth, and the
#: cheap positional pairing is used instead.
_MAX_ALIGN_CELLS = 15_000


def _align_block(
    block_a: list[str], block_b: list[str], threshold: float
) -> list[DiffLine]:
    """Pair up the lines of a replaced block, preserving reading order.

    Pairing by position fails as soon as one engine emits an extra line: every
    later line shifts by one, and genuinely matching lines get reported as two
    unrelated one-sided rows. That hides exactly the disagreements this feature
    exists to surface - a mistranscribed figure on the same table row.

    So the block is aligned with a weighted longest-common-subsequence: pair
    lines to maximise total similarity, allowing either side to skip a line,
    and only counting a pair when it clears ``threshold``.
    """
    rows_a, rows_b = len(block_a), len(block_b)
    if not rows_a:
        return [DiffLine(DiffKind.ONLY_B, text_b=line, similarity=0.0) for line in block_b]
    if not rows_b:
        return [DiffLine(DiffKind.ONLY_A, text_a=line, similarity=0.0) for line in block_a]

    if rows_a * rows_b > _MAX_ALIGN_CELLS:
        return _align_positionally(block_a, block_b, threshold)

    # scores[i][j] = similarity of block_a[i] to block_b[j], 0.0 when the pair
    # is too dissimilar to be the same line.
    scores: list[list[float]] = [[0.0] * rows_b for _ in range(rows_a)]
    for i, left in enumerate(block_a):
        for j, right in enumerate(block_b):
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            # Cheap upper bounds first: most pairs in a block are unrelated and
            # never need the full O(n*m) ratio.
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold:
                scores[i][j] = ratio

    # dp[i][j] = best total similarity achievable from block_a[i:] and block_b[j:]
    dp: list[list[float]] = [[0.0] * (rows_b + 1) for _ in range(rows_a + 1)]
    for i in range(rows_a - 1, -1, -1):
        for j in range(rows_b - 1, -1, -1):
            best = max(dp[i + 1][j], dp[i][j + 1])
            score = scores[i][j]
            if score > 0.0:
                best = max(best, score + dp[i + 1][j + 1])
            dp[i][j] = best

    lines: list[DiffLine] = []
    i = j = 0
    while i < rows_a and j < rows_b:
        score = scores[i][j]
        if score > 0.0 and dp[i][j] == score + dp[i + 1][j + 1]:
            lines.append(DiffLine(DiffKind.CHANGED, block_a[i], block_b[j], score))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            lines.append(DiffLine(DiffKind.ONLY_A, text_a=block_a[i], similarity=0.0))
            i += 1
        else:
            lines.append(DiffLine(DiffKind.ONLY_B, text_b=block_b[j], similarity=0.0))
            j += 1

    lines.extend(DiffLine(DiffKind.ONLY_A, text_a=line, similarity=0.0) for line in block_a[i:])
    lines.extend(DiffLine(DiffKind.ONLY_B, text_b=line, similarity=0.0) for line in block_b[j:])
    return lines


def _align_positionally(
    block_a: list[str], block_b: list[str], threshold: float
) -> list[DiffLine]:
    """Index-for-index fallback for blocks too large to align properly."""
    lines: list[DiffLine] = []
    for index in range(max(len(block_a), len(block_b))):
        left = block_a[index] if index < len(block_a) else ""
        right = block_b[index] if index < len(block_b) else ""
        if left and right:
            ratio = similarity_ratio(left, right)
            if ratio >= threshold:
                lines.append(DiffLine(DiffKind.CHANGED, left, right, ratio))
            else:
                lines.append(DiffLine(DiffKind.ONLY_A, text_a=left, similarity=0.0))
                lines.append(DiffLine(DiffKind.ONLY_B, text_b=right, similarity=0.0))
        elif left:
            lines.append(DiffLine(DiffKind.ONLY_A, text_a=left, similarity=0.0))
        elif right:
            lines.append(DiffLine(DiffKind.ONLY_B, text_b=right, similarity=0.0))
    return lines


def _record_numeric_conflict(result: ComparisonResult, left: str, right: str) -> None:
    """Flag a changed line whose digits disagree.

    A wrong word on an invoice is untidy; a wrong figure is expensive.
    """
    numbers_a = extract_numbers(left)
    numbers_b = extract_numbers(right)
    if numbers_a != numbers_b:
        result.numeric_conflicts.append(
            {
                "text_a": left,
                "text_b": right,
                "numbers_a": numbers_a,
                "numbers_b": numbers_b,
            }
        )


__all__ = [
    "ComparisonResult",
    "DiffKind",
    "DiffLine",
    "compare_texts",
    "extract_numbers",
    "normalise_line",
    "similarity_ratio",
]
