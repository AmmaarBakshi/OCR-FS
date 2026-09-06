"""Reconciliation of engine outputs into the final result.

Strategies are registered in :data:`FUSION_STRATEGIES` and chosen in Settings,
so the fusion policy is configurable rather than baked in (spec s5).

The rule every strategy obeys: **never introduce text that no engine produced.**
Fusion selects between transcriptions; it does not write new ones. The three
deterministic strategies cannot invent text by construction, and the LLM
strategy is constrained by prompt, verified against its inputs afterwards, and
falls back to a deterministic result if it drifts.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ocr_fusion.config.schema import AppSettings, FusionStrategy
from ocr_fusion.ocr.interface import OCRResult
from ocr_fusion.ocr.ollama_client import OllamaClient, OllamaError
from ocr_fusion.pipeline.comparison import DiffKind, compare_texts, normalise_line

logger = logging.getLogger(__name__)

#: Words for the containment check: letters, digits and internal separators, so
#: "INV-1024" and "1,325.50" stay single tokens.
_WORD_RE = re.compile(r"[\w][\w,.\-/]*")


@dataclass(slots=True)
class FusionResult:
    """The final reconciled transcription and how it was produced."""

    text: str
    strategy: str
    sources: list[str] = field(default_factory=list)
    """Display names of the engines that contributed."""

    duration_seconds: float = 0.0
    detail: str = ""
    """One line explaining the outcome, shown under the Fusion stage."""

    model_name: str | None = None
    """Set only when a model was used (the LLM strategy)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_used: bool = False
    """True when the chosen strategy could not run and another was substituted."""

    warnings: list[str] = field(default_factory=list)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "strategy": self.strategy,
            "sources": list(self.sources),
            "duration_seconds": self.duration_seconds,
            "detail": self.detail,
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "fallback_used": self.fallback_used,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "warnings": list(self.warnings),
        }


def _successful(results: list[OCRResult]) -> list[OCRResult]:
    """Engine results that actually produced text, in pipeline order."""
    return [r for r in results if r.succeeded and r.text.strip()]


# -- deterministic strategies ---------------------------------------------


def fuse_prefer_primary(
    results: list[OCRResult], settings: AppSettings
) -> FusionResult:
    """Take the first successful engine's text verbatim."""
    usable = _successful(results)
    if not usable:
        return _no_input_result("prefer_primary")
    winner = usable[0]
    return FusionResult(
        text=winner.text,
        strategy="prefer_primary",
        sources=[winner.provider_name],
        detail=f"Used {winner.provider_name} (first successful engine).",
    )


def fuse_prefer_longest(
    results: list[OCRResult], settings: AppSettings
) -> FusionResult:
    """Take the output with the most extracted characters.

    A crude but effective proxy for coverage: the common failure of a small VLM
    is stopping early, not writing too much.
    """
    usable = _successful(results)
    if not usable:
        return _no_input_result("prefer_longest")
    winner = max(usable, key=lambda r: r.character_count)
    return FusionResult(
        text=winner.text,
        strategy="prefer_longest",
        sources=[winner.provider_name],
        detail=(
            f"Used {winner.provider_name} ({winner.character_count} characters, "
            "the most complete extraction)."
        ),
    )


def fuse_line_vote(results: list[OCRResult], settings: AppSettings) -> FusionResult:
    """Merge two transcriptions line by line.

    Where the engines agree, the agreed line is kept. Where they disagree, the
    longer variant wins - a truncated line is the far more common OCR error
    than an over-long one. Lines only one engine produced are kept, since
    dropping them would lose real content.

    This is the default because it is the only deterministic strategy that can
    produce a result better than either input while remaining incapable of
    inventing text.
    """
    usable = _successful(results)
    if not usable:
        return _no_input_result("line_vote")
    if len(usable) == 1:
        only = usable[0]
        return FusionResult(
            text=only.text,
            strategy="line_vote",
            sources=[only.provider_name],
            detail=f"Only {only.provider_name} produced text, so it was used as is.",
        )

    first, second = usable[0], usable[1]
    comparison = compare_texts(
        first.text,
        second.text,
        engine_a=first.provider_name,
        engine_b=second.provider_name,
    )

    # Lines each engine produced anywhere in its output. Used to tell a line
    # that is genuinely unique to one engine from one both engines produced but
    # placed differently - the alignment is order-preserving, so it cannot match
    # the second kind and would otherwise emit it twice.
    lines_in_a = {normalise_line(line) for line in first.text.splitlines() if line.strip()}
    lines_in_b = {normalise_line(line) for line in second.text.splitlines() if line.strip()}

    merged: list[str] = []
    emitted: set[str] = set()
    agreed = resolved = kept_a = kept_b = duplicates = 0

    def emit(text: str) -> None:
        merged.append(text)
        emitted.add(normalise_line(text))

    for line in comparison.lines:
        if line.kind is DiffKind.EQUAL:
            emit(line.text_a)
            agreed += 1
        elif line.kind is DiffKind.CHANGED:
            chosen = _pick_line(line.text_a, line.text_b)
            emit(chosen)
            emitted.add(normalise_line(line.text_a))
            emitted.add(normalise_line(line.text_b))
            resolved += 1
        elif line.kind is DiffKind.ONLY_A:
            if _is_repositioned_duplicate(line.text_a, lines_in_b, emitted):
                duplicates += 1
                continue
            emit(line.text_a)
            kept_a += 1
        else:
            if _is_repositioned_duplicate(line.text_b, lines_in_a, emitted):
                duplicates += 1
                continue
            emit(line.text_b)
            kept_b += 1

    detail = (
        f"{agreed} lines agreed, {resolved} reconciled, "
        f"{kept_a + kept_b} kept from a single engine."
    )
    if duplicates:
        detail += (
            f" {duplicates} line{'s' if duplicates != 1 else ''} both engines "
            "placed differently were merged rather than repeated."
        )
    return FusionResult(
        text="\n".join(merged).strip(),
        strategy="line_vote",
        sources=[first.provider_name, second.provider_name],
        detail=detail,
    )


def _is_repositioned_duplicate(
    text: str, other_engine_lines: set[str], emitted: set[str]
) -> bool:
    """Whether a one-sided line is content already merged from the other engine.

    True only when both conditions hold: the other engine also produced this
    line somewhere, and it is already in the merged output. A line genuinely
    unique to one engine is always kept, and a line one engine legitimately
    repeats is only dropped when the other engine produced it too - a repeated
    table row, where losing the duplicate costs little and both raw outputs
    remain available.
    """
    key = normalise_line(text)
    return bool(key) and key in emitted and key in other_engine_lines


def _pick_line(left: str, right: str) -> str:
    """Choose between two transcriptions of the same line.

    Longer wins: OCR truncates far more often than it hallucinates extra words.
    Ties go to the first engine so the result is deterministic.
    """
    if len(right) > len(left):
        return right
    return left


# -- LLM strategy ----------------------------------------------------------


def fuse_with_llm(results: list[OCRResult], settings: AppSettings) -> FusionResult:
    """Ask a language model to reconcile the two transcriptions.

    The model can only ever improve a *selection* between two texts it is given,
    so its output is verified against them afterwards: if it drifts far from
    both inputs it is discarded in favour of the deterministic merge. That
    guarantees fusion cannot silently invent content (spec s5).
    """
    usable = _successful(results)
    if not usable:
        return _no_input_result("llm")
    if len(usable) == 1:
        only = usable[0]
        return FusionResult(
            text=only.text,
            strategy="llm",
            sources=[only.provider_name],
            detail=f"Only {only.provider_name} produced text, so no reconciliation was needed.",
        )

    first, second = usable[0], usable[1]
    client = OllamaClient(
        settings.ollama.host,
        connect_timeout=settings.ollama.connect_timeout_seconds,
    )
    model = settings.pipeline.fusion_model

    from ocr_fusion.config.prompts import render

    prompt = render(
        settings.prompts.fusion_user,
        engine_a=first.provider_name,
        text_a=first.text,
        engine_b=second.provider_name,
        text_b=second.text,
    )

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            system=settings.prompts.fusion_system,
            temperature=0.0,
            max_tokens=max(settings.qwen.max_tokens, 4096),
            timeout=settings.pipeline.fusion_timeout_seconds,
        )
    except OllamaError as exc:
        fallback = fuse_line_vote(results, settings)
        fallback.fallback_used = True
        fallback.warnings.append(
            f"LLM fusion was unavailable ({exc.message}), so the line-merge "
            "strategy was used instead."
        )
        fallback.detail = f"Fell back to line merge. {fallback.detail}"
        return fallback

    from ocr_fusion.ocr.postprocess import clean_ocr_text

    text = clean_ocr_text(response.text)
    result = FusionResult(
        text=text,
        strategy="llm",
        sources=[first.provider_name, second.provider_name],
        model_name=response.model,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        detail=f"Reconciled by {response.model}.",
    )

    verdict = _verify_against_sources(text, first.text, second.text)
    if verdict is not None:
        fallback = fuse_line_vote(results, settings)
        fallback.fallback_used = True
        fallback.strategy = "llm"
        fallback.warnings.append(verdict)
        fallback.detail = f"Discarded the model's output and merged by line. {fallback.detail}"
        return fallback
    return result


#: Minimum share of the fused text's words that must come from an engine.
_MIN_WORD_CONTAINMENT = 0.75


def _word_counts(text: str) -> Counter[str]:
    """Case-folded word multiset, punctuation-insensitive.

    Case and punctuation are exactly what a reconciler is allowed to adjust, so
    normalising them keeps the check focused on invented *content*.
    """
    return Counter(_WORD_RE.findall(text.lower()))


def _verify_against_sources(fused: str, text_a: str, text_b: str) -> str | None:
    """Return a warning when fused output does not look like its inputs.

    Guards the one way fusion could introduce content: a model that summarises,
    translates or answers the document instead of transcribing it.

    The measure is word containment - what share of the fused text's words came
    from one of the two engines - rather than sequence similarity. It targets
    invention directly (an invented word is one no engine wrote), and it is
    linear, where the character-level matcher is O(n*m) and would take close to
    a minute on a pair of full-page transcriptions.

    Returns ``None`` when the output is acceptable.
    """
    if not fused.strip():
        return "The fusion model returned no text."

    fused_words = _word_counts(fused)
    if not fused_words:
        return "The fusion model returned no readable words."

    available = _word_counts(text_a) | _word_counts(text_b)  # per-word maximum
    supported = sum(min(count, available[word]) for word, count in fused_words.items())
    containment = supported / sum(fused_words.values())

    if containment < _MIN_WORD_CONTAINMENT:
        return (
            f"Only {containment:.0%} of the fused text came from the engines, so "
            "it was discarded to avoid introducing content neither engine "
            "produced."
        )

    shortest = min(len(text_a), len(text_b))
    if shortest and len(fused) < shortest * 0.5:
        return (
            "The fusion model returned far less text than either engine, which "
            "suggests it summarised rather than transcribed, so it was discarded."
        )
    return None


def _no_input_result(strategy: str) -> FusionResult:
    return FusionResult(
        text="",
        strategy=strategy,
        detail="No engine produced text, so there was nothing to reconcile.",
    )


#: Strategy registry. Add an entry to offer a new fusion policy.
FUSION_STRATEGIES: dict[
    FusionStrategy, Callable[[list[OCRResult], AppSettings], FusionResult]
] = {
    FusionStrategy.PREFER_PRIMARY: fuse_prefer_primary,
    FusionStrategy.PREFER_LONGEST: fuse_prefer_longest,
    FusionStrategy.LINE_VOTE: fuse_line_vote,
    FusionStrategy.LLM: fuse_with_llm,
}


def fuse(results: list[OCRResult], settings: AppSettings) -> FusionResult:
    """Run the strategy configured in settings."""
    strategy = settings.pipeline.fusion_strategy
    handler = FUSION_STRATEGIES.get(strategy, fuse_line_vote)
    return handler(results, settings)


__all__ = [
    "FUSION_STRATEGIES",
    "FusionResult",
    "fuse",
    "fuse_line_vote",
    "fuse_prefer_longest",
    "fuse_prefer_primary",
    "fuse_with_llm",
]
