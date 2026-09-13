"""Answering questions about a document that has already been transcribed.

The pipeline's job ends at a transcription; this is what comes after it, when
someone wants to know what the document *says* rather than read all of it.

The answer is produced from the transcription, never from the page images. A
vision model would take minutes to re-read a scan the pipeline has already
read, and the transcription is the artefact the rest of the app is built on -
answering from anything else would mean the chat and the result could disagree.

Failures are returned, not raised, the same way a provider reports one: an
unreachable runtime is an expected condition with a remedy, not a bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ocr_fusion.config.prompts import CHAT_HISTORY_PROMPT, render
from ocr_fusion.config.schema import AppSettings
from ocr_fusion.ocr.interface import HealthStatus
from ocr_fusion.ocr.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaUnavailableError,
)

#: Marker left where a transcription was trimmed to fit the context budget.
#: Visible in the prompt on purpose - the model should know it is reading an
#: excerpt, so that "the document does not say" stays an honest answer.
_ELISION = "\n\n[... {count:,} characters omitted from the middle of the document ...]\n\n"

#: Share of the context budget given to the start of the document. Identity -
#: letterhead, reference numbers, dates - is at the top, and totals and
#: signatures are at the bottom; the middle is where the repetition lives.
_HEAD_SHARE = 0.6


@dataclass(slots=True)
class ChatTurn:
    """One exchange, kept so a follow-up question can resolve against it.

    ``note`` and ``warning`` are what the interface needs to redraw the
    exchange exactly as it first appeared - which model answered, or that the
    document had to be trimmed. Neither is sent back to the model: only the
    question and the answer are conversation.
    """

    question: str
    answer: str
    note: str = ""
    warning: str = ""


@dataclass(slots=True)
class ChatAnswer:
    """What the model said, or why it could not say anything."""

    text: str = ""
    model_name: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_seconds: float = 0.0
    error: str = ""
    remedy: str = ""
    context_was_trimmed: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return not self.error

    @classmethod
    def failure(
        cls, error: str, remedy: str = "", duration_seconds: float = 0.0
    ) -> ChatAnswer:
        return cls(error=error, remedy=remedy, duration_seconds=duration_seconds)


def trim_to_budget(text: str, budget: int) -> tuple[str, bool]:
    """Keep ``budget`` characters of ``text``, taken from both ends.

    Returns the kept text and whether anything was dropped. ``budget`` governs
    how much of the document survives; the elision marker is added on top of
    it, so the returned string is slightly longer when trimming happened.

    Both ends are kept because that is where a document identifies itself and
    where it totals itself; a plain head truncation would take the answer to
    "what is the total?" away from every long document.
    """
    if budget <= 0 or len(text) <= budget:
        return text, False

    head_size = int(budget * _HEAD_SHARE)
    tail_size = budget - head_size
    omitted = len(text) - budget
    return text[:head_size] + _ELISION.format(count=omitted) + text[-tail_size:], True


def format_history(turns: list[ChatTurn], limit: int) -> str:
    """The last ``limit`` exchanges, as a prompt fragment."""
    if limit <= 0 or not turns:
        return ""
    lines: list[str] = []
    for turn in turns[-limit:]:
        lines.append(f"Q: {turn.question}")
        lines.append(f"A: {turn.answer}")
    return render(CHAT_HISTORY_PROMPT, turns="\n".join(lines))


class DocumentChat:
    """Answers questions about one transcription."""

    def __init__(
        self, settings: AppSettings, client: OllamaClient | None = None
    ) -> None:
        self.settings = settings
        self.config = settings.chat
        self.client = client or OllamaClient(
            settings.ollama.host,
            connect_timeout=settings.ollama.connect_timeout_seconds,
        )

    @property
    def model(self) -> str:
        return self.config.model

    def health_check(self) -> HealthStatus:
        """Whether a question could be answered right now."""
        if not self.config.enabled:
            return HealthStatus(
                available=False,
                message="Asking questions about a document is switched off.",
                remedy="Turn it on in Settings > Chat.",
            )
        try:
            if not self.client.has_model(self.model):
                return HealthStatus(
                    available=False,
                    message=f"The model '{self.model}' has not been downloaded.",
                    remedy=f"Run: ollama pull {self.model}",
                )
        except OllamaUnavailableError as exc:
            return HealthStatus(available=False, message=exc.message, remedy=exc.remedy)
        return HealthStatus.ok(f"Ready ({self.model})", model=self.model)

    def ask(
        self,
        question: str,
        *,
        document_text: str,
        filename: str = "the document",
        history: list[ChatTurn] | None = None,
    ) -> ChatAnswer:
        """Answer ``question`` from ``document_text``."""
        question = question.strip()
        if not question:
            return ChatAnswer.failure("Ask a question about the document.")
        if not self.config.enabled:
            return ChatAnswer.failure(
                "Asking questions about a document is switched off.",
                remedy="Turn it on in Settings > Chat.",
            )
        if not document_text.strip():
            return ChatAnswer.failure(
                "There is no transcription to ask about yet.",
                remedy="Run a document through the OCR pipeline first.",
            )

        context, trimmed = trim_to_budget(
            document_text, self.config.max_context_characters
        )
        prompt = render(
            self.settings.prompts.chat_user,
            filename=filename,
            document=context,
            history=format_history(history or [], self.config.history_turns),
            question=question,
        )

        started = time.perf_counter()
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=self.settings.prompts.chat_system,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds,
                keep_alive=self.config.keep_alive,
            )
        except OllamaError as exc:
            return ChatAnswer.failure(
                exc.message, exc.remedy, time.perf_counter() - started
            )

        return ChatAnswer(
            text=response.text.strip(),
            model_name=response.model,
            input_tokens=response.prompt_tokens,
            output_tokens=response.completion_tokens,
            duration_seconds=time.perf_counter() - started,
            context_was_trimmed=trimmed,
        )


__all__ = [
    "ChatAnswer",
    "ChatTurn",
    "DocumentChat",
    "format_history",
    "trim_to_budget",
]
