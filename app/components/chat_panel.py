"""Ask questions about the transcribed document (the chat panel).

The panel is deliberately thin: it owns the conversation on screen and nothing
about how an answer is produced. :class:`~ocr_fusion.chat.DocumentChat` does
that, and knows nothing about Streamlit.

Answers come from the transcription the pipeline produced, which is why the
panel appears only once there is a result to ask about.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from app import state
from app.theme import empty_state, notice
from ocr_fusion.chat import ChatAnswer, DocumentChat
from ocr_fusion.config.schema import AppSettings
from ocr_fusion.pipeline.result import PipelineResult

#: Openers offered before the first question. They are the questions people
#: actually ask of a scanned document, and they save a first-time user from
#: having to guess what the box accepts.
STARTERS = [
    "What is in this document?",
    "Summarise it in a few lines",
    "List the dates, amounts and reference numbers",
]


def render(result: PipelineResult, settings: AppSettings) -> None:
    """Draw the conversation, the starters and the question box."""
    if not settings.chat.enabled:
        return

    transcription = result.final_text
    if not transcription.strip():
        empty_state(
            "?",
            "Nothing to ask about yet",
            "No engine produced a transcription, so there is no text to answer from.",
        )
        return

    chat = DocumentChat(settings)
    history = state.chat_history()

    message, remedy = state.chat_error()
    if message:
        notice(
            escape(message)
            + (f"<br><br><b>How to fix:</b> {escape(remedy)}" if remedy else ""),
            "err",
            "The question could not be answered",
        )
        # Cleared on display: it describes one failed attempt, not a state the
        # panel is stuck in.
        state.clear_chat_error()

    if not history:
        st.markdown(
            '<div class="ofs-chat-intro">Ask anything about this document. '
            "Answers come from the transcription above, not from the internet.</div>",
            unsafe_allow_html=True,
        )
        _render_starters()

    for turn in history:
        with st.chat_message("user"):
            st.markdown(escape(turn.question))
        with st.chat_message("assistant"):
            if turn.warning:
                notice(escape(turn.warning), "warn")
            st.markdown(turn.answer)
            if turn.note and state.developer_mode():
                st.caption(turn.note)

    pending = state.chat_pending()
    if pending:
        _answer(chat, pending, result, settings)
        return

    typed = st.chat_input(
        "Ask about this document", key="ofs_chat_input", max_chars=2000
    )
    if typed:
        state.set_chat_pending(typed)
        st.rerun()

    if history:
        if st.button("Clear conversation", key="ofs_chat_clear"):
            state.clear_chat()
            st.rerun()


def _render_starters() -> None:
    """One-click openers, laid out so they do not dominate the panel."""
    columns = st.columns(len(STARTERS))
    for index, starter in enumerate(STARTERS):
        with columns[index]:
            if st.button(
                starter,
                key=f"ofs_chat_starter_{index}",
                use_container_width=True,
            ):
                state.set_chat_pending(starter)
                st.rerun()


#: Shown when the document was too long to send whole.
_TRIM_WARNING = (
    "This document is longer than the model can read at once, so the middle "
    "was left out. Answers drawn from the start and the end are reliable; "
    "anything in the middle may be missing."
)


def _answer(
    chat: DocumentChat,
    question: str,
    result: PipelineResult,
    settings: AppSettings,
) -> None:
    """Send one question and record the exchange.

    Everything the answer needs on screen is stored on the turn rather than
    drawn here, because the rerun that follows would wipe anything written
    directly - and the exchange has to look the same on every later redraw.
    """
    with st.chat_message("user"):
        st.markdown(escape(question))

    with st.chat_message("assistant"), st.spinner("Reading the document…"):
        answer = chat.ask(
            question,
            document_text=result.final_text,
            filename=result.document.filename,
            history=list(state.chat_history()),
        )

    state.set_chat_pending("")

    if not answer.succeeded:
        # The question is not recorded: a failed round trip is not part of the
        # conversation, and keeping it would send an unanswered turn as
        # context for the next question.
        state.set_chat_error(answer.error, answer.remedy)
        st.rerun()

    state.add_chat_turn(
        question,
        answer.text or "The model returned an empty answer.",
        note=_answer_footnote(answer),
        warning=_TRIM_WARNING if answer.context_was_trimmed else "",
    )
    st.rerun()


def _answer_footnote(answer: ChatAnswer) -> str:
    """Model and cost of one answer, for Developer Mode."""
    parts = [answer.model_name or "unknown model", f"{answer.duration_seconds:.1f}s"]
    if answer.input_tokens is not None:
        parts.append(f"{answer.input_tokens:,} tokens in")
    if answer.output_tokens is not None:
        parts.append(f"{answer.output_tokens:,} out")
    return " · ".join(parts)


__all__ = ["STARTERS", "render"]
