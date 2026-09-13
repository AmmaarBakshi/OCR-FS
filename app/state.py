"""Session state for the Streamlit app.

Streamlit re-runs the whole script on every interaction, so anything that must
survive a click lives in ``st.session_state``. Centralising the keys here keeps
components from inventing their own and colliding.

Uploaded documents are held in memory only. Nothing is written to disk unless
the operator turns persistence on in Settings (spec s14).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ocr_fusion.chat import ChatTurn
from ocr_fusion.config import AppSettings, load_settings, save_settings
from ocr_fusion.documents.models import Document
from ocr_fusion.pipeline.result import PipelineResult

# Session keys, named once so a typo cannot silently create a second slot.
KEY_SETTINGS = "ofs_settings"
KEY_DOCUMENT = "ofs_document"
KEY_UPLOAD_ERROR = "ofs_upload_error"
KEY_RESULT = "ofs_result"
KEY_RUNNING = "ofs_running"
KEY_PAGE = "ofs_page"
KEY_VIEW = "ofs_view"
KEY_ZOOM = "ofs_zoom"
KEY_HEALTH = "ofs_health"
KEY_UPLOAD_TOKEN = "ofs_upload_token"
KEY_CHAT = "ofs_chat"
KEY_CHAT_PENDING = "ofs_chat_pending"
KEY_CHAT_ERROR = "ofs_chat_error"


def settings() -> AppSettings:
    """The live settings object, loaded once per session."""
    if KEY_SETTINGS not in st.session_state:
        st.session_state[KEY_SETTINGS] = load_settings()
    return st.session_state[KEY_SETTINGS]


def update_settings(new_settings: AppSettings, *, persist: bool = True) -> None:
    """Replace settings, optionally writing them to disk."""
    st.session_state[KEY_SETTINGS] = new_settings
    if persist:
        save_settings(new_settings)
    # Health results describe the previous configuration and are now stale.
    st.session_state.pop(KEY_HEALTH, None)


def document() -> Document | None:
    return st.session_state.get(KEY_DOCUMENT)


def set_document(doc: Document | None) -> None:
    """Store the loaded document and clear anything derived from the old one."""
    st.session_state[KEY_DOCUMENT] = doc
    st.session_state[KEY_RESULT] = None
    st.session_state[KEY_PAGE] = 1
    st.session_state[KEY_ZOOM] = 1.0
    # The conversation was about the previous document; keeping it would let a
    # follow-up question be answered against the wrong file.
    clear_chat()


def result() -> PipelineResult | None:
    return st.session_state.get(KEY_RESULT)


def set_result(value: PipelineResult | None) -> None:
    st.session_state[KEY_RESULT] = value


def is_running() -> bool:
    return bool(st.session_state.get(KEY_RUNNING, False))


def set_running(value: bool) -> None:
    st.session_state[KEY_RUNNING] = value


def current_page() -> int:
    return int(st.session_state.get(KEY_PAGE, 1))


def set_current_page(page: int) -> None:
    st.session_state[KEY_PAGE] = max(1, page)


def zoom() -> float:
    return float(st.session_state.get(KEY_ZOOM, 1.0))


def set_zoom(value: float) -> None:
    st.session_state[KEY_ZOOM] = max(0.25, min(3.0, value))


def chat_history() -> list[ChatTurn]:
    """Questions asked about the current document, oldest first."""
    return st.session_state.setdefault(KEY_CHAT, [])


def add_chat_turn(
    question: str, answer: str, *, note: str = "", warning: str = ""
) -> None:
    chat_history().append(
        ChatTurn(question=question, answer=answer, note=note, warning=warning)
    )


def clear_chat() -> None:
    st.session_state[KEY_CHAT] = []
    st.session_state[KEY_CHAT_PENDING] = ""
    clear_chat_error()


def chat_error() -> tuple[str, str]:
    """The last failed question, as ``(message, remedy)``.

    Held in state because reporting it happens on the rerun after the failure,
    by which point anything drawn during the failed attempt is gone.
    """
    return st.session_state.get(KEY_CHAT_ERROR, ("", ""))


def set_chat_error(message: str, remedy: str = "") -> None:
    st.session_state[KEY_CHAT_ERROR] = (message, remedy)


def clear_chat_error() -> None:
    st.session_state.pop(KEY_CHAT_ERROR, None)


def chat_pending() -> str:
    """A question submitted but not yet answered.

    Streamlit reruns the script on submit, so the question has to survive the
    rerun that draws the thinking state before the model is called.
    """
    return st.session_state.get(KEY_CHAT_PENDING, "")


def set_chat_pending(question: str) -> None:
    st.session_state[KEY_CHAT_PENDING] = question


def developer_mode() -> bool:
    return settings().general.developer_mode


def health() -> dict[str, Any] | None:
    """Cached engine readiness.

    Cached because a health check costs a round trip per engine and Streamlit
    re-runs the script on every widget interaction; without this, moving a
    slider would re-probe every engine.
    """
    return st.session_state.get(KEY_HEALTH)


def set_health(report: dict[str, Any] | None) -> None:
    st.session_state[KEY_HEALTH] = report


def upload_error() -> str:
    return st.session_state.get(KEY_UPLOAD_ERROR, "")


def set_upload_error(message: str) -> None:
    st.session_state[KEY_UPLOAD_ERROR] = message


def reset_run() -> None:
    """Clear the current document and results, keeping settings."""
    st.session_state[KEY_DOCUMENT] = None
    st.session_state[KEY_RESULT] = None
    st.session_state[KEY_UPLOAD_ERROR] = ""
    st.session_state[KEY_PAGE] = 1
    st.session_state[KEY_ZOOM] = 1.0
    clear_chat()


__all__ = [
    "KEY_CHAT",
    "KEY_UPLOAD_TOKEN",
    "KEY_VIEW",
    "add_chat_turn",
    "chat_error",
    "chat_history",
    "chat_pending",
    "clear_chat",
    "clear_chat_error",
    "current_page",
    "developer_mode",
    "document",
    "health",
    "is_running",
    "reset_run",
    "result",
    "set_chat_error",
    "set_chat_pending",
    "set_current_page",
    "set_document",
    "set_health",
    "set_result",
    "set_running",
    "set_upload_error",
    "set_zoom",
    "settings",
    "update_settings",
    "upload_error",
    "zoom",
]
