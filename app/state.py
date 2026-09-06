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


__all__ = [
    "KEY_UPLOAD_TOKEN",
    "KEY_VIEW",
    "current_page",
    "developer_mode",
    "document",
    "health",
    "is_running",
    "reset_run",
    "result",
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
