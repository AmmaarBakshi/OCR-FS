"""Document upload with drag-and-drop (spec s3)."""

from __future__ import annotations

import streamlit as st

from app import state
from app.theme import notice
from ocr_fusion.documents import load_document_bytes
from ocr_fusion.documents.errors import DocumentError

#: Extensions offered in the picker, without the leading dot.
ACCEPTED = ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "pdf"]


def render() -> None:
    """Draw the upload area and load whatever the user provides."""
    settings = state.settings()

    uploaded = st.file_uploader(
        "Upload a document",
        type=ACCEPTED,
        accept_multiple_files=False,
        key="ofs_uploader",
        label_visibility="collapsed",
        help="Drag a file here, or browse. PNG, JPG, WEBP, TIFF and PDF are supported.",
    )

    st.markdown(
        '<div style="font-size:12px;color:var(--ink-faint);margin-top:8px;'
        'text-align:center;">PNG · JPG · WEBP · TIFF · PDF &nbsp;·&nbsp; '
        f"up to {settings.documents.max_file_size_mb:.0f} MB &nbsp;·&nbsp; "
        "processed on this machine</div>",
        unsafe_allow_html=True,
    )

    if uploaded is None:
        return

    current = state.document()
    # Streamlit hands back the same object on every rerun, so reload only when
    # the file actually changed - re-rendering a 40-page PDF on each click
    # would make the app feel broken.
    token = f"{uploaded.name}:{uploaded.size}"
    if current is not None and st.session_state.get(state.KEY_UPLOAD_TOKEN) == token:
        return

    try:
        document = load_document_bytes(
            uploaded.getvalue(), uploaded.name, settings.documents
        )
    except DocumentError as exc:
        state.set_upload_error(exc.user_message)
        state.set_document(None)
        st.session_state[state.KEY_UPLOAD_TOKEN] = None
        return
    except Exception as exc:  # noqa: BLE001 - an unexpected decode failure must
        # still produce a readable message rather than a stack trace.
        state.set_upload_error(f"The file could not be read: {exc}")
        state.set_document(None)
        st.session_state[state.KEY_UPLOAD_TOKEN] = None
        return

    state.set_upload_error("")
    state.set_document(document)
    st.session_state[state.KEY_UPLOAD_TOKEN] = token
    st.rerun()


def render_error() -> None:
    """Show the last upload failure, if any."""
    message = state.upload_error()
    if message:
        notice(message, "err", "That file could not be opened")


__all__ = ["ACCEPTED", "render", "render_error"]
