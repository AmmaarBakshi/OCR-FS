"""Document preview with page navigation and zoom (spec s3, left panel)."""

from __future__ import annotations

import io
from html import escape

import streamlit as st

from app import state
from app.theme import badge
from ocr_fusion.documents.models import Document

#: Zoom is applied by resizing the image before display rather than with CSS,
#: because Streamlit re-lays-out on rerun and a CSS transform would overflow
#: the column rather than scroll inside it.
_ZOOM_STEPS = (0.5, 0.75, 1.0, 1.5, 2.0)


def render(document: Document) -> None:
    """Draw the preview panel for ``document``."""
    _render_metadata(document)

    page_number = state.current_page()
    if page_number > document.page_count:
        page_number = 1
        state.set_current_page(1)

    if document.page_count > 1:
        _render_page_nav(document, page_number)

    _render_zoom_controls()
    _render_image(document, page_number)


def _render_metadata(document: Document) -> None:
    """File facts a client cares about: name, type, size, pages."""
    size_mb = document.source_bytes_size / (1024 * 1024)
    size_text = (
        f"{size_mb:.1f} MB" if size_mb >= 0.1 else f"{document.source_bytes_size / 1024:.0f} KB"
    )
    badges = [
        badge(document.kind.value.upper(), "info"),
        badge(f"{document.page_count} page{'s' if document.page_count != 1 else ''}", "neutral"),
        badge(size_text, "neutral"),
    ]
    if document.has_text_layer:
        badges.append(badge("Contains selectable text", "ok"))

    st.markdown(
        f'<div style="font-size:14.5px;font-weight:600;color:var(--ink);'
        f'overflow-wrap:anywhere;margin-bottom:7px;">{escape(document.filename)}</div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px;">'
        f'{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )

    for warning in document.warnings:
        st.markdown(
            f'<div class="ofs-notice warn">{escape(warning)}</div>',
            unsafe_allow_html=True,
        )


def _render_page_nav(document: Document, page_number: int) -> None:
    previous, indicator, following = st.columns([1, 2, 1])
    with previous:
        if st.button(
            "‹ Previous",
            disabled=page_number <= 1,
            use_container_width=True,
            key="ofs_prev_page",
        ):
            state.set_current_page(page_number - 1)
            st.rerun()
    with indicator:
        st.markdown(
            f'<div style="text-align:center;padding-top:7px;font-size:13px;'
            f'font-weight:600;color:var(--ink-soft);">Page {page_number} '
            f"of {document.page_count}</div>",
            unsafe_allow_html=True,
        )
    with following:
        if st.button(
            "Next ›",
            disabled=page_number >= document.page_count,
            use_container_width=True,
            key="ofs_next_page",
        ):
            state.set_current_page(page_number + 1)
            st.rerun()


def _render_zoom_controls() -> None:
    current = state.zoom()
    labels = [f"{int(step * 100)}%" for step in _ZOOM_STEPS]
    try:
        index = _ZOOM_STEPS.index(current)
    except ValueError:
        index = _ZOOM_STEPS.index(1.0)

    chosen = st.radio(
        "Zoom",
        options=labels,
        index=index,
        horizontal=True,
        key="ofs_zoom_radio",
        label_visibility="collapsed",
    )
    new_zoom = _ZOOM_STEPS[labels.index(chosen)]
    if new_zoom != current:
        state.set_zoom(new_zoom)


def _render_image(document: Document, page_number: int) -> None:
    try:
        page = document.page(page_number)
    except KeyError:
        st.warning("That page is not available.")
        return

    zoom = state.zoom()
    if zoom == 1.0:
        st.image(page.image_bytes, use_container_width=True)
        return

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(page.image_bytes))
        width = max(1, int(image.width * zoom))
        height = max(1, int(image.height * zoom))
        st.image(image.resize((width, height), Image.LANCZOS), use_container_width=(zoom < 1.0))
    except Exception:  # noqa: BLE001 - a preview problem must not break the page
        st.image(page.image_bytes, use_container_width=True)


__all__ = ["render"]
