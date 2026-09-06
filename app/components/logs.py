"""Processing log panel with clear, copy and export (spec s9)."""

from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from ocr_fusion.pipeline.events import EventLog, LogLevel


def render(log: EventLog, *, expanded: bool = False) -> None:
    """Draw the log console and its controls."""
    with st.expander(f"Processing details ({len(log.events)} events)", expanded=expanded):
        if not log.events:
            st.caption("No events recorded yet.")
            return

        st.markdown(_console_html(log), unsafe_allow_html=True)

        if log.redact_text:
            st.caption(
                "Document text is deliberately kept out of the log; only "
                "counts are recorded. Change this in Settings › Privacy."
            )

        clear, download = st.columns([1, 1])
        with clear:
            if st.button("Clear log", use_container_width=True, key="ofs_clear_log"):
                log.clear()
                st.rerun()
        with download:
            st.download_button(
                "Export log",
                data=log.as_text().encode("utf-8"),
                file_name=f"ocr-log-{datetime.now().strftime('%Y%m%d-%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="ofs_export_log",
            )

        # st.code gives a working copy button without shipping custom JS.
        with st.expander("Copy log as text", expanded=False):
            st.code(log.as_text(), language="text")


def _console_html(log: EventLog) -> str:
    lines: list[str] = []
    for event in log.events:
        css = event.level.value if event.level in (LogLevel.WARNING, LogLevel.ERROR) else ""
        lines.append(
            f'<div class="ofs-log-line {css}">'
            f'<span class="t">[{event.clock}]</span> {escape(event.message)}</div>'
        )
    return f'<div class="ofs-log">{"".join(lines)}</div>'


__all__ = ["render"]
