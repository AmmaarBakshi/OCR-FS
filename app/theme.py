"""Visual theme for the Streamlit app.

Streamlit's defaults read as a developer tool, and the brief is a product a
non-technical client can be shown (spec s3). So the app hides Streamlit's own
chrome and renders its main surfaces as cards carrying our own class names.

Targeting our own classes rather than Streamlit's generated ones matters: the
internal class names change between releases, so styling built on them breaks
on upgrade. Only a handful of stable ``data-testid`` hooks are used.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

#: Brand palette. Deep indigo reads as considered and professional without
#: looking like a default framework blue.
PALETTE = {
    "ink": "#12141c",
    "ink_soft": "#4a4f61",
    "ink_faint": "#7c8296",
    "surface": "#ffffff",
    "canvas": "#f6f7fb",
    "line": "#e4e7f0",
    "brand": "#4338ca",
    "brand_soft": "#eef2ff",
    "success": "#0f9d58",
    "success_soft": "#e7f6ee",
    "warning": "#b45309",
    "warning_soft": "#fef3c7",
    "danger": "#c02626",
    "danger_soft": "#fdecec",
    "muted": "#9aa1b4",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink: #12141c;
  --ink-soft: #4a4f61;
  --ink-faint: #7c8296;
  --surface: #ffffff;
  --canvas: #f6f7fb;
  --line: #e4e7f0;
  --brand: #4338ca;
  --brand-soft: #eef2ff;
  --success: #0f9d58;
  --success-soft: #e7f6ee;
  --warning: #b45309;
  --warning-soft: #fef3c7;
  --danger: #c02626;
  --danger-soft: #fdecec;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(18,20,28,.04), 0 8px 24px rgba(18,20,28,.06);
}

/* Hide Streamlit chrome: the hamburger menu, footer and deploy button read as
   "developer tool" and have no meaning for a client. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

html, body, [class*="css"] { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: var(--canvas); }
[data-testid="stAppViewContainer"] > .main .block-container {
  padding: 1.6rem 2.4rem 4rem; max-width: 1500px;
}
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.018em; font-weight: 650; }

/* ---------- brand header ---------- */
.ofs-brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.ofs-logo {
  width: 42px; height: 42px; border-radius: 12px; flex: none;
  background: linear-gradient(135deg, #4338ca 0%, #6d28d9 55%, #9333ea 100%);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 700; font-size: 17px; letter-spacing: -0.5px;
  box-shadow: 0 4px 14px rgba(67,56,202,.32);
}
.ofs-title { font-size: 25px; font-weight: 680; color: var(--ink); line-height: 1.15; letter-spacing: -0.03em; }
.ofs-subtitle { font-size: 13.5px; color: var(--ink-faint); margin-top: 1px; }

/* ---------- cards ----------
   A card must contain real Streamlit widgets, so it is a st.container()
   rather than a hand-written <div>: separate st.markdown calls each render into
   their own element, so an opening tag is closed immediately and nothing nests
   inside it.

   Streamlit gives every container the same data-testid, so the card is
   identified by a marker element that card() emits as its first child. The
   :has() selector then styles only the wrapper that directly contains it. */
[data-testid="stLayoutWrapper"]:has(> div > [data-testid="stElementContainer"] .ofs-card-tag) {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 18px 22px;
  box-shadow: var(--shadow); margin-bottom: 16px;
}
/* Streamlit also styles the vertical block inside a container. Left alone it
   draws a second, nested frame inside the card, so it is flattened here. */
[data-testid="stLayoutWrapper"]:has(> div > [data-testid="stElementContainer"] .ofs-card-tag)
  > [data-testid="stVerticalBlock"] {
  border: none !important; border-radius: 0 !important;
  padding: 0 !important; background: transparent !important;
}
.ofs-card-tag { display: none; }
.ofs-card-title { font-size: 12px; font-weight: 650; text-transform: uppercase;
  letter-spacing: .09em; color: var(--ink-faint); display: block; margin-bottom: 2px; }
/* The marker's own element container would otherwise add a blank row. */
[data-testid="stElementContainer"]:has(.ofs-card-tag:only-child) { display: none; }

/* ---------- pipeline visualisation ---------- */
.ofs-pipe { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; }
.ofs-stage {
  flex: 1 1 0; min-width: 140px; padding: 14px 12px; border-radius: 12px;
  border: 1px solid var(--line); background: var(--canvas); text-align: center;
  position: relative; transition: all .18s ease;
}
.ofs-stage + .ofs-stage { margin-left: 26px; }
.ofs-stage + .ofs-stage::before {
  content: ""; position: absolute; left: -22px; top: 50%;
  width: 18px; height: 2px; background: var(--line); border-radius: 2px;
}
.ofs-stage-icon { font-size: 16px; line-height: 1; margin-bottom: 7px; display: block; }
.ofs-stage-name { font-size: 12.5px; font-weight: 600; color: var(--ink);
  overflow-wrap: anywhere; }
.ofs-stage-time { font-size: 11.5px; color: var(--ink-faint); margin-top: 3px;
  font-family: 'JetBrains Mono', monospace; }
.ofs-stage-detail { font-size: 10.5px; color: var(--muted); margin-top: 4px;
  overflow-wrap: anywhere; line-height: 1.35; }

.ofs-stage.is-running { border-color: var(--brand); background: var(--brand-soft); }
.ofs-stage.is-running .ofs-stage-icon { animation: ofs-pulse 1.1s ease-in-out infinite; }
.ofs-stage.is-completed { border-color: #bfe6cf; background: var(--success-soft); }
.ofs-stage.is-completed .ofs-stage-icon { color: var(--success); }
.ofs-stage.is-failed { border-color: #f3c2c2; background: var(--danger-soft); }
.ofs-stage.is-failed .ofs-stage-icon { color: var(--danger); }
.ofs-stage.is-skipped { opacity: .55; }
@keyframes ofs-pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

/* ---------- metric tiles ---------- */
.ofs-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 10px; }
.ofs-metric { background: var(--canvas); border: 1px solid var(--line);
  border-radius: 11px; padding: 12px 13px; }
.ofs-metric-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--ink-faint); font-weight: 600; }
.ofs-metric-value { font-size: 18px; font-weight: 650; color: var(--ink); margin-top: 3px;
  font-family: 'JetBrains Mono', monospace; letter-spacing: -.02em; overflow-wrap: anywhere; }
.ofs-metric-value.is-na { color: var(--muted); font-weight: 500; }

/* ---------- badges ---------- */
.ofs-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px;
  font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.ofs-badge.ok { background: var(--success-soft); color: #0b7a45; }
.ofs-badge.warn { background: var(--warning-soft); color: var(--warning); }
.ofs-badge.err { background: var(--danger-soft); color: var(--danger); }
.ofs-badge.info { background: var(--brand-soft); color: var(--brand); }
.ofs-badge.neutral { background: #eef0f5; color: var(--ink-soft); }

/* ---------- comparison diff ---------- */
.ofs-diff { border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
  font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.ofs-diff-head { display: grid; grid-template-columns: 1fr 1fr; background: var(--canvas);
  border-bottom: 1px solid var(--line); font-family: 'Inter', sans-serif; }
.ofs-diff-head > div { padding: 9px 13px; font-size: 11px; font-weight: 650;
  text-transform: uppercase; letter-spacing: .07em; color: var(--ink-faint); }
.ofs-diff-head > div:first-child { border-right: 1px solid var(--line); }
.ofs-diff-row { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid #f0f2f7; }
.ofs-diff-row:last-child { border-bottom: none; }
.ofs-diff-cell { padding: 7px 13px; white-space: pre-wrap; overflow-wrap: anywhere;
  line-height: 1.5; }
.ofs-diff-cell:first-child { border-right: 1px solid #f0f2f7; }
.ofs-diff-row.changed { background: #fffbeb; }
.ofs-diff-row.only-a .ofs-diff-cell:first-child { background: var(--danger-soft); }
.ofs-diff-row.only-b .ofs-diff-cell:last-child { background: var(--success-soft); }
.ofs-diff-row.numeric { background: var(--danger-soft); }
.ofs-diff-empty { color: var(--muted); font-style: italic; }

/* ---------- log console ---------- */
.ofs-log {
  background: #14161f; border-radius: 12px; padding: 14px 16px;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; line-height: 1.75;
  max-height: 340px; overflow-y: auto; color: #cbd2e0;
}
.ofs-log-line { white-space: pre-wrap; overflow-wrap: anywhere; }
.ofs-log-line .t { color: #6b7590; }
.ofs-log-line.warning { color: #fcd34d; }
.ofs-log-line.error { color: #fca5a5; }

/* ---------- text output ---------- */
.ofs-output {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 18px 20px; font-size: 13.5px; line-height: 1.72; color: var(--ink);
  max-height: 620px; overflow-y: auto; white-space: pre-wrap;
  overflow-wrap: anywhere; font-family: 'Inter', sans-serif;
}
.ofs-output.mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

/* ---------- empty state ---------- */
.ofs-empty { text-align: center; padding: 44px 20px; color: var(--ink-faint); }
.ofs-empty-icon { font-size: 40px; margin-bottom: 10px; }
.ofs-empty-title { font-size: 15px; font-weight: 600; color: var(--ink-soft); }
.ofs-empty-text { font-size: 13px; margin-top: 5px; }

/* ---------- notices ---------- */
.ofs-notice { border-radius: 11px; padding: 12px 15px; font-size: 13px;
  line-height: 1.6; border: 1px solid; margin-bottom: 12px; }
.ofs-notice.err { background: var(--danger-soft); border-color: #f3c2c2; color: #8c1c1c; }
.ofs-notice.warn { background: var(--warning-soft); border-color: #f5d98a; color: #8a4708; }
.ofs-notice.info { background: var(--brand-soft); border-color: #ccd3fb; color: #33299e; }
.ofs-notice-title { font-weight: 650; margin-bottom: 3px; }
.ofs-notice code { background: rgba(0,0,0,.07); padding: 1px 6px; border-radius: 5px;
  font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }

/* ---------- Streamlit control polish ---------- */
.stButton > button {
  border-radius: 10px; font-weight: 600; font-size: 13.5px;
  border: 1px solid var(--line); transition: all .15s ease;
}
.stButton > button[kind="primary"] {
  background: var(--brand); border-color: var(--brand);
  box-shadow: 0 2px 10px rgba(67,56,202,.26);
}
.stButton > button[kind="primary"]:hover {
  background: #3730a3; border-color: #3730a3;
  transform: translateY(-1px); box-shadow: 0 4px 16px rgba(67,56,202,.34);
}
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed #c9cfe2; border-radius: var(--radius);
  background: var(--surface); padding: 26px 18px; transition: all .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--brand); background: var(--brand-soft);
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  height: 40px; padding: 0 16px; font-size: 13.5px; font-weight: 600;
  color: var(--ink-faint); border-radius: 9px 9px 0 0;
}
.stTabs [aria-selected="true"] { color: var(--brand); background: var(--brand-soft); }
[data-testid="stExpander"] {
  border: 1px solid var(--line); border-radius: 12px; background: var(--surface);
}
[data-testid="stExpander"] summary { font-size: 13.5px; font-weight: 600; }
hr { margin: 14px 0; border-color: var(--line); }

/* Dark mode: the viewer's OS preference must not produce unreadable cards. */
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #eceff7; --ink-soft: #b3bacd; --ink-faint: #8c94ab;
    --surface: #191b24; --canvas: #101219; --line: #2b2f3d;
    --brand: #8b7cf6; --brand-soft: #23213c;
    --success-soft: #14301f; --warning-soft: #33270d; --danger-soft: #351919;
    --muted: #737b91;
  }
  .ofs-diff-row.changed { background: #2a2410; }
  .ofs-diff-row:not(:last-child) { border-bottom-color: #23262f; }
  .ofs-diff-cell:first-child { border-right-color: #23262f; }
  .ofs-notice.err { color: #f6b8b8; } .ofs-notice.warn { color: #f2cf8a; }
  .ofs-notice.info { color: #c3bbfb; }
  .ofs-badge.ok { color: #7fd6a5; } .ofs-badge.err { color: #f5a3a3; }
}
</style>
"""


def apply_theme() -> None:
    """Inject the stylesheet. Call once, immediately after ``set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


@contextmanager
def card(title: str = "") -> Iterator[None]:
    """A styled card that real Streamlit widgets can live inside.

    Used as a context manager::

        with card("Results"):
            st.write("anything, including widgets")

    Implemented as a container plus a marker element, because HTML written with
    ``st.markdown`` cannot wrap subsequent widgets - each markdown call renders
    into its own element, so an unclosed ``<div>`` is closed immediately and
    nothing nests inside it.
    """
    # border=False: the card's frame comes entirely from our own CSS below.
    # Streamlit's built-in border would draw a second, nested box inside it.
    container = st.container(border=False)
    with container:
        st.markdown('<span class="ofs-card-tag"></span>', unsafe_allow_html=True)
        if title:
            st.markdown(
                f'<span class="ofs-card-title">{title}</span>', unsafe_allow_html=True
            )
        yield


def badge(text: str, kind: str = "neutral") -> str:
    """Return badge HTML. ``kind`` is one of ok/warn/err/info/neutral."""
    return f'<span class="ofs-badge {kind}">{text}</span>'


def notice(message: str, kind: str = "info", title: str = "") -> None:
    """Render an inline notice. ``message`` may contain simple HTML."""
    heading = f'<div class="ofs-notice-title">{title}</div>' if title else ""
    st.markdown(
        f'<div class="ofs-notice {kind}">{heading}{message}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, text: str = "") -> None:
    st.markdown(
        f'<div class="ofs-empty"><div class="ofs-empty-icon">{icon}</div>'
        f'<div class="ofs-empty-title">{title}</div>'
        f'<div class="ofs-empty-text">{text}</div></div>',
        unsafe_allow_html=True,
    )


__all__ = [
    "PALETTE",
    "apply_theme",
    "badge",
    "card",
    "empty_state",
    "notice",
]
