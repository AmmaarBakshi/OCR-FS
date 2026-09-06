"""The pipeline visualisation (spec s4).

Renders every stage with its status and execution time:

    Document -> Qwen2.5-VL -> Unlimited-OCR -> Comparison -> Fusion

Stages are drawn from the pipeline's own stage list rather than a hardcoded
sequence, so an added engine appears here automatically.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from ocr_fusion.metrics import format_duration
from ocr_fusion.pipeline.events import Stage, StageStatus

#: Status -> (css modifier, glyph). Glyphs stay simple so they render in every
#: browser without an icon font.
_STATUS_STYLE: dict[StageStatus, tuple[str, str]] = {
    StageStatus.WAITING: ("is-waiting", "○"),
    StageStatus.RUNNING: ("is-running", "◐"),
    StageStatus.COMPLETED: ("is-completed", "✓"),
    StageStatus.FAILED: ("is-failed", "✕"),
    StageStatus.SKIPPED: ("is-skipped", "–"),
}


def render_stage_row(stages: list[Stage], *, show_detail: bool = True) -> None:
    """Draw the pipeline as a connected row of stage tiles."""
    if not stages:
        return

    tiles: list[str] = []
    for stage in stages:
        modifier, glyph = _STATUS_STYLE.get(stage.status, ("is-waiting", "○"))
        if stage.status is StageStatus.WAITING:
            timing = "Waiting"
        elif stage.status is StageStatus.RUNNING:
            timing = "Running…"
        elif stage.status is StageStatus.SKIPPED:
            timing = "Skipped"
        else:
            timing = format_duration(stage.duration_seconds)

        detail = ""
        if show_detail and stage.detail:
            detail = f'<div class="ofs-stage-detail">{escape(stage.detail[:80])}</div>'

        tiles.append(
            f'<div class="ofs-stage {modifier}">'
            f'<span class="ofs-stage-icon">{glyph}</span>'
            f'<div class="ofs-stage-name">{escape(stage.label)}</div>'
            f'<div class="ofs-stage-time">{escape(timing)}</div>'
            f"{detail}</div>"
        )

    st.markdown(
        f'<div class="ofs-pipe">{"".join(tiles)}</div>', unsafe_allow_html=True
    )


def render_total(total_seconds: float, stages: list[Stage]) -> None:
    """Total runtime plus the per-stage breakdown the spec asks for."""
    completed = [
        s for s in stages if s.duration_seconds is not None and s.duration_seconds > 0
    ]
    parts = " · ".join(
        f"{escape(s.label)} {format_duration(s.duration_seconds)}" for s in completed
    )
    st.markdown(
        '<div style="margin-top:14px;display:flex;justify-content:space-between;'
        'align-items:baseline;gap:16px;flex-wrap:wrap;">'
        f'<span style="font-size:13px;color:var(--ink-faint);">{parts}</span>'
        f'<span style="font-size:15px;font-weight:650;color:var(--ink);">'
        f"Total {format_duration(total_seconds)}</span></div>",
        unsafe_allow_html=True,
    )


def render_placeholder(labels: list[str]) -> None:
    """The pipeline before a run, with every stage waiting.

    Showing the shape up front tells a client what is about to happen, which is
    most of the point of the visualisation.
    """
    tiles = "".join(
        f'<div class="ofs-stage is-waiting">'
        f'<span class="ofs-stage-icon">○</span>'
        f'<div class="ofs-stage-name">{escape(label)}</div>'
        f'<div class="ofs-stage-time">Waiting</div></div>'
        for label in labels
    )
    st.markdown(f'<div class="ofs-pipe">{tiles}</div>', unsafe_allow_html=True)


__all__ = ["render_placeholder", "render_stage_row", "render_total"]
