"""Side-by-side comparison of the two engine outputs (spec s5)."""

from __future__ import annotations

from html import escape

import streamlit as st

from app.theme import badge, notice
from ocr_fusion.config.schema import AppSettings
from ocr_fusion.pipeline.comparison import ComparisonResult, DiffKind, extract_numbers

_ROW_CLASS = {
    DiffKind.EQUAL: "",
    DiffKind.CHANGED: "changed",
    DiffKind.ONLY_A: "only-a",
    DiffKind.ONLY_B: "only-b",
}

#: Cap on rendered rows. A long document produces thousands of diff lines, and
#: pushing them all through one markdown call makes the page crawl.
_MAX_ROWS = 400


def render(comparison: ComparisonResult, settings: AppSettings) -> None:
    """Draw the agreement summary, numeric conflicts and the aligned diff."""
    if comparison.note:
        notice(escape(comparison.note), "info")

    _render_summary(comparison)

    if comparison.numeric_conflicts:
        _render_numeric_conflicts(comparison)

    show_all = st.checkbox(
        "Show lines both engines agreed on",
        value=False,
        key="ofs_show_equal",
        help="Off by default so the disagreements are easy to find.",
    )
    rows = comparison.lines if show_all else comparison.disagreements()

    if not rows:
        notice("The two engines produced identical text.", "info")
        return

    _render_diff(comparison, rows)


def _render_summary(comparison: ComparisonResult) -> None:
    agreement = comparison.agreement_percent
    kind = "ok" if agreement >= 90 else "warn" if agreement >= 70 else "err"
    tiles = [
        ("Agreement", f"{agreement}%"),
        ("Identical lines", f"{comparison.equal_lines:,}"),
        ("Differing lines", f"{comparison.differing_lines:,}"),
        ("Numeric conflicts", f"{len(comparison.numeric_conflicts):,}"),
    ]
    cells = "".join(
        f'<div class="ofs-metric"><div class="ofs-metric-label">{label}</div>'
        f'<div class="ofs-metric-value">{value}</div></div>'
        for label, value in tiles
    )
    st.markdown(
        f'<div style="margin-bottom:12px;">{badge(f"{agreement}% agreement", kind)}</div>'
        f'<div class="ofs-metrics" style="margin-bottom:16px;">{cells}</div>',
        unsafe_allow_html=True,
    )


def _render_numeric_conflicts(comparison: ComparisonResult) -> None:
    """Digit disagreements, shown first because they are the costly ones."""
    count = len(comparison.numeric_conflicts)
    notice(
        f"The engines disagree on numbers in {count} "
        f"line{'s' if count != 1 else ''}. Check these before relying on any "
        "figure from this document.",
        "warn",
        "Numeric disagreements",
    )
    with st.expander(f"Review {count} numeric disagreement{'s' if count != 1 else ''}", expanded=count <= 5):
        rows = "".join(
            '<div class="ofs-diff-row numeric">'
            f'<div class="ofs-diff-cell">{escape(conflict["text_a"])}</div>'
            f'<div class="ofs-diff-cell">{escape(conflict["text_b"])}</div></div>'
            for conflict in comparison.numeric_conflicts[:100]
        )
        st.markdown(
            f'<div class="ofs-diff">'
            f'<div class="ofs-diff-head"><div>{escape(comparison.engine_a)}</div>'
            f"<div>{escape(comparison.engine_b)}</div></div>{rows}</div>",
            unsafe_allow_html=True,
        )


def _render_diff(comparison: ComparisonResult, rows) -> None:
    truncated = len(rows) > _MAX_ROWS
    visible = rows[:_MAX_ROWS]

    numeric_pairs = {
        (conflict["text_a"], conflict["text_b"])
        for conflict in comparison.numeric_conflicts
    }

    html_rows: list[str] = []
    for line in visible:
        css = _ROW_CLASS.get(line.kind, "")
        if (line.text_a, line.text_b) in numeric_pairs:
            css = "numeric"
        left = (
            escape(line.text_a)
            if line.text_a
            else '<span class="ofs-diff-empty">— not detected —</span>'
        )
        right = (
            escape(line.text_b)
            if line.text_b
            else '<span class="ofs-diff-empty">— not detected —</span>'
        )
        html_rows.append(
            f'<div class="ofs-diff-row {css}">'
            f'<div class="ofs-diff-cell">{left}</div>'
            f'<div class="ofs-diff-cell">{right}</div></div>'
        )

    st.markdown(
        f'<div class="ofs-diff">'
        f'<div class="ofs-diff-head"><div>{escape(comparison.engine_a)}</div>'
        f"<div>{escape(comparison.engine_b)}</div></div>"
        f'{"".join(html_rows)}</div>',
        unsafe_allow_html=True,
    )

    if truncated:
        st.caption(
            f"Showing the first {_MAX_ROWS:,} of {len(rows):,} rows. "
            "Export as JSON for the complete comparison."
        )


__all__ = ["render"]
