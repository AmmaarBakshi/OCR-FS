"""Metric tiles and the technical details panel (spec s8, s16)."""

from __future__ import annotations

import json
from html import escape

import streamlit as st

from ocr_fusion.config.schema import AppSettings
from ocr_fusion.metrics import NOT_AVAILABLE, Metric, engine_metrics, total_metrics
from ocr_fusion.ocr.interface import OCRResult
from ocr_fusion.pipeline.result import PipelineResult


def _tiles(metrics: list[Metric]) -> str:
    """Render metrics as a responsive grid, greying out anything unavailable."""
    cells: list[str] = []
    for metric in metrics:
        css = "" if metric.is_available else " is-na"
        title = f' title="{escape(metric.help_text)}"' if metric.help_text else ""
        cells.append(
            f'<div class="ofs-metric"{title}>'
            f'<div class="ofs-metric-label">{escape(metric.label)}</div>'
            f'<div class="ofs-metric-value{css}">{escape(metric.value)}</div></div>'
        )
    return f'<div class="ofs-metrics">{"".join(cells)}</div>'


def render_metric_grid(engine: OCRResult, settings: AppSettings) -> None:
    """Metrics for one engine."""
    st.markdown(_tiles(engine_metrics(engine, settings)), unsafe_allow_html=True)
    if any(m.value == NOT_AVAILABLE for m in engine_metrics(engine, settings)):
        st.caption(
            "N/A means this engine does not report that measurement. "
            "Nothing is estimated."
        )


def render_totals(result: PipelineResult, settings: AppSettings) -> None:
    """Run-level totals."""
    st.markdown(_tiles(total_metrics(result, settings)), unsafe_allow_html=True)


def render_technical_details(result: PipelineResult, settings: AppSettings) -> None:
    """The collapsed Developer Mode panel (spec s16).

    Hidden by default so a client sees a clean result, and complete when opened
    so an engineer never has to leave the app to debug a run.
    """
    with st.expander("Technical details", expanded=False):
        detail_tabs = st.tabs(["Per engine", "Stages", "Configuration", "Raw output"])

        with detail_tabs[0]:
            for engine in result.engine_results:
                st.markdown(
                    f'<div style="font-weight:650;margin:6px 0 8px;">'
                    f"{escape(engine.provider_name)}</div>",
                    unsafe_allow_html=True,
                )
                render_metric_grid(engine, settings)
                if engine.metadata:
                    st.json(engine.metadata, expanded=False)
                st.divider()

        with detail_tabs[1]:
            st.dataframe(
                [
                    {
                        "Stage": stage.label,
                        "Status": stage.status.value,
                        "Seconds": (
                            round(stage.duration_seconds, 3)
                            if stage.duration_seconds is not None
                            else None
                        ),
                        "Detail": stage.detail,
                    }
                    for stage in result.stages
                ],
                use_container_width=True,
                hide_index=True,
            )

        with detail_tabs[2]:
            st.caption(
                "The exact configuration this run used. API keys are redacted."
            )
            st.json(result.settings_snapshot, expanded=False)

        with detail_tabs[3]:
            st.caption("Unmodified engine responses, before any cleanup.")
            for engine in result.engine_results:
                payload = {
                    page.page_number: page.raw_response
                    for page in engine.pages
                    if page.raw_response
                }
                if payload:
                    st.markdown(f"**{escape(engine.provider_name)}**")
                    st.code(json.dumps(payload, indent=2, default=str), language="json")


__all__ = ["render_metric_grid", "render_technical_details", "render_totals"]
