"""Result panels: per-engine output, the final result, and exports (spec s3, s5)."""

from __future__ import annotations

from html import escape

import streamlit as st

from app import state
from app.components import comparison_view, metrics_view
from app.theme import badge, empty_state, notice
from ocr_fusion.config.schema import AppSettings, OutputFormat
from ocr_fusion.export import MIME_TYPES, export, export_filename
from ocr_fusion.ocr.interface import OCRResult, OCRStatus
from ocr_fusion.pipeline.result import PipelineResult

_STATUS_BADGE = {
    OCRStatus.SUCCESS: ("Completed", "ok"),
    OCRStatus.PARTIAL: ("Partly completed", "warn"),
    OCRStatus.FAILED: ("Failed", "err"),
    OCRStatus.SKIPPED: ("Skipped", "neutral"),
}


def render(result: PipelineResult, settings: AppSettings) -> None:
    """Draw the results panel: final result, each engine, comparison, exports."""
    tabs: list[str] = ["Final result"]
    for engine in result.engine_results:
        tabs.append(_short_name(engine.provider_name))
    if settings.output.show_comparison and result.comparison is not None:
        tabs.append("Comparison")
    tabs.append("Export")

    rendered = st.tabs(tabs)
    index = 0

    with rendered[index]:
        _render_final(result, settings)
    index += 1

    for engine in result.engine_results:
        with rendered[index]:
            _render_engine(engine, result, settings)
        index += 1

    if settings.output.show_comparison and result.comparison is not None:
        with rendered[index]:
            comparison_view.render(result.comparison, settings)
        index += 1

    with rendered[index]:
        _render_export(result, settings)


def _short_name(name: str) -> str:
    """Trim a provider's parenthetical model suffix so tab labels stay readable."""
    return name.split(" (")[0]


def _render_final(result: PipelineResult, settings: AppSettings) -> None:
    text = result.final_text
    if not text.strip():
        failures = [e for e in result.engine_results if e.status is OCRStatus.FAILED]
        empty_state(
            "⚠",
            "No text could be extracted",
            "Every OCR engine failed. Open Technical details for the reason.",
        )
        for engine in failures:
            notice(
                f"<b>{escape(engine.provider_name)}</b>: {escape(engine.error or 'failed')}"
                + (f"<br><code>{escape(engine.remedy)}</code>" if engine.remedy else ""),
                "err",
            )
        return

    if result.fusion:
        chips = [badge(f"Strategy: {result.fusion.strategy}", "info")]
        for source in result.fusion.sources:
            chips.append(badge(_short_name(source), "neutral"))
        if result.fusion.fallback_used:
            chips.append(badge("Fallback used", "warn"))
        st.markdown(
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">'
            f'{"".join(chips)}</div>',
            unsafe_allow_html=True,
        )
        if state.developer_mode() and result.fusion.detail:
            st.caption(result.fusion.detail)
        for warning in result.fusion.warnings:
            notice(escape(warning), "warn")

    _render_text(text)
    _render_counts(text, settings)


def _render_engine(
    engine: OCRResult, result: PipelineResult, settings: AppSettings
) -> None:
    label, kind = _STATUS_BADGE.get(engine.status, ("Unknown", "neutral"))
    chips = [badge(label, kind)]
    if settings.output.show_model_name and engine.model_name:
        chips.append(badge(engine.model_name, "neutral"))
    if engine.backend:
        chips.append(badge(f"via {engine.backend}", "neutral"))
    chips.append(
        badge(
            f"{engine.processing_location.value.title()} processing",
            "ok" if engine.processing_location.value == "local" else "warn",
        )
    )
    st.markdown(
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;">'
        f'{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )

    # A substitute engine must never be mistaken for the real one (spec s16).
    if engine.metadata.get("is_substitute"):
        notice(
            escape(
                engine.metadata.get("substitute_note")
                or "This stage ran a substitute model, not upstream Unlimited-OCR."
            ),
            "warn",
            "Substitute engine",
        )

    if engine.status is OCRStatus.FAILED:
        notice(
            escape(engine.error or "This engine failed.")
            + (
                f"<br><br><b>How to fix:</b> {escape(engine.remedy)}"
                if engine.remedy
                else ""
            ),
            "err",
            f"{_short_name(engine.provider_name)} could not run",
        )
        return

    if engine.status is OCRStatus.PARTIAL:
        failed = [p for p in engine.pages if p.status is OCRStatus.FAILED]
        notice(
            f"{len(failed)} of {len(engine.pages)} pages could not be read: "
            + escape(", ".join(f"page {p.page_number}" for p in failed[:8])),
            "warn",
        )

    if settings.output.show_page_numbers and len(engine.pages) > 1:
        _render_paged(engine, settings)
    else:
        _render_text(engine.text)
        _render_counts(engine.text, settings)

    if settings.output.show_processing_time or settings.output.show_token_usage:
        with st.expander("Metrics for this engine"):
            metrics_view.render_metric_grid(engine, settings)


def _render_paged(engine: OCRResult, settings: AppSettings) -> None:
    """Per-page output, so a multi-page document stays navigable (spec s2)."""
    page_numbers = [p.page_number for p in sorted(engine.pages, key=lambda p: p.page_number)]
    chosen = st.selectbox(
        "Page",
        page_numbers,
        key=f"ofs_page_select_{engine.provider_id}",
        format_func=lambda number: f"Page {number}",
    )
    page = engine.page(chosen)
    if page is None:
        return
    if page.status is OCRStatus.FAILED:
        notice(escape(page.error or "This page could not be read."), "err")
        return
    if page.error:
        notice(escape(page.error), "warn")
    _render_text(page.text)
    _render_counts(page.text, settings)


def _render_text(text: str) -> None:
    st.markdown(
        f'<div class="ofs-output">{escape(text)}</div>', unsafe_allow_html=True
    )


def _render_counts(text: str, settings: AppSettings) -> None:
    parts: list[str] = []
    if settings.output.show_character_count:
        parts.append(f"{len(text):,} characters")
    if settings.output.show_word_count:
        parts.append(f"{len(text.split()):,} words")
    if parts:
        st.markdown(
            f'<div style="margin-top:8px;font-size:12px;color:var(--ink-faint);">'
            f'{" · ".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def _render_export(result: PipelineResult, settings: AppSettings) -> None:
    """Download buttons for every supported format (spec s10)."""
    st.markdown(
        '<div style="font-size:13px;color:var(--ink-soft);margin-bottom:14px;">'
        "Exports follow the Output settings, so anything switched off there is "
        "left out of the file as well.</div>",
        unsafe_allow_html=True,
    )

    descriptions = {
        OutputFormat.TXT: "Plain transcription",
        OutputFormat.MARKDOWN: "Formatted report",
        OutputFormat.JSON: "Full structured data",
        OutputFormat.CSV: "One row per page",
    }

    for output_format in OutputFormat:
        try:
            payload = export(result, settings, output_format)
        except Exception as exc:  # noqa: BLE001 - one broken format must not
            # hide the others.
            notice(f"{output_format.value.upper()} export failed: {escape(str(exc))}", "err")
            continue

        left, right = st.columns([3, 1])
        with left:
            st.markdown(
                f'<div style="padding-top:6px;"><b>{output_format.value.upper()}</b> '
                f'<span style="color:var(--ink-faint);font-size:12.5px;">'
                f"— {descriptions[output_format]} · {len(payload):,} bytes</span></div>",
                unsafe_allow_html=True,
            )
        with right:
            st.download_button(
                "Download",
                data=payload.encode("utf-8"),
                file_name=export_filename(result, output_format),
                mime=MIME_TYPES[output_format],
                key=f"ofs_download_{output_format.value}",
                use_container_width=True,
            )


__all__ = ["render"]
