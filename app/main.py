"""OCR Fusion Studio - Streamlit application entry point.

Run with::

    streamlit run app/main.py

The UI depends only on :mod:`ocr_fusion`'s public interfaces - the provider
registry, the pipeline and the settings schema. It never imports a concrete OCR
engine, which is what lets a new engine appear here without a UI change
(spec s11).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app/main.py` from a checkout without installing the
# package: Streamlit puts the script's directory on sys.path, not the project
# root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import state  # noqa: E402
from app.components import (  # noqa: E402
    document_preview,
    logs,
    metrics_view,
    pipeline_view,
    results,
    settings_page,
    upload,
)
from app.theme import apply_theme, badge, card, empty_state, notice  # noqa: E402
from ocr_fusion.documents.models import Document  # noqa: E402
from ocr_fusion.ocr.registry import default_registry  # noqa: E402
from ocr_fusion.pipeline import build_pipeline  # noqa: E402
from ocr_fusion.version import __version__  # noqa: E402

import ocr_fusion.ocr.providers  # noqa: E402,F401  (registers the built-in engines)

PAGE_STUDIO = "Studio"
PAGE_SETTINGS = "Settings"


def main() -> None:
    st.set_page_config(
        page_title="OCR Fusion Studio",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    page = _render_sidebar()
    _render_header()

    if page == PAGE_SETTINGS:
        settings_page.render()
        return

    _render_studio()


# -- chrome ---------------------------------------------------------------


def _render_header() -> None:
    st.markdown(
        '<div class="ofs-brand">'
        '<div class="ofs-logo">◈</div>'
        "<div><div class=\"ofs-title\">OCR Fusion Studio</div>"
        '<div class="ofs-subtitle">Multi-engine document transcription with '
        "result comparison</div></div></div>",
        unsafe_allow_html=True,
    )
    st.write("")


def _render_sidebar() -> str:
    settings = state.settings()

    with st.sidebar:
        st.markdown(
            '<div style="font-size:12px;font-weight:650;text-transform:uppercase;'
            'letter-spacing:.09em;color:var(--ink-faint);margin-bottom:10px;">'
            "Workspace</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Page",
            [PAGE_STUDIO, PAGE_SETTINGS],
            label_visibility="collapsed",
            key="ofs_page_nav",
        )

        st.divider()

        developer = st.toggle(
            "Developer Mode",
            value=settings.general.developer_mode,
            help="Show metrics, logs, raw output and configuration.",
            key="ofs_dev_toggle",
        )
        if developer != settings.general.developer_mode:
            settings.general.developer_mode = developer
            state.update_settings(settings)
            st.rerun()
        st.caption(
            "Developer Mode" if developer else "Demo Mode - a clean view for clients"
        )

        st.divider()
        _render_engine_status(settings)

        st.divider()
        st.markdown(
            '<div style="font-size:11.5px;color:var(--ink-faint);line-height:1.6;">'
            "<b>Processing is local.</b><br>Documents stay in memory on this "
            "machine and are not uploaded anywhere.</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Version {__version__}")

    return page


def _render_engine_status(settings) -> None:
    """Sidebar readiness summary, refreshed on demand."""
    st.markdown(
        '<div style="font-size:12px;font-weight:650;text-transform:uppercase;'
        'letter-spacing:.09em;color:var(--ink-faint);margin-bottom:8px;">'
        "Engines</div>",
        unsafe_allow_html=True,
    )

    report = state.health()
    if report is None:
        if st.button("Check engines", use_container_width=True, key="ofs_check_engines"):
            with st.spinner("Checking…"):
                state.set_health(build_pipeline(settings).health_report())
            st.rerun()
        st.caption("Not checked yet.")
        return

    for info in report.values():
        kind = "ok" if info["available"] else "err"
        label = "Ready" if info["available"] else "Unavailable"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;gap:8px;margin-bottom:7px;">'
            f'<span style="font-size:12.5px;color:var(--ink-soft);overflow-wrap:anywhere;">'
            f'{info["name"].split(" (")[0]}</span>{badge(label, kind)}</div>',
            unsafe_allow_html=True,
        )
        if not info["available"] and state.developer_mode():
            st.caption(info["message"])

    if st.button("Re-check", use_container_width=True, key="ofs_recheck_engines"):
        state.set_health(None)
        st.rerun()


# -- studio page ----------------------------------------------------------


def _render_studio() -> None:
    settings = state.settings()
    document = state.document()
    result = state.result()

    if document is None:
        _render_intake(settings)
        return

    _render_run_bar(document, settings)

    if result is None:
        _render_pending(settings)
    else:
        _render_results(result, settings)


def _render_intake(settings) -> None:
    """First-run view: upload area plus what is about to happen (spec s15)."""
    upload.render_error()

    with card("Upload a document"):
        upload.render()

    with card("How it works"):
        pipeline_view.render_placeholder(_planned_stage_labels(settings))
        st.markdown(
            '<div style="margin-top:14px;font-size:13px;color:var(--ink-soft);'
            'line-height:1.65;">Each engine transcribes the document independently. '
            "Their results are compared side by side, and a single combined result "
            "is produced from both. Every stage reports how long it took.</div>",
            unsafe_allow_html=True,
        )


def _planned_stage_labels(settings) -> list[str]:
    """Stage names for the pre-run preview, taken from the registry."""
    labels = ["Document"]
    from ocr_fusion.ocr.providers import PIPELINE_ORDER

    enabled = {spec.provider_id: spec for spec in default_registry.enabled_specs(settings)}
    for provider_id in PIPELINE_ORDER:
        if provider_id in enabled:
            labels.append(enabled[provider_id].display_name)
    if settings.pipeline.run_comparison:
        labels.append("Comparison")
    if settings.pipeline.run_fusion:
        labels.append("Fusion")
    return labels


def _render_run_bar(document: Document, settings) -> None:
    """Document summary and the Run OCR control."""
    with card():
        left, right = st.columns([3, 1])
        with left:
            st.markdown(
                f'<div style="font-size:15px;font-weight:650;color:var(--ink);'
                f'overflow-wrap:anywhere;">{document.filename}</div>'
                f'<div style="font-size:12.5px;color:var(--ink-faint);margin-top:3px;">'
                f'{document.page_count} page{"s" if document.page_count != 1 else ""} · '
                f"{document.kind.value.upper()}</div>",
                unsafe_allow_html=True,
            )
        with right:
            if st.button("Run OCR", type="primary", use_container_width=True):
                _execute(document, settings)
            if st.button("Choose another file", use_container_width=True):
                state.reset_run()
                st.session_state[state.KEY_UPLOAD_TOKEN] = None
                st.rerun()


def _render_pending(settings) -> None:
    """Between upload and run: preview on the left, waiting pipeline on the right."""
    left, right = st.columns([1, 1], gap="large")
    with left:
        with card("Document"):
            document_preview.render(state.document())
    with right:
        with card("Pipeline"):
            pipeline_view.render_placeholder(_planned_stage_labels(settings))
        empty_state(
            "▶",
            "Ready when you are",
            "Select Run OCR to transcribe this document with every enabled engine.",
        )


def _execute(document: Document, settings) -> None:
    """Run the pipeline, updating the stage display as it advances."""
    pipeline = build_pipeline(settings)
    if not pipeline.providers:
        notice(
            "No OCR engines are switched on. Enable at least one in "
            "Settings › Pipeline.",
            "err",
        )
        return

    with card("Running"):
        stage_slot = st.empty()
        status_slot = st.empty()

    def on_event(event) -> None:
        """Redraw the pipeline as each stage reports in."""
        with stage_slot.container():
            pipeline_view.render_stage_row(pipeline.log.ordered_stages())
        status_slot.markdown(
            f'<div style="font-size:13px;color:var(--ink-soft);margin-top:6px;">'
            f"{event.message}</div>",
            unsafe_allow_html=True,
        )

    pipeline.subscribe(on_event)
    state.set_running(True)
    try:
        with st.spinner("Transcribing… the first run also loads the models, which takes longer."):
            result = pipeline.execute(document)
    except Exception as exc:  # noqa: BLE001 - the app must survive any failure
        # the pipeline did not already convert into a stage result.
        notice(
            f"The run could not be completed: {exc}",
            "err",
            "Something went wrong",
        )
        return
    finally:
        state.set_running(False)

    state.set_result(result)
    st.rerun()


def _render_results(result, settings) -> None:
    with card("Pipeline"):
        pipeline_view.render_stage_row(result.stages)
        pipeline_view.render_total(result.total_duration_seconds, result.stages)

    if not result.succeeded:
        notice(
            "No engine was able to read this document. The details below explain "
            "why, and each one includes what to change.",
            "err",
            "Nothing could be extracted",
        )

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with card("Document"):
            document_preview.render(result.document)
    with right:
        with card("Results"):
            results.render(result, settings)

    if state.developer_mode():
        with card("Metrics"):
            metrics_view.render_totals(result, settings)

        metrics_view.render_technical_details(result, settings)

        if settings.output.show_processing_logs and result.log is not None:
            logs.render(result.log)


if __name__ == "__main__":
    main()
