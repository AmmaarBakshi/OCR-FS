"""The Settings page (spec s6, s7).

Categories mirror the brief: General, OCR Models, Unlimited OCR, Prompts,
Pipeline, Output and Privacy. Every control is a labelled widget with a plain
explanation, so the page stays usable by someone who does not know what a
temperature or a token is.

Edits are applied to a copy of the settings and only committed on Save, so an
abandoned change never leaks into the next run.
"""

from __future__ import annotations

import streamlit as st

from app import state
from app.theme import badge, notice
from ocr_fusion.config import AppSettings, load_settings, save_settings
from ocr_fusion.config.prompts import DEFAULT_PROMPTS
from ocr_fusion.config.schema import (
    FusionStrategy,
    OutputFormat,
    Theme,
    UnlimitedBackend,
)

CATEGORIES = [
    "General",
    "OCR Models",
    "Unlimited OCR",
    "Prompts",
    "Pipeline",
    "Output",
    "Privacy",
]

_BACKEND_LABELS = {
    UnlimitedBackend.HTTP: "HTTP server (vLLM / SGLang) — recommended with a GPU",
    UnlimitedBackend.TRANSFORMERS: "In-process transformers — needs a CUDA GPU",
    UnlimitedBackend.CLI: "External command (infer.py, franken_ocr, container)",
    UnlimitedBackend.OLLAMA: "Local substitute model via Ollama — runs without a GPU",
}

_STRATEGY_LABELS = {
    FusionStrategy.LINE_VOTE: "Merge line by line (recommended)",
    FusionStrategy.PREFER_PRIMARY: "Always use the first engine",
    FusionStrategy.PREFER_LONGEST: "Use the most complete result",
    FusionStrategy.LLM: "Reconcile with a language model",
}


def render() -> None:
    """Draw the settings page."""
    current = state.settings()
    # Work on a copy: nothing takes effect until Save.
    draft = current.model_copy(deep=True)

    st.markdown("### Settings")
    category = st.radio(
        "Category",
        CATEGORIES,
        horizontal=True,
        key="ofs_settings_category",
        label_visibility="collapsed",
    )
    st.divider()

    renderers = {
        "General": _general,
        "OCR Models": _models,
        "Unlimited OCR": _unlimited,
        "Prompts": _prompts,
        "Pipeline": _pipeline,
        "Output": _output,
        "Privacy": _privacy,
    }
    renderers[category](draft)

    st.divider()
    _render_actions(draft)


def _render_actions(draft: AppSettings) -> None:
    save, cancel, reset = st.columns([1, 1, 1])
    with save:
        if st.button("Save settings", type="primary", use_container_width=True):
            state.update_settings(draft)
            st.success("Settings saved.")
            st.rerun()
    with cancel:
        if st.button("Cancel changes", use_container_width=True):
            st.rerun()
    with reset:
        if st.button("Restore defaults", use_container_width=True):
            fresh = AppSettings()
            state.update_settings(fresh)
            st.success("All settings restored to their defaults.")
            st.rerun()


# -- categories -----------------------------------------------------------


def _general(draft: AppSettings) -> None:
    general = draft.general
    left, right = st.columns(2)
    with left:
        general.theme = Theme(
            st.selectbox(
                "Theme",
                [t.value for t in Theme],
                index=[t.value for t in Theme].index(general.theme.value),
                help="Follows your operating system by default.",
            )
        )
        general.default_output_format = OutputFormat(
            st.selectbox(
                "Default export format",
                [f.value for f in OutputFormat],
                index=[f.value for f in OutputFormat].index(
                    general.default_output_format.value
                ),
            )
        )
    with right:
        general.language_hint = st.text_input(
            "Document language",
            value=general.language_hint,
            help="Passed to engines that accept a language hint. 'auto' suits most documents.",
        )
        general.auto_save_results = st.checkbox(
            "Save results automatically",
            value=general.auto_save_results,
            help="Off by default. Results stay in memory unless you turn this on.",
        )

    st.markdown("#### Interface mode")
    general.developer_mode = st.toggle(
        "Developer Mode",
        value=general.developer_mode,
        help=(
            "Off: a clean view for showing a client. "
            "On: metrics, logs, raw output and configuration."
        ),
    )
    st.caption(
        "Developer Mode adds the metrics grid, processing log and raw engine "
        "responses to the results page."
    )


def _models(draft: AppSettings) -> None:
    st.markdown("#### Connection")
    draft.ollama.host = st.text_input(
        "Ollama address",
        value=draft.ollama.host,
        help="Where Ollama is running. The default suits a local installation.",
    )

    st.markdown("#### Qwen2.5-VL (first engine)")
    qwen = draft.qwen
    qwen.enabled = st.checkbox("Use this engine", value=qwen.enabled)
    left, right = st.columns(2)
    with left:
        qwen.model = st.text_input(
            "Model name",
            value=qwen.model,
            help="Must already be pulled: ollama pull qwen2.5vl:3b",
        )
        qwen.temperature = st.slider(
            "Creativity",
            0.0,
            1.0,
            value=qwen.temperature,
            step=0.05,
            help="Keep at 0 for transcription. Higher values invite invention.",
        )
        qwen.max_tokens = st.number_input(
            "Maximum output length (tokens)",
            min_value=256,
            max_value=32768,
            value=qwen.max_tokens,
            step=256,
            help="Raise this if long pages come back cut off.",
        )
    with right:
        qwen.timeout_seconds = float(
            st.number_input(
                "Timeout (seconds)",
                min_value=30,
                max_value=3600,
                value=int(qwen.timeout_seconds),
                step=30,
                help="Without a GPU, a first run needs several minutes to load the model.",
            )
        )
        qwen.retry_count = st.number_input(
            "Retries after a failure",
            min_value=0,
            max_value=5,
            value=qwen.retry_count,
        )
        qwen.keep_alive = st.text_input(
            "Keep model loaded for",
            value=qwen.keep_alive,
            help=(
                "How long Ollama holds the model in memory. '5m' avoids reloading "
                "between pages. On a machine with limited RAM, '0' frees memory "
                "before the second engine loads."
            ),
        )

    _render_health("qwen_vl", draft)

    st.markdown("#### Tesseract (optional third engine)")
    tesseract = draft.tesseract
    tesseract.enabled = st.checkbox(
        "Use Tesseract as well", value=tesseract.enabled,
        help="Requires the Tesseract program to be installed separately.",
    )
    if tesseract.enabled:
        left, right = st.columns(2)
        with left:
            tesseract.executable = st.text_input(
                "Program path", value=tesseract.executable
            )
            tesseract.languages = st.text_input(
                "Languages", value=tesseract.languages, help="For example: eng, or eng+hin"
            )
        with right:
            tesseract.psm = st.number_input(
                "Page segmentation mode", 0, 13, value=tesseract.psm
            )
        _render_health("tesseract", draft)


def _unlimited(draft: AppSettings) -> None:
    config = draft.unlimited_ocr
    config.enabled = st.checkbox("Use this engine", value=config.enabled)

    options = list(UnlimitedBackend)
    config.backend = options[
        st.selectbox(
            "How should Unlimited-OCR run?",
            range(len(options)),
            index=options.index(config.backend),
            format_func=lambda i: _BACKEND_LABELS[options[i]],
        )
    ]

    if config.backend is UnlimitedBackend.HTTP:
        notice(
            "Run the model on a machine with an NVIDIA GPU, then point this "
            "address at it. See <code>docs/PROVIDERS.md</code> for the exact "
            "server commands.",
            "info",
        )
        config.endpoint = st.text_input("Server address", value=config.endpoint)
        config.http_model = st.text_input("Model name", value=config.http_model)
        config.api_key = st.text_input(
            "API key (optional)", value=config.api_key, type="password"
        )
    elif config.backend is UnlimitedBackend.TRANSFORMERS:
        notice(
            "This loads the full model into this machine's memory and requires "
            "a CUDA GPU. Roughly 14 GB is downloaded on first use.",
            "warn",
        )
        config.model_id = st.text_input("Model id", value=config.model_id)
        left, right = st.columns(2)
        with left:
            config.device = st.text_input("Device", value=config.device)
            config.base_size = st.number_input(
                "Base size", 256, 2048, value=config.base_size, step=64
            )
        with right:
            config.torch_dtype = st.selectbox(
                "Precision",
                ["bfloat16", "float16", "float32"],
                index=["bfloat16", "float16", "float32"].index(config.torch_dtype),
            )
            config.image_size = st.number_input(
                "Image size", 256, 2048, value=config.image_size, step=64
            )
        config.crop_mode = st.checkbox("Crop mode", value=config.crop_mode)
    elif config.backend is UnlimitedBackend.CLI:
        config.executable = st.text_input(
            "Command",
            value=config.executable,
            placeholder="python C:/tools/Unlimited-OCR/infer.py",
        )
        config.cli_args = st.text_input(
            "Arguments",
            value=config.cli_args,
            help="Use {image_path} and {output_path}; both are filled in per page.",
        )
    else:
        notice(
            "Upstream Unlimited-OCR needs a CUDA GPU. This option runs a "
            "different OCR model that is available locally, so the second stage "
            "still produces a real result. The interface labels it as a "
            "substitute wherever it appears.",
            "warn",
            "Substitute engine",
        )
        config.ollama_model = st.text_input(
            "Local model", value=config.ollama_model, help="For example: deepseek-ocr:3b"
        )

    left, right = st.columns(2)
    with left:
        config.timeout_seconds = float(
            st.number_input(
                "Timeout (seconds)", 30, 3600, value=int(config.timeout_seconds), step=30
            )
        )
    with right:
        config.max_tokens = st.number_input(
            "Maximum output length (tokens)", 256, 32768, value=config.max_tokens, step=256
        )

    _render_health("unlimited_ocr", draft)


def _prompts(draft: AppSettings) -> None:
    st.caption(
        "Prompts control how each model is instructed. They are stored separately "
        "from the application, so changes here need no code edit."
    )
    fields = [
        ("qwen_system", "Qwen system prompt", 260),
        ("qwen_user", "Qwen instruction", 180),
        ("unlimited_ocr_task", "Unlimited-OCR task prompt", 90),
        ("fusion_system", "Fusion system prompt", 240),
        ("fusion_user", "Fusion instruction", 200),
    ]
    for key, label, height in fields:
        value = st.text_area(
            label, value=getattr(draft.prompts, key), height=height, key=f"ofs_prompt_{key}"
        )
        setattr(draft.prompts, key, value)
        changed = value != DEFAULT_PROMPTS[key]
        columns = st.columns([3, 1])
        with columns[0]:
            st.markdown(
                badge("Modified", "warn") if changed else badge("Default", "neutral"),
                unsafe_allow_html=True,
            )
        with columns[1]:
            if st.button("Reset", key=f"ofs_reset_{key}", use_container_width=True):
                draft.prompts.reset_field(key)
                state.update_settings(draft)
                st.rerun()
        st.divider()


def _pipeline(draft: AppSettings) -> None:
    pipeline = draft.pipeline
    st.markdown("#### Stages to run")
    left, right = st.columns(2)
    with left:
        draft.qwen.enabled = st.checkbox("Qwen2.5-VL", value=draft.qwen.enabled)
        draft.unlimited_ocr.enabled = st.checkbox(
            "Unlimited-OCR", value=draft.unlimited_ocr.enabled
        )
        pipeline.run_comparison = st.checkbox(
            "Compare the results", value=pipeline.run_comparison
        )
    with right:
        pipeline.run_fusion = st.checkbox(
            "Produce a combined final result", value=pipeline.run_fusion
        )
        pipeline.collect_confidence = st.checkbox(
            "Collect confidence information", value=pipeline.collect_confidence
        )
        pipeline.collect_logs = st.checkbox(
            "Record a processing log", value=pipeline.collect_logs
        )

    st.markdown("#### How results are combined")
    strategies = list(FusionStrategy)
    pipeline.fusion_strategy = strategies[
        st.selectbox(
            "Strategy",
            range(len(strategies)),
            index=strategies.index(pipeline.fusion_strategy),
            format_func=lambda i: _STRATEGY_LABELS[strategies[i]],
        )
    ]
    if pipeline.fusion_strategy is FusionStrategy.LLM:
        pipeline.fusion_model = st.text_input(
            "Model used to reconcile", value=pipeline.fusion_model
        )
        st.caption(
            "The reconciled text is checked against both engines afterwards and "
            "discarded if it introduces content neither produced."
        )

    st.markdown("#### Safeguards")
    left, right = st.columns(2)
    with left:
        pipeline.continue_on_provider_error = st.checkbox(
            "Continue when an engine fails",
            value=pipeline.continue_on_provider_error,
            help="Recommended. One failed engine still lets you see the others.",
        )
    with right:
        pipeline.max_pages = st.number_input(
            "Page limit (0 = no limit)",
            min_value=0,
            max_value=500,
            value=pipeline.max_pages,
            help="Useful during a live demo so a long PDF cannot run away.",
        )


def _output(draft: AppSettings) -> None:
    output = draft.output
    st.caption("Choose what appears in the results and in exported files.")
    fields = [
        ("show_extracted_text", "Extracted text"),
        ("show_page_numbers", "Page numbers"),
        ("show_confidence", "Confidence"),
        ("show_token_usage", "Token usage"),
        ("show_processing_time", "Processing time"),
        ("show_model_name", "Model name"),
        ("show_engine_name", "OCR engine"),
        ("show_character_count", "Character count"),
        ("show_word_count", "Word count"),
        ("show_raw_results", "Raw engine results"),
        ("show_comparison", "Comparison"),
        ("show_processing_logs", "Processing logs"),
    ]
    columns = st.columns(3)
    for index, (key, label) in enumerate(fields):
        with columns[index % 3]:
            setattr(output, key, st.checkbox(label, value=getattr(output, key)))


def _privacy(draft: AppSettings) -> None:
    privacy = draft.privacy
    notice(
        "Documents are held in memory and processed on this machine. Nothing is "
        "written to disk or sent anywhere unless you switch it on below.",
        "info",
        "How your documents are handled",
    )
    privacy.persist_documents = st.checkbox(
        "Keep uploaded documents on disk", value=privacy.persist_documents
    )
    privacy.persist_results = st.checkbox(
        "Keep OCR results on disk", value=privacy.persist_results
    )
    privacy.redact_text_in_logs = st.checkbox(
        "Keep document text out of the log",
        value=privacy.redact_text_in_logs,
        help="Recommended. The log records counts rather than content.",
    )
    if not privacy.redact_text_in_logs:
        privacy.log_preview_chars = st.slider(
            "Characters of text allowed in the log", 0, 200, value=privacy.log_preview_chars
        )

    st.markdown("#### Document handling")
    documents = draft.documents
    left, right = st.columns(2)
    with left:
        documents.pdf_render_dpi = st.slider(
            "PDF quality (DPI)",
            72,
            400,
            value=documents.pdf_render_dpi,
            step=25,
            help="Higher is sharper but much slower without a GPU.",
        )
    with right:
        documents.max_file_size_mb = float(
            st.number_input(
                "Maximum file size (MB)", 1, 500, value=int(documents.max_file_size_mb)
            )
        )


def _render_health(provider_id: str, draft: AppSettings) -> None:
    """Live readiness check for one engine, run on demand.

    On demand rather than automatically: probing costs a round trip, and
    Streamlit re-runs this script on every keystroke in a text box.
    """
    if st.button("Test connection", key=f"ofs_health_{provider_id}"):
        from ocr_fusion.ocr.registry import default_registry

        try:
            provider = default_registry.create(provider_id, draft)
            status = provider.health_check()
        except Exception as exc:  # noqa: BLE001
            notice(f"The check could not run: {exc}", "err")
            return

        if status.available:
            notice(status.message, "info", "Ready")
        else:
            body = status.message
            if status.remedy:
                body += f"<br><br><b>How to fix:</b> <code>{status.remedy}</code>"
            notice(body, "err", "Not available")


__all__ = ["CATEGORIES", "render"]
