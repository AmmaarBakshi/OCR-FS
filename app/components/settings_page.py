"""The Settings page (spec s6, s7).

Categories mirror the brief: General, OCR Models, Unlimited OCR, Prompts,
Pipeline, Output and Privacy. Every control is a labelled widget with a plain
explanation, so the page stays usable by someone who does not know what a
temperature or a token is.

Edits are applied to a copy of the settings and only committed on Save, so an
abandoned change never leaks into the next run.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from app import state
from app.theme import badge, notice
from ocr_fusion.chat import DocumentChat
from ocr_fusion.config import AppSettings, load_settings, save_settings
from ocr_fusion.config.prompts import DEFAULT_PROMPTS
from ocr_fusion.config.schema import (
    DeliveryMode,
    EngineMode,
    FusionStrategy,
    OutputFormat,
    Theme,
    UnlimitedBackend,
)

CATEGORIES = [
    "General",
    "Speed",
    "OCR Models",
    "Unlimited OCR",
    "Prompts",
    "Pipeline",
    "Chat",
    "Output",
    "Privacy",
]

_BACKEND_LABELS = {
    UnlimitedBackend.HTTP: "HTTP server (vLLM / SGLang) — recommended with a GPU",
    UnlimitedBackend.TRANSFORMERS: "In-process transformers — needs a CUDA GPU",
    UnlimitedBackend.CLI: "External command (infer.py, franken_ocr, container)",
    UnlimitedBackend.OLLAMA: "Local substitute model via Ollama — runs without a GPU",
}

_ENGINE_MODE_LABELS = {
    EngineMode.CASCADE: "Cascade - each engine handles what the last could not",
    EngineMode.ALL_ENGINES: "Every engine reads every page - needed for comparison",
}

#: Which settings section holds each engine's own "enabled" switch.
_ENGINE_SETTING = {"qwen_vl": "qwen", "unlimited_ocr": "unlimited_ocr"}

_STRATEGY_LABELS = {
    FusionStrategy.LINE_VOTE: "Merge line by line (recommended)",
    FusionStrategy.PREFER_PRIMARY: "Always use the first engine",
    FusionStrategy.PREFER_LONGEST: "Use the most complete result",
    FusionStrategy.LLM: "Reconcile with a language model",
}


#: One line of orientation per category, shown under the heading so the page
#: says what you are looking at instead of making you infer it from the fields.
_CATEGORY_BLURBS = {
    "General": "Appearance, language and the interface mode.",
    "Speed": "How much work each page is worth, and what that costs.",
    "OCR Models": "Where Ollama runs, and how the first engine reads a page.",
    "Unlimited OCR": "How the second engine runs, and which model it uses.",
    "Prompts": "The exact instructions each model is given.",
    "Pipeline": "Which stages run, and how two results become one.",
    "Chat": "Asking questions about a document after it is transcribed.",
    "Output": "How you get the result, and what it includes.",
    "Privacy": "What is kept, and what is written to disk.",
}


def render() -> None:
    """Draw the settings page."""
    current = state.settings()
    # Work on a copy: nothing takes effect until Save.
    draft = current.model_copy(deep=True)

    st.markdown("### Settings")
    category = _render_nav()
    st.caption(_CATEGORY_BLURBS.get(category, ""))
    st.divider()

    renderers = {
        "General": _general,
        "Speed": _speed,
        "OCR Models": _models,
        "Unlimited OCR": _unlimited,
        "Prompts": _prompts,
        "Pipeline": _pipeline,
        "Chat": _chat,
        "Output": _output,
        "Privacy": _privacy,
    }
    renderers[category](draft)

    st.divider()
    _render_actions(draft)


def _render_nav() -> str:
    """Category navigation.

    A segmented control rather than a row of radios: eight radio dials read as
    a question with eight answers, which is not what picking a settings page
    is. Clicking the active segment clears the selection, so the previous
    category is remembered and reused rather than silently snapping back to
    the first one.
    """
    chosen = st.segmented_control(
        "Category",
        CATEGORIES,
        default=st.session_state.get(_ACTIVE_CATEGORY, CATEGORIES[0]),
        key="ofs_settings_category",
        label_visibility="collapsed",
    )
    if chosen is None:
        chosen = st.session_state.get(_ACTIVE_CATEGORY, CATEGORIES[0])
    st.session_state[_ACTIVE_CATEGORY] = chosen
    return chosen


#: Remembers the open category across the rerun that a deselection causes.
_ACTIVE_CATEGORY = "ofs_settings_category_active"


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


def _speed(draft: AppSettings) -> None:
    """The settings that decide how much computation a page is worth.

    Gathered on one page because they trade against each other: raising the
    render resolution and sending every page to every engine are the same
    decision made twice, and seeing them apart is how a configuration ends up
    slow for no reason anybody chose.
    """
    from ocr_fusion.config.profiles import (
        PROFILE_MEASUREMENTS,
        PerformanceProfile,
        apply_profile,
        current_profile,
    )

    st.markdown("#### Profile")
    profiles = list(PerformanceProfile)
    active = current_profile(draft)

    def label(index: int) -> str:
        profile = profiles[index]
        measured = PROFILE_MEASUREMENTS[profile]
        return (
            f"{profile.value.title()} - about "
            f"{measured['seconds_per_scanned_page']:.0f}s a scanned page, "
            f"{measured['recall']:.0%} of words recovered"
        )

    chosen = st.selectbox(
        "Speed and accuracy",
        range(len(profiles)),
        index=profiles.index(active) if active else 1,
        format_func=label,
        help=(
            "Measured on a real form page scored against the text that page "
            "was authored with. Per scanned page - a page with a usable text "
            "layer costs milliseconds under every profile."
        ),
    )
    if st.button("Apply this profile", use_container_width=False):
        apply_profile(draft, profiles[chosen])
        state.update_settings(draft)
        st.rerun()
    if active is None:
        notice(
            "The current settings do not match a profile. That is fine - the "
            "controls below are the ones a profile sets.",
            "info",
        )

    st.markdown("#### Which pages reach a model")
    routing = draft.routing
    routing.enabled = st.checkbox(
        "Decide per page how much work it is worth",
        value=routing.enabled,
        help=(
            "Off means every page goes to every engine. On this project's "
            "reference corpus that is about six times the work for no gain."
        ),
    )
    if routing.enabled:
        left, right = st.columns(2)
        with left:
            routing.use_text_layer = st.checkbox(
                "Use the text a PDF already contains",
                value=routing.use_text_layer,
                help=(
                    "On a born-digital page that text is what the file says, "
                    "rather than a transcription of a picture of it - so it is "
                    "both free and more accurate."
                ),
            )
            routing.skip_blank_pages = st.checkbox(
                "Skip blank pages", value=routing.skip_blank_pages
            )
        with right:
            routing.detect_duplicates = st.checkbox(
                "Reuse identical pages",
                value=routing.detect_duplicates,
                help=(
                    "Exact matches only. Two statement pages can look alike "
                    "and differ only in the figures."
                ),
            )
            routing.text_layer_min_words = st.number_input(
                "Words needed to trust a text layer",
                min_value=0,
                max_value=500,
                value=routing.text_layer_min_words,
                help="Guards against a page number stamped on a scan.",
            )

    st.markdown("#### How much the engines do")
    pipeline = draft.pipeline
    modes = list(EngineMode)
    pipeline.engine_mode = modes[
        st.selectbox(
            "Engine mode",
            range(len(modes)),
            index=modes.index(pipeline.engine_mode),
            format_func=lambda i: _ENGINE_MODE_LABELS[modes[i]],
        )
    ]
    if pipeline.engine_mode is EngineMode.CASCADE:
        pipeline.fallback_enabled = st.checkbox(
            "Let a second engine re-read doubtful pages",
            value=pipeline.fallback_enabled,
            help=(
                "Only pages that came back empty, cut off, repetitive or too "
                "thin for the ink on the page."
            ),
        )
    else:
        notice(
            "Every enabled engine will read every page. That is what the "
            "side-by-side comparison needs, and it costs one full pass per "
            "engine.",
            "warn",
        )

    st.markdown("#### Which engine reads first")
    ocr_engines = [
        ("qwen_vl", "Qwen2.5-VL", "Slower, and more careful with figures."),
        ("unlimited_ocr", "Unlimited-OCR / substitute", "About 3x faster; measured to lose more figures."),
    ]
    available = [e for e in ocr_engines if getattr(draft, _ENGINE_SETTING[e[0]]).enabled]
    if len(available) < 2:
        st.caption(
            "Enable both OCR engines to choose which one leads and which one "
            "re-reads doubtful pages."
        )
    else:
        current = next(
            (i for i, e in enumerate(available) if e[0] in draft.pipeline.engine_order[:2]),
            0,
        )
        chosen = st.selectbox(
            "Primary engine",
            range(len(available)),
            index=current,
            format_func=lambda i: f"{available[i][1]} - {available[i][2]}",
            help=(
                "Under cascade the primary reads every page that needs a "
                "model, and the other engine only re-reads pages the "
                "confidence check flagged."
            ),
        )
        primary = available[chosen][0]
        others = [e[0] for e in available if e[0] != primary]
        draft.pipeline.engine_order = ["text_layer", primary, *others]
        notice(
            "Measured here on a bank statement page: Qwen took 444s and "
            "recovered 99.7% of words and 89% of figures; the substitute took "
            "142s for 98.1% of words and 50% of figures. Speed against "
            "figures - choose for the documents you actually process.",
            "info",
        )

    st.markdown("#### Render resolution")
    draft.documents.pdf_render_dpi = st.slider(
        "DPI",
        min_value=72,
        max_value=300,
        value=draft.documents.pdf_render_dpi,
        step=6,
        help=(
            "The single biggest lever on inference cost: a vision model's "
            "prompt is mostly image tokens, and image tokens scale with pixel "
            "area."
        ),
    )
    st.caption(
        "Measured here: 150 DPI 551s a page at 99.7% recall, 120 DPI 398s at "
        "99.0%, 96 DPI 326s at 92.7%. Below 120 accuracy falls away quickly - "
        "blurred text does not just get misread, it makes the model ramble."
    )

    st.markdown("#### Remembering pages")
    draft.cache.enabled = st.checkbox(
        "Never read the same page twice",
        value=draft.cache.enabled,
        help=(
            "Writes transcribed text to disk so a repeated page, or a re-run "
            "after a crash, costs nothing."
        ),
    )
    if draft.cache.enabled:
        draft.cache.directory = st.text_input(
            "Where to keep it", value=draft.cache.directory
        )
        notice(
            "This writes document text to disk. Keep it wherever client data "
            "is allowed to live.",
            "warn",
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
        config.keep_alive = st.text_input(
            "Keep model loaded for",
            value=config.keep_alive,
            help=(
                "How long Ollama holds the model in memory. '5m' avoids reloading "
                "between pages. On a machine with limited RAM, set this and the "
                "same Qwen setting to '0' so each engine frees memory before the "
                "next one loads."
            ),
            key="ofs_unlimited_keep_alive",
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
        ("chat_system", "Question-answering system prompt", 260),
        ("chat_user", "Question-answering instruction", 160),
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


def _chat(draft: AppSettings) -> None:
    chat = draft.chat
    st.caption(
        "Ask questions about a document once it has been transcribed. Answers "
        "are drawn from the transcription, so nothing is sent anywhere the OCR "
        "run did not already go."
    )
    chat.enabled = st.checkbox("Offer the question box after a run", value=chat.enabled)

    left, right = st.columns(2)
    with left:
        chat.model = st.text_input(
            "Model",
            value=chat.model,
            help=(
                "A text model, not a vision one - the question is answered from "
                "the transcription. Must already be pulled: ollama pull "
                f"{chat.model or 'qwen2.5:1.5b'}"
            ),
            key="ofs_chat_model",
        )
        chat.max_tokens = int(
            st.number_input(
                "Maximum answer length (tokens)",
                min_value=128,
                max_value=16384,
                value=chat.max_tokens,
                step=128,
                help="Raise this if long answers come back cut off.",
            )
        )
        chat.keep_alive = st.text_input(
            "Keep model loaded for",
            value=chat.keep_alive,
            help=(
                "How long Ollama holds this model in memory between questions. "
                "Set to '0' on a machine that cannot hold it alongside the OCR "
                "models."
            ),
            key="ofs_chat_keep_alive",
        )
    with right:
        chat.timeout_seconds = float(
            st.number_input(
                "Timeout (seconds)",
                min_value=15,
                max_value=1800,
                value=int(chat.timeout_seconds),
                step=15,
                help="The first question also loads the model, so it takes longest.",
                key="ofs_chat_timeout",
            )
        )
        chat.history_turns = int(
            st.number_input(
                "Earlier questions to remember",
                min_value=0,
                max_value=50,
                value=chat.history_turns,
                help=(
                    "How much of the conversation is sent back, so a follow-up "
                    "like 'and the date?' knows what it refers to. 0 treats "
                    "every question as the first."
                ),
            )
        )
        chat.max_context_characters = int(
            st.number_input(
                "Most characters of the document to send",
                min_value=1000,
                max_value=500_000,
                value=chat.max_context_characters,
                step=1000,
                help=(
                    "A longer document is trimmed from the middle, keeping the "
                    "start and the end, and the answer says so."
                ),
            )
        )

    if st.button("Check the chat model", key="ofs_chat_health"):
        with st.spinner("Checking…"):
            status = DocumentChat(draft).health_check()
        if status.available:
            notice(escape(status.message), "info", "Ready")
        else:
            notice(
                escape(status.message)
                + (f"<br><br><b>How to fix:</b> {escape(status.remedy)}" if status.remedy else ""),
                "err",
                "Not ready",
            )


_DELIVERY_LABELS = {
    DeliveryMode.ON_SITE: "On site - read the result here",
    DeliveryMode.OFF_SITE: "Off site - hand it over as a file",
}

_FORMAT_LABELS = {
    OutputFormat.PDF: "PDF - a report anyone can open",
    OutputFormat.MARKDOWN: "Markdown - a readable report",
    OutputFormat.HTML: "HTML - a self-contained web page",
    OutputFormat.TXT: "Plain text - the transcription only",
    OutputFormat.JSON: "JSON - the full structured data",
    OutputFormat.XML: "XML - the same data as XML",
    OutputFormat.CSV: "CSV - one row per page, per engine",
}


def _format_choice(
    label: str, current: OutputFormat, key: str, help_text: str = ""
) -> OutputFormat:
    """A format picker labelled in terms of what the file is for."""
    formats = list(OutputFormat)
    return formats[
        st.selectbox(
            label,
            range(len(formats)),
            index=formats.index(current),
            format_func=lambda index: _FORMAT_LABELS[formats[index]],
            key=key,
            help=help_text or None,
        )
    ]


def _output(draft: AppSettings) -> None:
    output = draft.output

    st.markdown("#### How you get the result")
    modes = list(DeliveryMode)
    output.delivery_mode = modes[
        st.radio(
            "Delivery",
            range(len(modes)),
            index=modes.index(output.delivery_mode),
            format_func=lambda index: _DELIVERY_LABELS[modes[index]],
            key="ofs_delivery_mode",
            label_visibility="collapsed",
        )
    ]

    if output.delivery_mode is DeliveryMode.ON_SITE:
        st.caption(
            "The transcription is shown in the page, and you can still download "
            "it in any format from the Export tab."
        )
    else:
        output.download_format = _format_choice(
            "File type",
            output.download_format,
            key="ofs_download_format",
            help_text=(
                "The file the results panel offers first. Every other format "
                "stays available in the Export tab."
            ),
        )
        st.caption(
            "The results panel leads with this file instead of printing the "
            "transcription."
        )

    st.divider()
    st.markdown("#### What the result includes")
    st.caption(
        "These apply to what you see and to exported files alike, so anything "
        "switched off here is left out of the file as well."
    )
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
