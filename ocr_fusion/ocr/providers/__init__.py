"""Built-in OCR providers, registered on the default registry.

Importing this module registers every bundled engine. Nothing else in the
codebase imports a concrete provider class - the pipeline and the UI resolve
engines by id through :data:`~ocr_fusion.ocr.registry.default_registry`, which
is what keeps a new engine from requiring UI changes (spec s11).

To add an engine, see ``docs/PROVIDERS.md``.
"""

from __future__ import annotations

from ocr_fusion.ocr.registry import ProviderSpec, default_registry, register_provider

#: Stage order. The pipeline runs enabled providers in this sequence, and
#: cheapest-first is deliberate: text-layer extraction answers most pages
#: outright, so the vision models only ever see what it could not.
PIPELINE_ORDER: tuple[str, ...] = (
    "text_layer",
    "qwen_vl",
    "unlimited_ocr",
    "tesseract",
)


def _register_builtin_providers() -> None:
    """Register the bundled engines, ignoring an already-populated registry.

    Streamlit re-executes modules on every rerun, so registration must be
    idempotent rather than raising on the second import.
    """
    from ocr_fusion.ocr.providers.qwen_vl import build_qwen_provider
    from ocr_fusion.ocr.providers.text_layer import build_text_layer_provider
    from ocr_fusion.ocr.providers.tesseract import build_tesseract_provider
    from ocr_fusion.ocr.providers.unlimited import build_unlimited_provider

    specs = [
        ProviderSpec(
            provider_id="text_layer",
            display_name="PDF text layer",
            factory=build_text_layer_provider,
            description=(
                "Reads the text a PDF was authored with. No model, no "
                "recognition and no inference cost - and on a born-digital "
                "page the result is exact rather than transcribed."
            ),
            enabled_check=lambda s: s.text_layer.enabled,
            enable_setter=lambda s, on: setattr(s.text_layer, "enabled", on),
            tags=("extraction", "local", "free"),
        ),
        ProviderSpec(
            provider_id="qwen_vl",
            display_name="Qwen2.5-VL",
            factory=build_qwen_provider,
            description=(
                "Vision-language transcription through Ollama. Runs locally and "
                "reports exact token and timing metrics."
            ),
            enabled_check=lambda s: s.qwen.enabled,
            enable_setter=lambda s, on: setattr(s.qwen, "enabled", on),
            tags=("vlm", "local", "ollama"),
        ),
        ProviderSpec(
            provider_id="unlimited_ocr",
            display_name="Unlimited-OCR",
            factory=build_unlimited_provider,
            description=(
                "Baidu Unlimited-OCR, executed through a configurable backend "
                "(HTTP server, transformers, external command, or a local "
                "substitute model when no Unlimited-OCR runtime is available)."
            ),
            enabled_check=lambda s: s.unlimited_ocr.enabled,
            enable_setter=lambda s, on: setattr(s.unlimited_ocr, "enabled", on),
            tags=("vlm", "pluggable"),
        ),
        ProviderSpec(
            provider_id="tesseract",
            display_name="Tesseract",
            factory=build_tesseract_provider,
            description=(
                "Classical OCR engine. Off by default; included to demonstrate "
                "that adding an engine needs no pipeline or UI change."
            ),
            enabled_check=lambda s: s.tesseract.enabled,
            enable_setter=lambda s, on: setattr(s.tesseract, "enabled", on),
            tags=("classical", "local", "optional"),
        ),
    ]
    for spec in specs:
        register_provider(spec, replace=True)


_register_builtin_providers()

__all__ = ["PIPELINE_ORDER", "default_registry"]
