"""Built-in OCR providers, registered on the default registry.

Importing this module registers every bundled engine. Nothing else in the
codebase imports a concrete provider class - the pipeline and the UI resolve
engines by id through :data:`~ocr_fusion.ocr.registry.default_registry`, which
is what keeps a new engine from requiring UI changes (spec s11).

To add an engine, see ``docs/PROVIDERS.md``.
"""

from __future__ import annotations

from ocr_fusion.ocr.registry import ProviderSpec, default_registry, register_provider

#: Stage order. The pipeline runs enabled providers in this sequence.
PIPELINE_ORDER: tuple[str, ...] = ("qwen_vl", "unlimited_ocr", "tesseract")


def _register_builtin_providers() -> None:
    """Register the bundled engines, ignoring an already-populated registry.

    Streamlit re-executes modules on every rerun, so registration must be
    idempotent rather than raising on the second import.
    """
    from ocr_fusion.ocr.providers.qwen_vl import build_qwen_provider
    from ocr_fusion.ocr.providers.tesseract import build_tesseract_provider
    from ocr_fusion.ocr.providers.unlimited import build_unlimited_provider

    specs = [
        ProviderSpec(
            provider_id="qwen_vl",
            display_name="Qwen2.5-VL",
            factory=build_qwen_provider,
            description=(
                "Vision-language transcription through Ollama. Runs locally and "
                "reports exact token and timing metrics."
            ),
            enabled_check=lambda s: s.qwen.enabled,
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
            tags=("classical", "local", "optional"),
        ),
    ]
    for spec in specs:
        register_provider(spec, replace=True)


_register_builtin_providers()

__all__ = ["PIPELINE_ORDER", "default_registry"]
