"""OCR Fusion Studio - a modular multi-engine OCR framework.

The package is deliberately UI-agnostic: everything needed to run a full OCR
pipeline is importable without Streamlit, so the same core can be reused by a
CLI, a web service, a batch job or another application.

Typical headless usage::

    from ocr_fusion import build_default_pipeline, load_document, load_settings

    settings = load_settings()
    document = load_document("invoice.pdf")
    result = build_default_pipeline(settings).execute(document)
    print(result.final_result.text)
"""

from ocr_fusion.version import __version__

__all__ = ["__version__"]
