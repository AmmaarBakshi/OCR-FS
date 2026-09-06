"""Centralised configuration for OCR Fusion Studio.

Import settings from here rather than reaching into submodules::

    from ocr_fusion.config import AppSettings, load_settings, save_settings
"""

from ocr_fusion.config.prompts import DEFAULT_PROMPTS, render
from ocr_fusion.config.schema import (
    AppSettings,
    DocumentSettings,
    FusionStrategy,
    GeneralSettings,
    OllamaSettings,
    OutputFormat,
    OutputSettings,
    PipelineSettings,
    PrivacySettings,
    ProcessingLocation,
    PromptSettings,
    QwenSettings,
    TesseractSettings,
    Theme,
    UnlimitedBackend,
    UnlimitedOCRSettings,
)
from ocr_fusion.config.store import (
    default_settings_path,
    load_settings,
    reset_settings,
    save_settings,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "AppSettings",
    "DocumentSettings",
    "FusionStrategy",
    "GeneralSettings",
    "OllamaSettings",
    "OutputFormat",
    "OutputSettings",
    "PipelineSettings",
    "PrivacySettings",
    "ProcessingLocation",
    "PromptSettings",
    "QwenSettings",
    "TesseractSettings",
    "Theme",
    "UnlimitedBackend",
    "UnlimitedOCRSettings",
    "default_settings_path",
    "load_settings",
    "render",
    "reset_settings",
    "save_settings",
]
