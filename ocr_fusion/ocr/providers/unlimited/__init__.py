"""Unlimited-OCR provider and its interchangeable execution backends."""

from ocr_fusion.ocr.providers.unlimited.base import (
    BackendError,
    PageOutput,
    UnlimitedBackendBase,
)
from ocr_fusion.ocr.providers.unlimited.cli_backend import CliBackend
from ocr_fusion.ocr.providers.unlimited.http_backend import HttpBackend
from ocr_fusion.ocr.providers.unlimited.ollama_backend import OllamaBackend
from ocr_fusion.ocr.providers.unlimited.provider import (
    BACKENDS,
    UnlimitedOCRProvider,
    build_backend,
    build_unlimited_provider,
)
from ocr_fusion.ocr.providers.unlimited.transformers_backend import TransformersBackend

__all__ = [
    "BACKENDS",
    "BackendError",
    "CliBackend",
    "HttpBackend",
    "OllamaBackend",
    "PageOutput",
    "TransformersBackend",
    "UnlimitedBackendBase",
    "UnlimitedOCRProvider",
    "build_backend",
    "build_unlimited_provider",
]
