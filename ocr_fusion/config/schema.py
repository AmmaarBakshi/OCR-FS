"""Typed settings schema for OCR Fusion Studio.

Everything the application can be tuned with lives in :class:`AppSettings`.
Nothing in ``ocr_fusion`` reads an environment variable or a hardcoded model
name directly - it receives a settings object. That keeps the framework
embeddable (a host application can construct settings however it likes) and
makes every knob reachable from the Settings UI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ocr_fusion.config import prompts as default_prompts


class ProcessingLocation(str, Enum):
    """Where a provider sends document bytes. Surfaced in the UI (spec s14)."""

    LOCAL = "local"
    """Runs on this machine or a host the operator controls."""

    CLOUD = "cloud"
    """Leaves the machine for a third-party service."""

    UNKNOWN = "unknown"


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


class OutputFormat(str, Enum):
    TXT = "txt"
    MARKDOWN = "md"
    JSON = "json"
    CSV = "csv"


class FusionStrategy(str, Enum):
    """How two engine outputs are reconciled into the final result."""

    PREFER_PRIMARY = "prefer_primary"
    """Take the first successful provider's text verbatim."""

    PREFER_LONGEST = "prefer_longest"
    """Take the output with the most extracted characters."""

    LINE_VOTE = "line_vote"
    """Merge line by line, keeping agreed lines and picking a winner otherwise."""

    LLM = "llm"
    """Ask a language model to reconcile the two transcriptions."""


class UnlimitedBackend(str, Enum):
    """Execution strategies for the Unlimited-OCR engine.

    The upstream project (github.com/baidu/Unlimited-OCR) is a ~6.7B MoE VLM
    that expects CUDA. Rather than binding the app to one deployment shape, the
    provider supports several interchangeable backends selected in Settings.
    """

    HTTP = "http"
    """An OpenAI-compatible server (vLLM / SGLang / unlimited-ocr-server)."""

    TRANSFORMERS = "transformers"
    """In-process ``transformers`` inference. Requires a CUDA GPU."""

    CLI = "cli"
    """An external executable or script (infer.py, franken_ocr, container)."""

    OLLAMA = "ollama"
    """A locally pulled OCR-capable VLM served by Ollama.

    Used when no Unlimited-OCR runtime is available. This is a *substitute*
    engine, not Unlimited-OCR itself; the provider reports the real model name
    so the UI can label it honestly.
    """


class GeneralSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: Theme = Theme.SYSTEM
    default_output_format: OutputFormat = OutputFormat.TXT
    auto_save_results: bool = False
    """Persist run results to disk. Off by default (spec s14: no silent storage)."""

    language_hint: str = Field(
        default="auto",
        description="Document language hint passed to engines that accept one.",
    )
    developer_mode: bool = False
    """False = Demo Mode (clean, client facing). True = Developer Mode."""


class OllamaSettings(BaseModel):
    """Connection settings shared by every Ollama-backed provider."""

    model_config = ConfigDict(extra="forbid")

    host: str = "http://localhost:11434"
    connect_timeout_seconds: float = Field(default=10.0, gt=0)

    @field_validator("host")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class QwenSettings(BaseModel):
    """Stage 1 - Qwen2.5-VL through Ollama."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = "qwen2.5vl:3b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=600.0, gt=0)
    """Generous by default: a cold 3B VLM load on CPU can exceed five
    minutes before the first token, and a timeout there wastes the load."""

    retry_count: int = Field(default=1, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=2.0, ge=0)
    keep_alive: str = "5m"
    """How long Ollama keeps the model resident. Avoids reloading between pages."""


class UnlimitedOCRSettings(BaseModel):
    """Stage 2 - Unlimited-OCR, behind a pluggable backend."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: UnlimitedBackend = UnlimitedBackend.OLLAMA

    # --- http backend -----------------------------------------------------
    endpoint: str = "http://localhost:8000/v1"
    api_key: str = ""
    http_model: str = "baidu/Unlimited-OCR"

    # --- transformers backend --------------------------------------------
    model_id: str = "baidu/Unlimited-OCR"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True

    # --- cli backend ------------------------------------------------------
    executable: str = ""
    """Path to an inference script/binary, e.g. an infer.py wrapper or franken_ocr."""

    cli_args: str = "--image {image_path} --output {output_path}"

    # --- ollama backend (local substitute) --------------------------------
    ollama_model: str = "deepseek-ocr:3b"

    # --- shared -----------------------------------------------------------
    max_tokens: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=600.0, gt=0)
    retry_count: int = Field(default=1, ge=0, le=5)

    @field_validator("endpoint")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


class TesseractSettings(BaseModel):
    """An optional third engine, included to prove the provider abstraction."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    executable: str = "tesseract"
    languages: str = "eng"
    psm: int = Field(default=3, ge=0, le=13)
    timeout_seconds: float = Field(default=120.0, gt=0)


class PromptSettings(BaseModel):
    """User-editable prompts. Defaults come from :mod:`ocr_fusion.config.prompts`."""

    model_config = ConfigDict(extra="forbid")

    qwen_system: str = default_prompts.QWEN_SYSTEM_PROMPT
    qwen_user: str = default_prompts.QWEN_USER_PROMPT
    unlimited_ocr_task: str = default_prompts.UNLIMITED_OCR_PROMPT
    fusion_system: str = default_prompts.FUSION_SYSTEM_PROMPT
    fusion_user: str = default_prompts.FUSION_USER_PROMPT

    def reset_field(self, name: str) -> None:
        """Restore a single prompt to its packaged default."""
        if name not in default_prompts.DEFAULT_PROMPTS:
            raise KeyError(f"Unknown prompt: {name!r}")
        setattr(self, name, default_prompts.DEFAULT_PROMPTS[name])


class PipelineSettings(BaseModel):
    """Which pipeline stages run (spec s7 - Pipeline category)."""

    model_config = ConfigDict(extra="forbid")

    run_qwen: bool = True
    run_unlimited_ocr: bool = True
    run_comparison: bool = True
    run_fusion: bool = True
    collect_confidence: bool = True
    collect_logs: bool = True

    fusion_strategy: FusionStrategy = FusionStrategy.LINE_VOTE
    fusion_model: str = "qwen2.5:1.5b"
    """Model used when ``fusion_strategy`` is ``llm``."""

    fusion_timeout_seconds: float = Field(default=180.0, gt=0)
    continue_on_provider_error: bool = True
    """A failing engine must not abort the run (spec s13)."""

    max_pages: int = Field(default=0, ge=0)
    """0 = no limit. Guards against a 500-page PDF in a live demo."""


class OutputSettings(BaseModel):
    """What appears in the final result (spec s7 - Output category)."""

    model_config = ConfigDict(extra="forbid")

    show_extracted_text: bool = True
    show_page_numbers: bool = True
    show_confidence: bool = True
    show_token_usage: bool = True
    show_processing_time: bool = True
    show_model_name: bool = True
    show_engine_name: bool = True
    show_character_count: bool = True
    show_word_count: bool = True
    show_raw_results: bool = True
    show_comparison: bool = True
    show_processing_logs: bool = True


class DocumentSettings(BaseModel):
    """How input documents are decoded before they reach a provider."""

    model_config = ConfigDict(extra="forbid")

    pdf_render_dpi: int = Field(default=150, ge=72, le=600)
    """150 DPI keeps a rendered A4 page near 1240x1755, which transcribes
    accurately while roughly halving image tokens against 200 DPI - the
    difference between a tolerable and a painful CPU demo."""

    max_image_dimension: int = Field(default=2048, ge=256)
    """Longest edge, in pixels. Larger pages are downscaled before inference."""

    detect_text_layer: bool = True
    """Report whether a PDF already contains selectable text."""

    max_file_size_mb: float = Field(default=50.0, gt=0)


class PrivacySettings(BaseModel):
    """Confidentiality controls (spec s14)."""

    model_config = ConfigDict(extra="forbid")

    persist_documents: bool = False
    """Never write uploaded documents to disk unless explicitly enabled."""

    persist_results: bool = False
    redact_text_in_logs: bool = True
    """Log lengths and counts, not document contents."""

    log_preview_chars: int = Field(default=0, ge=0, le=200)
    """Characters of OCR text allowed into logs when redaction is off."""


class AppSettings(BaseModel):
    """Root settings object handed to every component."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    schema_version: Literal[1] = 1
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    qwen: QwenSettings = Field(default_factory=QwenSettings)
    unlimited_ocr: UnlimitedOCRSettings = Field(default_factory=UnlimitedOCRSettings)
    tesseract: TesseractSettings = Field(default_factory=TesseractSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    documents: DocumentSettings = Field(default_factory=DocumentSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def redacted(self) -> dict[str, Any]:
        """Settings safe to embed in an export or a log: secrets removed."""
        data = self.to_dict()
        if data.get("unlimited_ocr", {}).get("api_key"):
            data["unlimited_ocr"]["api_key"] = "***"
        return data


__all__ = [
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
]
