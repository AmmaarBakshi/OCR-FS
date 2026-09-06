"""The OCR provider contract.

Every engine - a local VLM, a hosted API, a CLI binary - implements
:class:`OCRProvider`. The pipeline and the UI depend on this interface only,
which is what makes engines swappable without touching either (spec s11).

Two rules run through the result types:

* **Unknown metrics are ``None``, never zero.** A provider that cannot report
  token counts leaves them unset so the UI can render "N/A" instead of
  fabricating a number (spec s8).
* **Failure is data, not an exception.** A provider returns a failed
  :class:`OCRResult` rather than raising, so one dead engine cannot abort a run
  (spec s13).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ocr_fusion.config.schema import ProcessingLocation
from ocr_fusion.documents.models import Document


class OCRStatus(str, Enum):
    """Outcome of a provider run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    """Some pages succeeded and others failed."""

    FAILED = "failed"
    SKIPPED = "skipped"
    """Disabled in settings, or not reached because an earlier stage stopped."""


@dataclass(slots=True)
class HealthStatus:
    """Result of a provider readiness probe.

    ``remedy`` is the actionable instruction shown to the user - "run
    ``ollama pull qwen2.5vl:3b``" is far more useful than "model not found".
    """

    available: bool
    message: str
    remedy: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    checked_in_seconds: float | None = None

    @classmethod
    def ok(cls, message: str = "Ready", **details: Any) -> HealthStatus:
        return cls(available=True, message=message, details=details)

    @classmethod
    def unavailable(cls, message: str, remedy: str = "", **details: Any) -> HealthStatus:
        return cls(available=False, message=message, remedy=remedy, details=details)


@dataclass(slots=True)
class TokenUsage:
    """Token accounting. ``None`` means the engine does not report it."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    def as_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum usages, preserving ``None`` when neither side reported a value."""

        def _add(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return TokenUsage(
            input_tokens=_add(self.input_tokens, other.input_tokens),
            output_tokens=_add(self.output_tokens, other.output_tokens),
        )


@dataclass(slots=True)
class PageResult:
    """OCR output for a single page."""

    page_number: int
    text: str = ""
    status: OCRStatus = OCRStatus.SUCCESS
    error: str | None = None
    duration_seconds: float | None = None
    tokens: TokenUsage = field(default_factory=TokenUsage)
    confidence: float | None = None
    """0.0-1.0 when the engine reports one. VLMs generally do not."""

    load_duration_seconds: float | None = None
    """Model load time, when the runtime distinguishes it from inference."""

    inference_duration_seconds: float | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    """Unmodified engine response, shown under Developer Mode."""

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class OCRResult:
    """Everything one provider produced for one document."""

    provider_id: str
    provider_name: str
    status: OCRStatus = OCRStatus.SUCCESS
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None
    """Client-safe explanation when the provider failed as a whole."""

    remedy: str = ""
    duration_seconds: float = 0.0
    model_name: str | None = None
    backend: str | None = None
    """Which execution strategy actually ran (spec s16: report what really ran)."""

    processing_location: ProcessingLocation = ProcessingLocation.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status in (OCRStatus.SUCCESS, OCRStatus.PARTIAL)

    @property
    def text(self) -> str:
        """All successful page text, joined in page order."""
        parts = [p.text for p in sorted(self.pages, key=lambda p: p.page_number) if p.text]
        return "\n\n".join(parts)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def tokens(self) -> TokenUsage:
        total = TokenUsage()
        for page in self.pages:
            total = total + page.tokens
        return total

    @property
    def tokens_per_second(self) -> float | None:
        """Output throughput, or ``None`` when tokens or timing are unknown."""
        output = self.tokens.output_tokens
        if not output or self.duration_seconds <= 0:
            return None
        return output / self.duration_seconds

    def page(self, number: int) -> PageResult | None:
        return next((p for p in self.pages if p.page_number == number), None)

    @classmethod
    def failure(
        cls,
        provider_id: str,
        provider_name: str,
        error: str,
        *,
        remedy: str = "",
        duration_seconds: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> OCRResult:
        """Build a failed result. Providers return this instead of raising.

        ``metadata`` is an explicit dict rather than ``**kwargs``: provider
        metadata legitimately contains a ``provider_id`` key, and collecting it
        as keyword arguments made that collide with the positional parameter,
        raising TypeError on exactly the health-check failure path this method
        exists to report.
        """
        return cls(
            provider_id=provider_id,
            provider_name=provider_name,
            status=OCRStatus.FAILED,
            error=error,
            remedy=remedy,
            duration_seconds=duration_seconds,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def skipped(cls, provider_id: str, provider_name: str, reason: str) -> OCRResult:
        return cls(
            provider_id=provider_id,
            provider_name=provider_name,
            status=OCRStatus.SKIPPED,
            error=reason,
        )

    def summary(self) -> dict[str, Any]:
        """Metrics view for the UI and the JSON export. ``None`` renders as N/A."""
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "status": self.status.value,
            "model_name": self.model_name,
            "backend": self.backend,
            "processing_location": self.processing_location.value,
            "pages": len(self.pages),
            "duration_seconds": self.duration_seconds,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "tokens": self.tokens.as_dict(),
            "tokens_per_second": self.tokens_per_second,
            "error": self.error,
        }


class OCRProvider(ABC):
    """Base class for OCR engines.

    Subclasses implement :meth:`process`, :meth:`health_check` and
    :meth:`get_metadata`. Nothing else in the codebase may import a concrete
    provider by name; the pipeline resolves them through the registry.
    """

    #: Stable machine identifier, used in exports and settings.
    provider_id: str = "base"

    #: Human-readable name shown in the UI.
    provider_name: str = "OCR Provider"

    #: Whether document bytes stay on this machine (spec s14).
    processing_location: ProcessingLocation = ProcessingLocation.UNKNOWN

    @abstractmethod
    def process(self, document: Document) -> OCRResult:
        """Transcribe every page of ``document``.

        Implementations must not raise for expected failures (unreachable
        runtime, timeout, missing model); they return a failed
        :class:`OCRResult` so the pipeline can carry on with other engines.
        """

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Report whether the engine can run right now, and how to fix it if not."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Describe the engine: model, backend, runtime and effective settings."""

    # -- helpers available to every provider ------------------------------

    def _timer(self) -> float:
        """Monotonic start time. Wall clock is unsuitable for measuring latency."""
        return time.perf_counter()

    def _elapsed(self, start: float) -> float:
        return time.perf_counter() - start

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.provider_id!r}>"


__all__ = [
    "HealthStatus",
    "OCRProvider",
    "OCRResult",
    "OCRStatus",
    "PageResult",
    "TokenUsage",
]
