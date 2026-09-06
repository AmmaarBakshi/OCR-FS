"""Backend contract for the Unlimited-OCR provider.

Upstream Unlimited-OCR (github.com/baidu/Unlimited-OCR) is a ~6.7B MoE VLM that
expects CUDA, and it is deployed in several different shapes in practice: a
vLLM/SGLang server, in-process ``transformers``, or a batch CLI. Binding the
provider to one of those would make the app undemonstrable everywhere else.

So the provider owns *policy* (page loop, retries, status roll-up, metrics) and
a backend owns *mechanism* (how one page becomes text). Adding a new deployment
shape means writing one :class:`UnlimitedBackendBase` subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ocr_fusion.config.schema import ProcessingLocation, UnlimitedOCRSettings
from ocr_fusion.documents.models import DocumentPage
from ocr_fusion.ocr.interface import HealthStatus


class BackendError(Exception):
    """A backend failure that the provider turns into a page or run failure.

    ``remedy`` is the actionable next step shown to the user; ``retryable``
    tells the provider whether attempting the page again could plausibly help.
    """

    def __init__(self, message: str, remedy: str = "", *, retryable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.retryable = retryable


@dataclass(slots=True)
class PageOutput:
    """What a backend returns for one page.

    Metrics are optional because backends differ in what they can report: an
    Ollama backend gives exact token counts, a CLI backend gives none. Unknown
    stays ``None`` so the UI can render "N/A" (spec s8).
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    load_duration_seconds: float | None = None
    inference_duration_seconds: float | None = None
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class UnlimitedBackendBase(ABC):
    """One way of executing Unlimited-OCR."""

    #: Value of :class:`~ocr_fusion.config.schema.UnlimitedBackend` this implements.
    backend_id: str = "base"

    #: Shown in the UI so the user always knows what actually ran (spec s16).
    display_name: str = "Unlimited-OCR"

    #: Whether page bytes stay on this machine.
    processing_location: ProcessingLocation = ProcessingLocation.UNKNOWN

    #: True when this backend runs a model other than Unlimited-OCR itself.
    #: The UI must label such a run as a substitute rather than imply upstream
    #: Unlimited-OCR produced it.
    is_substitute: bool = False

    def __init__(self, config: UnlimitedOCRSettings) -> None:
        self.config = config

    @abstractmethod
    def run_page(self, page: DocumentPage, prompt: str) -> PageOutput:
        """Transcribe one page. Raise :class:`BackendError` on failure."""

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Report whether this backend can run right now, and how to fix it."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Backend identity and effective configuration, for Developer Mode."""

    @property
    def model_label(self) -> str:
        """The model name to report. Subclasses override with the real value."""
        return self.config.model_id


__all__ = ["BackendError", "PageOutput", "UnlimitedBackendBase"]
