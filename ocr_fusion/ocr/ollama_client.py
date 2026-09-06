"""A small, typed Ollama HTTP client.

Shared by every Ollama-backed component (the Qwen provider, the Unlimited-OCR
ollama backend and LLM fusion) so connection handling, error translation and
metric extraction are written once.

The client deliberately does not depend on the ``ollama`` Python package: the
REST API is stable and small, and one less dependency means one less thing to
install before a demo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Nanoseconds per second - Ollama reports all durations in ns.
_NS = 1_000_000_000


class OllamaError(Exception):
    """Base class for Ollama failures, carrying a client-safe explanation."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy


class OllamaUnavailableError(OllamaError):
    """The Ollama server could not be reached."""


class OllamaModelMissingError(OllamaError):
    """The server is up but the requested model has not been pulled."""


class OllamaTimeoutError(OllamaError):
    """Generation exceeded the configured timeout."""


@dataclass(slots=True)
class GenerateResponse:
    """One completion, with whatever metrics the server reported.

    Ollama omits timing fields in some code paths, so every metric is optional
    and stays ``None`` when absent rather than defaulting to zero.
    """

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_seconds: float | None = None
    load_duration_seconds: float | None = None
    eval_duration_seconds: float | None = None
    done_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True when generation stopped at the token limit rather than finishing."""
        return self.done_reason == "length"

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or not self.eval_duration_seconds:
            return None
        return self.completion_tokens / self.eval_duration_seconds


def _ns_to_seconds(value: Any) -> float | None:
    """Convert an Ollama nanosecond duration, tolerating missing/odd values."""
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return value / _NS


class OllamaClient:
    """Blocking Ollama client.

    The pipeline runs synchronously inside Streamlit's script thread, so a
    blocking client keeps call sites simple; ``httpx`` handles the transport.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        *,
        connect_timeout: float = 10.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.connect_timeout = connect_timeout

    # -- server introspection ---------------------------------------------

    def is_reachable(self) -> bool:
        try:
            with httpx.Client(timeout=self.connect_timeout) as client:
                return client.get(f"{self.host}/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        """Names of every pulled model.

        Raises :class:`OllamaUnavailableError` when the server is unreachable,
        which is the single failure the caller must distinguish from "the model
        is missing".
        """
        try:
            with httpx.Client(timeout=self.connect_timeout) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host}.",
                remedy=(
                    "Start Ollama (run 'ollama serve', or launch the Ollama app) "
                    "and confirm the host in Settings > OCR Models."
                ),
            ) from exc
        except ValueError as exc:
            raise OllamaUnavailableError(
                f"Ollama at {self.host} returned an unreadable response."
            ) from exc

        models = payload.get("models") or []
        return [m.get("name", "") for m in models if m.get("name")]

    def has_model(self, model: str) -> bool:
        """Whether ``model`` is pulled, tolerating an implicit ``:latest`` tag."""
        available = self.list_models()
        if model in available:
            return True
        wanted = model if ":" in model else f"{model}:latest"
        return wanted in available

    def show_model(self, model: str) -> dict[str, Any]:
        """Model card details (family, parameter size, quantisation)."""
        try:
            with httpx.Client(timeout=self.connect_timeout) as client:
                response = client.post(f"{self.host}/api/show", json={"model": model})
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError):
            return {}

    # -- generation --------------------------------------------------------

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        images: list[str] | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300.0,
        keep_alive: str | None = None,
    ) -> GenerateResponse:
        """Run a single non-streaming completion.

        ``images`` are base64-encoded PNGs, the form Ollama's vision models
        expect. Streaming is not used: the pipeline reports progress per stage
        and per page, so partial tokens would add complexity without changing
        what the user sees.
        """
        options: dict[str, Any] = {"temperature": temperature}
        if top_p is not None:
            options["top_p"] = top_p
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images
        if keep_alive:
            payload["keep_alive"] = keep_alive

        try:
            # Connect quickly but allow a long read: a 3B VLM on CPU can take
            # minutes for a dense page.
            timeouts = httpx.Timeout(timeout, connect=self.connect_timeout)
            with httpx.Client(timeout=timeouts) as client:
                response = client.post(f"{self.host}/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"{model} did not respond within {timeout:.0f} seconds.",
                remedy=(
                    "Raise the timeout in Settings > OCR Models, lower the PDF "
                    "render DPI, or use a smaller model."
                ),
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host}.",
                remedy="Start Ollama and check the host in Settings > OCR Models.",
            ) from exc

        if response.status_code == 404:
            raise OllamaModelMissingError(
                f"The model '{model}' has not been downloaded.",
                remedy=f"Run: ollama pull {model}",
            )
        if response.status_code >= 400:
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code} for model '{model}'.",
                remedy=_detail_from_error(response),
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned an unreadable response for '{model}'.") from exc

        return GenerateResponse(
            text=data.get("response", "") or "",
            model=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_duration_seconds=_ns_to_seconds(data.get("total_duration")),
            load_duration_seconds=_ns_to_seconds(data.get("load_duration")),
            eval_duration_seconds=_ns_to_seconds(data.get("eval_duration")),
            done_reason=data.get("done_reason"),
            raw=data,
        )


def _detail_from_error(response: httpx.Response) -> str:
    """Pull Ollama's own error string out of a failed response, if present."""
    try:
        return str(response.json().get("error", ""))[:300]
    except ValueError:
        return ""


__all__ = [
    "GenerateResponse",
    "OllamaClient",
    "OllamaError",
    "OllamaModelMissingError",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
]
