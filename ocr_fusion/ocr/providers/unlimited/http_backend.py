"""HTTP backend - Unlimited-OCR behind an OpenAI-compatible server.

This is the recommended production shape: run the model on a GPU host with
vLLM, SGLang or unlimited-ocr-server, and point the app at it. The app itself
then needs no CUDA, no torch and no model weights.

    python -m sglang.launch_server --model baidu/Unlimited-OCR --context-length 32768
    # then set Settings > Unlimited OCR > Endpoint to http://<host>:30000/v1
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ocr_fusion.config.schema import ProcessingLocation
from ocr_fusion.documents.models import DocumentPage
from ocr_fusion.ocr.interface import HealthStatus
from ocr_fusion.ocr.providers.unlimited.base import (
    BackendError,
    PageOutput,
    UnlimitedBackendBase,
)

logger = logging.getLogger(__name__)


class HttpBackend(UnlimitedBackendBase):
    """Calls a chat-completions endpoint that accepts an image part."""

    backend_id = "http"
    display_name = "Unlimited-OCR (HTTP server)"

    @property
    def processing_location(self) -> ProcessingLocation:  # type: ignore[override]
        """Local only when the endpoint is on this machine.

        Sending pages to a remote host is a privacy-relevant fact, so it is
        derived from the endpoint rather than assumed (spec s14).
        """
        endpoint = self.config.endpoint.lower()
        if any(host in endpoint for host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")):
            return ProcessingLocation.LOCAL
        return ProcessingLocation.CLOUD

    @property
    def model_label(self) -> str:
        return self.config.http_model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def health_check(self) -> HealthStatus:
        url = f"{self.config.endpoint}/models"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            return HealthStatus.unavailable(
                f"No Unlimited-OCR server responded at {self.config.endpoint}.",
                "Start a vLLM, SGLang or unlimited-ocr-server instance on a GPU "
                "host and set its address in Settings > Unlimited OCR. See "
                "docs/PROVIDERS.md for the exact commands.",
                endpoint=self.config.endpoint,
                error=str(exc)[:200],
            )

        if response.status_code in (401, 403):
            return HealthStatus.unavailable(
                "The Unlimited-OCR server rejected the API key.",
                "Check the API key in Settings > Unlimited OCR.",
                endpoint=self.config.endpoint,
            )
        if response.status_code >= 400:
            return HealthStatus.unavailable(
                f"The Unlimited-OCR server returned HTTP {response.status_code}.",
                "Confirm the endpoint points at an OpenAI-compatible /v1 path.",
                endpoint=self.config.endpoint,
            )

        served: list[str] = []
        try:
            served = [m.get("id", "") for m in response.json().get("data", [])]
        except ValueError:
            pass

        # A mismatched model name is a common misconfiguration and produces a
        # confusing 404 at inference time, so flag it during the health check.
        if served and self.config.http_model not in served:
            return HealthStatus.unavailable(
                f"The server does not serve '{self.config.http_model}'.",
                f"Set the model name to one of: {', '.join(served[:5])}",
                endpoint=self.config.endpoint,
                served_models=served[:10],
            )

        return HealthStatus.ok(
            f"Unlimited-OCR server ready at {self.config.endpoint}.",
            endpoint=self.config.endpoint,
            model=self.config.http_model,
            served_models=served[:10],
        )

    def run_page(self, page: DocumentPage, prompt: str) -> PageOutput:
        payload: dict[str, Any] = {
            "model": self.config.http_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{page.image_base64()}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        try:
            timeouts = httpx.Timeout(self.config.timeout_seconds, connect=10.0)
            with httpx.Client(timeout=timeouts) as client:
                response = client.post(
                    f"{self.config.endpoint}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.TimeoutException as exc:
            raise BackendError(
                f"The Unlimited-OCR server did not respond within "
                f"{self.config.timeout_seconds:.0f} seconds.",
                "Raise the timeout in Settings > Unlimited OCR, or lower the "
                "PDF render DPI.",
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(
                f"Could not reach the Unlimited-OCR server at {self.config.endpoint}.",
                "Confirm the server is running and reachable from this machine.",
            ) from exc

        if response.status_code >= 400:
            raise BackendError(
                f"The Unlimited-OCR server returned HTTP {response.status_code}.",
                _error_detail(response),
                retryable=response.status_code >= 500,
            )

        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError) as exc:
            raise BackendError(
                "The Unlimited-OCR server returned an unexpected response shape.",
                "Confirm the endpoint is an OpenAI-compatible chat-completions API.",
                retryable=False,
            ) from exc

        usage = data.get("usage") or {}
        return PageOutput(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            truncated=choice.get("finish_reason") == "length",
            raw={
                "model": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "engine": "Unlimited-OCR",
            "runtime": "OpenAI-compatible HTTP server",
            "endpoint": self.config.endpoint,
            "model": self.config.http_model,
            "authenticated": bool(self.config.api_key),
            "timeout_seconds": self.config.timeout_seconds,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "processing_location": self.processing_location.value,
        }


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message", ""))[:300]
    return str(error or body)[:300]


__all__ = ["HttpBackend"]
