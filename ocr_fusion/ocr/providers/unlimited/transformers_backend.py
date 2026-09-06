"""Transformers backend - upstream Unlimited-OCR in-process.

Mirrors the usage documented by github.com/baidu/Unlimited-OCR::

    model.infer(tokenizer, prompt='<image>\\nFree OCR.', image_file=...,
                base_size=1024, image_size=640, crop_mode=True)

torch and transformers are imported lazily inside methods, never at module
import time: the app must start on a machine with neither installed, and the
Settings UI has to be able to list this backend in order to explain why it is
unavailable.

The model is cached on the instance because loading ~14 GB of weights per page
would be unusable.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from ocr_fusion.config.schema import ProcessingLocation
from ocr_fusion.documents.models import DocumentPage
from ocr_fusion.ocr.interface import HealthStatus
from ocr_fusion.ocr.providers.unlimited.base import (
    BackendError,
    PageOutput,
    UnlimitedBackendBase,
)

logger = logging.getLogger(__name__)


class TransformersBackend(UnlimitedBackendBase):
    """Loads Unlimited-OCR locally with ``transformers``. Requires a GPU."""

    backend_id = "transformers"
    display_name = "Unlimited-OCR (local transformers)"
    processing_location = ProcessingLocation.LOCAL

    def __init__(self, config) -> None:  # noqa: ANN001 - typed by the base class
        super().__init__(config)
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model_label(self) -> str:
        return self.config.model_id

    # -- readiness ---------------------------------------------------------

    def health_check(self) -> HealthStatus:
        try:
            import torch
        except ImportError:
            return HealthStatus.unavailable(
                "PyTorch is not installed, so Unlimited-OCR cannot run in this process.",
                "Install the extra: pip install -e .[transformers] - and note "
                "this backend also needs a CUDA GPU.",
            )
        try:
            import transformers  # noqa: F401
        except ImportError:
            return HealthStatus.unavailable(
                "The transformers library is not installed.",
                "Install the extra: pip install -e .[transformers]",
            )

        device = self.config.device.lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            return HealthStatus.unavailable(
                "No CUDA GPU is available, and Unlimited-OCR requires one to run "
                "in this process.",
                "Use the HTTP backend against a GPU host instead, or switch "
                "Settings > Unlimited OCR > Backend to a runnable option.",
                torch_version=torch.__version__,
                cuda_available=False,
                requested_device=self.config.device,
            )

        return HealthStatus.ok(
            f"Ready to load {self.config.model_id} on {self.config.device}.",
            model_id=self.config.model_id,
            device=self.config.device,
            torch_version=torch.__version__,
            weights_loaded=self._model is not None,
        )

    # -- inference ---------------------------------------------------------

    def _ensure_model(self) -> tuple[Any, Any]:
        """Load and cache the model. First call downloads several GB."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise BackendError(
                "PyTorch and transformers are required for this backend.",
                "Install with: pip install -e .[transformers]",
                retryable=False,
            ) from exc

        dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            model = model.eval().to(self.config.device)
        except Exception as exc:  # noqa: BLE001 - surface any load failure verbatim
            raise BackendError(
                f"Could not load {self.config.model_id}: {exc}",
                "Check the model id, available disk space and GPU memory.",
                retryable=False,
            ) from exc

        self._model, self._tokenizer = model, tokenizer
        return model, tokenizer

    def run_page(self, page: DocumentPage, prompt: str) -> PageOutput:
        model, tokenizer = self._ensure_model()

        # model.infer() takes a file path, so the page is written to a temp file
        # that is removed immediately afterwards - it is never persisted.
        with tempfile.TemporaryDirectory(prefix="ocrfs-ulim-") as tmpdir:
            image_path = Path(tmpdir) / f"page-{page.number}.png"
            image_path.write_bytes(page.image_bytes)

            try:
                output = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    base_size=self.config.base_size,
                    image_size=self.config.image_size,
                    crop_mode=self.config.crop_mode,
                    max_length=self.config.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if "out of memory" in message.lower():
                    raise BackendError(
                        "The GPU ran out of memory while transcribing this page.",
                        "Lower Base size / Image size in Settings > Unlimited OCR, "
                        "or reduce the PDF render DPI.",
                        retryable=False,
                    ) from exc
                raise BackendError(f"Unlimited-OCR inference failed: {message}") from exc

        return PageOutput(
            text=_as_text(output),
            raw={"model_id": self.config.model_id, "device": self.config.device},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "engine": "Unlimited-OCR",
            "runtime": "transformers (in-process)",
            "model_id": self.config.model_id,
            "device": self.config.device,
            "torch_dtype": self.config.torch_dtype,
            "base_size": self.config.base_size,
            "image_size": self.config.image_size,
            "crop_mode": self.config.crop_mode,
            "weights_loaded": self._model is not None,
            "processing_location": self.processing_location.value,
        }


def _as_text(output: Any) -> str:
    """Normalise whatever ``infer`` returned into a string.

    The upstream API returns a string, but forks and versions differ, so common
    container shapes are unwrapped rather than stringified into noise.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "result", "output", "content"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    if isinstance(output, (list, tuple)) and output:
        return "\n".join(str(item) for item in output)
    return "" if output is None else str(output)


__all__ = ["TransformersBackend"]
