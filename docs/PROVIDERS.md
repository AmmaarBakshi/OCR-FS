# Writing an OCR provider

A provider is one class and one registry entry. Once registered it appears in
the pipeline visualisation, the results tabs, the comparison, the metrics and
every export format — with no changes to the pipeline or the UI.

---

## The contract

```python
class OCRProvider(ABC):
    provider_id: str                             # stable, used in exports
    provider_name: str                           # shown to the user
    processing_location: ProcessingLocation      # LOCAL or CLOUD

    def process(self, document: Document) -> OCRResult: ...
    def health_check(self) -> HealthStatus: ...
    def get_metadata(self) -> dict[str, Any]: ...
```

Three rules govern the implementation:

1. **Do not raise for expected failures.** An unreachable server, a missing
   model, a timeout — return a failed `OCRResult`. The pipeline keeps going, and
   the user still sees whatever other engines produced.
2. **Leave unknown metrics as `None`.** If your engine cannot count tokens, do
   not set them. `None` renders as `N/A`; `0` is a fabricated measurement.
3. **Take configuration from settings.** No hardcoded model names, hosts,
   timeouts or paths — they belong in the settings schema so they reach the
   Settings page.

---

## A complete example: PaddleOCR

### 1. Settings

`ocr_fusion/config/schema.py`:

```python
class PaddleOCRSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    language: str = "en"
    use_angle_classifier: bool = True
    timeout_seconds: float = Field(default=120.0, gt=0)


class AppSettings(BaseModel):
    ...
    paddle: PaddleOCRSettings = Field(default_factory=PaddleOCRSettings)
```

Optionally add an environment override in `config/store.py`:

```python
ENV_OVERRIDES = {
    ...
    "OCRFS_PADDLE_LANGUAGE": "paddle.language",
}
```

### 2. The provider

`ocr_fusion/ocr/providers/paddle.py`:

```python
"""PaddleOCR provider."""

from __future__ import annotations

from typing import Any

from ocr_fusion.config.schema import AppSettings, ProcessingLocation
from ocr_fusion.documents.models import Document
from ocr_fusion.ocr.interface import (
    HealthStatus, OCRProvider, OCRResult, OCRStatus, PageResult,
)
from ocr_fusion.ocr.postprocess import normalise_whitespace


class PaddleOCRProvider(OCRProvider):
    provider_id = "paddle"
    provider_name = "PaddleOCR"
    processing_location = ProcessingLocation.LOCAL

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.config = settings.paddle
        self._engine = None          # loaded lazily; see health_check

    def health_check(self) -> HealthStatus:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return HealthStatus.unavailable(
                "PaddleOCR is not installed.",
                "Install it with: pip install paddleocr paddlepaddle",
            )
        return HealthStatus.ok(
            f"PaddleOCR ready ({self.config.language}).",
            language=self.config.language,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "engine": "PaddleOCR",
            "runtime": "paddlepaddle",
            "backend": "native",
            "language": self.config.language,
            "processing_location": self.processing_location.value,
            "reports_tokens": False,
        }

    def process(self, document: Document) -> OCRResult:
        started = self._timer()
        result = OCRResult(
            provider_id=self.provider_id,
            provider_name=f"{self.provider_name} ({self.config.language})",
            model_name=f"paddleocr:{self.config.language}",
            backend="native",
            processing_location=self.processing_location,
            metadata=self.get_metadata(),
        )

        # One check up front, not once per page.
        health = self.health_check()
        if not health.available:
            return OCRResult.failure(
                self.provider_id,
                self.provider_name,
                health.message,
                remedy=health.remedy,
                duration_seconds=self._elapsed(started),
                metadata=self.get_metadata(),
            )

        for page in document.pages:
            result.pages.append(self._process_page(page))

        result.duration_seconds = self._elapsed(started)
        succeeded = sum(1 for p in result.pages if p.status is OCRStatus.SUCCESS)
        if succeeded == 0:
            result.status = OCRStatus.FAILED
            result.error = "No text could be extracted."
        elif succeeded < len(result.pages):
            result.status = OCRStatus.PARTIAL
        return result

    def _process_page(self, page) -> PageResult:
        import numpy as np
        from PIL import Image
        import io

        started = self._timer()
        try:
            engine = self._ensure_engine()
            image = np.array(Image.open(io.BytesIO(page.image_bytes)).convert("RGB"))
            lines = engine.ocr(image, cls=self.config.use_angle_classifier)
        except Exception as exc:                      # noqa: BLE001
            return PageResult(
                page_number=page.number,
                status=OCRStatus.FAILED,
                error=f"PaddleOCR failed on this page: {exc}",
                duration_seconds=self._elapsed(started),
            )

        text_parts, confidences = [], []
        for block in (lines or []):
            for _box, (text, confidence) in (block or []):
                text_parts.append(text)
                confidences.append(confidence)

        text = normalise_whitespace("\n".join(text_parts))
        return PageResult(
            page_number=page.number,
            text=text,
            status=OCRStatus.SUCCESS if text else OCRStatus.FAILED,
            error=None if text else "No text found on this page.",
            duration_seconds=self._elapsed(started),
            # PaddleOCR *does* report confidence, so set it. It has no notion of
            # tokens, so TokenUsage is deliberately left unset -> N/A.
            confidence=sum(confidences) / len(confidences) if confidences else None,
        )

    def _ensure_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang=self.config.language,
                use_angle_cls=self.config.use_angle_classifier,
                show_log=False,
            )
        return self._engine


def build_paddle_provider(settings: AppSettings) -> PaddleOCRProvider:
    return PaddleOCRProvider(settings)
```

### 3. Register it

In `ocr_fusion/ocr/providers/__init__.py`, add to `_register_builtin_providers`:

```python
from ocr_fusion.ocr.providers.paddle import build_paddle_provider

ProviderSpec(
    provider_id="paddle",
    display_name="PaddleOCR",
    factory=build_paddle_provider,
    description="Classical OCR with strong multilingual coverage.",
    enabled_check=lambda s: s.paddle.enabled,
    tags=("classical", "local"),
),
```

and add it to the stage order:

```python
PIPELINE_ORDER = ("qwen_vl", "unlimited_ocr", "paddle", "tesseract")
```

### 4. Settings UI (optional)

Add a block in `app/components/settings_page.py` if the engine needs options
beyond enable/disable. Write labels for someone who does not know the jargon —
"Languages", not "lang codes".

### 5. Tests

Nothing about your provider needs a real model to be tested:

```python
def test_reports_missing_dependency(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    status = PaddleOCRProvider(settings).health_check()
    assert status.available is False
    assert "pip install" in status.remedy


def test_missing_dependency_fails_cleanly(settings, single_page_document):
    result = PaddleOCRProvider(settings).process(single_page_document)
    assert result.status is OCRStatus.FAILED     # returned, not raised
```

The contract tests in `tests/unit/test_providers.py::TestContract` walk the
registry automatically, so your provider is checked for interface compliance and
a non-raising `health_check` the moment you register it.

---

## Providers with several deployments

If your engine can run as a server *and* locally *and* as a CLI tool, split
policy from mechanism the way `UnlimitedOCRProvider` does: the provider owns the
page loop, retries and status roll-up; a `Backend` subclass owns turning one
page into text. See `ocr_fusion/ocr/providers/unlimited/`.

```python
class MyBackend(UnlimitedBackendBase):
    backend_id = "grpc"
    display_name = "My Engine (gRPC)"

    def run_page(self, page, prompt) -> PageOutput: ...
    def health_check(self) -> HealthStatus: ...
    def describe(self) -> dict[str, Any]: ...
```

---

## Checklist

- [ ] `provider_id` is stable and unique — it appears in saved exports
- [ ] `health_check()` never raises, and returns a **remedy** when unavailable
- [ ] `process()` returns a failed `OCRResult` rather than raising
- [ ] Unknown metrics are left `None`
- [ ] Every option comes from settings
- [ ] Heavy imports are inside methods, not at module level
- [ ] Temporary files are cleaned up; documents are never persisted
- [ ] `processing_location` is honest — `CLOUD` if bytes leave the machine
- [ ] If the engine is a stand-in for another, say so in its metadata and name

---

## Common mistakes

**Importing a heavy dependency at module level.** The registry is imported at
start-up, and the Settings page must list backends it cannot run in order to
explain why. Import inside `health_check` and the inference path.

**Reporting `0` for a metric you cannot measure.** Leave it `None`.

**Raising on a missing dependency.** Return a failed result with the install
command in `remedy`. The user reads that message; they do not read your
traceback.

**Reloading a model per page.** Cache it on the instance.

**Writing the document to a permanent path.** Use `tempfile.TemporaryDirectory`.
