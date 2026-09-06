"""Tesseract provider - a classical OCR engine.

Included mainly as proof that the abstraction holds: Tesseract has nothing in
common with a vision-language model - no prompts, no tokens, no sampling
parameters - yet it implements the same :class:`OCRProvider` interface and
therefore needs no pipeline or UI change to appear in a run (spec s11).

It also demonstrates the "unknown metrics are None" rule from a different
angle: Tesseract reports a per-word confidence but no token counts at all.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ocr_fusion.config.schema import AppSettings, ProcessingLocation
from ocr_fusion.documents.models import Document, DocumentPage
from ocr_fusion.ocr.interface import (
    HealthStatus,
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
)
from ocr_fusion.ocr.postprocess import normalise_whitespace

logger = logging.getLogger(__name__)


class TesseractProvider(OCRProvider):
    """Runs the Tesseract binary once per page."""

    provider_id = "tesseract"
    provider_name = "Tesseract"
    processing_location = ProcessingLocation.LOCAL

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.config = settings.tesseract

    def health_check(self) -> HealthStatus:
        executable = shutil.which(self.config.executable)
        if executable is None:
            return HealthStatus.unavailable(
                f"The Tesseract executable '{self.config.executable}' was not found.",
                "Install Tesseract (Windows: winget install UB-Mannheim.TesseractOCR) "
                "and make sure it is on PATH, or set the full path in Settings.",
            )
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, no shell
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return HealthStatus.unavailable(
                f"Tesseract could not be started: {exc}",
                "Reinstall Tesseract or correct the path in Settings.",
            )

        version = (completed.stdout or "").splitlines()
        return HealthStatus.ok(
            f"Tesseract ready ({version[0] if version else 'unknown version'}).",
            executable=executable,
            languages=self.config.languages,
            psm=self.config.psm,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "engine": "Tesseract",
            "runtime": "native binary",
            "backend": "cli",
            "executable": self.config.executable,
            "languages": self.config.languages,
            "psm": self.config.psm,
            "processing_location": self.processing_location.value,
            "reports_tokens": False,
        }

    def process(self, document: Document) -> OCRResult:
        started = self._timer()
        result = OCRResult(
            provider_id=self.provider_id,
            provider_name=f"{self.provider_name} ({self.config.languages})",
            model_name=f"tesseract:{self.config.languages}",
            backend="cli",
            processing_location=self.processing_location,
            metadata=self.get_metadata(),
        )

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
        if not result.pages or succeeded == 0:
            result.status = OCRStatus.FAILED
            result.error = next(
                (p.error for p in result.pages if p.error), "No text could be extracted."
            )
        elif succeeded < len(result.pages):
            result.status = OCRStatus.PARTIAL
        return result

    def _process_page(self, page: DocumentPage) -> PageResult:
        started = self._timer()
        executable = shutil.which(self.config.executable) or self.config.executable

        with tempfile.TemporaryDirectory(prefix="ocrfs-tess-") as tmpdir:
            image_path = Path(tmpdir) / f"page-{page.number}.png"
            image_path.write_bytes(page.image_bytes)

            command = [
                executable,
                str(image_path),
                "stdout",
                "-l",
                self.config.languages,
                "--psm",
                str(self.config.psm),
            ]
            try:
                completed = subprocess.run(  # noqa: S603 - argv list, no shell
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return PageResult(
                    page_number=page.number,
                    status=OCRStatus.FAILED,
                    error=(
                        f"Tesseract did not finish within "
                        f"{self.config.timeout_seconds:.0f} seconds."
                    ),
                    duration_seconds=self._elapsed(started),
                )
            except OSError as exc:
                return PageResult(
                    page_number=page.number,
                    status=OCRStatus.FAILED,
                    error=f"Tesseract could not be started: {exc}",
                    duration_seconds=self._elapsed(started),
                )

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()[-200:]
            return PageResult(
                page_number=page.number,
                status=OCRStatus.FAILED,
                error=f"Tesseract exited with code {completed.returncode}. {detail}",
                duration_seconds=self._elapsed(started),
            )

        text = normalise_whitespace(completed.stdout or "")
        return PageResult(
            page_number=page.number,
            text=text,
            status=OCRStatus.SUCCESS if text.strip() else OCRStatus.FAILED,
            error=None if text.strip() else "Tesseract found no text on this page.",
            duration_seconds=self._elapsed(started),
            # Deliberately no TokenUsage: Tesseract has no concept of tokens, so
            # the metrics stay None and the UI renders N/A (spec s8).
            raw_response={"psm": self.config.psm, "languages": self.config.languages},
        )


def build_tesseract_provider(settings: AppSettings) -> TesseractProvider:
    """Registry factory."""
    return TesseractProvider(settings)


__all__ = ["TesseractProvider", "build_tesseract_provider"]
