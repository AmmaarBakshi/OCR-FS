"""CLI backend - Unlimited-OCR through an external command.

Covers the deployment shapes that are neither a server nor an in-process model:
the upstream batch script, a container wrapper, or a third-party engine such as
franken_ocr (a CPU-only Rust implementation for Unlimited-OCR weights).

The command template is configured in Settings, for example::

    Executable: python C:/tools/Unlimited-OCR/infer.py
    Arguments:  --image {image_path} --output {output_path}

``{image_path}`` and ``{output_path}`` are substituted per page. The command is
run without a shell and its arguments are tokenised with :mod:`shlex`, so a path
containing spaces or shell metacharacters cannot alter the command.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
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


class CliBackend(UnlimitedBackendBase):
    """Runs a configured executable once per page."""

    backend_id = "cli"
    display_name = "Unlimited-OCR (external command)"
    processing_location = ProcessingLocation.LOCAL

    @property
    def model_label(self) -> str:
        return self.config.executable or "external command"

    def _command_parts(self) -> list[str]:
        """Tokenise the configured executable, honouring quoted paths."""
        return shlex.split(self.config.executable, posix=False)

    def health_check(self) -> HealthStatus:
        if not self.config.executable.strip():
            return HealthStatus.unavailable(
                "No Unlimited-OCR command is configured.",
                "Set the executable in Settings > Unlimited OCR, for example "
                "'python /path/to/Unlimited-OCR/infer.py'.",
            )

        parts = self._command_parts()
        if not parts:
            return HealthStatus.unavailable(
                "The configured command could not be parsed.",
                "Check Settings > Unlimited OCR > Executable.",
            )

        binary = parts[0].strip('"')
        resolved = shutil.which(binary) or (binary if Path(binary).exists() else None)
        if resolved is None:
            return HealthStatus.unavailable(
                f"The command '{binary}' was not found on this system.",
                "Give the full path to the executable in Settings > Unlimited OCR.",
                executable=self.config.executable,
            )

        # A script path passed to an interpreter must exist too, otherwise the
        # failure only appears on the first page.
        for argument in parts[1:]:
            candidate = argument.strip('"')
            if candidate.lower().endswith(".py") and not Path(candidate).exists():
                return HealthStatus.unavailable(
                    f"The script '{candidate}' does not exist.",
                    "Correct the path in Settings > Unlimited OCR.",
                    executable=self.config.executable,
                )

        return HealthStatus.ok(
            f"Command ready: {self.config.executable}",
            executable=self.config.executable,
            resolved=resolved,
            arguments=self.config.cli_args,
        )

    def run_page(self, page: DocumentPage, prompt: str) -> PageOutput:
        with tempfile.TemporaryDirectory(prefix="ocrfs-cli-") as tmpdir:
            workdir = Path(tmpdir)
            image_path = workdir / f"page-{page.number}.png"
            output_path = workdir / f"page-{page.number}.txt"
            image_path.write_bytes(page.image_bytes)

            try:
                arguments = shlex.split(
                    self.config.cli_args.format(
                        image_path=str(image_path),
                        output_path=str(output_path),
                        prompt=prompt,
                        page_number=page.number,
                    ),
                    posix=False,
                )
            except (KeyError, ValueError) as exc:
                raise BackendError(
                    f"The argument template could not be built: {exc}",
                    "Use only {image_path}, {output_path}, {prompt} and "
                    "{page_number} in Settings > Unlimited OCR > Arguments.",
                    retryable=False,
                ) from exc

            command = self._command_parts() + arguments
            try:
                completed = subprocess.run(  # noqa: S603 - argv list, no shell
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                    cwd=workdir,
                )
            except FileNotFoundError as exc:
                raise BackendError(
                    f"The command '{command[0]}' could not be started.",
                    "Check the executable path in Settings > Unlimited OCR.",
                    retryable=False,
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise BackendError(
                    f"The command did not finish within "
                    f"{self.config.timeout_seconds:.0f} seconds.",
                    "Raise the timeout in Settings > Unlimited OCR.",
                ) from exc

            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise BackendError(
                    f"The Unlimited-OCR command exited with code {completed.returncode}.",
                    detail[-300:] if detail else "No error output was produced.",
                )

            # Prefer the output file; fall back to stdout for tools that stream.
            if output_path.exists():
                text = output_path.read_text(encoding="utf-8", errors="replace")
            else:
                text = completed.stdout or ""

            return PageOutput(
                text=text,
                raw={
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "used_output_file": output_path.exists(),
                },
            )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "engine": "Unlimited-OCR",
            "runtime": "external command",
            "executable": self.config.executable,
            "arguments": self.config.cli_args,
            "timeout_seconds": self.config.timeout_seconds,
            "processing_location": self.processing_location.value,
        }


__all__ = ["CliBackend"]
