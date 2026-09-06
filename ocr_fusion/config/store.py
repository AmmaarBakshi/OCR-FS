"""Loading, persisting and layering of :class:`AppSettings`.

Precedence, lowest to highest:

1. Packaged defaults (the field defaults in :mod:`ocr_fusion.config.schema`)
2. A JSON settings file (user edits made in the Settings UI)
3. Environment variables prefixed ``OCRFS_``

Environment variables win so that a container or CI run can point the app at a
different Ollama host without editing any file. The settings file is written
atomically so an interrupted save cannot leave a truncated, unparseable file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ocr_fusion.config.schema import AppSettings

logger = logging.getLogger(__name__)

ENV_PREFIX = "OCRFS_"

#: Maps ``OCRFS_*`` environment variables onto dotted settings paths.
ENV_OVERRIDES: dict[str, str] = {
    "OCRFS_OLLAMA_HOST": "ollama.host",
    "OCRFS_QWEN_MODEL": "qwen.model",
    "OCRFS_QWEN_TIMEOUT": "qwen.timeout_seconds",
    "OCRFS_QWEN_TEMPERATURE": "qwen.temperature",
    "OCRFS_QWEN_MAX_TOKENS": "qwen.max_tokens",
    "OCRFS_UNLIMITED_BACKEND": "unlimited_ocr.backend",
    "OCRFS_UNLIMITED_ENDPOINT": "unlimited_ocr.endpoint",
    "OCRFS_UNLIMITED_API_KEY": "unlimited_ocr.api_key",
    "OCRFS_UNLIMITED_MODEL_ID": "unlimited_ocr.model_id",
    "OCRFS_UNLIMITED_OLLAMA_MODEL": "unlimited_ocr.ollama_model",
    "OCRFS_UNLIMITED_EXECUTABLE": "unlimited_ocr.executable",
    "OCRFS_UNLIMITED_DEVICE": "unlimited_ocr.device",
    "OCRFS_FUSION_STRATEGY": "pipeline.fusion_strategy",
    "OCRFS_FUSION_MODEL": "pipeline.fusion_model",
    "OCRFS_PDF_DPI": "documents.pdf_render_dpi",
    "OCRFS_MAX_PAGES": "pipeline.max_pages",
    "OCRFS_DEVELOPER_MODE": "general.developer_mode",
    "OCRFS_PERSIST_DOCUMENTS": "privacy.persist_documents",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def default_settings_path() -> Path:
    """Where user settings are stored.

    ``OCRFS_SETTINGS_PATH`` overrides the location; otherwise settings live in a
    ``runtime/`` directory beside the project, which ``.gitignore`` excludes.
    """
    override = os.environ.get("OCRFS_SETTINGS_PATH")
    if override:
        return Path(override).expanduser()
    return _project_root() / "runtime" / "settings.json"


def _project_root() -> Path:
    # ocr_fusion/config/store.py -> ocr_fusion/config -> ocr_fusion -> root
    return Path(__file__).resolve().parents[2]


def _coerce(raw: str) -> Any:
    """Turn an environment string into a bool/int/float where unambiguous."""
    lowered = raw.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    return raw


def _assign(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``dotted`` ("a.b.c") inside a nested dict, creating levels as needed."""
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):  # pragma: no cover - guards malformed files
            return
    node[parts[-1]] = value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def env_overrides() -> dict[str, Any]:
    """Collect ``OCRFS_*`` variables into a nested settings dict."""
    overrides: dict[str, Any] = {}
    for env_name, dotted in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        _assign(overrides, dotted, _coerce(raw))
    return overrides


def read_settings_file(path: Path) -> dict[str, Any]:
    """Read a settings JSON file, returning ``{}`` when absent or unreadable."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable settings file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_settings(path: Path | str | None = None) -> AppSettings:
    """Build settings from defaults, then the settings file, then the environment.

    Invalid stored values never prevent start-up: the app falls back to defaults
    and logs the problem, because a corrupt settings file should not brick a
    client demo.
    """
    settings_path = Path(path) if path is not None else default_settings_path()
    layered = _deep_merge(read_settings_file(settings_path), env_overrides())
    if not layered:
        return AppSettings()
    try:
        return AppSettings.model_validate(layered)
    except ValidationError as exc:
        logger.warning("Stored settings failed validation, using defaults: %s", exc)
        return AppSettings()


def save_settings(settings: AppSettings, path: Path | str | None = None) -> Path:
    """Persist settings atomically. Returns the path written."""
    settings_path = Path(path) if path is not None else default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.to_dict(), indent=2, ensure_ascii=False)

    # Write to a temp file in the same directory, then replace: a crash midway
    # leaves the previous settings intact rather than a half-written file.
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=settings_path.parent,
        prefix=".settings-",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle as stream:
            stream.write(payload)
        os.replace(handle.name, settings_path)
    except OSError:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return settings_path


def reset_settings(path: Path | str | None = None) -> AppSettings:
    """Delete the stored settings file and return packaged defaults."""
    settings_path = Path(path) if path is not None else default_settings_path()
    settings_path.unlink(missing_ok=True)
    return AppSettings()


__all__ = [
    "ENV_OVERRIDES",
    "default_settings_path",
    "env_overrides",
    "load_settings",
    "read_settings_file",
    "reset_settings",
    "save_settings",
]
