"""Pipeline stage model and the event log.

Two jobs:

* **Stage tracking** - each stage carries a status and timing so the UI can
  draw the pipeline visualisation (spec s4).
* **Event logging** - a timestamped record of what happened, shown in the
  Processing Details panel (spec s9).

The log is privacy-aware by default: events record counts and lengths, never
document text, unless the operator explicitly lowers that guard (spec s14).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    """Lifecycle of one pipeline stage, as rendered in the UI."""

    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def symbol(self) -> str:
        return {
            StageStatus.WAITING: "○",
            StageStatus.RUNNING: "◐",
            StageStatus.COMPLETED: "✓",
            StageStatus.FAILED: "✕",
            StageStatus.SKIPPED: "–",
        }[self]


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class Stage:
    """One step of the pipeline visualisation."""

    key: str
    label: str
    status: StageStatus = StageStatus.WAITING
    duration_seconds: float | None = None
    detail: str = ""
    """One line shown under the stage, e.g. the model that ran or why it failed."""

    started_at: float | None = field(default=None, repr=False)

    def start(self) -> None:
        self.status = StageStatus.RUNNING
        self.started_at = time.perf_counter()

    def finish(self, status: StageStatus, detail: str = "") -> None:
        if self.started_at is not None:
            self.duration_seconds = time.perf_counter() - self.started_at
        self.status = status
        if detail:
            self.detail = detail

    def skip(self, reason: str) -> None:
        self.status = StageStatus.SKIPPED
        self.detail = reason
        self.duration_seconds = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "detail": self.detail,
        }


@dataclass(slots=True)
class LogEvent:
    """One line in the processing log."""

    timestamp: datetime
    level: LogLevel
    message: str
    stage: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def clock(self) -> str:
        """``HH:MM:SS``, matching the format in the spec's example log."""
        return self.timestamp.strftime("%H:%M:%S")

    def format(self) -> str:
        return f"[{self.clock}] {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "clock": self.clock,
            "level": self.level.value,
            "stage": self.stage,
            "message": self.message,
            "data": self.data,
        }


class EventLog:
    """Collects stages and log events for a single run.

    A subscriber callback is invoked on every event, which is how the Streamlit
    UI updates its pipeline display while a run is in progress.
    """

    def __init__(self, *, redact_text: bool = True, preview_chars: int = 0) -> None:
        self.events: list[LogEvent] = []
        self.stages: dict[str, Stage] = {}
        self._order: list[str] = []
        self.redact_text = redact_text
        self.preview_chars = preview_chars
        self._subscribers: list[Callable[[LogEvent], None]] = []

    # -- subscription ------------------------------------------------------

    def subscribe(self, callback: Callable[[LogEvent], None]) -> None:
        self._subscribers.append(callback)

    def _emit(self, event: LogEvent) -> None:
        self.events.append(event)
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - a broken UI callback must not
                # abort an OCR run that is already minutes deep.
                pass

    # -- logging -----------------------------------------------------------

    def log(
        self,
        message: str,
        *,
        level: LogLevel = LogLevel.INFO,
        stage: str = "",
        **data: Any,
    ) -> LogEvent:
        event = LogEvent(
            timestamp=datetime.now(), level=level, message=message, stage=stage, data=data
        )
        self._emit(event)
        return event

    def info(self, message: str, **kwargs: Any) -> LogEvent:
        return self.log(message, level=LogLevel.INFO, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> LogEvent:
        return self.log(message, level=LogLevel.WARNING, **kwargs)

    def error(self, message: str, **kwargs: Any) -> LogEvent:
        return self.log(message, level=LogLevel.ERROR, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> LogEvent:
        return self.log(message, level=LogLevel.DEBUG, **kwargs)

    def text_summary(self, text: str) -> str:
        """Describe OCR output without reproducing it.

        Documents may be confidential, so the log records shape - characters and
        words - and includes a short preview only when the operator has turned
        redaction off (spec s14).
        """
        summary = f"{len(text)} characters, {len(text.split())} words"
        if not self.redact_text and self.preview_chars > 0:
            preview = " ".join(text[: self.preview_chars].split())
            if preview:
                summary += f' - "{preview}..."'
        return summary

    # -- stages ------------------------------------------------------------

    def add_stage(self, key: str, label: str) -> Stage:
        stage = Stage(key=key, label=label)
        self.stages[key] = stage
        self._order.append(key)
        return stage

    def stage(self, key: str) -> Stage:
        return self.stages[key]

    def ordered_stages(self) -> list[Stage]:
        return [self.stages[key] for key in self._order]

    def start_stage(self, key: str, message: str = "") -> Stage:
        stage = self.stages[key]
        stage.start()
        self.info(message or f"{stage.label} started", stage=key)
        return stage

    def finish_stage(
        self, key: str, status: StageStatus, detail: str = "", message: str = ""
    ) -> Stage:
        stage = self.stages[key]
        stage.finish(status, detail)
        level = LogLevel.ERROR if status is StageStatus.FAILED else LogLevel.INFO
        elapsed = (
            f" in {stage.duration_seconds:.2f}s" if stage.duration_seconds else ""
        )
        self.log(
            message or f"{stage.label} {status.value}{elapsed}",
            level=level,
            stage=key,
            duration_seconds=stage.duration_seconds,
        )
        return stage

    def skip_stage(self, key: str, reason: str) -> Stage:
        stage = self.stages[key]
        stage.skip(reason)
        self.info(f"{stage.label} skipped - {reason}", stage=key)
        return stage

    # -- export ------------------------------------------------------------

    def clear(self) -> None:
        self.events.clear()

    def as_text(self) -> str:
        """Copyable/exportable log, in the format the spec illustrates."""
        return "\n".join(event.format() for event in self.events)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.events]


__all__ = ["EventLog", "LogEvent", "LogLevel", "Stage", "StageStatus"]
