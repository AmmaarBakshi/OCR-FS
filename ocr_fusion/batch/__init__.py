"""Processing many documents without losing work.

A thousand PDFs is not a big version of one PDF. At the rate this hardware
manages it is days of work, so the property that matters is not peak speed
but that no minute of it is ever spent twice: results are written per
document, state is checkpointed on every change, and a resumed run picks up
exactly where the last one stopped.
"""

from __future__ import annotations

from ocr_fusion.batch.jobs import (
    BatchState,
    DocumentJob,
    JobState,
    available_memory_mb,
    elapsed_text,
    wait_for_memory,
)
from ocr_fusion.batch.runner import (
    DEFAULT_MEMORY_FLOOR_MB,
    BatchCancelled,
    BatchRunner,
    collect_documents,
    memory_floor_for,
)

__all__ = [
    "DEFAULT_MEMORY_FLOOR_MB",
    "BatchCancelled",
    "BatchRunner",
    "BatchState",
    "DocumentJob",
    "JobState",
    "available_memory_mb",
    "collect_documents",
    "elapsed_text",
    "memory_floor_for",
    "wait_for_memory",
]
