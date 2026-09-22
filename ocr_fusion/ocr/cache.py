"""Never transcribe the same page twice.

OCR on a fixed page with a fixed model at temperature zero is a pure
function: the same pixels and the same prompt produce the same text. Anything
pure and this expensive - about eight minutes a page on this machine - is
worth remembering.

At one document the saving is occasional. Across a batch it is structural:
client bundles repeat pages constantly (the same driving licence attached to
three applications, the same trust deed in two folders, a statement submitted
twice), and a re-run after a crash or a settings tweak would otherwise pay
the full cost again for pages that have not changed.

The cache is a provider wrapper, not a change to any provider. It looks up
each page, hands the engine a document containing only the pages it does not
already know, and merges the two sets of results back together. Every engine
therefore gets caching without knowing the cache exists - the same seam that
lets a new engine join a run unmodified.

**It is off by default.** A cache is persistence, and this project's rule is
that documents and their text stay in memory unless the operator turns
storage on deliberately. When it is on, what lands on disk is transcribed
text, so the location is configurable and the operator is told.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr_fusion.documents.models import Document
from ocr_fusion.ocr.interface import (
    HealthStatus,
    OCRProvider,
    OCRResult,
    OCRStatus,
    PageResult,
    TokenUsage,
)

logger = logging.getLogger(__name__)

#: Bump when a change to this module would make stored rows mean something
#: different. Entries written under an older version are ignored rather than
#: deleted, so rolling back does not throw away a warm cache.
CACHE_FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    key           TEXT PRIMARY KEY,
    format        INTEGER NOT NULL,
    provider_id   TEXT NOT NULL,
    model         TEXT,
    text          TEXT NOT NULL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    seconds       REAL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS pages_provider ON pages (provider_id);
"""


@dataclass(slots=True)
class CacheStats:
    """Hits and misses for one run, for the log and the metrics panel."""

    hits: int = 0
    misses: int = 0
    seconds_saved: float = 0.0
    """Sum of the recorded cost of the pages that were served from the cache."""

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "seconds_saved": round(self.seconds_saved, 2),
        }


def page_cache_key(
    image_bytes: bytes,
    provider_id: str,
    model: str | None,
    options: dict[str, Any],
) -> str:
    """Content address for one page under one exact configuration.

    Everything that can change the output goes into the key: the pixels, the
    engine, the model and the sampling options including the prompt. Change
    any of them and the entry is a miss, which is the point - a cache that
    returned yesterday's prompt's answer would be worse than no cache.
    """
    digest = hashlib.sha256()
    digest.update(image_bytes)
    digest.update(b"\x00")
    digest.update(provider_id.encode("utf-8"))
    digest.update(b"\x00")
    digest.update((model or "").encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(options, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


class PageCache:
    """SQLite-backed store of page transcriptions.

    SQLite rather than a directory of files because a batch of ten thousand
    documents is hundreds of thousands of pages, and because it gives
    durability and concurrent readers without writing any of that here.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One connection guarded by a lock: the pipeline is synchronous, and a
        # connection per call would pay SQLite's open cost on every page.
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT text, input_tokens, output_tokens, seconds FROM pages "
                "WHERE key = ? AND format = ?",
                (key, CACHE_FORMAT_VERSION),
            ).fetchone()
        if row is None:
            return None
        return {
            "text": row[0],
            "input_tokens": row[1],
            "output_tokens": row[2],
            "seconds": row[3],
        }

    def put(
        self,
        key: str,
        provider_id: str,
        model: str | None,
        page: PageResult,
    ) -> None:
        """Store a page. Only a clean success is worth remembering."""
        if page.status is not OCRStatus.SUCCESS or not page.text.strip():
            return
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO pages "
                "(key, format, provider_id, model, text, input_tokens, "
                " output_tokens, seconds, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    CACHE_FORMAT_VERSION,
                    provider_id,
                    model,
                    page.text,
                    page.tokens.input_tokens,
                    page.tokens.output_tokens,
                    page.duration_seconds,
                    time.time(),
                ),
            )
            self._connection.commit()

    def count(self) -> int:
        with self._lock:
            return int(
                self._connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            )

    def clear(self) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM pages")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachingProvider(OCRProvider):
    """Wraps an engine so a page it has already read is never read again.

    Implements :class:`~ocr_fusion.ocr.interface.OCRProvider` and delegates
    everything else to the engine it wraps, so the pipeline, the registry and
    the UI cannot tell the difference - except that the wrapped engine reports
    its own identity, so the UI still names the model that really ran.
    """

    def __init__(self, inner: OCRProvider, cache: PageCache) -> None:
        self.inner = inner
        self.cache = cache
        self.stats = CacheStats()

    # -- identity is the wrapped engine's ---------------------------------

    @property
    def provider_id(self) -> str:  # type: ignore[override]
        return self.inner.provider_id

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return self.inner.provider_name

    @property
    def processing_location(self):  # type: ignore[override]
        return self.inner.processing_location

    def health_check(self) -> HealthStatus:
        return self.inner.health_check()

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.inner.get_metadata())
        metadata["page_cache"] = str(self.cache.path)
        return metadata

    # -- caching ----------------------------------------------------------

    def _key_for(self, image_bytes: bytes) -> str:
        metadata = self.inner.get_metadata()
        options = {
            key: metadata.get(key)
            for key in ("temperature", "top_p", "max_tokens", "prompt", "task_prompt")
            if metadata.get(key) is not None
        }
        return page_cache_key(
            image_bytes, self.inner.provider_id, metadata.get("model"), options
        )

    def process(
        self,
        document: Document,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> OCRResult:
        from ocr_fusion.pipeline.routing import subset

        self.stats = CacheStats()
        cached: dict[int, PageResult] = {}
        keys: dict[int, str] = {}

        for page in document.pages:
            key = self._key_for(page.image_bytes)
            keys[page.number] = key
            entry = self.cache.get(key)
            if entry is None:
                self.stats.misses += 1
                continue
            self.stats.hits += 1
            self.stats.seconds_saved += entry["seconds"] or 0.0
            cached[page.number] = PageResult(
                page_number=page.number,
                text=entry["text"],
                status=OCRStatus.SUCCESS,
                # The cost was paid on a previous run, not this one. Reporting
                # the original duration here would claim this run spent time it
                # did not (spec s8).
                duration_seconds=0.0,
                tokens=TokenUsage(entry["input_tokens"], entry["output_tokens"]),
                raw_response={"page_cache": "hit", "original_seconds": entry["seconds"]},
            )

        remaining = [p.number for p in document.pages if p.number not in cached]
        if not remaining:
            return self._result_from(document, cached, fresh=None)

        fresh = self.inner.process(subset(document, remaining), on_progress=on_progress)
        for page in fresh.pages:
            self.cache.put(keys[page.page_number], self.provider_id, fresh.model_name, page)

        if not cached:
            return fresh
        return self._result_from(document, cached, fresh=fresh)

    def _result_from(
        self,
        document: Document,
        cached: dict[int, PageResult],
        fresh: OCRResult | None,
    ) -> OCRResult:
        """Merge cached pages and freshly read ones back into one result."""
        if fresh is not None:
            result = OCRResult(
                provider_id=fresh.provider_id,
                provider_name=fresh.provider_name,
                pages=list(fresh.pages),
                duration_seconds=fresh.duration_seconds,
                model_name=fresh.model_name,
                backend=fresh.backend,
                processing_location=fresh.processing_location,
                metadata=dict(fresh.metadata),
                remedy=fresh.remedy,
            )
        else:
            metadata = self.get_metadata()
            result = OCRResult(
                provider_id=self.provider_id,
                provider_name=self.provider_name,
                model_name=metadata.get("model"),
                backend=metadata.get("backend"),
                processing_location=self.processing_location,
                metadata=metadata,
            )

        result.pages.extend(cached.values())
        result.pages.sort(key=lambda p: p.page_number)
        result.metadata["page_cache"] = self.stats.as_dict()

        succeeded = sum(1 for p in result.pages if p.status is OCRStatus.SUCCESS)
        if succeeded == len(result.pages):
            result.status = OCRStatus.SUCCESS
        elif succeeded == 0:
            result.status = OCRStatus.FAILED
        else:
            result.status = OCRStatus.PARTIAL
        return result


def wrap_with_cache(provider: OCRProvider, cache: PageCache) -> OCRProvider:
    """Add caching to an engine, leaving free engines alone.

    Text-layer extraction and routing cost milliseconds; a cache lookup would
    cost more than the work, and storing their output would put document text
    on disk for no gain.
    """
    if provider.provider_id in {"text_layer", "routing"}:
        return provider
    return CachingProvider(provider, cache)


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CacheStats",
    "CachingProvider",
    "PageCache",
    "page_cache_key",
    "wrap_with_cache",
]
