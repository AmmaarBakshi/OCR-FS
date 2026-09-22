"""Timing a configuration, so a claim about speed can be checked.

Two things are measured and they answer different questions.

:func:`benchmark_documents` runs whole documents through the pipeline and
reports what an operator experiences: wall time, pages per minute, peak
memory, and where the time went stage by stage. This is the number that has
to come down.

:func:`benchmark_accuracy` renders gold pages at a given DPI and scores the
transcription against the text layer that page was authored with. This is the
number that must not come down with it.

Neither invents a measurement. A metric the run could not observe - peak RSS
without psutil, tokens from an engine that reports none - comes back as
``None`` and prints as ``n/a`` rather than as a zero.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocr_fusion.config.schema import AppSettings
from ocr_fusion.documents import load_document
from ocr_fusion.pipeline.result import PipelineResult


def _peak_rss_mb() -> float | None:
    """Resident memory in MB, or ``None`` when it cannot be observed."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - a metric, not a feature
        return None


@dataclass(slots=True)
class PageTiming:
    """What one page cost, and which engine spent it."""

    page_number: int
    engine: str
    seconds: float | None
    words: int
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_number,
            "engine": self.engine,
            "seconds": None if self.seconds is None else round(self.seconds, 2),
            "words": self.words,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(slots=True)
class DocumentTiming:
    """One document's run."""

    filename: str
    pages: int
    seconds: float
    load_seconds: float
    pages_needing_ocr: int
    pages_avoided: int
    stages: dict[str, float] = field(default_factory=dict)
    page_timings: list[PageTiming] = field(default_factory=list)
    peak_rss_mb: float | None = None
    final_words: int = 0
    error: str | None = None

    @property
    def pages_per_minute(self) -> float:
        return (self.pages / self.seconds * 60) if self.seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "pages": self.pages,
            "seconds": round(self.seconds, 2),
            "load_seconds": round(self.load_seconds, 3),
            "pages_needing_ocr": self.pages_needing_ocr,
            "pages_avoided": self.pages_avoided,
            "pages_per_minute": round(self.pages_per_minute, 3),
            "peak_rss_mb": None if self.peak_rss_mb is None else round(self.peak_rss_mb, 1),
            "final_words": self.final_words,
            "stages": {k: round(v, 2) for k, v in self.stages.items()},
            "pages_detail": [p.as_dict() for p in self.page_timings],
            "error": self.error,
        }


@dataclass(slots=True)
class BenchmarkReport:
    """Everything one configuration produced, ready to print or diff."""

    label: str
    settings_summary: dict[str, Any]
    documents: list[DocumentTiming] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return sum(d.pages for d in self.documents)

    @property
    def total_seconds(self) -> float:
        return sum(d.seconds for d in self.documents)

    @property
    def total_ocr_pages(self) -> int:
        return sum(d.pages_needing_ocr for d in self.documents)

    @property
    def pages_per_minute(self) -> float:
        return (self.total_pages / self.total_seconds * 60) if self.total_seconds else 0.0

    def page_seconds(self) -> list[float]:
        return [
            t.seconds
            for d in self.documents
            for t in d.page_timings
            if t.seconds is not None
        ]

    def percentiles(self) -> dict[str, float | None]:
        """Per-page timing spread. One difficult page can dominate a document,
        so a mean on its own hides the thing worth optimising."""
        values = sorted(self.page_seconds())
        if not values:
            return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}

        def at(fraction: float) -> float:
            index = min(len(values) - 1, int(round(fraction * (len(values) - 1))))
            return values[index]

        return {
            "mean": round(statistics.fmean(values), 2),
            "median": round(statistics.median(values), 2),
            "p90": round(at(0.90), 2),
            "p95": round(at(0.95), 2),
            "max": round(values[-1], 2),
        }

    def as_dict(self) -> dict[str, Any]:
        peaks = [d.peak_rss_mb for d in self.documents if d.peak_rss_mb is not None]
        return {
            "label": self.label,
            "settings": self.settings_summary,
            "totals": {
                "documents": len(self.documents),
                "pages": self.total_pages,
                "pages_needing_ocr": self.total_ocr_pages,
                "pages_avoided": self.total_pages - self.total_ocr_pages,
                "seconds": round(self.total_seconds, 2),
                "pages_per_minute": round(self.pages_per_minute, 3),
                "peak_rss_mb": round(max(peaks), 1) if peaks else None,
            },
            "page_seconds": self.percentiles(),
            "documents": [d.as_dict() for d in self.documents],
        }


def settings_summary(settings: AppSettings) -> dict[str, Any]:
    """The settings that actually move the number, for the report header."""
    return {
        "engine_mode": settings.pipeline.engine_mode.value,
        "routing_enabled": settings.routing.enabled,
        "use_text_layer": settings.routing.use_text_layer,
        "fallback_enabled": settings.pipeline.fallback_enabled,
        "pdf_render_dpi": settings.documents.pdf_render_dpi,
        "max_image_dimension": settings.documents.max_image_dimension,
        "qwen_model": settings.qwen.model,
        "qwen_max_tokens": settings.qwen.max_tokens,
        "unlimited_enabled": settings.unlimited_ocr.enabled,
        "unlimited_model": settings.unlimited_ocr.ollama_model,
        "fusion_strategy": settings.pipeline.fusion_strategy.value,
    }


def _collect(result: PipelineResult) -> tuple[dict[str, float], list[PageTiming]]:
    stages = {
        stage.key: stage.duration_seconds
        for stage in result.stages
        if stage.duration_seconds is not None
    }
    timings: list[PageTiming] = []
    for engine in result.engine_results:
        for page in engine.pages:
            timings.append(
                PageTiming(
                    page_number=page.page_number,
                    engine=engine.provider_id,
                    seconds=page.duration_seconds,
                    words=page.word_count,
                    input_tokens=page.tokens.input_tokens,
                    output_tokens=page.tokens.output_tokens,
                )
            )
    return stages, timings


def benchmark_documents(
    paths: list[Path],
    settings: AppSettings,
    *,
    label: str = "run",
    on_document: Any = None,
) -> BenchmarkReport:
    """Run each document through the pipeline and record what it cost."""
    from ocr_fusion.pipeline import build_pipeline

    report = BenchmarkReport(label=label, settings_summary=settings_summary(settings))

    for path in paths:
        if on_document:
            on_document(path)
        gc.collect()
        started = time.perf_counter()
        try:
            load_started = time.perf_counter()
            document = load_document(path, settings.documents)
            load_seconds = time.perf_counter() - load_started
        except Exception as exc:  # noqa: BLE001 - a broken file must not end a batch
            report.documents.append(
                DocumentTiming(
                    filename=path.name,
                    pages=0,
                    seconds=time.perf_counter() - started,
                    load_seconds=0.0,
                    pages_needing_ocr=0,
                    pages_avoided=0,
                    error=f"could not be loaded: {exc}",
                )
            )
            continue

        # A fresh pipeline per document, so one document's event log cannot
        # grow into the next one's measurement.
        result = build_pipeline(settings).execute(document)
        elapsed = time.perf_counter() - started
        stages, timings = _collect(result)

        report.documents.append(
            DocumentTiming(
                filename=path.name,
                pages=document.page_count,
                seconds=elapsed,
                load_seconds=load_seconds,
                pages_needing_ocr=len(result.routing.ocr_pages) if result.routing else document.page_count,
                pages_avoided=result.routing.pages_avoided if result.routing else 0,
                stages=stages,
                page_timings=timings,
                peak_rss_mb=_peak_rss_mb(),
                final_words=len(result.final_text.split()),
            )
        )
    return report


def benchmark_accuracy(
    gold: Any,
    settings: AppSettings,
    *,
    label: str = "accuracy",
    on_page: Any = None,
) -> dict[str, Any]:
    """Transcribe each gold page and score it against its own text layer.

    The page is rendered from the source PDF at the configured DPI and sent
    through the first enabled OCR engine - deliberately not through routing,
    which would answer the page from the very text layer being used as the
    answer key.
    """
    import fitz

    from ocr_fusion.bench.accuracy import score_page
    from ocr_fusion.documents.models import Document, DocumentKind, DocumentPage
    from ocr_fusion.ocr.registry import default_registry

    provider_id = "qwen_vl" if settings.qwen.enabled else "unlimited_ocr"
    provider = default_registry.create(provider_id, settings)

    scores = []
    durations = []
    for entry in gold.pages:
        if on_page:
            on_page(entry)
        pdf = fitz.open(entry.source)
        try:
            zoom = settings.documents.pdf_render_dpi / 72.0
            pixmap = pdf.load_page(entry.page_number - 1).get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), alpha=False
            )
            image = pixmap.tobytes("png")
            width, height = pixmap.width, pixmap.height
        finally:
            pdf.close()

        # A one-page document through the provider's public interface. Going
        # through process() rather than a private per-page method means this
        # measures the same path a real run takes, including retries.
        single = Document(
            filename=Path(entry.source).name,
            kind=DocumentKind.PDF,
            pages=[
                DocumentPage(
                    number=entry.page_number,
                    image_bytes=image,
                    width=width,
                    height=height,
                )
            ],
        )
        started = time.perf_counter()
        result = provider.process(single)
        elapsed = time.perf_counter() - started
        durations.append(elapsed)

        page_result = result.pages[0] if result.pages else None
        text = page_result.text if page_result else ""
        score = score_page(entry.page_number, entry.reference, text)
        scores.append(
            {
                "source": Path(entry.source).name,
                **score.as_dict(),
                "seconds": round(elapsed, 2),
                "input_tokens": page_result.tokens.input_tokens if page_result else None,
                "output_tokens": page_result.tokens.output_tokens if page_result else None,
                "error": result.error,
            }
        )

    def mean_of(key: str) -> float | None:
        values = [s[key] for s in scores if s.get(key) is not None]
        return round(statistics.fmean(values), 4) if values else None

    return {
        "label": label,
        "settings": settings_summary(settings),
        "pages": len(scores),
        "mean_seconds": round(statistics.fmean(durations), 2) if durations else None,
        "mean_wer": mean_of("wer"),
        "mean_cer": mean_of("cer"),
        "mean_recall": mean_of("recall"),
        "mean_figure_recall": mean_of("figure_recall"),
        "mean_input_tokens": mean_of("input_tokens"),
        "scores": scores,
    }


__all__ = [
    "BenchmarkReport",
    "DocumentTiming",
    "PageTiming",
    "benchmark_accuracy",
    "benchmark_documents",
    "settings_summary",
]
