"""Pipeline orchestration: stages, execution, comparison and fusion."""

from ocr_fusion.pipeline.comparison import (
    ComparisonResult,
    DiffKind,
    DiffLine,
    compare_texts,
)
from ocr_fusion.pipeline.events import EventLog, LogEvent, LogLevel, Stage, StageStatus
from ocr_fusion.pipeline.fusion import FUSION_STRATEGIES, FusionResult, fuse
from ocr_fusion.pipeline.pipeline import (
    STAGE_COMPARISON,
    STAGE_FUSION,
    STAGE_UPLOAD,
    OCRPipeline,
    build_pipeline,
)
from ocr_fusion.pipeline.result import SCHEMA_VERSION, PipelineResult

__all__ = [
    "FUSION_STRATEGIES",
    "SCHEMA_VERSION",
    "STAGE_COMPARISON",
    "STAGE_FUSION",
    "STAGE_UPLOAD",
    "ComparisonResult",
    "DiffKind",
    "DiffLine",
    "EventLog",
    "FusionResult",
    "LogEvent",
    "LogLevel",
    "OCRPipeline",
    "PipelineResult",
    "Stage",
    "StageStatus",
    "build_pipeline",
    "compare_texts",
    "fuse",
]
