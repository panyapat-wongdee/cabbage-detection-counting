"""Small immutable records shared by metrics and framework adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Box:
    coordinates: tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int = 1


@dataclass(frozen=True)
class PostprocessingMetadata:
    confidence_owner: str
    nms_owner: str
    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int = 100
    area_threshold: float = 0.0
    image_size: tuple[int, int] | None = None
    checkpoint_provenance: str | None = None
    stage: str | None = None
    native_nms_applied: bool = False
    native_nms_iou_threshold: float | None = None
    native_score_threshold: float | None = None
    native_detections_per_image: int | None = None


@dataclass(frozen=True)
class MatchSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    predicted_count: int
    actual_count: int


@dataclass(frozen=True)
class CountingMetrics:
    predicted_count: int
    actual_count: int
    count_error: float
    fdr: float
    fnr: float
    f1: float


@dataclass(frozen=True)
class PredictionRecord:
    image_id: str
    detections: tuple[Detection, ...]
    postprocessing: PostprocessingMetadata | None = field(default=None, compare=True)


@dataclass(frozen=True)
class TargetRecord:
    image_id: str
    boxes: tuple[Box, ...]
