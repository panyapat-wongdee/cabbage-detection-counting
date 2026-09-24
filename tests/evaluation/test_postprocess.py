from __future__ import annotations

import pytest

from cabbage_detection.config import PostprocessStageConfig
from cabbage_detection.evaluation.postprocess import filter_torchvision_outputs


torch = pytest.importorskip("torch")


def _stage(**kwargs):
    values = {
        "confidence_threshold": 0.5,
        "iou_threshold": 0.5,
        "area_threshold": 100.0,
        "confidence_owner": "repository",
        "nms_owner": "repository",
    }
    values.update(kwargs)
    return PostprocessStageConfig(**values)


def test_historical_filters_use_strict_score_and_area() -> None:
    output = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 11.0, 11.0]]),
        "scores": torch.tensor([0.5, 0.9]),
        "labels": torch.tensor([1, 1]),
    }
    result = filter_torchvision_outputs([output], _stage(), max_detections=100)[0]
    assert result["scores"].tolist() == pytest.approx([0.9])


def test_historical_filter_applies_repository_nms_and_max_detections() -> None:
    output = {
        "boxes": torch.tensor([[0.0, 0.0, 20.0, 20.0], [1.0, 1.0, 19.0, 19.0], [40.0, 40.0, 60.0, 60.0]]),
        "scores": torch.tensor([0.8, 0.9, 0.7]),
        "labels": torch.tensor([1, 1, 1]),
    }
    result = filter_torchvision_outputs([output], _stage(area_threshold=0.0), max_detections=2)[0]
    assert result["scores"].tolist() == pytest.approx([0.9, 0.7])


def test_filter_preserves_empty_output() -> None:
    empty = {"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.int64)}
    result = filter_torchvision_outputs([empty], _stage(), max_detections=100)[0]
    assert tuple(result["boxes"].shape) == (0, 4)
