"""Explicit torchvision output filtering shared by validation and prediction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..config import PostprocessStageConfig


def filter_torchvision_outputs(
    outputs: Iterable[dict[str, Any]],
    stage: PostprocessStageConfig,
    *,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Apply the historical score/area/NMS order to detector outputs.

    The preserved engine uses strict ``>`` comparisons for both confidence and
    area, then performs repository NMS and truncates by score.
    """

    try:
        import torch
        from torchvision.ops import nms
    except ImportError as error:
        raise RuntimeError("install torch and torchvision for torchvision post-processing") from error
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    filtered: list[dict[str, Any]] = []
    for output in outputs:
        boxes = output.get("boxes", torch.empty((0, 4)))
        scores = output.get("scores", torch.empty((0,)))
        labels = output.get("labels", torch.empty((0,), dtype=torch.int64))
        keep: list[int] = []
        for index, score in enumerate(scores.tolist()):
            if float(score) <= stage.confidence_threshold:
                continue
            box = boxes[index]
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if stage.area_threshold > 0 and area <= stage.area_threshold:
                continue
            keep.append(index)
        if stage.nms_owner == "repository" and keep:
            kept = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
            keep = nms(boxes[kept], scores[kept], stage.iou_threshold).tolist()
            keep = [int(kept[index]) for index in keep]
        keep.sort(key=lambda index: float(scores[index]), reverse=True)
        keep = keep[:max_detections]
        filtered.append(
            {
                "boxes": boxes[keep] if keep else boxes.new_empty((0, 4)),
                "scores": scores[keep] if keep else scores.new_empty((0,)),
                "labels": labels[keep] if keep else labels.new_empty((0,), dtype=labels.dtype),
            }
        )
    return filtered
