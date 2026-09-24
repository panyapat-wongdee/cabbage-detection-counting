"""Bounding-box validation and IoU."""

from __future__ import annotations

from collections.abc import Sequence


def _validated(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    values = tuple(float(value) for value in box)
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError("box coordinates must be ordered x1<=x2 and y1<=y2")
    return values


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = _validated(box_a)
    bx1, by1, bx2, by2 = _validated(box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
