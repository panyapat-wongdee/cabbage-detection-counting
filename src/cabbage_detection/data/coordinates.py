"""Auditable conversions between COCO, XYXY, and YOLO box formats."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _values(box: Sequence[float], message: str) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(message)
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box values must be finite")
    return values


def coco_xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = _values(box, "COCO bbox must contain four values")
    if width < 0 or height < 0:
        raise ValueError("COCO bbox dimensions must be non-negative")
    return (x, y, x + width, y + height)


def xyxy_to_coco_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _values(box, "xyxy box must contain four values")
    if x2 < x1 or y2 < y1:
        raise ValueError("xyxy coordinates must be ordered")
    return (x1, y1, x2 - x1, y2 - y1)


def xyxy_to_yolo(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = _values(box, "xyxy box must contain four values")
    if x2 < x1 or y2 < y1:
        raise ValueError("xyxy coordinates must be ordered")
    return (((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height, (x2 - x1) / width, (y2 - y1) / height)


def coco_xywh_to_yolo(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    return xyxy_to_yolo(coco_xywh_to_xyxy(box), width, height)


def yolo_to_coco_xywh(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    center_x, center_y, box_width, box_height = _values(box, "YOLO box must contain four values")
    if min(box_width, box_height) < 0:
        raise ValueError("YOLO box dimensions must be non-negative")
    pixel_width, pixel_height = box_width * width, box_height * height
    return ((center_x * width) - pixel_width / 2, (center_y * height) - pixel_height / 2, pixel_width, pixel_height)


def format_yolo_box(box: Sequence[float], decimals: int = 10) -> str:
    if decimals < 1:
        raise ValueError("decimals must be positive")
    return " ".join(f"{value:.{decimals}f}" for value in _values(box, "YOLO box must contain four values"))


def assert_round_trip(original: Sequence[float], restored: Sequence[float], absolute_tolerance: float = 1e-6) -> None:
    expected = _values(original, "original box must contain four values")
    actual = _values(restored, "restored box must contain four values")
    if absolute_tolerance < 0 or any(abs(a - b) > absolute_tolerance for a, b in zip(expected, actual)):
        raise AssertionError(f"box round-trip exceeded tolerance {absolute_tolerance}: {expected} != {actual}")
