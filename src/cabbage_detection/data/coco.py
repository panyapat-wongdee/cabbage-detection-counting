"""Minimal, strict COCO detection parsing and coordinate conversion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Sequence

from .coordinates import coco_xywh_to_xyxy, xyxy_to_yolo


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    relative_path: Path | None = None


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class CocoDetectionDataset:
    images: dict[int, CocoImage]
    annotations: tuple[CocoAnnotation, ...]


def read_coco_detection(path: Path) -> CocoDetectionDataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("images"), list) or not isinstance(raw.get("annotations"), list):
        raise ValueError("COCO file must contain images and annotations lists")
    images: dict[int, CocoImage] = {}
    for item in raw["images"]:
        try:
            file_name = str(item["file_name"])
            if item.get("path") is not None:
                relative_path = _parse_coco_relative_path(item["path"], field="path")
            else:
                _parse_coco_relative_path(file_name, field="file_name")
                relative_path = None
            image = CocoImage(
                int(item["id"]), file_name, int(item["width"]), int(item["height"]), relative_path
            )
        except ValueError as error:
            if "safe relative path" in str(error):
                raise
            raise ValueError("invalid COCO image record") from error
        except (KeyError, TypeError) as error:
            raise ValueError("invalid COCO image record") from error
        if image.width <= 0 or image.height <= 0:
            raise ValueError("COCO image dimensions must be positive")
        if image.id in images:
            raise ValueError(f"duplicate COCO image id: {image.id}")
        images[image.id] = image
    annotations: list[CocoAnnotation] = []
    for index, item in enumerate(raw["annotations"], start=1):
        try:
            image_id = int(item["image_id"])
            bbox = tuple(float(value) for value in item["bbox"])
            annotation = CocoAnnotation(int(item.get("id", index)), image_id, int(item.get("category_id", 1)), bbox)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid COCO annotation record") from error
        if image_id not in images:
            raise ValueError(f"annotation references unknown image_id: {image_id}")
        coco_xywh_to_xyxy(annotation.bbox)
        annotations.append(annotation)
    return CocoDetectionDataset(images=images, annotations=tuple(annotations))


def _parse_coco_relative_path(value: object, *, field: str) -> Path:
    """Parse an official slash-delimited path and reject unsafe components."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"COCO image {field} must be a safe relative path")
    if "\\" in value or re.match(r"^[A-Za-z]:", value) or value.startswith("//"):
        raise ValueError(f"COCO image {field} must be a safe relative path")
    if value.startswith("/"):
        value = value[1:]
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".."} for part in parsed.parts):
        raise ValueError(f"COCO image {field} must be a safe relative path")
    return Path(*parsed.parts)


def resolve_coco_image_path(images_root: Path, image: CocoImage) -> Path:
    """Resolve a COCO image beneath ``images_root`` with containment checks."""

    root = Path(images_root).resolve()
    relative = image.relative_path or _parse_coco_relative_path(image.file_name, field="file_name")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"COCO image path escapes images root: {relative}")
    return candidate
