"""Schema-less JSON I/O for canonical prediction and target records."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .records import Box, Detection, PostprocessingMetadata, PredictionRecord, TargetRecord


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"evaluation {label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"evaluation {label} must be finite")
    return number


def _coordinates(value: object, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"evaluation {label} must be a four-value list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"evaluation {label} must contain numeric coordinates")
    coordinates = tuple(_number(item, f"{label} coordinate") for item in value)
    return coordinates  # type: ignore[return-value]


def write_prediction_records(path: Path, records: tuple[PredictionRecord, ...] | list[PredictionRecord]) -> Path:
    """Write canonical prediction records without a version marker."""
    payload = {
        "records": [
            {
                "image_id": record.image_id,
                "detections": [
                    {"box": list(detection.box), "score": detection.score, "class_id": detection.class_id}
                    for detection in record.detections
                ],
                "postprocessing": (
                    {
                        "confidence_owner": record.postprocessing.confidence_owner,
                        "nms_owner": record.postprocessing.nms_owner,
                        "confidence_threshold": record.postprocessing.confidence_threshold,
                        "nms_iou_threshold": record.postprocessing.nms_iou_threshold,
                        "max_detections": record.postprocessing.max_detections,
                        "area_threshold": record.postprocessing.area_threshold,
                        "image_size": list(record.postprocessing.image_size) if record.postprocessing.image_size else None,
                        "checkpoint_provenance": record.postprocessing.checkpoint_provenance,
                        "stage": record.postprocessing.stage,
                        "native_nms_applied": record.postprocessing.native_nms_applied,
                    }
                    if record.postprocessing is not None
                    else None
                ),
            }
            for record in records
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def write_target_records(path: Path, records: tuple[TargetRecord, ...] | list[TargetRecord]) -> Path:
    """Write canonical target records without a version marker."""
    payload = {
        "records": [
            {"image_id": record.image_id, "boxes": [list(box.coordinates) for box in record.boxes]}
            for record in records
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _read(path: Path, key: str) -> list[dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read current evaluation format from {path}") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"records"}
        or not isinstance(payload["records"], list)
    ):
        raise ValueError("evaluation JSON must be a schema-less object containing only a records list")
    records = payload["records"]
    if any(not isinstance(record, dict) or "image_id" not in record or key not in record for record in records):
        raise ValueError(f"every evaluation record requires image_id and {key}")
    allowed_fields = {"image_id", key}
    if key == "detections":
        allowed_fields.add("postprocessing")
    if any(set(record) - allowed_fields for record in records):
        raise ValueError(f"evaluation records contain fields outside the current {key} format")
    return records


def read_prediction_records(path: Path) -> tuple[PredictionRecord, ...]:
    records = _read(path, "detections")
    output = []
    for record in records:
        if not isinstance(record["detections"], list):
            raise ValueError("evaluation detections must be a list")
        detections = []
        for item in record["detections"]:
            if not isinstance(item, dict) or "box" not in item or "score" not in item:
                raise ValueError("evaluation detections must contain box and score")
            if set(item) - {"box", "score", "class_id"}:
                raise ValueError("evaluation detections contain fields outside the current format")
            score_value = item["score"]
            class_id_value = item.get("class_id", 1)
            if isinstance(class_id_value, bool) or not isinstance(class_id_value, int):
                raise ValueError("evaluation detections must contain numeric score and class_id")
            score = _number(score_value, "detection score")
            class_id = int(class_id_value)
            detections.append(Detection(_coordinates(item["box"], "detection box"), score, class_id))
        detections = tuple(detections)
        metadata_raw = record.get("postprocessing")
        if metadata_raw is not None and not isinstance(metadata_raw, dict):
            raise ValueError("evaluation postprocessing must be an object or null")
        if metadata_raw is None:
            metadata = None
        else:
            if set(metadata_raw) - {
                "confidence_owner",
                "nms_owner",
                "confidence_threshold",
                "nms_iou_threshold",
                "max_detections",
                "area_threshold",
                "image_size",
                "checkpoint_provenance",
                "stage",
                "native_nms_applied",
            }:
                raise ValueError("evaluation postprocessing contains fields outside the current format")
            try:
                if not isinstance(metadata_raw["confidence_owner"], str) or not isinstance(
                    metadata_raw["nms_owner"], str
                ):
                    raise ValueError("evaluation postprocessing owners must be strings")
                if not isinstance(metadata_raw.get("max_detections", 100), int) or isinstance(
                    metadata_raw.get("max_detections", 100), bool
                ):
                    raise ValueError("evaluation postprocessing max_detections must be an integer")
                image_size_raw = metadata_raw.get("image_size")
                if image_size_raw is not None and (
                    not isinstance(image_size_raw, list)
                    or len(image_size_raw) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in image_size_raw)
                ):
                    raise ValueError("evaluation postprocessing image_size must be a two-value integer list or null")
                checkpoint_provenance = metadata_raw.get("checkpoint_provenance")
                if checkpoint_provenance is not None and not isinstance(checkpoint_provenance, str):
                    raise ValueError("evaluation postprocessing checkpoint_provenance must be a string or null")
                stage = metadata_raw.get("stage")
                if stage is not None and not isinstance(stage, str):
                    raise ValueError("evaluation postprocessing stage must be a string or null")
                native_nms_applied = metadata_raw.get("native_nms_applied", False)
                if not isinstance(native_nms_applied, bool):
                    raise ValueError("evaluation postprocessing native_nms_applied must be boolean")
                metadata = PostprocessingMetadata(
                    confidence_owner=metadata_raw["confidence_owner"],
                    nms_owner=metadata_raw["nms_owner"],
                    confidence_threshold=_number(metadata_raw["confidence_threshold"], "confidence_threshold"),
                    nms_iou_threshold=_number(metadata_raw["nms_iou_threshold"], "nms_iou_threshold"),
                    max_detections=metadata_raw.get("max_detections", 100),
                    area_threshold=_number(metadata_raw.get("area_threshold", 0.0), "area_threshold"),
                    image_size=tuple(image_size_raw) if image_size_raw is not None else None,
                    checkpoint_provenance=checkpoint_provenance,
                    stage=stage,
                    native_nms_applied=native_nms_applied,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("evaluation postprocessing has an invalid current format") from error
        output.append(PredictionRecord(str(record["image_id"]), detections, metadata))
    return tuple(output)


def read_target_records(path: Path) -> tuple[TargetRecord, ...]:
    records = _read(path, "boxes")
    if any(not isinstance(record["boxes"], list) for record in records):
        raise ValueError("evaluation boxes must be a list of box coordinates")
    try:
        return tuple(
            TargetRecord(
                str(record["image_id"]),
                tuple(Box(_coordinates(item, "target box")) for item in record["boxes"]),
            )
            for record in records
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evaluation boxes must be a list of box coordinates") from error
