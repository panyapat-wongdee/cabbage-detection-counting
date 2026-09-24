"""Native Ultralytics validation with explicit paper-method arguments."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..progress import status


SUPPORTED_SPLITS = {"train", "val", "test"}


def _load_model(config: ExperimentConfig, checkpoint: Path) -> Any:
    """Load the configured Ultralytics model lazily."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as error:
        raise RuntimeError("install the optional ultralytics dependency to run native validation") from error
    constructor = RTDETR if config.model_name == "rt-detr-l" else YOLO
    return constructor(str(checkpoint))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_digest(config: ExperimentConfig) -> str:
    encoded = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision() -> str:
    try:
        repository = str(Path.cwd().resolve()).replace("\\", "/")
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _native_device(config: ExperimentConfig) -> str:
    if config.execution.device == "cuda":
        return "0"
    if config.execution.device.startswith("cuda:"):
        return config.execution.device.split(":", 1)[1]
    return config.execution.device


def _finite_metric(metrics: Any, name: str) -> float:
    value = getattr(metrics, name, None)
    if value is None:
        raise ValueError(f"Ultralytics validation did not return metrics.box.{name}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Ultralytics validation returned a non-finite metrics.box.{name}")
    return value


def evaluate_ultralytics_native(
    config: ExperimentConfig,
    checkpoint: Path,
    dataset_yaml: Path,
    split: str,
    output_dir: Path,
) -> Path:
    """Run native ``model.val`` and write a separately labeled metrics artifact."""
    if config.framework != "ultralytics":
        raise ValueError("native Ultralytics validation requires framework=ultralytics")
    if split not in SUPPORTED_SPLITS:
        raise ValueError("split must be one of train, val, or test")
    checkpoint = Path(checkpoint)
    dataset_yaml = Path(dataset_yaml)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    console_path = destination / "console.log"
    status(
        f"Ultralytics native evaluation started: split={split} device={config.execution.device}",
        log_file=console_path,
    )
    model = _load_model(config, checkpoint)
    device = _native_device(config)
    validation_args: dict[str, object] = {
        "data": str(dataset_yaml),
        "split": split,
        "conf": config.confidence_threshold,
        "iou": config.iou_threshold,
        "max_det": config.evaluation.max_detections,
        "batch": config.batch_size,
        "imgsz": config.image_size[0],
        "device": device,
        "project": str(destination),
        "name": "framework",
        "plots": False,
        "save_json": False,
    }
    native_result = model.val(**validation_args)
    box_metrics = getattr(native_result, "box", None)
    if box_metrics is None:
        raise ValueError("Ultralytics validation did not return box metrics")
    map50 = _finite_metric(box_metrics, "map50")
    map50_95 = _finite_metric(box_metrics, "map")

    metrics_payload: dict[str, object] = {
        "task": "detection",
        "input_schema": "native_ultralytics",
        "scope": {"split": split},
        "metrics": {"map50": map50, "map50_95": map50_95},
        "evaluation": {
            "backend": "ultralytics_native_val",
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
            "max_detections": config.evaluation.max_detections,
            "split": split,
        },
        "units": {"map50": "fraction", "map50_95": "fraction"},
    }
    metrics_path = destination / "native_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    metadata_payload: dict[str, object] = {
        "task": "detection",
        "backend": "ultralytics_native_val",
        "command": list(sys.argv),
        "git": {"revision": _git_revision()},
        "config": {"digest": _config_digest(config), "resolved": config.to_dict()},
        "inputs": {
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256_file(checkpoint)},
            "dataset_yaml": {"path": str(dataset_yaml.resolve()), "sha256": _sha256_file(dataset_yaml)},
        },
        "validation_args": validation_args,
        "metrics_file": metrics_path.name,
    }
    (destination / "evaluation_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    status(
        f"Ultralytics native evaluation complete: map50={map50:.6f} map50_95={map50_95:.6f} output={metrics_path}",
        log_file=console_path,
    )
    return metrics_path
