"""Ultralytics argument mapping kept separate from model loading."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..progress import status
from .ultralytics_artifacts import write_ultralytics_artifacts


ULTRALYTICS_WEIGHTS = {
    "yolov8n": "yolov8n.pt",
    "yolov8m": "yolov8m.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11m": "yolo11m.pt",
    "rt-detr-l": "rtdetr-l.pt",
    "yolo12n": "yolo12n.pt",
    "yolo12m": "yolo12m.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26m": "yolo26m.pt",
}


@dataclass(frozen=True)
class UltralyticsTrainingResult:
    output_dir: Path
    native_result: Any
    train_args: dict[str, object]


def build_ultralytics_train_args(
    config: ExperimentConfig, dataset_yaml: Path, run_dir: Path
) -> dict[str, object]:
    """Map the recovered script settings to one native ``model.train`` call."""
    if config.framework != "ultralytics":
        raise ValueError("Ultralytics training requires framework=ultralytics")
    if config.model_name not in ULTRALYTICS_WEIGHTS:
        raise ValueError(f"unsupported Ultralytics model: {config.model_name}")
    run_dir = Path(run_dir)
    device = config.execution.device
    if device == "cuda":
        device = "0"
    elif device.startswith("cuda:"):
        device = device.split(":", 1)[1]
    args: dict[str, object] = {
        "data": str(Path(dataset_yaml)),
        "name": run_dir.name,
        "project": run_dir.parent.resolve().as_posix(),
        "batch": config.batch_size,
        "epochs": config.epochs,
        "patience": config.native_training.patience,
        "imgsz": config.image_size[0],
        "device": device,
        "lr0": config.learning_rate,
        "lrf": config.scheduler.final_lr_factor if config.scheduler.final_lr_factor is not None else 1.0,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer,
        "cache": config.native_training.cache,
        # The run directory is created exist_ok=False by create_run() before
        # any framework adapter runs, so a second exist_ok=False here would
        # make Ultralytics auto-increment name="framework" to "framework2"
        # rather than writing directly into it.
        "exist_ok": True,
        "hsv_h": config.augmentation.hsv_h,
        "hsv_s": config.augmentation.hsv_s,
        "hsv_v": config.augmentation.hsv_v,
        "degrees": config.augmentation.degrees,
        "translate": config.augmentation.translate,
        "scale": config.augmentation.scale,
        "flipud": config.augmentation.vertical_flip,
        "fliplr": config.augmentation.horizontal_flip,
        "mosaic": config.augmentation.mosaic,
    }
    if config.execution.workers is not None:
        args["workers"] = config.execution.workers
    if config.execution.seed is not None:
        args["seed"] = config.execution.seed
    return args


def train_ultralytics(
    config: ExperimentConfig, model: Any, dataset_yaml: Path, run_dir: Path
) -> UltralyticsTrainingResult:
    """Invoke the native trainer once and return its call contract."""
    run_root = Path(run_dir)
    # Deliberately not created ahead of time: with exist_ok=True Ultralytics
    # creates "framework/" itself, so the native output lands there directly
    # instead of behind an auto-incremented sibling.
    output_dir = run_root / "framework"
    # A logged-in W&B installation must not exfiltrate a local smoke/full run
    # unless the caller explicitly opts in with WANDB_DISABLED=false.
    os.environ.setdefault("WANDB_DISABLED", "true")
    args = build_ultralytics_train_args(config, Path(dataset_yaml), output_dir)
    status(
        f"ultralytics native training started: model={config.model_name} output={output_dir}"
    )
    native_result = model.train(**args)
    save_dir = getattr(native_result, "save_dir", None)
    if save_dir is not None:
        candidate = Path(str(save_dir)).resolve()
        try:
            candidate.relative_to(run_root.resolve())
        except ValueError as error:
            raise ValueError("Ultralytics native output escaped the run directory") from error
        output_dir = candidate
    status(f"ultralytics native training complete: output={output_dir}")
    write_ultralytics_artifacts(config, run_root, output_dir)
    return UltralyticsTrainingResult(output_dir, native_result, args)
