import hashlib
from pathlib import Path

import pytest

from cabbage_detection.config import (
    AugmentationConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    InitializationConfig,
    NativeTrainingConfig,
    SchedulerConfig,
)
from cabbage_detection.training.artifact_writer import EPOCH_COLUMNS
from cabbage_detection.training.ultralytics_artifacts import write_ultralytics_artifacts


_RESULTS_CSV_HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n"
)
# Epoch 2 has the highest fitness (0.1*map50 + 0.9*map50_95), so it must be
# the epoch reported as best, not simply the last row.
_RESULTS_CSV_ROWS = (
    "1,9.5,2.1,2.9,1.9,0.07,0.58,0.07,0.02,1.7,2.9,1.7,0.0002,0.0002,0.08\n"
    "2,18.9,1.8,2.5,1.7,0.09,0.67,0.40,0.30,1.6,2.6,1.6,0.0004,0.0004,0.06\n"
    "3,28.1,1.7,2.4,1.6,0.10,0.70,0.35,0.25,1.5,2.5,1.5,0.0003,0.0003,0.05\n"
)


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="yolov8n",
        framework="ultralytics",
        dataset=DatasetConfig("yolo", Path("dataset"), tmp_path / "missing-manifest.csv", Path("data.yaml"), None, ("Cabbage",)),
        initialization=InitializationConfig("yolov8n.pt", 1),
        image_size=(512, 512),
        optimizer="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=16,
        epochs=3,
        confidence_threshold=0.5,
        iou_threshold=0.5,
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114),
        scheduler=SchedulerConfig("constant", None, None, 1.0),
        evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"),
        native_training=NativeTrainingConfig(500, True, True, False),
        output_dir=tmp_path / "run",
    )


def _write_native_run(native_dir: Path) -> tuple[bytes, bytes]:
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "results.csv").write_text(_RESULTS_CSV_HEADER + _RESULTS_CSV_ROWS, encoding="utf-8")
    weights_dir = native_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_bytes = b"best-checkpoint-bytes"
    last_bytes = b"last-checkpoint-bytes"
    (weights_dir / "best.pt").write_bytes(best_bytes)
    (weights_dir / "last.pt").write_bytes(last_bytes)
    return best_bytes, last_bytes


def test_write_ultralytics_artifacts_matches_torchvision_shape(tmp_path: Path):
    config = _config(tmp_path)
    run_dir = config.output_dir
    native_dir = run_dir / "framework"
    best_bytes, last_bytes = _write_native_run(native_dir)

    write_ultralytics_artifacts(config, run_dir, native_dir)

    training_log = (run_dir / "logs" / "training_log.csv").read_text(encoding="utf-8").splitlines()
    assert training_log[0].split(",") == list(EPOCH_COLUMNS)
    assert len(training_log) == 1 + 3  # header + 3 epochs

    import json

    summary = json.loads((run_dir / "logs" / "best_epoch.json").read_text(encoding="utf-8"))
    assert summary["epoch"] == 2
    assert summary["selection_metric"] == "ultralytics_fitness"
    assert summary["selection_mode"] == "max"
    assert summary["checkpoint"] == "checkpoints/best.pt"
    assert summary["last_checkpoint"] == "checkpoints/last.pt"
    assert summary["sha256"] == hashlib.sha256(best_bytes).hexdigest()
    assert summary["last_sha256"] == hashlib.sha256(last_bytes).hexdigest()

    assert (run_dir / "checkpoints" / "best.pt").read_bytes() == best_bytes
    assert (run_dir / "checkpoints" / "last.pt").read_bytes() == last_bytes

    assert (run_dir / "plots" / "training_loss.png").is_file()
    assert (run_dir / "plots" / "validation_metrics.png").is_file()

    console_log = (run_dir / "logs" / "console.log").read_text(encoding="utf-8")
    assert "epoch=1" in console_log
    assert "ultralytics artifacts complete" in console_log


def test_write_ultralytics_artifacts_requires_results_csv(tmp_path: Path):
    config = _config(tmp_path)
    native_dir = config.output_dir / "framework"
    native_dir.mkdir(parents=True)
    (native_dir / "weights").mkdir()
    (native_dir / "weights" / "best.pt").write_bytes(b"x")
    (native_dir / "weights" / "last.pt").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="results.csv"):
        write_ultralytics_artifacts(config, config.output_dir, native_dir)


def test_write_ultralytics_artifacts_requires_weights(tmp_path: Path):
    config = _config(tmp_path)
    native_dir = config.output_dir / "framework"
    native_dir.mkdir(parents=True)
    (native_dir / "results.csv").write_text(_RESULTS_CSV_HEADER + _RESULTS_CSV_ROWS, encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="weights"):
        write_ultralytics_artifacts(config, config.output_dir, native_dir)


# RT-DETR reports a different training-loss decomposition (giou/cls/l1) than
# YOLO (box/cls/dfl); the writer must sum whatever train/*loss columns are
# actually present instead of a YOLO-specific fixed set.
_RTDETR_RESULTS_CSV_HEADER = (
    "epoch,time,train/giou_loss,train/cls_loss,train/l1_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/giou_loss,val/cls_loss,val/l1_loss,lr/pg0,lr/pg1,lr/pg2\n"
)
_RTDETR_RESULTS_CSV_ROW = "1,9.5,1.1,0.9,0.6,0.07,0.58,0.07,0.03,1.0,0.8,0.5,0.0002,0.0002,0.08\n"


def test_write_ultralytics_artifacts_sums_rtdetr_loss_columns(tmp_path: Path):
    config = _config(tmp_path)
    run_dir = config.output_dir
    native_dir = run_dir / "framework"
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "results.csv").write_text(_RTDETR_RESULTS_CSV_HEADER + _RTDETR_RESULTS_CSV_ROW, encoding="utf-8")
    weights_dir = native_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    (weights_dir / "last.pt").write_bytes(b"last")

    write_ultralytics_artifacts(config, run_dir, native_dir)

    training_log = (run_dir / "logs" / "training_log.csv").read_text(encoding="utf-8").splitlines()
    row = dict(zip(training_log[0].split(","), training_log[1].split(",")))
    assert float(row["train_loss_mean"]) == pytest.approx(1.1 + 0.9 + 0.6)


def test_write_ultralytics_artifacts_falls_back_to_last_epoch_when_no_epoch_is_scored(tmp_path: Path):
    """When no epoch reports a finite mAP (e.g. a validation-less smoke run),
    the last epoch must be reported as best, not accidentally the first one
    via a ``None == None`` match."""
    config = _config(tmp_path)
    run_dir = config.output_dir
    native_dir = run_dir / "framework"
    native_dir.mkdir(parents=True, exist_ok=True)
    unscored_rows = (
        "1,9.5,2.1,2.9,1.9,,,,,1.7,2.9,1.7,0.0002,0.0002,0.08\n"
        "2,18.9,1.8,2.5,1.7,,,,,1.6,2.6,1.6,0.0004,0.0004,0.06\n"
        "3,28.1,1.7,2.4,1.6,,,,,1.5,2.5,1.5,0.0003,0.0003,0.05\n"
    )
    (native_dir / "results.csv").write_text(_RESULTS_CSV_HEADER + unscored_rows, encoding="utf-8")
    weights_dir = native_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"best")
    (weights_dir / "last.pt").write_bytes(b"last")

    write_ultralytics_artifacts(config, run_dir, native_dir)

    import json

    summary = json.loads((run_dir / "logs" / "best_epoch.json").read_text(encoding="utf-8"))
    assert summary["epoch"] == 3
    assert summary["selection_value"] is None
