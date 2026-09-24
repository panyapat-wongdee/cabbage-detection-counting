from pathlib import Path
import os

import pytest

from cabbage_detection.config import (
    AugmentationConfig,
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    InitializationConfig,
    NativeTrainingConfig,
    SchedulerConfig,
)
from cabbage_detection.training.ultralytics_trainer import ULTRALYTICS_WEIGHTS, build_ultralytics_train_args, train_ultralytics


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="yolo11m", framework="ultralytics",
        dataset=DatasetConfig("yolo", Path("dataset"), Path("manifest.csv"), Path("data.yaml"), None, ("Cabbage",)),
        initialization=InitializationConfig("yolo11m.pt", 1), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=16, epochs=3,
        confidence_threshold=0.5, iou_threshold=0.5,
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114),
        scheduler=SchedulerConfig("constant", None, None, 1.0),
        evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"),
        native_training=NativeTrainingConfig(500, True, True, False), output_dir=tmp_path / "run",
        execution=ExecutionConfig(device="cpu", workers=8, seed=17),
        checkpoint=CheckpointConfig(metric="train_loss", mode="min"),
    )


def test_ultralytics_args_map_recovered_training_settings(tmp_path: Path):
    native_dir = tmp_path / "run" / "framework"
    args = build_ultralytics_train_args(_config(tmp_path), Path("data.yaml"), native_dir)
    assert args["data"] == "data.yaml"
    assert args["imgsz"] == 512
    assert args["batch"] == 16
    assert args["epochs"] == 3
    assert args["optimizer"] == "SGD"
    assert args["lr0"] == 0.001
    assert args["lrf"] == 1.0
    assert args["momentum"] == 0.9
    assert args["weight_decay"] == 0.0005
    assert args["workers"] == 8
    assert args["device"] == "cpu"
    assert args["patience"] == 500
    assert args["cache"] is True
    assert args["project"] == (tmp_path / "run").as_posix()
    assert args["name"] == "framework"
    assert args["seed"] == 17
    assert args["exist_ok"] is True
    assert args["hsv_h"] == args["hsv_s"] == args["hsv_v"] == 0.0
    assert args["degrees"] == args["translate"] == args["scale"] == 0.0
    assert args["flipud"] == args["fliplr"] == args["mosaic"] == 0.0


@pytest.mark.parametrize(("configured", "native"), [("cuda", "0"), ("cuda:1", "1")])
def test_ultralytics_device_mapping_matches_native_index(tmp_path: Path, configured: str, native: str):
    config = _config(tmp_path)
    config = ExperimentConfig(**{**config.__dict__, "execution": ExecutionConfig(device=configured, workers=8, seed=17)})
    assert build_ultralytics_train_args(config, Path("data.yaml"), tmp_path / "run" / "framework")["device"] == native


def test_ultralytics_weights_identifiers_cover_every_ultralytics_model():
    assert ULTRALYTICS_WEIGHTS == {
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


_RESULTS_CSV_HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n"
)
_RESULTS_CSV_ROW = "1,9.5,2.1,2.9,1.9,0.07,0.58,0.07,0.03,1.7,2.9,1.7,0.0002,0.0002,0.08\n"


def _write_fake_native_run(output_dir: Path) -> None:
    """Populate the native output directory the way a real Ultralytics run would."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.csv").write_text(_RESULTS_CSV_HEADER + _RESULTS_CSV_ROW, encoding="utf-8")
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_bytes(b"best-weights")
    (weights_dir / "last.pt").write_bytes(b"last-weights")


def test_ultralytics_trainer_calls_native_train_once(tmp_path: Path, capsys):
    calls = []

    class FakeModel:
        def train(self, **kwargs):
            calls.append(kwargs)
            _write_fake_native_run(Path(kwargs["project"]) / kwargs["name"])
            return "native-result"

    result = train_ultralytics(_config(tmp_path), FakeModel(), Path("data.yaml"), tmp_path / "run")
    assert result.native_result == "native-result"
    assert len(calls) == 1
    assert calls[0]["data"] == "data.yaml"
    assert calls[0]["project"] == (tmp_path / "run").as_posix()
    assert calls[0]["name"] == "framework"
    assert result.output_dir == tmp_path / "run" / "framework"
    terminal = capsys.readouterr().err
    assert "native training started" in terminal
    assert "native training complete" in terminal
    assert (tmp_path / "run" / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "run" / "logs" / "training_log.csv").is_file()


def test_ultralytics_training_disables_external_logging_by_default(tmp_path: Path, monkeypatch):
    class FakeModel:
        def train(self, **kwargs):
            _write_fake_native_run(Path(kwargs["project"]) / kwargs["name"])
            return None

    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    train_ultralytics(_config(tmp_path), FakeModel(), Path("data.yaml"), tmp_path / "run")
    assert os.environ["WANDB_DISABLED"] == "true"
