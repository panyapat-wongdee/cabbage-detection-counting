from pathlib import Path
import json
from dataclasses import replace

from cabbage_detection.config import AugmentationConfig, DatasetConfig, EvaluationConfig, ExperimentConfig, InitializationConfig, NativeTrainingConfig, SchedulerConfig, load_config
from cabbage_detection.training.runner import run_prediction
from cabbage_detection.training.runner import run_training


def test_run_prediction_uses_adapter_contract(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        model_name="yolo11m", framework="ultralytics", dataset=DatasetConfig("yolo", Path("dataset"), Path("splits.csv"), Path("data.yaml"), None, ("Cabbage",)), initialization=InitializationConfig("yolo11m.pt", 1), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=16,
        epochs=500, confidence_threshold=0.5, iou_threshold=0.5,
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114), scheduler=SchedulerConfig("constant", None, None, 1.0), evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"), native_training=NativeTrainingConfig(500, True, True, False),
        output_dir=tmp_path / "run",
    )

    class StubAdapter:
        def predict(self, source, checkpoint):
            assert source == tmp_path / "image.png"
            assert checkpoint == tmp_path / "best.pt"
            return [{"box": [0, 0, 1, 1], "score": 0.9}]

    monkeypatch.setattr("cabbage_detection.training.runner.create_model", lambda _: StubAdapter())
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert run_prediction(config, tmp_path / "image.png", checkpoint)[0]["score"] == 0.9


def test_run_training_records_failure_metadata(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        model_name="yolo11m", framework="ultralytics", dataset=DatasetConfig("yolo", Path("dataset"), tmp_path / "missing-manifest.csv", Path("data.yaml"), None, ("Cabbage",)), initialization=InitializationConfig("yolo11m.pt", 1), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=16,
        epochs=1, confidence_threshold=0.5, iou_threshold=0.5, output_dir=tmp_path / "run",
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114), scheduler=SchedulerConfig("constant", None, None, 1.0), evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"), native_training=NativeTrainingConfig(500, True, True, False),
    )

    class StubAdapter:
        def train(self):
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr("cabbage_detection.training.runner.create_model", lambda _: StubAdapter())
    try:
        run_training(config, ["train"])
    except RuntimeError as error:
        assert str(error) == "synthetic failure"
    else:
        raise AssertionError("training failure was swallowed")
    import json

    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["error"]["type"] == "RuntimeError"


def test_run_training_uses_smoke_status_from_config(tmp_path: Path, monkeypatch):
    config = replace(
        load_config(Path("configs/ultralytics/yolov8n.yaml")),
        output_dir=tmp_path / "run",
        result_status="smoke",
    )

    class StubAdapter:
        def train(self):
            checkpoint = config.output_dir / "checkpoints" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"best")
            return checkpoint

    monkeypatch.setattr("cabbage_detection.training.runner.create_model", lambda _: StubAdapter())
    run_training(config, ["train", "--config", "configs/smoke/faster_rcnn.yaml"])

    metadata = json.loads((config.output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "succeeded"
    assert metadata["result_status"] == "smoke"


def test_run_training_records_best_last_and_training_artifacts(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        model_name="yolo11m", framework="ultralytics", dataset=DatasetConfig("yolo", Path("dataset"), tmp_path / "missing-manifest.csv", Path("data.yaml"), None, ("Cabbage",)), initialization=InitializationConfig("yolo11m.pt", 1), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=16,
        epochs=1, confidence_threshold=0.5, iou_threshold=0.5, output_dir=tmp_path / "run",
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114), scheduler=SchedulerConfig("constant", None, None, 1.0), evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"), native_training=NativeTrainingConfig(500, True, True, False),
    )

    class StubAdapter:
        def train(self):
            best = config.output_dir / "checkpoints" / "best.pt"
            last = config.output_dir / "checkpoints" / "last.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"best")
            last.write_bytes(b"last")
            (config.output_dir / "logs").mkdir()
            (config.output_dir / "logs" / "training_log.csv").write_text("epoch\n1\n", encoding="utf-8")
            (config.output_dir / "logs" / "console.log").write_text("epoch=1\n", encoding="utf-8")
            (config.output_dir / "plots").mkdir()
            (config.output_dir / "plots" / "training_loss.png").write_bytes(b"png")
            (config.output_dir / "plots" / "validation_metrics.png").write_bytes(b"png")
            return best

    monkeypatch.setattr("cabbage_detection.training.runner.create_model", lambda _: StubAdapter())
    result = run_training(config, ["train"])

    assert result == config.output_dir / "checkpoints" / "best.pt"
    metadata = json.loads((config.output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["outputs"]["best_checkpoint"]["path"] == "checkpoints/best.pt"
    assert metadata["outputs"]["last_checkpoint"]["path"] == "checkpoints/last.pt"
    assert metadata["outputs"]["training_log"]["path"] == "logs/console.log"
    assert metadata["outputs"]["epoch_metrics"]["path"] == "logs/training_log.csv"
    assert metadata["outputs"]["training_loss_plot"]["path"] == "plots/training_loss.png"
    assert metadata["outputs"]["validation_metrics_plot"]["path"] == "plots/validation_metrics.png"


def test_run_training_indexes_ultralytics_native_evidence(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(
        model_name="yolo11m", framework="ultralytics", dataset=DatasetConfig("yolo", Path("dataset"), tmp_path / "missing-manifest.csv", Path("data.yaml"), None, ("Cabbage",)), initialization=InitializationConfig("yolo11m.pt", 1), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=16,
        epochs=1, confidence_threshold=0.5, iou_threshold=0.5, output_dir=tmp_path / "run",
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114), scheduler=SchedulerConfig("constant", None, None, 1.0), evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"), native_training=NativeTrainingConfig(500, True, True, False),
    )

    class StubAdapter:
        def train(self):
            native = config.output_dir / "framework" / "ultralytics"
            (native / "weights").mkdir(parents=True)
            (native / "args.yaml").write_text("epochs: 1\n", encoding="utf-8")
            (native / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
            (native / "weights" / "best.pt").write_bytes(b"best")
            (native / "weights" / "last.pt").write_bytes(b"last")
            return native

    monkeypatch.setattr("cabbage_detection.training.runner.create_model", lambda _: StubAdapter())
    run_training(config, ["train"])

    metadata = json.loads((config.output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["outputs"]["ultralytics_args"]["path"] == "framework/ultralytics/args.yaml"
    assert metadata["outputs"]["ultralytics_results"]["path"] == "framework/ultralytics/results.csv"
    assert metadata["outputs"]["ultralytics_best_checkpoint"]["path"] == "framework/ultralytics/weights/best.pt"
    assert metadata["outputs"]["best_checkpoint"]["path"] == "framework/ultralytics/weights/best.pt"
    assert metadata["outputs"]["last_checkpoint"]["path"] == "framework/ultralytics/weights/last.pt"
