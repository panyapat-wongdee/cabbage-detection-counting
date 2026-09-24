from pathlib import Path
from dataclasses import replace

from cabbage_detection.config import ExperimentConfig
from cabbage_detection.config import AugmentationConfig, DatasetConfig, EvaluationConfig, InitializationConfig, NativeTrainingConfig, SchedulerConfig
from cabbage_detection.models.ultralytics_adapter import UltralyticsAdapter
from cabbage_detection.evaluation.records import PostprocessingMetadata


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="yolov8n",
        framework="ultralytics",
        dataset=DatasetConfig("yolo", Path("data"), Path("manifest.csv"), Path("data.yaml"), None, ("Cabbage",)),
        initialization=InitializationConfig("yolov8n.pt", 1),
        image_size=(512, 512),
        optimizer="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=16,
        epochs=1,
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114),
        scheduler=SchedulerConfig("constant", None, None, 1.0),
        evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics"),
        native_training=NativeTrainingConfig(500, True, True, False),
        confidence_threshold=0.5,
        iou_threshold=0.5,
        output_dir=Path("runs/test"),
    )


class _Array:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return _Array(self.value[index])

    def tolist(self):
        return self.value

    def item(self):
        return self.value


class _Box:
    xyxy = _Array([[1, 2, 10, 20]])
    conf = _Array([0.9])
    cls = _Array([0])


class _Result:
    path = "frame_001.jpg"
    boxes = [_Box()]


def test_prediction_conversion_preserves_image_id_and_native_filtering():
    metadata = PostprocessingMetadata("ultralytics", "ultralytics", 0.5, 0.5)
    records = UltralyticsAdapter._canonical_predictions([_Result()], Path("source.jpg"), metadata)
    assert records[0].image_id == "frame_001"
    assert records[0].detections[0].box == (1.0, 2.0, 10.0, 20.0)
    assert records[0].detections[0].score == 0.9
    assert records[0].detections[0].class_id == 1
    assert records[0].postprocessing == metadata


def test_adapter_prediction_passes_execution_device_once(monkeypatch, tmp_path: Path):
    calls = []

    class FakeModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return [_Result()]

    adapter = UltralyticsAdapter(_config())
    monkeypatch.setattr(adapter, "_load", lambda: FakeModel())
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(adapter, "_load_checkpoint", lambda path: FakeModel())
    adapter.predict(Path("source.jpg"), checkpoint)
    assert calls[0]["device"] == "cpu"
    assert calls[0]["imgsz"] == 512
    assert calls[0]["conf"] == 0.5
    assert calls[0]["iou"] == 0.5
    assert calls[0]["max_det"] == 100


def test_rtdetr_prediction_records_no_nms(monkeypatch, tmp_path: Path):
    class FakeModel:
        def predict(self, **kwargs):
            return [_Result()]

    config = replace(
        _config(),
        model_name="rt-detr-l",
        initialization=type(_config().initialization)("rtdetr-l.pt", 1),
        evaluation=type(_config().evaluation)(100, 0.0, "ultralytics", "none"),
    )
    adapter = UltralyticsAdapter(config)
    monkeypatch.setattr(adapter, "_load_checkpoint", lambda path: FakeModel())
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    record = adapter.predict(Path("source.jpg"), checkpoint)[0]

    assert record.postprocessing is not None
    assert record.postprocessing.nms_owner == "none"
    assert record.postprocessing.native_nms_applied is False


def test_adapter_training_uses_framework_subdirectory(monkeypatch, tmp_path: Path):
    class FakeModel:
        pass

    captured = {}

    def fake_train(config, model, dataset_yaml, run_dir):
        captured["run_dir"] = run_dir
        return type("Result", (), {"output_dir": run_dir})()

    adapter = UltralyticsAdapter(replace(_config(), output_dir=tmp_path / "run"))
    monkeypatch.setattr(adapter, "_load", lambda: FakeModel())
    monkeypatch.setattr("cabbage_detection.models.ultralytics_adapter.train_ultralytics", fake_train)
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\n", encoding="utf-8")
    adapter.train(dataset_yaml)
    assert captured["run_dir"] == tmp_path / "run"
