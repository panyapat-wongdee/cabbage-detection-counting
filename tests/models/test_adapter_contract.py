import pytest

from cabbage_detection.config import (
    AugmentationConfig, DatasetConfig, EvaluationConfig, ExperimentConfig,
    InitializationConfig, NativeTrainingConfig, SchedulerConfig,
)
from cabbage_detection.models.factory import create_model


def _config(model_name: str, framework: str) -> ExperimentConfig:
    from pathlib import Path

    ultralytics = framework == "ultralytics"
    return ExperimentConfig(
        model_name=model_name,
        framework=framework,
        dataset=DatasetConfig("yolo" if ultralytics else "pascal_voc", Path("dataset"), Path("splits.csv"), Path("data.yaml") if ultralytics else None, None if ultralytics else "labels", ("Cabbage",) if ultralytics else ("background", "cabbage")),
        initialization=InitializationConfig({"yolo11m": "yolo11m.pt", "rt-detr-l": "rtdetr-l.pt", "faster_rcnn": "COCO_V1", "ssdlite": "COCO_V1"}[model_name], 1 if ultralytics else 2),
        image_size=(512, 512),
        optimizer="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=16,
        epochs=500,
        augmentation=AugmentationConfig("reproduction_augmented", 0.5, 0.5, 0.1, 0.1, 15.0, 0.1, 0.05, 0.05, 0.0, True, 114),
        scheduler=SchedulerConfig("constant", None, None, 1.0),
        evaluation=EvaluationConfig(100, 0.0, "ultralytics" if ultralytics else "repository", "ultralytics" if ultralytics else "torchvision"),
        native_training=NativeTrainingConfig(500, ultralytics, True, False),
        confidence_threshold=0.5,
        iou_threshold=0.5,
        output_dir=Path("results/run"),
    )


@pytest.mark.parametrize(
    ("model_name", "framework"),
    [("yolo11m", "ultralytics"), ("rt-detr-l", "ultralytics"), ("faster_rcnn", "torchvision"), ("ssdlite", "torchvision")],
)
def test_create_model_routes_supported_models(model_name: str, framework: str):
    adapter = create_model(_config(model_name, framework))

    assert adapter.model_name == model_name
    assert adapter.framework == framework


def test_create_model_rejects_framework_model_mismatch():
    with pytest.raises(ValueError, match="framework"):
        create_model(_config("yolo11m", "torchvision"))
