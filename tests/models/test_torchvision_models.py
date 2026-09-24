from pathlib import Path

import pytest

from cabbage_detection.config import ExperimentConfig
from cabbage_detection.config import (
    AugmentationConfig,
    DatasetConfig,
    EvaluationConfig,
    InitializationConfig,
    NativeTrainingConfig,
    SchedulerConfig,
)
from dataclasses import replace

from cabbage_detection.models.torchvision_models import build_torchvision_model, detector_input_size


def _config(model_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        model_name=model_name,
        framework="torchvision",
        dataset=DatasetConfig("pascal_voc", Path("data"), Path("manifest.csv"), None, "labels", ("background", "cabbage")),
        initialization=InitializationConfig("COCO_V1", 2),
        image_size=(512, 512),
        optimizer="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=16,
        epochs=1,
        augmentation=AugmentationConfig("reproduction_augmented", 0.5, 0.5, 0.1, 0.1, 15.0, 0.1, 0.05, 0.05, 0.0, True, 114),
        scheduler=SchedulerConfig("step_lr", 1000, 0.1, None),
        evaluation=EvaluationConfig(100, 1.0, "repository", "repository"),
        native_training=NativeTrainingConfig(500, False, True, False),
        confidence_threshold=0.5,
        iou_threshold=0.5,
        output_dir=Path("runs/test"),
    )


def test_model_builder_rejects_wrong_framework_without_importing_torchvision():
    config = _config("faster_rcnn")
    config = ExperimentConfig(**{**config.__dict__, "framework": "ultralytics"})
    with pytest.raises(ValueError, match="framework=torchvision"):
        build_torchvision_model(config)


@pytest.mark.parametrize("model_name", ["faster_rcnn", "ssd", "retinanet", "fcos", "ssdlite"])
def test_model_builder_replaces_head_for_two_classes(model_name: str):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    model = build_torchvision_model(_config(model_name), weights=None)
    if model_name == "faster_rcnn":
        assert model.roi_heads.box_predictor.cls_score.out_features == 2
    elif model_name == "ssd":
        head = model.head.classification_head
        assert head.module_list[0].out_channels // model.anchor_generator.num_anchors_per_location()[0] == 2
    elif model_name == "ssdlite":
        assert model.head.classification_head.num_columns == 2
    else:
        assert model.head.classification_head.num_classes == 2


def test_model_builder_uses_configured_initialization(monkeypatch):
    config = _config("faster_rcnn")
    config = ExperimentConfig(**{**config.__dict__, "initialization": InitializationConfig("unsupported", 2)})
    with pytest.raises(ValueError, match="unsupported torchvision weights"):
        build_torchvision_model(config)


def test_ssdlite_does_not_request_a_separate_backbone(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    import torchvision.models.detection as detection

    calls = {}
    original = detection.ssdlite320_mobilenet_v3_large

    def constructor(**kwargs):
        calls.update(kwargs)
        return original(weights=None, weights_backbone=None)

    monkeypatch.setattr(detection, "ssdlite320_mobilenet_v3_large", constructor)
    build_torchvision_model(_config("ssdlite"), weights=None)

    assert calls["weights_backbone"] is None


# The configured image_size resize is followed by the detector's own
# transform. These are the inputs the released runs' backbones received.
FRAMEWORK_DEFAULT_INPUT = {
    "faster_rcnn": (800, 800),
    "retinanet": (800, 800),
    "fcos": (800, 800),
    "ssd": (300, 300),
    "ssdlite": (320, 320),
}


@pytest.mark.parametrize("model_name", sorted(FRAMEWORK_DEFAULT_INPUT))
def test_framework_default_detector_input_is_what_released_runs_used(model_name: str):
    pytest.importorskip("torchvision")
    model = build_torchvision_model(_config(model_name), weights=None)
    assert detector_input_size(model, (512, 512)) == FRAMEWORK_DEFAULT_INPUT[model_name]


@pytest.mark.parametrize("model_name", sorted(FRAMEWORK_DEFAULT_INPUT))
def test_image_size_detector_input_keeps_the_configured_size(model_name: str):
    pytest.importorskip("torchvision")
    config = replace(_config(model_name), detector_input="image_size")
    model = build_torchvision_model(config, weights=None)
    assert detector_input_size(model, (512, 512)) == (512, 512)
    if model_name == "ssd":
        # SSD300's steps place default boxes for a 300 px image.
        assert model.anchor_generator.steps is None
