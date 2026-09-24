from pathlib import Path
import importlib.util
import pickle
from dataclasses import replace

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
from cabbage_detection.data.transforms import build_transforms


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        model_name="faster_rcnn", framework="torchvision",
        dataset=DatasetConfig("pascal_voc", Path("data"), Path("manifest.csv"), None, "labels", ("background", "cabbage")),
        initialization=InitializationConfig("COCO_V1", 2), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=16, epochs=1,
        augmentation=AugmentationConfig("reproduction_augmented", 0.5, 0.5, 0.1, 0.1, 15.0, 0.1, 0.05, 0.05, 0.0, True, 114),
        scheduler=SchedulerConfig("step_lr", 1000, 0.1, None),
        evaluation=EvaluationConfig(100, 1.0, "repository", "repository"),
        native_training=NativeTrainingConfig(500, False, True, False),
        confidence_threshold=0.5, iou_threshold=0.5, output_dir=Path("runs/test"),
    )


def test_transform_builder_reports_optional_dependency_when_unavailable():
    if importlib.util.find_spec("albumentations") is None:
        with pytest.raises(RuntimeError, match="optional Albumentations"):
            build_transforms(_config(), training=True)
    else:
        assert callable(build_transforms(_config(), training=True))


def test_published_transform_parameters_match_notebook():
    if importlib.util.find_spec("albumentations") is None:
        pytest.skip("albumentations is optional")
    transform = build_transforms(_config(), training=True)
    compose = transform.compose
    by_name = {type(item).__name__: item for item in compose.transforms}
    assert [type(item).__name__ for item in compose.transforms] == [
        "Resize",
        "ShiftScaleRotate",
        "HueSaturationValue",
        "VerticalFlip",
        "HorizontalFlip",
        "Normalize",
    ]
    assert by_name["ShiftScaleRotate"].shift_limit_x == (-0.1, 0.1)
    assert by_name["ShiftScaleRotate"].scale_limit == (0.9, 1.1)
    assert by_name["ShiftScaleRotate"].rotate_limit == (-15.0, 15.0)
    assert by_name["ShiftScaleRotate"].value == 114.0
    assert by_name["HueSaturationValue"].hue_shift_limit == (-18.0, 18.0)
    assert by_name["HueSaturationValue"].sat_shift_limit == (-12.75, 12.75)
    assert by_name["HueSaturationValue"].val_shift_limit == (-12.75, 12.75)
    assert by_name["ShiftScaleRotate"].p == pytest.approx(0.5)
    assert by_name["HueSaturationValue"].p == pytest.approx(0.5)
    assert by_name["VerticalFlip"].p == pytest.approx(0.5)
    assert by_name["HorizontalFlip"].p == pytest.approx(0.5)
    assert "Normalize" in by_name
    assert compose.processors["bboxes"].params.label_fields == ["labels"]


def test_no_augmentation_profile_disables_stochastic_transforms():
    if importlib.util.find_spec("albumentations") is None:
        pytest.skip("albumentations is optional")
    config = replace(
        _config(),
        augmentation=replace(
            _config().augmentation,
            profile="recovered_no_augmentation",
            horizontal_flip=0.0,
            vertical_flip=0.0,
            translate=0.0,
            scale=0.0,
            degrees=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
        ),
    )
    by_name = {type(item).__name__: item for item in build_transforms(config, training=True).compose.transforms}
    assert by_name["VerticalFlip"].p == pytest.approx(0.0)
    assert by_name["HorizontalFlip"].p == pytest.approx(0.0)


def test_evaluation_keeps_configured_normalization():
    if importlib.util.find_spec("albumentations") is None:
        pytest.skip("albumentations is optional")
    transform = build_transforms(_config(), training=False)
    assert [type(item).__name__ for item in transform.compose.transforms] == ["Resize", "Normalize"]


def test_transform_is_picklable_for_multiprocess_dataloader():
    if importlib.util.find_spec("albumentations") is None:
        pytest.skip("albumentations is optional")

    transform = build_transforms(_config(), training=True)

    assert pickle.loads(pickle.dumps(transform)).compose.transforms
