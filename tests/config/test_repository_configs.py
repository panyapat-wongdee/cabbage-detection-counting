from pathlib import Path

import pytest

from dataclasses import replace

from cabbage_detection.config import ExecutionConfig, load_config
from cabbage_detection.data.manifests import SplitManifest, _digest, write_manifest


PRIMARY = tuple(sorted(Path("configs").glob("torchvision/*.yaml"))) + tuple(
    sorted(Path("configs").glob("ultralytics/*.yaml"))
)
NO_AUG_ULTRALYTICS = tuple(sorted(Path("configs/ultralytics/no_augmentation").glob("*.yaml")))
NO_AUG_TORCHVISION = tuple(sorted(Path("configs/torchvision/no_augmentation").glob("*.yaml")))
SMOKE_PARENTS = {
    Path("configs/smoke/faster_rcnn.yaml"): Path("configs/torchvision/faster_rcnn.yaml"),
    Path("configs/smoke/yolov8n.yaml"): Path("configs/ultralytics/yolov8n.yaml"),
}


def _flatten(prefix: str, value: object) -> dict[str, object]:
    if isinstance(value, dict):
        flattened: dict[str, object] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child_prefix, child))
        return flattened
    return {prefix: value}


@pytest.mark.parametrize("path", PRIMARY)
def test_primary_configs_encode_reproduction_augmented_protocol(path: Path) -> None:
    config = load_config(path)
    assert config.image_size == (512, 512)
    assert config.optimizer == "SGD"
    assert config.learning_rate == pytest.approx(0.001)
    assert config.momentum == pytest.approx(0.9)
    assert config.weight_decay == pytest.approx(0.0005)
    assert config.batch_size == 16
    assert config.epochs == 500
    assert config.augmentation.profile == "reproduction_augmented"
    assert config.confidence_threshold == pytest.approx(0.5)
    assert config.iou_threshold == pytest.approx(0.5)
    assert config.evaluation.max_detections == 100
    assert config.dataset.split_manifest == Path("splits/fold1_recovered.csv")
    if config.framework == "torchvision":
        assert config.dataset.format == "coco"
        assert config.dataset.root == Path("original_dataset")
        assert config.dataset.annotations is None
        assert config.provenance["augmentation"] == "repository_protocol"
        assert config.execution.device == "cuda:0"
        assert config.execution.workers == 2
        assert config.execution.seed == 42
        assert config.provenance["execution.workers"] == "repository_default"
        assert config.provenance["execution.device"] == "repository_default"
        assert config.provenance["training.seed"] == "repository_default"
        assert config.native_training.cache is True
        assert config.evaluation.backend == "pycocotools_coco_eval"
        assert config.validation_postprocess is not None
        assert config.validation_postprocess.area_threshold == pytest.approx(0.0)
        assert config.final_postprocess is not None
        assert config.final_postprocess.area_threshold == pytest.approx(
            0.0 if config.model_name == "faster_rcnn" else 100.0
        )
        assert config.native_training.warmup_first_epoch is True
    else:
        assert config.dataset.root == Path("prepared/fold1_yolo")
        assert config.provenance["training.checkpoint_selection"] == "framework_default"
        assert config.execution.device == "cuda:0"
        assert config.execution.workers == 2
        assert config.execution.seed == 42
        assert config.provenance["execution.workers"] == "repository_default"
        assert config.provenance["execution.device"] == "repository_default"
        assert config.provenance["training.seed"] == "repository_default"
        assert config.native_training.cache is True
        assert config.evaluation.backend == "repository_ap_101"
        if config.model_name == "rt-detr-l":
            assert config.evaluation.nms_owner == "none"


@pytest.mark.parametrize(("smoke_path", "parent_path"), SMOKE_PARENTS.items())
def test_smoke_configs_only_override_operational_fields(smoke_path: Path, parent_path: Path) -> None:
    smoke = load_config(smoke_path)
    parent = load_config(parent_path)
    smoke_values = _flatten("", smoke.to_dict())
    parent_values = _flatten("", parent.to_dict())
    differences = {
        key
        for key in smoke_values.keys() | parent_values.keys()
        if smoke_values.get(key) != parent_values.get(key)
    }
    allowed = {
        "training.batch_size",
        "training.epochs",
        "native_training.cache",
        "artifacts.output_dir",
        "artifacts.result_status",
        "execution.workers",
        "provenance.training.epochs",
        "provenance.training.batch_size",
        "provenance.native_training.cache",
    }

    assert differences <= allowed
    assert smoke.epochs == 1
    assert smoke.batch_size == 2
    assert smoke.execution.workers == 0
    assert smoke.native_training.cache is False
    assert smoke.result_status == "smoke"
    assert smoke.output_dir == Path("runs/reproduced/smoke") / smoke.model_name
    assert smoke.provenance["training.epochs"] == "repository_default"
    assert smoke.provenance["training.batch_size"] == "repository_default"
    assert smoke.provenance["native_training.cache"] == "repository_default"


def test_recovered_ultralytics_no_augmentation_matrix_is_complete() -> None:
    assert {load_config(path).model_name for path in NO_AUG_ULTRALYTICS} == {
        "yolov8n",
        "yolov8m",
        "yolo11n",
        "yolo11m",
        "rt-detr-l",
    }
    for path in NO_AUG_ULTRALYTICS:
        config = load_config(path)
        assert config.augmentation.profile == "recovered_no_augmentation"
        assert config.augmentation.normalize is False
        assert config.execution.device == "cuda:0"
        assert config.execution.workers == 2
        assert config.execution.seed == 42
        assert config.provenance["execution.device"] == "repository_default"
        assert config.provenance["augmentation"] == "historical_code"
        assert config.provenance["training.checkpoint_selection"] == "framework_default"
        assert config.provenance["execution.workers"] == "repository_default"
        assert config.provenance["training.seed"] == "repository_default"
        assert config.native_training.cache is True
        assert config.native_training.patience == 500


def test_torchvision_no_augmentation_matrix_is_complete() -> None:
    assert {load_config(path).model_name for path in NO_AUG_TORCHVISION} == {
        "faster_rcnn",
        "ssd",
        "retinanet",
        "fcos",
        "ssdlite",
    }
    for path in NO_AUG_TORCHVISION:
        config = load_config(path)
        assert config.framework == "torchvision"
        assert config.dataset.format == "coco"
        assert config.dataset.root == Path("original_dataset")
        assert config.dataset.annotations is None
        assert config.image_size == (512, 512)
        assert config.optimizer == "SGD"
        assert config.learning_rate == pytest.approx(0.001)
        assert config.momentum == pytest.approx(0.9)
        assert config.weight_decay == pytest.approx(0.0005)
        assert config.batch_size == 16
        assert config.epochs == 500
        assert config.augmentation.profile == "recovered_no_augmentation"
        assert config.augmentation.horizontal_flip == 0.0
        assert config.augmentation.vertical_flip == 0.0
        assert config.augmentation.translate == 0.0
        assert config.augmentation.scale == 0.0
        assert config.augmentation.degrees == 0.0
        assert config.augmentation.hsv_h == 0.0
        assert config.augmentation.hsv_s == 0.0
        assert config.augmentation.hsv_v == 0.0
        assert config.augmentation.mosaic == 0.0
        assert config.augmentation.normalize is True
        assert config.provenance["augmentation"] == "historical_code"
        assert config.scheduler.name == "step_lr"
        assert config.scheduler.step_size == 1000
        assert config.scheduler.gamma == pytest.approx(0.1)
        assert config.confidence_threshold == pytest.approx(0.5)
        assert config.iou_threshold == pytest.approx(0.5)
        assert config.evaluation.max_detections == 100
        assert config.execution.device == "cuda:0"
        assert config.execution.workers == 2
        assert config.execution.seed == 42
        assert config.provenance["execution.device"] == "repository_default"
        assert config.native_training.cache is True
        assert config.native_training.patience == 500


def test_torchvision_loader_discovers_bundle_from_outer_root(tmp_path: Path):
    pytest.importorskip("torch")

    bundle = tmp_path / "An annotated image dataset of cabbages for instance segmentation"
    images = bundle / "images" / "OkinaSP" / "Kaizu" / "202010"
    images.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png"):
        (images / name).write_bytes(b"image")
    annotations = bundle / "annotation.json"
    annotations.write_text(
        "{\"images\": ["
        "{\"id\": 1, \"file_name\": \"a.png\", \"path\": \"/OkinaSP/Kaizu/202010/a.png\", \"width\": 10, \"height\": 10},"
        "{\"id\": 2, \"file_name\": \"b.png\", \"path\": \"/OkinaSP/Kaizu/202010/b.png\", \"width\": 10, \"height\": 10},"
        "{\"id\": 3, \"file_name\": \"c.png\", \"path\": \"/OkinaSP/Kaizu/202010/c.png\", \"width\": 10, \"height\": 10}],"
        "\"annotations\": []}",
        encoding="utf-8",
    )
    rows = (("train", "a"), ("val", "b"), ("test", "c"))
    manifest = tmp_path / "fold.csv"
    write_manifest(SplitManifest(rows, "fixture", _digest(rows)), manifest)
    config = load_config(Path("configs/torchvision/faster_rcnn.yaml"))
    config = replace(
        config,
        dataset=replace(config.dataset, root=tmp_path, split_manifest=manifest, annotations=None),
        execution=ExecutionConfig(device="cpu", workers=0, seed=17),
    )

    from cabbage_detection.training.loaders import build_torchvision_loaders

    loaders = build_torchvision_loaders(config)

    assert loaders.train.dataset.cache_enabled is True
    assert loaders.validation.dataset.cache_enabled is True
    assert loaders.test.dataset.cache_enabled is True

    assert loaders.train.dataset.samples[0].path.is_relative_to(bundle / "images")

    cached_config = replace(config, execution=ExecutionConfig(device="cpu", workers=1, seed=17))
    cached_loaders = build_torchvision_loaders(cached_config)
    assert cached_loaders.train.persistent_workers is True
