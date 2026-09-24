from dataclasses import replace
from pathlib import Path

import pytest

from cabbage_detection.config import ExecutionConfig, load_config
from cabbage_detection.training.loaders import build_torchvision_loaders


torch = pytest.importorskip("torch")


def test_torchvision_loaders_follow_fold1_manifest_membership():
    dataset_root = Path("original_dataset")
    has_official_bundle = any(
        path.is_file() and (path.parent / "images").is_dir()
        for path in dataset_root.rglob("annotation.json")
    ) if dataset_root.is_dir() else False
    if not has_official_bundle:
        pytest.skip("official Mendeley dataset download is not available")
    config = load_config(Path("configs/torchvision/faster_rcnn.yaml"))
    config = replace(config, dataset=replace(config.dataset, root=dataset_root))
    config = replace(config, execution=ExecutionConfig(device="cpu", workers=0, seed=17))
    loaders = build_torchvision_loaders(config)
    assert len(loaders.train.dataset) == 320
    assert len(loaders.validation.dataset) == 46
    assert len(loaders.test.dataset) == 92
    assert loaders.train.batch_size == 16
