"""Framework-specific loader construction from explicit split manifests."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig
from ..data.official_layout import discover_official_dataset
from ..data.manifests import read_manifest
from ..data.torchvision_dataset import CabbageDetectionDataset, PascalDetectionDataset, collate_detection_batch
from ..data.transforms import build_transforms
from .torchvision_trainer import DetectionLoaders


def build_torchvision_loaders(config: ExperimentConfig) -> DetectionLoaders:
    """Build train/validation/test DataLoaders using manifest membership exactly."""
    if config.framework != "torchvision":
        raise ValueError("torchvision loaders require framework=torchvision")
    if config.dataset.format not in {"pascal_voc", "coco"}:
        raise ValueError("torchvision loaders require dataset.format=pascal_voc or coco")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PyTorch to build torchvision loaders") from error
    manifest = read_manifest(config.split_manifest)
    ids_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split, image_id in manifest.rows:
        if split not in ids_by_split:
            raise ValueError(f"unsupported manifest split: {split}")
        ids_by_split[split].append(image_id)
    if any(not ids for ids in ids_by_split.values()):
        raise ValueError("manifest must contain train, val, and test rows")
    root = config.dataset.root
    if config.dataset.format == "coco":
        if config.dataset.annotations is None:
            discovered = discover_official_dataset(root)
            root = discovered.images_root
            annotations = discovered.annotations_path
        else:
            annotations = config.dataset.annotations
        dataset_factory = lambda ids, training: CabbageDetectionDataset(
            root,
            annotations,
            ids,
            build_transforms(config, training=training),
            cache=config.native_training.cache,
        )
        train = dataset_factory(ids_by_split["train"], True)
        validation = dataset_factory(ids_by_split["val"], False)
        test = dataset_factory(ids_by_split["test"], False)
    else:
        train = PascalDetectionDataset(
            root / "train" / "images",
            root / "train" / config.dataset.labels_dirname,
            ids_by_split["train"],
            build_transforms(config, training=True),
            cache=config.native_training.cache,
        )
        validation = PascalDetectionDataset(
            root / "val" / "images",
            root / "val" / config.dataset.labels_dirname,
            ids_by_split["val"],
            build_transforms(config, training=False),
            cache=config.native_training.cache,
        )
        test = PascalDetectionDataset(
            root / "test" / "images",
            root / "test" / config.dataset.labels_dirname,
            ids_by_split["test"],
            build_transforms(config, training=False),
            cache=config.native_training.cache,
        )
    workers = config.execution.workers or 0
    common: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": config.execution.device != "cpu",
    }
    if workers > 0:
        common["persistent_workers"] = config.native_training.cache
    return DetectionLoaders(
        train=torch.utils.data.DataLoader(train, shuffle=config.native_training.train_shuffle, **common),
        validation=torch.utils.data.DataLoader(
            validation, shuffle=config.native_training.validation_shuffle, **common
        ),
        test=torch.utils.data.DataLoader(test, shuffle=False, **common),
    )


def build_detection_loaders(config: ExperimentConfig) -> DetectionLoaders:
    """Public framework-neutral entry point for configured detection loaders."""
    if config.framework == "torchvision":
        return build_torchvision_loaders(config)
    raise ValueError(f"loader construction is not implemented for framework={config.framework}")
