"""Measure images that become empty after configured torchvision transforms."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..config import load_config
from ..training.loaders import build_torchvision_loaders
from ..progress import progress


def _raw_sample(sample: Any) -> tuple[str, int]:
    if isinstance(sample, tuple):
        return str(sample[0]), len(sample[2])
    return str(sample.image_id), len(sample.annotations)


def probe_augmented_empty_targets(config_path: Path, repeats: int, seed: int) -> dict[str, Any]:
    """Probe transformed training samples without changing dataset files."""
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    config = load_config(Path(config_path))
    if config.framework != "torchvision":
        raise ValueError("augmented empty-target audit requires framework=torchvision")
    loaders = build_torchvision_loaders(config)
    dataset = loaders.train.dataset

    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError("install numpy and torch for the augmented empty-target audit") from error

    raw_empty_ids: list[str] = []
    image_ids: list[str] = []
    for sample in progress(dataset.samples, description="inspect raw targets", total=len(dataset.samples)):
        image_id, box_count = _raw_sample(sample)
        image_ids.append(image_id)
        if box_count == 0:
            raw_empty_ids.append(image_id)

    affected_ids: set[str] = set()
    per_repeat: list[dict[str, Any]] = []
    transformed_empty_images = 0
    for repeat in progress(range(repeats), description="probe augmentation", total=repeats):
        repeat_seed = int(seed) + repeat
        random.seed(repeat_seed)
        np.random.seed(repeat_seed % (2**32))
        torch.manual_seed(repeat_seed)
        empty_ids: list[str] = []
        for index, image_id in enumerate(
            progress(image_ids, description=f"transform repeat {repeat + 1}/{repeats}", total=len(image_ids))
        ):
            _, target = dataset[index]
            boxes = target.get("boxes")
            if boxes is None or len(boxes) == 0:
                empty_ids.append(image_id)
                affected_ids.add(image_id)
        transformed_empty_images += len(empty_ids)
        per_repeat.append({"repeat": repeat, "seed": repeat_seed, "empty_image_ids": empty_ids})

    return {
        "config": str(Path(config_path)),
        "raw_image_count": len(image_ids),
        "raw_empty_images": len(raw_empty_ids),
        "raw_empty_image_ids": raw_empty_ids,
        "transformed_empty_images": transformed_empty_images,
        "affected_image_ids": sorted(affected_ids),
        "repeats": repeats,
        "seed": int(seed),
        "per_repeat": per_repeat,
    }
