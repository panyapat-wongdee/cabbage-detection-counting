"""Albumentations boundary used by the torchvision dataset."""

from __future__ import annotations

from typing import Any

from ..config import ExperimentConfig


class DetectionTransform:
    """Pickle-friendly Albumentations transform for DataLoader workers."""

    def __init__(self, compose: Any) -> None:
        self.compose = compose

    def __call__(self, image: Any, target: dict[str, Any]):
        result = self.compose(image=image, bboxes=target["boxes"], labels=target["labels"])
        return result["image"], {
            **target,
            "boxes": list(result["bboxes"]),
            "labels": list(result["labels"]),
        }


def build_transforms(config: ExperimentConfig, training: bool):
    """Build the published resize/augmentation policy lazily.

    The operation parameters mirror the preserved torchvision notebook.
    """
    try:
        import albumentations as A
    except ImportError as error:
        raise RuntimeError("install the optional Albumentations dependency to build transforms") from error

    operations = [A.Resize(height=config.image_size[0], width=config.image_size[1], p=1.0)]
    if training:
        operations.extend(
            [
                A.ShiftScaleRotate(
                    shift_limit=config.augmentation.translate,
                    scale_limit=config.augmentation.scale,
                    rotate_limit=config.augmentation.degrees,
                    border_mode=0,
                    value=config.augmentation.border_value,
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=180 * config.augmentation.hsv_h,
                    sat_shift_limit=255 * config.augmentation.hsv_s,
                    val_shift_limit=255 * config.augmentation.hsv_v,
                    p=0.5,
                ),
                A.VerticalFlip(p=config.augmentation.vertical_flip),
                A.HorizontalFlip(p=config.augmentation.horizontal_flip),
            ]
        )
    if config.augmentation.normalize:
        operations.append(A.Normalize(p=1.0))
    compose = A.Compose(
        operations,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0),
    )

    return DetectionTransform(compose)
