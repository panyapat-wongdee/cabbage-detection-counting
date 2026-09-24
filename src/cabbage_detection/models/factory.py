"""Model adapter routing."""

from __future__ import annotations

from ..config import ExperimentConfig
from .base import ModelAdapter
from .torchvision_adapter import TorchvisionAdapter
from .ultralytics_adapter import UltralyticsAdapter


def create_model(config: ExperimentConfig) -> ModelAdapter:
    """Create a framework-specific adapter after validating model routing."""
    ultralytics_models = {
        "yolov8n", "yolov8m", "yolo11n", "yolo11m", "rt-detr-l",
        "yolo12n", "yolo12m", "yolo26n", "yolo26m",
    }
    torchvision_models = {"faster_rcnn", "ssd", "ssdlite", "retinanet", "fcos"}
    if config.model_name in ultralytics_models and config.framework != "ultralytics":
        raise ValueError(f"model {config.model_name} requires ultralytics framework")
    if config.model_name in torchvision_models and config.framework != "torchvision":
        raise ValueError(f"model {config.model_name} requires torchvision framework")
    if config.framework == "ultralytics":
        return UltralyticsAdapter(config)
    if config.framework == "torchvision":
        return TorchvisionAdapter(config)
    raise ValueError(f"unsupported framework: {config.framework}")
