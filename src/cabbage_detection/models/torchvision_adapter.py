"""Lazy torchvision adapter boundary."""

from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig, PostprocessStageConfig
from ..evaluation.records import PredictionRecord
from ..evaluation.records import Detection, PostprocessingMetadata
from ..evaluation.postprocess import filter_torchvision_outputs
from ..data.transforms import build_transforms
from ..training.loaders import build_detection_loaders
from ..training.torchvision_trainer import evaluate_torchvision_map50_95, train_torchvision
from ..progress import status
from .torchvision_models import build_torchvision_model
from .base import ModelAdapter


def apply_native_score_threshold(model: object, threshold: float) -> None:
    """Lower the detector's own score gate so AP can see the full score range.

    This changes model behaviour: the published runs keep each detector's
    torchvision default. Callers must record it as a deviation.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("native score threshold must be between 0 and 1")
    holder = getattr(model, "roi_heads", None) or model
    if not hasattr(holder, "score_thresh"):
        raise ValueError("detector does not expose score_thresh")
    holder.score_thresh = float(threshold)


def _native_postprocess_settings(model: object) -> dict[str, object]:
    """Read the detector's own score/NMS settings.

    torchvision detectors filter and run NMS inside ``postprocess_detections``
    before repository post-processing sees an output, so these values are part
    of the prediction's provenance. Faster R-CNN keeps them on ``roi_heads``.
    """
    holder = getattr(model, "roi_heads", None) or model
    return {
        "score_thresh": _optional_float(getattr(holder, "score_thresh", None)),
        "nms_thresh": _optional_float(getattr(holder, "nms_thresh", None)),
        "detections_per_img": _optional_int(getattr(holder, "detections_per_img", None)),
    }


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


class TorchvisionAdapter(ModelAdapter):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        if config.framework != "torchvision":
            raise ValueError("torchvision adapter requires torchvision framework")

    def train(self) -> Path:
        status(f"torchvision initialization: building model={self.config.model_name}")
        model = self.build()
        status("torchvision initialization: building data loaders")
        loaders = build_detection_loaders(self.config)
        status("torchvision initialization complete; starting trainer")
        result = train_torchvision(
            self.config,
            model,
            loaders,
            validation_evaluator=lambda detector, loader, device: evaluate_torchvision_map50_95(
                self.config, detector, loader, device
            ),
        )
        return result.checkpoint

    def build(self):
        """Construct the configured detector, loading torchvision lazily."""
        return build_torchvision_model(self.config)

    def _load_detector(self, checkpoint: Path):
        """Build the detector, apply the checkpoint, and move it to the device."""
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install torch and torchvision for torchvision prediction") from error
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        model = self.build()
        if self.config.native_score_threshold is not None:
            apply_native_score_threshold(model, self.config.native_score_threshold)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        device = torch.device(self.config.execution.device)
        return model.to(device).eval(), device, checkpoint_path

    def _final_stage(self) -> PostprocessStageConfig:
        return self.config.final_postprocess or PostprocessStageConfig(
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            area_threshold=self.config.evaluation.area_threshold,
            confidence_owner=self.config.evaluation.confidence_owner,
            nms_owner=self.config.evaluation.nms_owner,
            native_nms_applied=self.config.evaluation.nms_owner in {"torchvision", "ultralytics"},
        )

    def _predict_one(self, model, device, checkpoint_path: Path, source: Path) -> PredictionRecord:
        """Run one image through an already loaded detector."""
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("install torch, torchvision, numpy, and Pillow for torchvision prediction") from error
        image = np.array(Image.open(source).convert("RGB"), copy=True)
        image, _ = build_transforms(self.config, training=False)(image, {"boxes": [], "labels": []})
        tensor = torch.as_tensor(image).permute(2, 0, 1).contiguous().float()
        if getattr(image, "dtype", None) is not None and image.dtype.kind in "ui":
            tensor = tensor / 255.0
        with torch.no_grad():
            output = model([tensor.to(device)])[0]
        stage = self._final_stage()
        output = filter_torchvision_outputs(
            [output], stage, max_detections=self.config.evaluation.max_detections
        )[0]
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]
        keep = list(range(len(scores)))
        detections = tuple(
            Detection(tuple(float(value) for value in boxes[index].detach().cpu().tolist()), float(scores[index]), int(labels[index]))
            for index in keep
        )
        native = _native_postprocess_settings(model)
        metadata = PostprocessingMetadata(
            confidence_owner=stage.confidence_owner,
            nms_owner=stage.nms_owner,
            confidence_threshold=stage.confidence_threshold,
            nms_iou_threshold=stage.iou_threshold,
            max_detections=self.config.evaluation.max_detections,
            area_threshold=stage.area_threshold,
            image_size=self.config.image_size,
            checkpoint_provenance=str(checkpoint_path),
            stage="final",
            native_nms_applied=stage.native_nms_applied,
            native_nms_iou_threshold=native["nms_thresh"],
            native_score_threshold=native["score_thresh"],
            native_detections_per_image=native["detections_per_img"],
        )
        return PredictionRecord(Path(source).stem, detections, metadata)

    def predict(self, source: Path, checkpoint: Path) -> list[PredictionRecord]:
        model, device, checkpoint_path = self._load_detector(checkpoint)
        return [self._predict_one(model, device, checkpoint_path, Path(source))]

    def predict_iter(self, sources, checkpoint: Path):
        """Yield a record per source with one detector build and checkpoint load."""
        model, device, checkpoint_path = self._load_detector(checkpoint)
        for source in sources:
            yield self._predict_one(model, device, checkpoint_path, Path(source))
