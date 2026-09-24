"""Lazy Ultralytics adapter; importing this module does not import Ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..config import ExperimentConfig
from ..evaluation.records import Detection, PostprocessingMetadata, PredictionRecord
from ..training.ultralytics_trainer import ULTRALYTICS_WEIGHTS, train_ultralytics
from .base import ModelAdapter


def _check_nms_owner(nms_owner: str, model: Any) -> None:
    """Refuse a config whose NMS owner contradicts the loaded detection head.

    YOLO26 heads are end-to-end (one-to-one) and skip NMS, like RT-DETR's
    decoder; YOLOv8/11/12 heads need it. Recording the wrong owner would
    misstate the post-processing behind every prediction.
    """
    end2end = getattr(model.model.model[-1], "end2end", None)
    if end2end is None:
        return
    if bool(end2end) and nms_owner != "none":
        raise ValueError(f"end-to-end detection head applies no NMS; set evaluation.nms_owner: none, not {nms_owner}")
    if not end2end and nms_owner == "none":
        raise ValueError("this detection head needs NMS; evaluation.nms_owner cannot be none")


class UltralyticsAdapter(ModelAdapter):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        if config.framework != "ultralytics":
            raise ValueError("ultralytics adapter requires ultralytics framework")

    def _load(self):
        try:
            from ultralytics import RTDETR, YOLO
        except ImportError as error:
            raise RuntimeError("install the optional ultralytics dependency to run this adapter") from error
        model_file = self.config.initialization.weights
        expected = ULTRALYTICS_WEIGHTS[self.model_name]
        if model_file != expected:
            raise ValueError(
                f"initialization.weights for {self.model_name} must be {expected}, got {model_file}"
            )
        return (RTDETR if self.model_name == "rt-detr-l" else YOLO)(model_file)

    def train(self, dataset_yaml: Path | None = None, run_dir: Path | None = None) -> Path:
        model = self._load()
        if dataset_yaml is None:
            candidate = self.config.dataset.yaml
            if candidate is None:
                raise ValueError("Ultralytics training requires an explicit dataset YAML path")
            dataset_yaml = candidate
        if not Path(dataset_yaml).is_file():
            raise FileNotFoundError(dataset_yaml)
        framework_run_dir = run_dir or self.config.output_dir
        result = train_ultralytics(self.config, model, dataset_yaml, framework_run_dir)
        return result.output_dir

    @staticmethod
    def _canonical_class_id(native_class_id: int) -> int:
        """Map the single native Ultralytics class to the canonical namespace."""
        if native_class_id != 0:
            raise ValueError(f"unsupported Ultralytics class id: {native_class_id}")
        return 1

    @staticmethod
    def _canonical_predictions(
        results: Iterable[Any], source: Path, postprocessing: PostprocessingMetadata | None = None
    ) -> list[PredictionRecord]:
        """Convert native results without applying a second filter or NMS."""
        records: list[PredictionRecord] = []
        for index, result in enumerate(results):
            path = getattr(result, "path", None)
            image_id = Path(str(path)).stem if path else (source.stem if index == 0 else f"{source.stem}:{index}")
            detections: list[Detection] = []
            for box in getattr(result, "boxes", ()):
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                score = box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0]
                class_values = getattr(box, "cls", None)
                native_class_id = (
                    int(class_values[0].item() if hasattr(class_values[0], "item") else class_values[0])
                    if class_values is not None
                    else 0
                )
                class_id = UltralyticsAdapter._canonical_class_id(native_class_id)
                detections.append(Detection(tuple(float(value) for value in xyxy), float(score), class_id))
            records.append(PredictionRecord(image_id, tuple(detections), postprocessing))
        return records

    def _predict_arguments(self) -> dict[str, object]:
        """Native prediction arguments, identical for one or many sources."""
        device = self.config.execution.device
        if device == "cuda":
            device = "0"
        elif device.startswith("cuda:"):
            device = device.split(":", 1)[1]
        return {
            "imgsz": self.config.image_size[0],
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "max_det": self.config.evaluation.max_detections,
            "device": device,
            "verbose": False,
            "save": False,
        }

    def predict_iter(self, sources, checkpoint: Path):
        """Yield a record per source with one checkpoint load."""
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        paths = [Path(source) for source in sources]
        if not paths:
            return
        model = self._load_checkpoint(checkpoint)
        arguments = self._predict_arguments()
        metadata = self._postprocessing_metadata(checkpoint)
        for source in paths:
            results = model.predict(source=str(source), **arguments)
            yield from self._canonical_predictions(results, source, metadata)

    def _postprocessing_metadata(self, checkpoint: Path) -> PostprocessingMetadata:
        return PostprocessingMetadata(
            confidence_owner="ultralytics",
            nms_owner=self.config.evaluation.nms_owner,
            confidence_threshold=self.config.confidence_threshold,
            nms_iou_threshold=self.config.iou_threshold,
            max_detections=self.config.evaluation.max_detections,
            area_threshold=self.config.evaluation.area_threshold,
            image_size=self.config.image_size,
            checkpoint_provenance=str(checkpoint),
            native_nms_applied=self.config.evaluation.nms_owner == "ultralytics",
        )

    def predict(self, source: Path, checkpoint: Path) -> list[PredictionRecord]:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        results = self._load_checkpoint(checkpoint).predict(
            source=str(source), **self._predict_arguments()
        )
        return self._canonical_predictions(
            results, Path(source), self._postprocessing_metadata(checkpoint)
        )

    def _load_checkpoint(self, checkpoint: Path):
        try:
            from ultralytics import RTDETR, YOLO
        except ImportError as error:
            raise RuntimeError("install the optional ultralytics dependency to run this adapter") from error
        constructor = RTDETR if self.model_name == "rt-detr-l" else YOLO
        model = constructor(str(checkpoint))
        _check_nms_owner(self.config.evaluation.nms_owner, model)
        return model
