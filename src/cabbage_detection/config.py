"""Strict experiment configuration loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Literal

import yaml


SUPPORTED_MODELS = {
    "faster_rcnn",
    "ssd",
    "ssdlite",
    "retinanet",
    "fcos",
    "yolov8n",
    "yolov8m",
    "yolo11n",
    "yolo11m",
    "rt-detr-l",
    "yolo12n",
    "yolo12m",
    "yolo26n",
    "yolo26m",
}
SETTING_SOURCES = {
    "historical_code",
    "repository_protocol",
    "framework_default",
    "repository_default",
    "unknown",
}


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime settings; ``None`` means the historical value is unknown."""

    device: str = "cpu"
    workers: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint selection policy for training outputs."""

    metric: str | None = None
    mode: str | None = None
    retain_each_epoch: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    format: str
    root: Path
    split_manifest: Path
    yaml: Path | None
    labels_dirname: str | None
    class_names: tuple[str, ...]
    annotations: Path | None = None


@dataclass(frozen=True)
class InitializationConfig:
    weights: str
    num_classes: int


@dataclass(frozen=True)
class AugmentationConfig:
    profile: str
    horizontal_flip: float
    vertical_flip: float
    translate: float
    scale: float
    degrees: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    mosaic: float
    normalize: bool
    border_value: int


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    step_size: int | None
    gamma: float | None
    final_lr_factor: float | None


@dataclass(frozen=True)
class EvaluationConfig:
    max_detections: int
    area_threshold: float
    confidence_owner: str
    nms_owner: str
    backend: str = "repository_ap_101"
    counting_iou_threshold: float | None = None


@dataclass(frozen=True)
class PostprocessStageConfig:
    """Filtering and NMS ownership for one evaluation stage."""

    confidence_threshold: float
    iou_threshold: float
    area_threshold: float
    confidence_owner: str
    nms_owner: str
    native_nms_applied: bool = False


@dataclass(frozen=True)
class NativeTrainingConfig:
    patience: int
    cache: bool
    train_shuffle: bool
    validation_shuffle: bool
    warmup_first_epoch: bool = False
    empty_target_policy: str = "include"


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    framework: str
    dataset: DatasetConfig
    initialization: InitializationConfig
    image_size: tuple[int, int]
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    epochs: int
    confidence_threshold: float
    iou_threshold: float
    augmentation: AugmentationConfig
    scheduler: SchedulerConfig
    evaluation: EvaluationConfig
    native_training: NativeTrainingConfig
    output_dir: Path
    result_status: Literal["smoke", "reproduction_candidate"] = "reproduction_candidate"
    provenance: dict[str, str] = field(default_factory=dict)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    validation_postprocess: PostprocessStageConfig | None = None
    final_postprocess: PostprocessStageConfig | None = None
    # Set only by an explicit evaluation override, never by a shipped profile.
    # Lowering a detector's own score gate is a deviation from the published
    # model behaviour and is recorded as one.
    native_score_threshold: float | None = None
    # Which input size a torchvision detector's own GeneralizedRCNNTransform
    # uses after the configured ``image_size`` resize. ``framework_default``
    # keeps torchvision's constructor values (min_size 800 for Faster R-CNN,
    # RetinaNet and FCOS; fixed_size 300 for SSD and 320 for SSDLite), which
    # is what every released run used. ``image_size`` makes the detector see
    # ``image_size`` itself. Ultralytics models accept only the default.
    detector_input: Literal["framework_default", "image_size"] = "framework_default"

    @property
    def dataset_root(self) -> Path:
        """Backward-compatible access to the configured dataset root."""
        return self.dataset.root

    @property
    def split_manifest(self) -> Path:
        """Backward-compatible access to the configured split manifest."""
        return self.dataset.split_manifest

    @property
    def counting_iou_threshold(self) -> float:
        """IoU used to match predictions to targets when counting.

        This is a different quantity from ``postprocess.iou_threshold``, which
        owns NMS during prediction. Profiles that do not set
        ``evaluation.counting_iou_threshold`` keep the historical behaviour of
        reusing the NMS value.
        """
        if self.evaluation.counting_iou_threshold is None:
            return self.iou_threshold
        return self.evaluation.counting_iou_threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the configuration."""
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["image_size"] = list(self.image_size)
        provenance = values.pop("provenance")
        dataset = values.pop("dataset")
        dataset["root"] = str(self.dataset.root)
        dataset["split_manifest"] = str(self.dataset.split_manifest)
        dataset["annotations"] = str(self.dataset.annotations) if self.dataset.annotations else None
        dataset["yaml"] = str(self.dataset.yaml) if self.dataset.yaml else None
        initialization = values.pop("initialization")
        augmentation = values.pop("augmentation")
        scheduler = values.pop("scheduler")
        evaluation = values.pop("evaluation")
        native_training = values.pop("native_training")
        return {
            "model": {
                "name": values.pop("model_name"),
                "framework": values.pop("framework"),
                "detector_input": values.pop("detector_input"),
            },
            "dataset": dataset,
            "initialization": initialization,
            "training": {
                "image_size": values.pop("image_size"),
                "optimizer": values.pop("optimizer"),
                "learning_rate": values.pop("learning_rate"),
                "momentum": values.pop("momentum"),
                "weight_decay": values.pop("weight_decay"),
                "batch_size": values.pop("batch_size"),
                "epochs": values.pop("epochs"),
            },
            "augmentation": augmentation,
            "scheduler": scheduler,
            "postprocess": {
                "confidence_threshold": values.pop("confidence_threshold"),
                "iou_threshold": values.pop("iou_threshold"),
            },
            "evaluation": evaluation,
            "validation_postprocess": _stage_to_dict(self.validation_postprocess),
            "final_postprocess": _stage_to_dict(self.final_postprocess),
            "native_training": native_training,
            "artifacts": {
                "output_dir": values.pop("output_dir"),
                "result_status": values.pop("result_status"),
            },
            "execution": {
                "device": self.execution.device,
                "workers": self.execution.workers,
                "seed": self.execution.seed,
            },
            "checkpoint": {
                "metric": self.checkpoint.metric,
                "mode": self.checkpoint.mode,
                "retain_each_epoch": self.checkpoint.retain_each_epoch,
            },
            "provenance": provenance,
            "overrides": {"native_score_threshold": values.pop("native_score_threshold")},
        }


def _stage_to_dict(stage: PostprocessStageConfig | None) -> dict[str, Any] | None:
    if stage is None:
        return None
    return asdict(stage)


def _section(data: dict[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - keys
    if unknown:
        raise ValueError(f"unknown {name} key(s): {', '.join(sorted(unknown))}")
    missing = keys - set(value)
    if missing:
        raise ValueError(f"missing {name} key(s): {', '.join(sorted(missing))}")
    return value


def load_config(path: Path) -> ExperimentConfig:
    """Load and strictly validate an experiment YAML configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    required = {"model", "dataset", "training", "postprocess", "artifacts"}
    allowed = required | {
        "initialization",
        "augmentation",
        "scheduler",
        "evaluation",
        "native_training",
        "provenance",
        "execution",
        "checkpoint",
        "validation_postprocess",
        "final_postprocess",
        "overrides",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")
    if not required <= set(raw):
        missing = required - set(raw)
        raise ValueError(f"missing top-level key(s): {', '.join(sorted(missing))}")

    raw_model = raw.get("model")
    if isinstance(raw_model, dict) and "detector_input" not in raw_model:
        raw = dict(raw)
        raw["model"] = {**raw_model, "detector_input": "framework_default"}
    model = _section(raw, "model", {"name", "framework", "detector_input"})
    raw_dataset = raw.get("dataset")
    if isinstance(raw_dataset, dict) and "annotations" not in raw_dataset:
        raw_dataset = dict(raw_dataset)
        raw_dataset["annotations"] = None
        raw = dict(raw)
        raw["dataset"] = raw_dataset
    dataset = _section(raw, "dataset", {"format", "root", "split_manifest", "annotations", "yaml", "labels_dirname", "class_names"})
    initialization = _section(raw, "initialization", {"weights", "num_classes"})
    training = _section(
        raw,
        "training",
        {"image_size", "optimizer", "learning_rate", "momentum", "weight_decay", "batch_size", "epochs"},
    )
    postprocess = _section(raw, "postprocess", {"confidence_threshold", "iou_threshold"})
    augmentation = _section(
        raw,
        "augmentation",
        {"profile", "horizontal_flip", "vertical_flip", "translate", "scale", "degrees", "hsv_h", "hsv_s", "hsv_v", "mosaic", "normalize", "border_value"},
    )
    scheduler = _section(raw, "scheduler", {"name", "step_size", "gamma", "final_lr_factor"})
    raw_evaluation = raw.get("evaluation")
    if isinstance(raw_evaluation, dict) and "backend" not in raw_evaluation:
        raise ValueError("missing evaluation.backend")
    raw_evaluation_section = raw.get("evaluation")
    if isinstance(raw_evaluation_section, dict) and "counting_iou_threshold" not in raw_evaluation_section:
        raw["evaluation"] = dict(raw_evaluation_section)
        raw["evaluation"]["counting_iou_threshold"] = None
    evaluation = _section(
        raw,
        "evaluation",
        {"max_detections", "area_threshold", "confidence_owner", "nms_owner", "backend", "counting_iou_threshold"},
    )
    raw_native_training = raw.get("native_training")
    if isinstance(raw_native_training, dict):
        raw_native_training = dict(raw_native_training)
        raw_native_training.setdefault("warmup_first_epoch", str(model.get("framework")) == "torchvision")
        raw_native_training.setdefault("empty_target_policy", "include")
        raw = dict(raw)
        raw["native_training"] = raw_native_training
    native_training = _section(
        raw,
        "native_training",
        {"patience", "cache", "train_shuffle", "validation_shuffle", "warmup_first_epoch", "empty_target_policy"},
    )
    raw_artifacts = raw.get("artifacts")
    if isinstance(raw_artifacts, dict) and "result_status" not in raw_artifacts:
        raw_artifacts = dict(raw_artifacts)
        raw_artifacts["result_status"] = "reproduction_candidate"
        raw = dict(raw)
        raw["artifacts"] = raw_artifacts
    artifacts = _section(raw, "artifacts", {"output_dir", "result_status"})

    # torchvision detectors run their own NMS inside ``postprocess_detections``
    # regardless of who owns the repository-side stage, so a torchvision run
    # always has native NMS applied. Recording otherwise would understate the
    # duplication when ``nms_owner`` is ``repository``.
    native_nms_default = (
        str(evaluation["nms_owner"]) in {"torchvision", "ultralytics"}
        or str(model.get("framework")) == "torchvision"
    )

    def stage_config(name: str, default_area: float) -> PostprocessStageConfig:
        raw_stage = raw.get(name)
        if raw_stage is None:
            return PostprocessStageConfig(
                confidence_threshold=float(postprocess["confidence_threshold"]),
                iou_threshold=float(postprocess["iou_threshold"]),
                area_threshold=float(default_area),
                confidence_owner=str(evaluation["confidence_owner"]),
                nms_owner=str(evaluation["nms_owner"]),
                native_nms_applied=native_nms_default,
            )
        stage = _section(
            raw,
            name,
            {"confidence_threshold", "iou_threshold", "area_threshold", "confidence_owner", "nms_owner", "native_nms_applied"},
        )
        for key in ("confidence_threshold", "iou_threshold"):
            value = stage[key]
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name}.{key} must be between 0 and 1")
        if not isinstance(stage["area_threshold"], (int, float)) or stage["area_threshold"] < 0:
            raise ValueError(f"{name}.area_threshold must be non-negative")
        if stage["confidence_owner"] not in {"repository", "torchvision", "ultralytics"}:
            raise ValueError(f"invalid {name} confidence owner")
        if stage["nms_owner"] not in {"repository", "torchvision", "ultralytics", "none"}:
            raise ValueError(f"invalid {name} NMS owner")
        if not isinstance(stage["native_nms_applied"], bool):
            raise ValueError(f"{name}.native_nms_applied must be boolean")
        return PostprocessStageConfig(
            confidence_threshold=float(stage["confidence_threshold"]),
            iou_threshold=float(stage["iou_threshold"]),
            area_threshold=float(stage["area_threshold"]),
            confidence_owner=str(stage["confidence_owner"]),
            nms_owner=str(stage["nms_owner"]),
            native_nms_applied=bool(stage["native_nms_applied"]),
        )

    validation_postprocess = stage_config("validation_postprocess", float(evaluation["area_threshold"]))
    final_area_default = (
        100.0
        if str(model.get("name")) in {"retinanet", "ssd", "ssdlite", "fcos"}
        and str(model.get("framework")) == "torchvision"
        else float(evaluation["area_threshold"])
    )
    final_postprocess = stage_config("final_postprocess", final_area_default)
    provenance = {
        "training.epochs": "repository_protocol",
        "training.seed": "unknown",
        "training.workers": "unknown",
        "training.checkpoint_selection": "unknown",
        "execution.device": "repository_default",
        "execution.workers": "unknown",
        "execution.seed": "unknown",
    }
    supplied_provenance = raw.get("provenance", {})
    if not isinstance(supplied_provenance, dict):
        raise ValueError("provenance must be a mapping")
    invalid_sources = {
        key: value for key, value in supplied_provenance.items() if value not in SETTING_SOURCES
    }
    if invalid_sources:
        raise ValueError(
            "provenance source must be one of: historical_code, repository_protocol, framework_default, repository_default, unknown"
        )
    provenance.update({str(key): str(value) for key, value in supplied_provenance.items()})

    overrides = raw.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a mapping")
    unknown_overrides = set(overrides) - {"native_score_threshold"}
    if unknown_overrides:
        raise ValueError(f"unknown override(s): {', '.join(sorted(unknown_overrides))}")
    native_score_threshold = overrides.get("native_score_threshold")
    if native_score_threshold is not None and (
        not isinstance(native_score_threshold, (int, float)) or not 0 <= native_score_threshold <= 1
    ):
        raise ValueError("overrides.native_score_threshold must be between 0 and 1")

    model_name = str(model["name"])
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model: {model_name}")
    framework = str(model["framework"])
    if framework not in {"torchvision", "ultralytics"}:
        raise ValueError(f"unsupported framework: {framework}")
    detector_input = str(model["detector_input"])
    if detector_input not in {"framework_default", "image_size"}:
        raise ValueError("model.detector_input must be framework_default or image_size")
    if detector_input != "framework_default" and framework != "torchvision":
        raise ValueError("model.detector_input applies only to torchvision models")
    image_size = training["image_size"]
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("image_size must contain two integers")
    if any(not isinstance(value, int) or value <= 0 for value in image_size):
        raise ValueError("image_size must contain positive integers")
    dataset_format = str(dataset["format"])
    if dataset_format not in {"pascal_voc", "coco", "yolo"}:
        raise ValueError("dataset format must be pascal_voc, coco, or yolo")
    dataset_annotations = dataset["annotations"]
    if dataset_annotations is not None and not isinstance(dataset_annotations, (str, Path)):
        raise ValueError("dataset.annotations must be a path or null")
    dataset_yaml = dataset["yaml"]
    if dataset_yaml is not None and not isinstance(dataset_yaml, (str, Path)):
        raise ValueError("dataset.yaml must be a path or null")
    labels_dirname = dataset["labels_dirname"]
    if labels_dirname is not None and not isinstance(labels_dirname, str):
        raise ValueError("dataset.labels_dirname must be a string or null")
    if framework == "ultralytics" and dataset_yaml is None:
        raise ValueError("Ultralytics config requires dataset.yaml")
    if framework == "torchvision" and dataset_format == "pascal_voc" and labels_dirname != "labels":
        raise ValueError("torchvision config requires dataset.labels_dirname=labels")
    if framework == "torchvision" and dataset_format == "coco" and labels_dirname is not None:
        raise ValueError("COCO torchvision config requires dataset.labels_dirname=null")
    class_names = dataset["class_names"]
    if not isinstance(class_names, (list, tuple)) or not class_names or any(not isinstance(name, str) for name in class_names):
        raise ValueError("dataset.class_names must be a non-empty list of strings")
    weights = initialization["weights"]
    num_classes = initialization["num_classes"]
    if not isinstance(weights, str) or not weights:
        raise ValueError("initialization.weights must be a non-empty string")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("initialization.num_classes must be a positive integer")
    expected_classes = 2 if framework == "torchvision" else 1
    if num_classes != expected_classes:
        raise ValueError(f"{framework} initialization.num_classes must be {expected_classes}")
    profile = augmentation["profile"]
    if profile not in {"reproduction_augmented", "recovered_no_augmentation"}:
        raise ValueError("augmentation.profile must be reproduction_augmented or recovered_no_augmentation")
    for key in ("horizontal_flip", "vertical_flip", "translate", "scale", "mosaic"):
        value = augmentation[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"augmentation.{key} must be between 0 and 1")
    for key in ("degrees", "hsv_h", "hsv_s", "hsv_v"):
        value = augmentation[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"augmentation.{key} must be non-negative")
    if not isinstance(augmentation["normalize"], bool):
        raise ValueError("augmentation.normalize must be boolean")
    if not isinstance(augmentation["border_value"], int) or not 0 <= augmentation["border_value"] <= 255:
        raise ValueError("augmentation.border_value must be between 0 and 255")
    all_augmentation_values = tuple(float(augmentation[key]) for key in ("horizontal_flip", "vertical_flip", "translate", "scale", "degrees", "hsv_h", "hsv_s", "hsv_v", "mosaic"))
    if profile == "recovered_no_augmentation" and any(all_augmentation_values):
        raise ValueError("recovered_no_augmentation must disable every augmentation")
    if profile == "reproduction_augmented" and not any(all_augmentation_values):
        raise ValueError("reproduction_augmented must enable at least one augmentation")
    scheduler_name = str(scheduler["name"])
    if scheduler_name not in {"constant", "step_lr"}:
        raise ValueError("scheduler.name must be constant or step_lr")
    for key in ("step_size",):
        value = scheduler[key]
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"scheduler.{key} must be a positive integer or null")
    for key in ("gamma", "final_lr_factor"):
        value = scheduler[key]
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise ValueError(f"scheduler.{key} must be non-negative or null")
    if scheduler_name == "step_lr" and (scheduler["step_size"] is None or scheduler["gamma"] is None):
        raise ValueError("step_lr requires scheduler.step_size and scheduler.gamma")
    if scheduler_name == "constant" and scheduler["final_lr_factor"] is None:
        raise ValueError("constant requires scheduler.final_lr_factor")
    max_detections = evaluation["max_detections"]
    if not isinstance(max_detections, int) or max_detections <= 0:
        raise ValueError("evaluation.max_detections must be positive")
    area_threshold = evaluation["area_threshold"]
    if not isinstance(area_threshold, (int, float)) or area_threshold < 0:
        raise ValueError("evaluation.area_threshold must be non-negative")
    if evaluation["confidence_owner"] not in {"repository", "torchvision", "ultralytics"}:
        raise ValueError("invalid confidence owner")
    if evaluation["nms_owner"] not in {"repository", "torchvision", "ultralytics", "none"}:
        raise ValueError("invalid NMS owner")
    if evaluation["backend"] not in {"repository_ap_101", "pycocotools_coco_eval"}:
        raise ValueError("evaluation.backend must be repository_ap_101 or pycocotools_coco_eval")
    counting_iou = evaluation["counting_iou_threshold"]
    if counting_iou is not None and (not isinstance(counting_iou, (int, float)) or not 0 <= counting_iou <= 1):
        raise ValueError("evaluation.counting_iou_threshold must be between 0 and 1")
    patience = native_training["patience"]
    if not isinstance(patience, int) or patience <= 0:
        raise ValueError("native_training.patience must be positive")
    for key in ("cache", "train_shuffle", "validation_shuffle"):
        if not isinstance(native_training[key], bool):
            raise ValueError(f"native_training.{key} must be boolean")
    if not isinstance(native_training["warmup_first_epoch"], bool):
        raise ValueError("native_training.warmup_first_epoch must be boolean")
    if native_training["empty_target_policy"] not in {"include", "skip"}:
        raise ValueError("native_training.empty_target_policy must be include or skip")
    for key in ("confidence_threshold", "iou_threshold"):
        value = postprocess[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    for key in ("batch_size", "epochs"):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    result_status = str(artifacts["result_status"])
    if result_status not in {"smoke", "reproduction_candidate"}:
        raise ValueError("artifacts.result_status must be smoke or reproduction_candidate")

    execution_raw = raw.get("execution", {})
    if not isinstance(execution_raw, dict):
        raise ValueError("execution must be a mapping")
    unknown_execution = set(execution_raw) - {"device", "workers", "seed"}
    if unknown_execution:
        raise ValueError(f"unknown execution key(s): {', '.join(sorted(unknown_execution))}")
    device = str(execution_raw.get("device", "cpu"))
    if device != "cpu" and device != "cuda" and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    workers = execution_raw.get("workers")
    if workers is not None and (not isinstance(workers, int) or workers < 0):
        raise ValueError("workers must be a non-negative integer or null")
    seed = execution_raw.get("seed")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a non-negative integer or null")

    checkpoint_raw = raw.get("checkpoint", {})
    if not isinstance(checkpoint_raw, dict):
        raise ValueError("checkpoint must be a mapping")
    unknown_checkpoint = set(checkpoint_raw) - {"metric", "mode", "retain_each_epoch"}
    if unknown_checkpoint:
        raise ValueError(f"unknown checkpoint key(s): {', '.join(sorted(unknown_checkpoint))}")
    metric = checkpoint_raw.get("metric")
    mode = checkpoint_raw.get("mode")
    if (metric is None) != (mode is None):
        raise ValueError("checkpoint metric and mode must be provided together")
    if mode is not None and mode not in {"min", "max"}:
        raise ValueError("checkpoint mode must be min or max")
    retain_each_epoch = checkpoint_raw.get("retain_each_epoch", False)
    if not isinstance(retain_each_epoch, bool):
        raise ValueError("checkpoint.retain_each_epoch must be boolean")

    return ExperimentConfig(
        model_name=model_name,
        framework=framework,
        detector_input=detector_input,
        dataset=DatasetConfig(
            format=dataset_format,
            root=Path(str(dataset["root"])),
            split_manifest=Path(str(dataset["split_manifest"])),
            annotations=Path(str(dataset_annotations)) if dataset_annotations is not None else None,
            yaml=Path(str(dataset_yaml)) if dataset_yaml is not None else None,
            labels_dirname=labels_dirname,
            class_names=tuple(class_names),
        ),
        initialization=InitializationConfig(weights=weights, num_classes=num_classes),
        image_size=(int(image_size[0]), int(image_size[1])),
        optimizer=str(training["optimizer"]),
        learning_rate=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        confidence_threshold=float(postprocess["confidence_threshold"]),
        iou_threshold=float(postprocess["iou_threshold"]),
        augmentation=AugmentationConfig(
            profile=str(profile),
            horizontal_flip=float(augmentation["horizontal_flip"]),
            vertical_flip=float(augmentation["vertical_flip"]),
            translate=float(augmentation["translate"]),
            scale=float(augmentation["scale"]),
            degrees=float(augmentation["degrees"]),
            hsv_h=float(augmentation["hsv_h"]),
            hsv_s=float(augmentation["hsv_s"]),
            hsv_v=float(augmentation["hsv_v"]),
            mosaic=float(augmentation["mosaic"]),
            normalize=bool(augmentation["normalize"]),
            border_value=int(augmentation["border_value"]),
        ),
        scheduler=SchedulerConfig(
            name=scheduler_name,
            step_size=scheduler["step_size"],
            gamma=float(scheduler["gamma"]) if scheduler["gamma"] is not None else None,
            final_lr_factor=float(scheduler["final_lr_factor"]) if scheduler["final_lr_factor"] is not None else None,
        ),
        evaluation=EvaluationConfig(
            max_detections=max_detections,
            area_threshold=float(area_threshold),
            confidence_owner=str(evaluation["confidence_owner"]),
            nms_owner=str(evaluation["nms_owner"]),
            backend=str(evaluation["backend"]),
            counting_iou_threshold=None if counting_iou is None else float(counting_iou),
        ),
        native_training=NativeTrainingConfig(
            patience=patience,
            cache=bool(native_training["cache"]),
            train_shuffle=bool(native_training["train_shuffle"]),
            validation_shuffle=bool(native_training["validation_shuffle"]),
            warmup_first_epoch=bool(native_training["warmup_first_epoch"]),
            empty_target_policy=str(native_training["empty_target_policy"]),
        ),
        output_dir=Path(str(artifacts["output_dir"])),
        result_status=result_status,
        native_score_threshold=(
            None if native_score_threshold is None else float(native_score_threshold)
        ),
        provenance=provenance,
        execution=ExecutionConfig(device=device, workers=workers, seed=seed),
        checkpoint=CheckpointConfig(
            metric=str(metric) if metric is not None else None,
            mode=str(mode) if mode is not None else None,
            retain_each_epoch=retain_each_epoch,
        ),
        validation_postprocess=validation_postprocess,
        final_postprocess=final_postprocess,
    )
