"""Small, explicit torchvision training loop with lazy torch imports."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import ExperimentConfig, PostprocessStageConfig
from ..artifacts import write_resolved_config
from ..data.manifests import read_manifest
from ..evaluation.detection import evaluate_detection
from ..evaluation.detection import DetectionReport
from ..evaluation.coco_detection import evaluate_coco_detection
from ..artifact_paths import run_artifact_path
from ..progress import progress_task, status
from .artifact_writer import write_epoch, write_training_summary
from ..evaluation.records import Box, Detection, PostprocessingMetadata, PredictionRecord, TargetRecord
from ..evaluation.postprocess import filter_torchvision_outputs


def _postprocess_stage(config: ExperimentConfig, stage: str) -> PostprocessStageConfig:
    selected = config.validation_postprocess if stage == "validation" else config.final_postprocess
    if selected is not None:
        return selected
    return PostprocessStageConfig(
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        area_threshold=config.evaluation.area_threshold,
        confidence_owner=config.evaluation.confidence_owner,
        nms_owner=config.evaluation.nms_owner,
        native_nms_applied=config.evaluation.nms_owner in {"torchvision", "ultralytics"},
    )


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics and checkpoint-selection evidence for one completed epoch."""

    epoch: int
    train_loss_mean: float
    val_loss: float | None
    learning_rate: float
    val_map50: float | None
    val_map50_95: float | None
    selection_metric: str
    selection_value: float
    is_best: bool
    elapsed_seconds: float
    images_seen: int = 0
    images_trained: int = 0
    images_skipped: int = 0


@dataclass(frozen=True)
class DetectionLoaders:
    """DataLoader bundle kept framework-neutral at the orchestration boundary."""

    train: Any
    validation: Any
    test: Any


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    epochs_completed: int
    best_metric: float
    metric_name: str
    best_epoch: int
    last_checkpoint: Path
    history: tuple[EpochMetrics, ...]
    selection_mode: str


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install PyTorch to run the torchvision trainer") from error
    return torch


def resolve_device(requested: str):
    """Resolve a requested device without silently falling back to CPU."""
    torch = _torch()
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested!r} is not available")
    if requested == "cuda":
        return torch.device("cuda")
    index = int(requested.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"requested device {requested!r} is not available")
    return torch.device(requested)


def _manifest_digest(config: ExperimentConfig) -> str | None:
    try:
        return read_manifest(config.split_manifest).digest
    except (OSError, ValueError):
        return None


def _loss_for_batch(model: Any, batch: Any, device: Any):
    torch = _torch()
    images, targets = batch
    device_images = [image.to(device) for image in images]
    device_targets = [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]
    losses = model(device_images, device_targets)
    if not isinstance(losses, dict) or not losses:
        raise RuntimeError("torchvision training model must return a non-empty loss mapping")
    total = sum(losses.values())
    if not torch.isfinite(total).item():
        raise RuntimeError("non-finite torchvision training loss")
    return total


def _validation_loss(model: Any, loader: Any, device: Any) -> float:
    """Return the mean per-batch validation loss without updating weights.

    torchvision detection models only return a loss dict when called in
    train() mode with targets; eval() mode ignores targets and returns
    detections instead. Wrapping a train()-mode forward pass in
    ``torch.no_grad()`` computes the loss without backpropagating or
    stepping the optimizer. This is safe for every backbone used in this
    repository because they freeze BatchNorm statistics
    (``FrozenBatchNorm2d``), so the extra forward pass cannot contaminate
    running statistics the way it would with ordinary BatchNorm.
    """
    torch = _torch()
    model.train()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        with progress_task(
            description="validation loss",
            total=len(loader) if hasattr(loader, "__len__") else None,
            unit="batch",
        ) as loss_progress:
            for batch in loader:
                loss = _loss_for_batch(model, batch, device)
                total_loss += float(loss.detach().cpu().item())
                batches += 1
                loss_progress.advance(avg_loss=total_loss / batches)
    if batches == 0:
        raise ValueError("torchvision validation loader has no batches for loss computation")
    return total_loss / batches


def _gpu_memory_gib(torch: Any, device: Any) -> float | None:
    """Return reserved CUDA memory for live progress, or ``None`` on CPU."""

    if getattr(device, "type", None) != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_reserved(device) / (1024**3))


def train_torchvision(
    config: ExperimentConfig,
    model: Any,
    loaders: DetectionLoaders,
    validation_evaluator: Callable[[Any, Any, Any], float | DetectionReport] | None = None,
) -> TrainingResult:
    """Train a detector and select the best checkpoint by an explicit metric.

    A missing/unknown checkpoint policy is rejected before the first epoch. A
    train-loss checkpoint remains available for lightweight smoke tests.
    """
    torch = _torch()
    policy = config.checkpoint
    if policy.metric is None or policy.mode is None:
        raise ValueError("checkpoint metric and mode must be provided together before training")
    if policy.metric not in {"train_loss", "val_map_50_95"}:
        raise ValueError("unsupported torchvision checkpoint metric")
    if policy.metric == "val_map_50_95" and validation_evaluator is None:
        raise ValueError("val_map_50_95 checkpointing requires a validation evaluator")
    device = resolve_device(config.execution.device)
    if config.execution.seed is not None:
        torch.manual_seed(config.execution.seed)
    model = model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("torchvision model has no trainable parameters")
    if config.optimizer.upper() != "SGD":
        raise ValueError(f"unsupported torchvision optimizer: {config.optimizer}")
    optimizer = torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = None
    if config.scheduler.name == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler.step_size, gamma=config.scheduler.gamma
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    console_path = run_artifact_path(output_dir, "console_log")
    status(
        f"torchvision training started: model={config.model_name} device={device} "
        f"epochs={config.epochs}",
        log_file=console_path,
    )
    config_path = run_artifact_path(output_dir, "config")
    if not config_path.exists():
        write_resolved_config(config, output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_artifact_path(output_dir, "best_checkpoint")
    last_path = run_artifact_path(output_dir, "last_checkpoint")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_dir = checkpoint_dir / "epochs"
    if config.checkpoint.retain_each_epoch:
        epoch_dir.mkdir(parents=True, exist_ok=True)
    best_metric = math.inf if policy.mode == "min" else -math.inf
    best_checkpoint: Path | None = None
    best_epoch = 0
    history: list[EpochMetrics] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        status(f"epoch {epoch}/{config.epochs} training", log_file=console_path)
        model.train()
        total_loss = 0.0
        batches = 0
        images_seen = 0
        images_trained = 0
        images_skipped = 0
        warmup_scheduler = None
        if config.native_training.warmup_first_epoch and epoch == 1 and len(loaders.train) > 1:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / 1000,
                total_iters=min(1000, len(loaders.train) - 1),
            )
        with progress_task(
            description=f"train {epoch}/{config.epochs}",
            total=len(loaders.train),
            unit="batch",
        ) as train_progress:
            for batch in loaders.train:
                images, targets = batch
                images_seen += len(images)
                if config.native_training.empty_target_policy == "skip":
                    retained = [
                        (image, target)
                        for image, target in zip(images, targets)
                        if target.get("boxes") is not None and len(target["boxes"]) > 0
                    ]
                    images_skipped += len(images) - len(retained)
                    if not retained:
                        train_progress.advance(
                            avg_loss=total_loss / batches if batches else None,
                            lr=float(optimizer.param_groups[0]["lr"]),
                            images=images_seen,
                            skipped=images_skipped,
                            gpu_mem=_gpu_memory_gib(torch, device),
                        )
                        continue
                    images, targets = zip(*retained)
                    batch = (list(images), list(targets))
                images_trained += len(batch[0])
                optimizer.zero_grad(set_to_none=True)
                loss = _loss_for_batch(model, batch, device)
                loss.backward()
                optimizer.step()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
                loss_value = float(loss.detach().cpu().item())
                total_loss += loss_value
                batches += 1
                train_progress.advance(
                    loss=loss_value,
                    avg_loss=total_loss / batches,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    images=images_seen,
                    skipped=images_skipped,
                    gpu_mem=_gpu_memory_gib(torch, device),
                )
        if batches == 0:
            raise ValueError("torchvision train loader has no non-empty training batches")
        train_loss_mean = total_loss / batches
        metric = train_loss_mean
        val_map50: float | None = None
        val_map50_95: float | None = None
        val_loss: float | None = None
        if policy.metric == "val_map_50_95":
            assert validation_evaluator is not None
            status(f"epoch {epoch}/{config.epochs} validation", log_file=console_path)
            validation = validation_evaluator(model, loaders.validation, device)
            if isinstance(validation, DetectionReport):
                val_map50 = float(validation.map50)
                val_map50_95 = float(validation.map50_95)
                metric = val_map50_95
            else:
                val_map50_95 = float(validation)
                metric = val_map50_95
            status(f"epoch {epoch}/{config.epochs} validation loss", log_file=console_path)
            val_loss = _validation_loss(model, loaders.validation, device)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        is_better = metric < best_metric if policy.mode == "min" else metric > best_metric
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metric_name": policy.metric,
            "metric_value": metric,
            "config": config.to_dict(),
            "manifest_digest": _manifest_digest(config),
        }
        # Writing last after each completed epoch makes an interrupted run
        # resumable without retaining a full epoch file by default.
        torch.save(
            payload,
            last_path,
        )
        status(f"checkpoint last saved: {last_path}", log_file=console_path)
        if config.checkpoint.retain_each_epoch:
            torch.save(payload, epoch_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if is_better or best_checkpoint is None:
            best_metric = metric
            torch.save(payload, best_path)
            best_checkpoint = best_path
            best_epoch = epoch
            status(f"checkpoint best updated: epoch {epoch} metric={metric:.6f}", log_file=console_path)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss_mean=float(train_loss_mean),
                val_loss=val_loss,
                learning_rate=learning_rate,
                val_map50=val_map50,
                val_map50_95=val_map50_95,
                selection_metric=policy.metric,
                selection_value=float(metric),
                is_best=is_better,
                elapsed_seconds=time.perf_counter() - started,
                images_seen=images_seen,
                images_trained=images_trained,
                images_skipped=images_skipped,
            )
        )
        write_epoch(history[-1], output_dir)
    assert best_checkpoint is not None
    result = TrainingResult(
        best_path,
        config.epochs,
        best_metric,
        policy.metric,
        best_epoch,
        last_path,
        tuple(history),
        policy.mode,
    )
    write_training_summary(result, output_dir)
    status(
        f"training complete: best_epoch:{result.best_epoch} "
        f"best_metric={result.best_metric:.6f} checkpoint={result.checkpoint}",
        log_file=console_path,
    )
    return result


def evaluate_torchvision_map50_95(
    config: ExperimentConfig, model: Any, loader: Any, device: Any
) -> DetectionReport:
    """Evaluate torchvision outputs and retain both validation mAP values."""
    torch = _torch()
    predictions: list[PredictionRecord] = []
    targets: list[TargetRecord] = []
    model.eval()
    processed_images = 0
    with torch.no_grad():
        with progress_task(
            description="validation",
            total=len(loader) if hasattr(loader, "__len__") else None,
            unit="batch",
        ) as validation_progress:
            for images, batch_targets in loader:
                outputs = model([image.to(device) for image in images])
                outputs = filter_torchvision_outputs(
                    outputs,
                    _postprocess_stage(config, "validation"),
                    max_detections=config.evaluation.max_detections,
                )
                detected = 0
                for output, target in zip(outputs, batch_targets):
                    image_id = str(target.get("image_id_text", target.get("image_id")))
                    scores = output.get("scores", torch.empty((0,)))
                    keep = list(range(len(scores)))
                    detections = tuple(
                        Detection(
                            tuple(float(value) for value in output["boxes"][index].tolist()),
                            float(scores[index]),
                            int(output["labels"][index]),
                        )
                        for index in keep
                    )
                    detected += len(detections)
                    stage = _postprocess_stage(config, "validation")
                    predictions.append(
                        PredictionRecord(
                            image_id,
                            detections,
                            PostprocessingMetadata(
                                confidence_owner=stage.confidence_owner,
                                nms_owner=stage.nms_owner,
                                confidence_threshold=stage.confidence_threshold,
                                nms_iou_threshold=stage.iou_threshold,
                                max_detections=config.evaluation.max_detections,
                                area_threshold=stage.area_threshold,
                                image_size=config.image_size,
                                checkpoint_provenance=config.initialization.weights,
                                stage="validation",
                                native_nms_applied=stage.native_nms_applied,
                            ),
                        )
                    )
                    targets.append(
                        TargetRecord(
                            image_id,
                            tuple(Box(tuple(float(value) for value in box.tolist())) for box in target["boxes"]),
                        )
                    )
                processed_images += len(images)
                validation_progress.advance(images=processed_images, detections=detected)
    if config.evaluation.backend == "pycocotools_coco_eval":
        return evaluate_coco_detection(
            predictions,
            targets,
            max_detections=config.evaluation.max_detections,
        )
    return evaluate_detection(predictions, targets)
