"""End-to-end CPU smoke test for the torchvision split/train/counting path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image

from cabbage_detection.artifacts import create_run
from cabbage_detection.cli import evaluate_counts, prepare_dataset
from cabbage_detection.config import (
    AugmentationConfig,
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    InitializationConfig,
    NativeTrainingConfig,
    SchedulerConfig,
)
from cabbage_detection.evaluation.io import write_prediction_records, write_target_records
from cabbage_detection.evaluation.records import Box, Detection, PredictionRecord, TargetRecord
from cabbage_detection.training.loaders import build_torchvision_loaders
from cabbage_detection.training.torchvision_trainer import DetectionLoaders, train_torchvision
from cabbage_detection.evaluation.detection import DetectionReport


class SmokeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, images, targets=None):
        if self.training:
            return {"loss": (self.weight - 1.0).pow(2)}
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]


def _dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        image_id = f"{split}_sample"
        Image.new("RGB", (8, 8), color=(128, 128, 128)).save(image_dir / f"{image_id}.png")
        (label_dir / f"{image_id}.txt").write_text("0 0 0 8 8\n", encoding="utf-8")


def _config(dataset_root: Path, manifest: Path, output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="faster_rcnn",
        framework="torchvision",
        dataset=DatasetConfig("pascal_voc", dataset_root, manifest, None, "labels", ("background", "cabbage")),
        initialization=InitializationConfig("COCO_V1", 2),
        image_size=(8, 8),
        optimizer="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=1,
        epochs=1,
        augmentation=AugmentationConfig("reproduction_augmented", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0),
        scheduler=SchedulerConfig("step_lr", 1000, 0.1, None),
        evaluation=EvaluationConfig(100, 0.0, "repository", "repository"),
        native_training=NativeTrainingConfig(1, False, True, False),
        confidence_threshold=0.5,
        iou_threshold=0.5,
        output_dir=output_dir,
        execution=ExecutionConfig(device="cpu", workers=0, seed=42),
        checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"),
    )


@pytest.mark.integration
def test_torchvision_split_train_and_counting_smoke(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    manifest = tmp_path / "splits.csv"
    prepare_dataset(dataset_root, manifest)

    config = _config(dataset_root, manifest, tmp_path / "run")
    loaders = build_torchvision_loaders(config)
    assert len(loaders.train.dataset) == 1
    assert len(loaders.validation.dataset) == 1
    assert len(loaders.test.dataset) == 1
    assert set(loaders.train.dataset.image_ids).isdisjoint(loaders.validation.dataset.image_ids)
    assert set(loaders.validation.dataset.image_ids).isdisjoint(loaders.test.dataset.image_ids)

    context = create_run(config, ["smoke", "split", "train", "evaluate"])
    context.record_environment()
    model = SmokeDetector()
    initial = model.weight.item()
    result = train_torchvision(
        config,
        model,
        loaders,
        validation_evaluator=lambda detector, loader, device: DetectionReport(1.0, 1.0),
    )
    context.finalize(
        "succeeded",
        {
            "best_checkpoint": result.checkpoint,
            "last_checkpoint": result.last_checkpoint,
            "training_log": config.output_dir / "logs" / "console.log",
            "epoch_metrics": config.output_dir / "logs" / "training_log.csv",
        },
        result_status="smoke",
    )
    assert model.weight.item() != initial
    assert result.best_epoch == 1
    assert result.checkpoint.exists() and result.last_checkpoint.exists()

    image_id = loaders.test.dataset.image_ids[0]
    box = Box((0.0, 0.0, 8.0, 8.0))
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    write_prediction_records(predictions, [PredictionRecord(image_id, (Detection(box.coordinates, 0.9),))])
    write_target_records(targets, [TargetRecord(image_id, (box,))])
    metrics_path = evaluate_counts(predictions, targets, tmp_path / "counting" / "metrics.json")

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert report["metrics"]["actual_count"] == 1
    assert report["metrics"]["predicted_count"] == 1
    assert report["metrics"]["f1"] == 1.0
    metadata = json.loads((config.output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result_status"] == "smoke"
