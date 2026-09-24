from pathlib import Path
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import replace

import pytest

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
from cabbage_detection.training.torchvision_trainer import (
    DetectionLoaders,
    evaluate_torchvision_map50_95,
    resolve_device,
    train_torchvision,
)
from cabbage_detection.evaluation.detection import DetectionReport


torch = pytest.importorskip("torch")


def _config(tmp_path: Path, checkpoint: CheckpointConfig | None = None, device: str = "cpu") -> ExperimentConfig:
    return ExperimentConfig(
        model_name="faster_rcnn", framework="torchvision",
        dataset=DatasetConfig("pascal_voc", Path("data"), Path("manifest.csv"), None, "labels", ("background", "cabbage")),
        initialization=InitializationConfig("COCO_V1", 2), image_size=(512, 512), optimizer="SGD",
        learning_rate=0.001, momentum=0.9, weight_decay=0.0005, batch_size=2, epochs=2,
        augmentation=AugmentationConfig("reproduction_augmented", 0.5, 0.5, 0.1, 0.1, 15.0, 0.1, 0.05, 0.05, 0.0, True, 114),
        scheduler=SchedulerConfig("step_lr", 1000, 0.1, None),
        evaluation=EvaluationConfig(100, 1.0, "repository", "torchvision"),
        native_training=NativeTrainingConfig(500, False, True, False),
        confidence_threshold=0.5, iou_threshold=0.5, output_dir=tmp_path / "run",
        execution=ExecutionConfig(device=device),
        checkpoint=checkpoint or CheckpointConfig(metric="train_loss", mode="min"),
    )


class StubDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, images, targets=None):
        if self.training:
            return {"loss": (self.weight - 1.0).pow(2)}
        return [{"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,), dtype=torch.int64)} for _ in images]


class OneBatch:
    def __iter__(self):
        return iter([( [torch.zeros((3, 8, 8))], [{"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}] )])

    def __len__(self):
        return 1


class TwoBatches:
    def __iter__(self):
        batch = (
            [torch.zeros((3, 8, 8))],
            [{"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}],
        )
        return iter([batch, batch])

    def __len__(self):
        return 2


class OneBatchWithoutLength:
    def __iter__(self):
        return iter(
            [
                (
                    [torch.zeros((3, 8, 8))],
                    [{"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}],
                )
            ]
        )


class MixedBatch:
    def __iter__(self):
        return iter(
            [
                (
                    [torch.zeros((3, 8, 8)), torch.zeros((3, 8, 8))],
                    [
                        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)},
                        {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([1])},
                    ],
                )
            ]
        )

    def __len__(self):
        return 1


def test_resolve_device_rejects_unavailable_cuda_index():
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda:99")


def test_training_records_optimizer_and_best_checkpoint(tmp_path: Path):
    config = _config(tmp_path)
    result = train_torchvision(config, StubDetector(), DetectionLoaders(OneBatch(), OneBatch(), OneBatch()))

    assert result.epochs_completed == 2
    assert result.metric_name == "train_loss"
    assert result.checkpoint.exists()
    payload = torch.load(result.checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["training"]["learning_rate"] == 0.001
    assert payload["optimizer"]["param_groups"][0]["momentum"] == 0.9
    assert payload["epoch"] in {1, 2}


def test_training_refuses_unknown_checkpoint_policy_before_epoch(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig())
    model = StubDetector()
    with pytest.raises(ValueError, match="checkpoint metric and mode"):
        train_torchvision(config, model, DetectionLoaders(OneBatch(), OneBatch(), OneBatch()))
    assert model.weight.item() == 2.0


def test_training_can_select_checkpoint_by_validation_map(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"))
    calls = []

    def evaluate(model, loader, device):
        calls.append(device)
        return 0.25

    result = train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(OneBatch(), OneBatch(), OneBatch()),
        validation_evaluator=evaluate,
    )
    assert result.metric_name == "val_map_50_95"
    assert result.best_metric == 0.25
    assert result.checkpoint.name == "best.pt"
    assert len(calls) == 2


def test_training_records_epoch_history_best_epoch_and_last_checkpoint(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"))
    reports = iter((DetectionReport(0.4, 0.2), DetectionReport(0.3, 0.1)))

    result = train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(OneBatch(), OneBatch(), OneBatch()),
        validation_evaluator=lambda model, loader, device: next(reports),
    )

    assert result.best_epoch == 1
    assert result.epochs_completed == 2
    assert result.last_checkpoint.name == "last.pt"
    assert result.last_checkpoint.exists()
    assert [row.epoch for row in result.history] == [1, 2]
    assert [row.val_map50 for row in result.history] == [0.4, 0.3]
    assert [row.val_map50_95 for row in result.history] == [0.2, 0.1]
    assert [row.is_best for row in result.history] == [True, False]
    assert all(isinstance(row.val_loss, float) for row in result.history)


def test_training_keeps_first_epoch_on_validation_metric_tie(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"))
    reports = iter((DetectionReport(0.2, 0.2), DetectionReport(0.2, 0.2)))

    result = train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(OneBatch(), OneBatch(), OneBatch()),
        validation_evaluator=lambda model, loader, device: next(reports),
    )

    assert result.best_epoch == 1
    assert [row.is_best for row in result.history] == [True, False]


def test_training_writes_log_curves_best_summary_and_resolved_config(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"))
    reports = iter((DetectionReport(0.4, 0.2), DetectionReport(0.3, 0.1)))

    result = train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(OneBatch(), OneBatch(), OneBatch()),
        validation_evaluator=lambda model, loader, device: next(reports),
    )

    run_dir = config.output_dir
    epochs_path = run_dir / "logs" / "training_log.csv"
    assert epochs_path.exists()
    with epochs_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert rows[0]["val_map50_95"] == "0.2"
    assert rows[0]["is_best"] == "true"
    assert (run_dir / "logs" / "console.log").read_text(encoding="utf-8").count("epoch=") == 2

    summary = json.loads((run_dir / "logs" / "best_epoch.json").read_text(encoding="utf-8"))
    assert summary["epoch"] == 1
    assert summary["checkpoint"] == "checkpoints/best.pt"
    assert summary["sha256"]
    import yaml

    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["model"]["name"] == "faster_rcnn"
    assert (run_dir / "plots" / "training_loss.png").read_bytes().startswith(b"\x89PNG")
    assert (run_dir / "plots" / "validation_metrics.png").read_bytes().startswith(b"\x89PNG")

    import torch

    assert torch.load(result.checkpoint, map_location="cpu", weights_only=False)["epoch"] == 1
    assert torch.load(result.last_checkpoint, map_location="cpu", weights_only=False)["epoch"] == 2


def test_training_reports_stages_to_terminal_and_console_log(tmp_path: Path, capsys):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"))
    reports = iter((DetectionReport(0.4, 0.2), DetectionReport(0.3, 0.1)))

    train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(OneBatch(), OneBatch(), OneBatch()),
        validation_evaluator=lambda model, loader, device: next(reports),
    )

    terminal = capsys.readouterr().err
    log = (config.output_dir / "logs" / "console.log").read_text(encoding="utf-8")
    for fragment in ("epoch 1/2 training", "epoch 1/2 validation", "epoch=1", "training complete"):
        assert fragment in terminal
        assert fragment in log


def test_training_progress_reports_live_batch_metrics(tmp_path: Path, monkeypatch):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="train_loss", mode="min"))
    progress_calls: list[dict[str, object]] = []
    descriptions: list[str] = []

    class FakeTask:
        def advance(self, **fields):
            progress_calls.append(fields)

    @contextmanager
    def fake_progress_task(**kwargs):
        descriptions.append(kwargs["description"])
        assert kwargs["description"] in {"train 1/2", "train 2/2"}
        assert kwargs["total"] == 2
        assert kwargs["unit"] == "batch"
        yield FakeTask()

    import cabbage_detection.training.torchvision_trainer as trainer_module

    monkeypatch.setattr(trainer_module, "progress_task", fake_progress_task, raising=False)
    train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(TwoBatches(), OneBatch(), OneBatch()),
    )

    assert len(progress_calls) == 4
    assert descriptions == ["train 1/2", "train 2/2"]
    assert all({"loss", "avg_loss", "lr", "images", "skipped", "gpu_mem"} <= call.keys() for call in progress_calls)
    assert [call["images"] for call in progress_calls] == [1, 2, 1, 2]
    assert [call["skipped"] for call in progress_calls] == [0, 0, 0, 0]


def test_validation_progress_reports_images_and_detections(monkeypatch, tmp_path: Path):
    progress_calls: list[dict[str, object]] = []

    class FakeTask:
        def advance(self, **fields):
            progress_calls.append(fields)

    @contextmanager
    def fake_progress_task(**kwargs):
        assert kwargs == {"description": "validation", "total": 1, "unit": "batch"}
        yield FakeTask()

    import cabbage_detection.training.torchvision_trainer as trainer_module

    monkeypatch.setattr(trainer_module, "progress_task", fake_progress_task)
    config = _config(tmp_path)
    evaluate_torchvision_map50_95(config, StubDetector(), OneBatch(), torch.device("cpu"))

    assert progress_calls == [{"images": 1, "detections": 0}]


def test_validation_progress_keeps_unknown_total_open(monkeypatch, tmp_path: Path):
    @contextmanager
    def fake_progress_task(**kwargs):
        assert kwargs == {"description": "validation", "total": None, "unit": "batch"}
        yield type("Task", (), {"advance": lambda self, **fields: None})()

    import cabbage_detection.training.torchvision_trainer as trainer_module

    monkeypatch.setattr(trainer_module, "progress_task", fake_progress_task)
    evaluate_torchvision_map50_95(_config(tmp_path), StubDetector(), OneBatchWithoutLength(), torch.device("cpu"))


def test_training_does_not_keep_per_epoch_files_by_default(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="train_loss", mode="min"))
    train_torchvision(config, StubDetector(), DetectionLoaders(OneBatch(), OneBatch(), OneBatch()))
    assert not list((config.output_dir / "checkpoints").glob("checkpoint_epoch_*.pt"))


def test_training_can_keep_per_epoch_files_when_requested(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="train_loss", mode="min", retain_each_epoch=True))
    train_torchvision(config, StubDetector(), DetectionLoaders(OneBatch(), OneBatch(), OneBatch()))
    epoch_files = sorted((config.output_dir / "checkpoints" / "epochs").glob("checkpoint_epoch_*.pt"))
    assert [path.name for path in epoch_files] == ["checkpoint_epoch_0001.pt", "checkpoint_epoch_0002.pt"]


def test_training_curve_data_has_integer_epochs_and_no_fake_validation(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="train_loss", mode="min"))
    from cabbage_detection.training.artifact_writer import training_curve_data

    result = train_torchvision(config, StubDetector(), DetectionLoaders(OneBatch(), OneBatch(), OneBatch()))
    data = training_curve_data(result)
    assert data["epochs"] == [1, 2]
    assert data["train_loss"] == [row.train_loss_mean for row in result.history]
    assert data["val_loss"] == []
    assert data["val_map50"] == []
    assert data["val_map50_95"] == []


def test_training_curve_data_includes_val_loss_when_validation_runs(tmp_path: Path):
    config = _config(tmp_path, checkpoint=CheckpointConfig(metric="val_map_50_95", mode="max"))
    from cabbage_detection.training.artifact_writer import training_curve_data

    reports = iter((DetectionReport(0.4, 0.2), DetectionReport(0.3, 0.1)))
    result = train_torchvision(
        config,
        StubDetector(),
        DetectionLoaders(OneBatch(), OneBatch(), OneBatch()),
        validation_evaluator=lambda model, loader, device: next(reports),
    )
    data = training_curve_data(result)
    assert len(data["val_loss"]) == 2
    assert all(math.isfinite(value) for value in data["val_loss"])


def test_training_can_skip_empty_targets_and_records_counts(tmp_path: Path):
    config = _config(tmp_path)
    config = replace(
        config,
        native_training=replace(config.native_training, empty_target_policy="skip"),
    )
    result = train_torchvision(config, StubDetector(), DetectionLoaders(MixedBatch(), OneBatch(), OneBatch()))
    row = result.history[0]
    assert (row.images_seen, row.images_trained, row.images_skipped) == (2, 1, 1)


def test_warmup_policy_handles_single_batch_without_invalid_scheduler(tmp_path: Path):
    config = _config(tmp_path)
    config = replace(
        config,
        native_training=replace(config.native_training, warmup_first_epoch=True),
    )
    result = train_torchvision(config, StubDetector(), DetectionLoaders(OneBatch(), OneBatch(), OneBatch()))
    assert result.epochs_completed == 2


def test_historical_empty_target_predicate_is_shape_based():
    assert torch.empty((0, 4)).shape[1] == 4
    with pytest.raises(IndexError):
        _ = torch.empty((0,)).shape[1]
