"""Portable training logs, summaries, and curve plots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..artifact_paths import run_artifact_path
from ..progress import status

if TYPE_CHECKING:
    from ..config import ExperimentConfig
    from .torchvision_trainer import EpochMetrics, TrainingResult


EPOCH_COLUMNS = (
    "epoch",
    "train_loss_mean",
    "val_loss",
    "learning_rate",
    "val_map50",
    "val_map50_95",
    "selection_metric",
    "selection_value",
    "is_best",
    "elapsed_seconds",
    "images_seen",
    "images_trained",
    "images_skipped",
)


def csv_number(value: float | None) -> str:
    """Render an optional metric for the shared epoch-CSV schema.

    Shared by every framework adapter so a missing or non-finite value is
    written the same way (an empty cell) whether the row came from
    torchvision or Ultralytics.
    """
    return "" if value is None or not math.isfinite(float(value)) else str(float(value))


# Retained as a private alias so existing call sites in this module are unaffected.
_number = csv_number


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Retained as a private alias so existing call sites in this module are unaffected.
_sha256 = sha256_file


def write_epoch(row: "EpochMetrics", run_dir: Path) -> None:
    """Append one completed epoch to CSV and human-readable training log."""
    run_dir = Path(run_dir)
    epochs_path = run_artifact_path(run_dir, "training_log")
    console_path = run_artifact_path(run_dir, "console_log")
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not epochs_path.exists() or epochs_path.stat().st_size == 0
    values: dict[str, object] = {
        "epoch": row.epoch,
        "train_loss_mean": _number(row.train_loss_mean),
        "val_loss": _number(row.val_loss),
        "learning_rate": _number(row.learning_rate),
        "val_map50": _number(row.val_map50),
        "val_map50_95": _number(row.val_map50_95),
        "selection_metric": row.selection_metric,
        "selection_value": _number(row.selection_value),
        "is_best": str(bool(row.is_best)).lower(),
        "elapsed_seconds": _number(row.elapsed_seconds),
        "images_seen": row.images_seen,
        "images_trained": row.images_trained,
        "images_skipped": row.images_skipped,
    }
    with epochs_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPOCH_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(values)
    val_parts = []
    if row.val_loss is not None:
        val_parts.append(f"val_loss={row.val_loss:.6f}")
    if row.val_map50 is not None:
        val_parts.append(f"val_map50={row.val_map50:.6f}")
    if row.val_map50_95 is not None:
        val_parts.append(f"val_map50_95={row.val_map50_95:.6f}")
    suffix = " ".join(val_parts)
    if suffix:
        suffix = " " + suffix
    status(
        f"epoch={row.epoch} train_loss={row.train_loss_mean:.6f} "
        f"lr={row.learning_rate:.8g} selection={row.selection_metric}:{row.selection_value:.6f} "
        f"best={str(bool(row.is_best)).lower()} images={row.images_trained}/{row.images_seen} "
        f"skipped={row.images_skipped} elapsed={row.elapsed_seconds:.2f}s{suffix}",
        log_file=console_path,
    )


def render_training_plots(
    curve_data: dict[str, list[float] | list[int]],
    best_epoch: int,
    run_dir: Path,
    title_prefix: str = "Torchvision",
) -> None:
    """Render deterministic training-loss/validation-metric PNG curves.

    Shared by every framework adapter so torchvision and Ultralytics runs
    produce identically named, identically shaped ``plots/*.png`` artifacts.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install matplotlib to render training curves") from error

    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = curve_data["epochs"]

    figure, axis = plt.subplots(figsize=(7, 4.5), dpi=120)
    axis.plot(epochs, curve_data["train_loss"], marker="o", label="Mean training batch loss")
    val_loss = curve_data.get("val_loss", [])
    if val_loss:
        axis.plot(epochs, val_loss, marker="o", label="Mean validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(f"{title_prefix} training loss")
    axis.set_xticks(epochs)
    axis.set_xticklabels([str(epoch) for epoch in epochs])
    axis.axvline(best_epoch, linestyle=":", color="tab:green", label=f"best epoch ({best_epoch})")
    if len(epochs) == 1:
        axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "training_loss.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5), dpi=120)
    map50 = curve_data["val_map50"]
    map50_95 = curve_data["val_map50_95"]
    if map50:
        axis.plot(epochs, map50, marker="o", label="Validation mAP@50")
    if map50_95:
        axis.plot(epochs, map50_95, marker="o", label="Validation mAP@50:95")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("mAP")
    axis.set_title(f"{title_prefix} validation metrics")
    axis.set_ylim(0, 1)
    axis.set_xticks(epochs)
    axis.set_xticklabels([str(epoch) for epoch in epochs])
    if epochs:
        axis.axvline(best_epoch, linestyle=":", color="tab:green", label=f"best epoch ({best_epoch})")
    if len(epochs) == 1:
        axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
    axis.grid(True, alpha=0.3)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "validation_metrics.png")
    plt.close(figure)


def training_curve_data(result: "TrainingResult") -> dict[str, list[float] | list[int]]:
    """Return the exact finite series used by the training plot writer."""

    epochs = [int(row.epoch) for row in result.history]
    train_loss = [float(row.train_loss_mean) for row in result.history if math.isfinite(float(row.train_loss_mean))]
    if len(train_loss) != len(epochs):
        raise ValueError("training history contains a non-finite loss")
    def optional_series(values: list[float | None]) -> list[float]:
        if not any(value is not None for value in values):
            return []
        return [
            float(value) if value is not None and math.isfinite(float(value)) else float("nan")
            for value in values
        ]

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": optional_series([row.val_loss for row in result.history]),
        "val_map50": optional_series([row.val_map50 for row in result.history]),
        "val_map50_95": optional_series([row.val_map50_95 for row in result.history]),
    }


def write_training_summary(result: "TrainingResult", run_dir: Path) -> None:
    """Write best-epoch evidence and render plots after successful training."""
    run_dir = Path(run_dir)
    if not result.history:
        raise ValueError("cannot write a training summary without epoch history")
    best_row = next((row for row in result.history if row.epoch == result.best_epoch), None)
    if best_row is None:
        raise ValueError("best_epoch is not present in training history")
    best_path = Path(result.checkpoint)
    last_path = Path(result.last_checkpoint)
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("best and last checkpoints are required for training summary")
    try:
        checkpoint_name = best_path.relative_to(run_dir).as_posix()
        last_name = last_path.relative_to(run_dir).as_posix()
    except ValueError as error:
        raise ValueError("training checkpoints must be inside the run directory") from error
    summary = {
        "epoch": result.best_epoch,
        "selection_metric": result.metric_name,
        "selection_mode": result.selection_mode,
        "selection_value": result.best_metric,
        "checkpoint": checkpoint_name,
        "sha256": _sha256(best_path),
        "last_checkpoint": last_name,
        "last_sha256": _sha256(last_path),
    }
    run_artifact_path(run_dir, "best_epoch").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    render_training_plots(training_curve_data(result), result.best_epoch, run_dir)
