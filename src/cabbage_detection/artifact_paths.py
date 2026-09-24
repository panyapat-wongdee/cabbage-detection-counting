"""Stable relative paths for generated training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path


_RUN_ARTIFACTS: dict[str, Path] = {
    "config": Path("config.yaml"),
    "console_log": Path("logs") / "console.log",
    "training_log": Path("logs") / "training_log.csv",
    "best_epoch": Path("logs") / "best_epoch.json",
    "best_checkpoint": Path("checkpoints") / "best.pt",
    "last_checkpoint": Path("checkpoints") / "last.pt",
    "training_loss_plot": Path("plots") / "training_loss.png",
    "validation_metrics_plot": Path("plots") / "validation_metrics.png",
}


def run_artifact_path(run_dir: Path, name: str) -> Path:
    """Return a named artifact path below ``run_dir``.

    The map is deliberately closed so a typo cannot silently create a file in
    the wrong part of a run directory.
    """

    try:
        relative = _RUN_ARTIFACTS[name]
    except KeyError as error:
        raise KeyError(f"unknown run artifact: {name}") from error
    return Path(run_dir) / relative


def evaluation_artifact_path(output_dir: Path, name: str) -> Path:
    """Return a stable path for a task-specific evaluation artifact."""

    mapping = {
        "metrics": Path("metrics.json"),
        "per_image": Path("per_image.csv"),
        "metadata": Path("evaluation_metadata.json"),
        "count_scatter": Path("plots") / "count_scatter.png",
        "count_error_histogram": Path("plots") / "count_error_histogram.png",
    }
    try:
        relative = mapping[name]
    except KeyError as error:
        raise KeyError(f"unknown evaluation artifact: {name}") from error
    return Path(output_dir) / relative
