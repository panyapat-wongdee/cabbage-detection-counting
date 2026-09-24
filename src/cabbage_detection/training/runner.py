"""Thin orchestration layer that preserves framework-specific adapter behavior."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from ..artifacts import create_run
from ..artifact_paths import run_artifact_path
from ..config import ExperimentConfig
from ..evaluation.records import PredictionRecord
from ..models.factory import create_model
from ..progress import progress, status


def run_training(
    config: ExperimentConfig,
    command: Sequence[str] | None = None,
    *,
    result_status: str | None = None,
) -> Path:
    """Create an immutable run, delegate training, and retain failure metadata."""
    selected_status = config.result_status if result_status is None else result_status
    if selected_status not in {"smoke", "reproduction_candidate"}:
        raise ValueError("result_status must be smoke or reproduction_candidate")
    context = create_run(config, command or sys.argv)
    console_path = run_artifact_path(config.output_dir, "console_log")
    # Keep the initial status terminal-only. Framework adapters may create
    # their own run directories/logs during startup; creating ``logs/`` here
    # would change that contract. The durable start/failure/completion lines
    # are appended once the adapter has established its output layout.
    status(f"training run created: model={config.model_name} framework={config.framework}")
    try:
        context.record_environment()
        if config.split_manifest.exists():
            context.record_input("split_manifest", config.split_manifest)
        result = Path(create_model(config).train())
    except Exception as error:
        status(f"training failed: {type(error).__name__}: {error}", log_file=console_path)
        context.fail(error)
        raise
    outputs: dict[str, Path] = {}
    if result.is_file():
        outputs["training"] = result
        outputs["best_checkpoint"] = result
    elif result.is_dir():
        native_files = {
            "ultralytics_args": result / "args.yaml",
            "ultralytics_results": result / "results.csv",
            "ultralytics_best_checkpoint": result / "weights" / "best.pt",
            "ultralytics_last_checkpoint": result / "weights" / "last.pt",
            "best_checkpoint": result / "weights" / "best.pt",
            "last_checkpoint": result / "weights" / "last.pt",
        }
        outputs.update({name: path for name, path in native_files.items() if path.is_file()})
    for name, artifact_name in (
        ("best_checkpoint", "best_checkpoint"),
        ("last_checkpoint", "last_checkpoint"),
        ("training_log", "console_log"),
        ("epoch_metrics", "training_log"),
        ("best_epoch", "best_epoch"),
        ("resolved_config", "config"),
        ("training_loss_plot", "training_loss_plot"),
        ("validation_metrics_plot", "validation_metrics_plot"),
    ):
        candidate = run_artifact_path(config.output_dir, artifact_name)
        if candidate.is_file():
            outputs[name] = candidate
    status("training complete; finalizing run metadata", log_file=console_path)
    context.finalize("succeeded", outputs, result_status=selected_status)
    return result


def run_prediction(config: ExperimentConfig, source: Path, checkpoint: Path) -> list[PredictionRecord]:
    """Delegate prediction to the selected adapter using the canonical record shape."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return create_model(config).predict(Path(source), checkpoint)


def run_predictions(
    config: ExperimentConfig,
    sources: Sequence[Path],
    checkpoint: Path,
    progress_description: str | None = None,
) -> list[PredictionRecord]:
    """Predict many sources with one adapter, loading the checkpoint once.

    Rebuilding the detector per image dominated evaluation runtime. The
    adapters keep a single per-source inference path, so the records are
    unchanged.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    paths = [Path(source) for source in sources]
    records = create_model(config).predict_iter(paths, checkpoint)
    if progress_description is not None:
        records = progress(records, description=progress_description, total=len(paths))
    return list(records)
