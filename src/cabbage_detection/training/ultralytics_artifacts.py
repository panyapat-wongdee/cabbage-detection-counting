"""Portable training logs, summaries, and curve plots for Ultralytics runs.

Mirrors ``artifact_writer.py``'s torchvision-facing artifacts so a completed
Ultralytics run is inspectable the same way: ``logs/training_log.csv``,
``logs/best_epoch.json``, ``checkpoints/{best,last}.pt``, and ``plots/*.png``
alongside the native ``framework/`` output.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ..artifact_paths import run_artifact_path
from ..data.manifests import read_manifest
from ..progress import status
from .artifact_writer import EPOCH_COLUMNS, csv_number as _number, render_training_plots, sha256_file

if TYPE_CHECKING:
    from ..config import ExperimentConfig


def _read_results_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _train_image_count(config: "ExperimentConfig") -> int:
    """Return the training-split image count, or ``0`` if the manifest is unavailable."""
    try:
        manifest = read_manifest(config.split_manifest)
    except (OSError, ValueError):
        return 0
    return sum(1 for split, _ in manifest.rows if split == "train")


def _fitness(map50: float | None, map50_95: float | None) -> float | None:
    """Ultralytics' own weighted-fitness formula, used to select ``best.pt``."""
    if map50 is None or map50_95 is None:
        return None
    return 0.1 * map50 + 0.9 * map50_95


def _train_loss_mean(row: dict[str, str]) -> float:
    """Sum every ``train/*loss`` column present in the row.

    Different Ultralytics trainers report different loss decompositions
    (YOLO: box/cls/dfl; RT-DETR: giou/cls/l1), so summing by column-name
    prefix instead of a YOLO-specific fixed set keeps this correct across
    model families instead of silently dropping unrecognized components.
    """
    return sum(_float(row, key) or 0.0 for key in row if key.startswith("train/") and key.endswith("loss"))


def _val_loss_mean(row: dict[str, str]) -> float | None:
    """Sum every ``val/*loss`` column present in the row, or ``None`` if absent.

    Mirrors ``_train_loss_mean`` but returns ``None`` rather than ``0.0`` when
    every val/*loss column is missing or unparseable, since a genuinely
    absent validation pass should not be plotted as a zero loss.
    """
    keys = [key for key in row if key.startswith("val/") and key.endswith("loss")]
    values = [_float(row, key) for key in keys]
    if not keys or all(value is None for value in values):
        return None
    return sum(value or 0.0 for value in values)


def write_ultralytics_artifacts(config: "ExperimentConfig", run_dir: Path, native_dir: Path) -> None:
    """Write torchvision-shaped artifacts from a completed Ultralytics run.

    Reads the native ``results.csv`` under ``native_dir`` and copies the
    native best/last weights into ``checkpoints/``. Raises if the native run
    did not produce the expected files.
    """
    run_dir = Path(run_dir)
    native_dir = Path(native_dir)
    results_csv = native_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Ultralytics run is missing results.csv: {results_csv}")
    weights_dir = native_dir / "weights"
    native_best = weights_dir / "best.pt"
    native_last = weights_dir / "last.pt"
    if not native_best.is_file() or not native_last.is_file():
        raise FileNotFoundError(f"Ultralytics run is missing best/last weights under {weights_dir}")

    rows = _read_results_csv(results_csv)
    if not rows:
        raise ValueError(f"Ultralytics results.csv has no epoch rows: {results_csv}")

    console_path = run_artifact_path(run_dir, "console_log")
    train_images = _train_image_count(config)

    epoch_records = []
    for row in rows:
        map50 = _float(row, "metrics/mAP50(B)")
        map50_95 = _float(row, "metrics/mAP50-95(B)")
        epoch_records.append(
            {
                "epoch": int(float(row["epoch"])),
                "train_loss_mean": _train_loss_mean(row),
                "val_loss": _val_loss_mean(row),
                "learning_rate": _float(row, "lr/pg0") or 0.0,
                "val_map50": map50,
                "val_map50_95": map50_95,
                "fitness": _fitness(map50, map50_95),
                # Ultralytics' "time" column is already cumulative seconds
                # since training started, matching torchvision's
                # ``elapsed_seconds`` semantics (time.perf_counter() - started).
                "elapsed_seconds": _float(row, "time") or 0.0,
            }
        )

    scored = [
        (index, record["fitness"]) for index, record in enumerate(epoch_records) if record["fitness"] is not None
    ]
    if scored:
        best_index, best_fitness = max(scored, key=lambda item: item[1])
    else:
        # No epoch reported a finite fitness (e.g. a validation-less smoke
        # run): fall back to the last epoch rather than accidentally
        # matching the first record via ``None == None``.
        best_index, best_fitness = len(epoch_records) - 1, None
    best_epoch = epoch_records[best_index]["epoch"]

    epochs_path = run_artifact_path(run_dir, "training_log")
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    with epochs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPOCH_COLUMNS)
        writer.writeheader()
        for index, record in enumerate(epoch_records):
            is_best = index == best_index
            writer.writerow(
                {
                    "epoch": record["epoch"],
                    "train_loss_mean": _number(record["train_loss_mean"]),
                    "val_loss": _number(record["val_loss"]),
                    "learning_rate": _number(record["learning_rate"]),
                    "val_map50": _number(record["val_map50"]),
                    "val_map50_95": _number(record["val_map50_95"]),
                    "selection_metric": "ultralytics_fitness",
                    "selection_value": _number(record["fitness"]),
                    "is_best": str(is_best).lower(),
                    "elapsed_seconds": _number(record["elapsed_seconds"]),
                    "images_seen": train_images,
                    "images_trained": train_images,
                    "images_skipped": 0,
                }
            )
            fitness_text = _number(record["fitness"]) or "n/a"
            val_loss_suffix = f" val_loss={record['val_loss']:.6f}" if record["val_loss"] is not None else ""
            status(
                f"epoch={record['epoch']} train_loss={record['train_loss_mean']:.6f}{val_loss_suffix} "
                f"lr={record['learning_rate']:.8g} selection=ultralytics_fitness:{fitness_text} "
                f"best={str(is_best).lower()} images={train_images}/{train_images} "
                f"skipped=0 elapsed={record['elapsed_seconds']:.2f}s",
                log_file=console_path,
            )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_artifact_path(run_dir, "best_checkpoint")
    last_path = run_artifact_path(run_dir, "last_checkpoint")
    shutil.copy2(native_best, best_path)
    shutil.copy2(native_last, last_path)

    summary = {
        "epoch": best_epoch,
        "selection_metric": "ultralytics_fitness",
        "selection_mode": "max",
        "selection_value": best_fitness,
        "checkpoint": best_path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(best_path),
        "last_checkpoint": last_path.relative_to(run_dir).as_posix(),
        "last_sha256": sha256_file(last_path),
    }
    run_artifact_path(run_dir, "best_epoch").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    def optional_series(values: list[float | None]) -> list[float]:
        if not any(value is not None for value in values):
            return []
        return [value if value is not None else float("nan") for value in values]

    curve_data = {
        "epochs": [record["epoch"] for record in epoch_records],
        "train_loss": [record["train_loss_mean"] for record in epoch_records],
        "val_loss": optional_series([record["val_loss"] for record in epoch_records]),
        "val_map50": optional_series([record["val_map50"] for record in epoch_records]),
        "val_map50_95": optional_series([record["val_map50_95"] for record in epoch_records]),
    }
    render_training_plots(curve_data, best_epoch, run_dir, title_prefix="Ultralytics")
    status(
        f"ultralytics artifacts complete: best_epoch={best_epoch} best_fitness={_number(best_fitness) or 'n/a'}",
        log_file=console_path,
    )
