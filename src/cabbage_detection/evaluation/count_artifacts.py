"""Counting evaluation reports and plots for canonical records."""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..artifacts import write_run_metadata
from .counting import CountingReport, evaluate_counting
from .io import read_prediction_records, read_target_records
from ..progress import status


METRIC_UNITS = {
    "count_error": "signed_fraction_of_actual_count",
    "fdr": "fraction",
    "fnr": "fraction",
    "f1": "fraction",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Represent an input path without leaking a machine-specific absolute path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def count_plot_data(report: CountingReport) -> dict[str, list[float] | list[int]]:
    """Return sorted, frequency-collapsed values used by both counting plots."""

    ordered = [report.per_image[key] for key in sorted(report.per_image)]
    pairs = Counter((item.actual_count, item.predicted_count) for item in ordered)
    points = sorted(pairs.items())
    return {
        "actual": [pair[0][0] for pair in points],
        "predicted": [pair[0][1] for pair in points],
        "errors": [item.predicted_count - item.actual_count for item in ordered],
        "point_sizes": [40 + 20 * pair[1] for pair in points],
    }


def count_histogram_bins(errors: list[int], max_bins: int = 60) -> list[float]:
    """Return bounded histogram edges that cover all integer count errors."""

    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    if not errors:
        return [0.0, 1.0]
    low, high = min(errors), max(errors)
    width = high - low + 1
    edge_count = min(max_bins, width + 1)
    if edge_count == width + 1:
        return [float(value) for value in range(low, high + 2)]
    return [low + (high + 1 - low) * index / (edge_count - 1) for index in range(edge_count)]


def _write_plots(report: CountingReport, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install matplotlib to render counting plots") from error

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = count_plot_data(report)
    actual = data["actual"]
    predicted = data["predicted"]

    figure, axis = plt.subplots(figsize=(6, 5), dpi=120)
    axis.scatter(actual, predicted, s=data["point_sizes"], alpha=0.75, edgecolors="none")
    upper = max(actual + predicted + [1])
    axis.plot([0, upper], [0, upper], linestyle="--", color="black", linewidth=1, label="perfect count")
    axis.set_xlabel("Actual count")
    axis.set_ylabel("Predicted count")
    axis.set_title("Actual vs predicted cabbage count")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "count_scatter.png")
    plt.close(figure)

    errors = data["errors"]
    figure, axis = plt.subplots(figsize=(6, 4.5), dpi=120)
    axis.hist(errors, bins=count_histogram_bins(errors), align="left", rwidth=0.8)
    axis.axvline(0, linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Predicted count − actual count")
    axis.set_ylabel("Images")
    axis.set_title("Counting error distribution")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(plot_dir / "count_error_histogram.png")
    plt.close(figure)


def _write_per_image_csv(report: CountingReport, output_dir: Path) -> Path:
    destination = output_dir / "per_image.csv"
    columns = ("image_id", "predicted_count", "actual_count", "count_error", "fdr", "fnr", "f1")
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for image_id in sorted(report.per_image):
            writer.writerow({"image_id": image_id, **asdict(report.per_image[image_id])})
    return destination


def macro_metrics(report: CountingReport) -> dict[str, float]:
    """Average the per-image metrics.

    This is a macro average over images. It is reported beside the aggregate
    (micro) metrics because the two differ, and it is never the headline
    figure; which one an external publication used is not recorded here.
    """
    rows = list(report.per_image.values())
    if not rows:
        return {name: 0.0 for name in METRIC_UNITS}
    return {
        name: sum(getattr(row, name) for row in rows) / len(rows) for name in METRIC_UNITS
    }


def write_counting_report(
    report: CountingReport,
    output_path: Path,
    iou_threshold: float,
    confidence_threshold: float | None = None,
    console_path: Path | None = None,
) -> Path:
    """Write the counting metrics, per-image rows, and plots.

    Metadata is composed by the caller so one evaluation writes a single
    metadata file instead of one per task.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    status(
        f"counting evaluation started: images={len(report.per_image)}",
        log_file=console_path,
    )
    _write_per_image_csv(report, output_dir)
    _write_plots(report, output_dir)
    if report.match_totals is None:
        raise ValueError("canonical counting report is missing aggregate match totals")
    payload = {
        "task": "counting",
        "input_schema": "canonical",
        "scope": {"image_count": len(report.per_image)},
        "metrics": {
            "actual_count": report.totals.actual_count,
            "predicted_count": report.totals.predicted_count,
            "true_positives": report.match_totals.true_positives,
            "false_positives": report.match_totals.false_positives,
            "false_negatives": report.match_totals.false_negatives,
            "count_error": report.totals.count_error,
            "fdr": report.totals.fdr,
            "fnr": report.totals.fnr,
            "f1": report.totals.f1,
        },
        "metrics_macro": macro_metrics(report),
        "evaluation": {
            "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold,
            "matching": "one_to_one_greedy_highest_iou",
            "aggregation": "micro",
            "macro_aggregation": "mean_over_images",
        },
        "units": METRIC_UNITS,
        "files": {"per_image_csv": "per_image.csv", "plots": "plots"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status(
        f"counting evaluation complete: images={len(report.per_image)} report={output_path.name}",
        log_file=console_path,
    )
    return output_path


def counting_inputs_metadata(predictions_path: Path, targets_path: Path) -> dict[str, object]:
    """Describe the record files a counting report was computed from."""
    predictions_path = Path(predictions_path)
    targets_path = Path(targets_path)
    return {
        "predictions": {"path": _portable_path(predictions_path), "sha256": _sha256(predictions_path)},
        "targets": {"path": _portable_path(targets_path), "sha256": _sha256(targets_path)},
    }


def evaluate_count_files(
    predictions_path: Path,
    targets_path: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
) -> Path:
    """Evaluate canonical dataset-level prediction and target records."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    predictions_path = Path(predictions_path)
    targets_path = Path(targets_path)
    # The canonical readers enforce the one current schema-less records format.
    # Versioned, list-based, and malformed payloads are rejected.
    predictions = read_prediction_records(predictions_path)
    targets = read_target_records(targets_path)
    report = evaluate_counting(predictions, targets, iou_threshold)
    output_path = Path(output_path)
    result = write_counting_report(report, output_path, iou_threshold)
    metadata_path = write_run_metadata(
        output_path.parent, config=None, filename="metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "input_schema": "canonical",
            "iou_threshold": iou_threshold,
            "metric_units": METRIC_UNITS,
            "inputs": counting_inputs_metadata(predictions_path, targets_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return result
