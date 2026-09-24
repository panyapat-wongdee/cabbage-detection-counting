"""Repository-root command implementations for lightweight workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Literal, Sequence

from .artifact_paths import run_artifact_path
from .config import ExperimentConfig, load_config
from .data.layout import validate_dataset_root
from .data.manifests import build_manifest, read_manifest, write_manifest
from .data.coco import read_coco_detection
from .data.official_layout import discover_official_dataset
from .data.validation import validate_official_dataset
from .data.yolo import prepare_yolo_dataset
from .data.audit import audit_dataset_migration
from .evaluation.io import (
    read_prediction_records,
    read_target_records,
    write_prediction_records,
    write_target_records,
)
from .evaluation.confidence import filter_by_confidence, lower_export_confidence
from .evaluation.targets import (
    EvaluationDataset,
    build_target_records,
    load_evaluation_dataset,
    split_image_ids,
)
from .evaluation.records import PredictionRecord, TargetRecord
from .evaluation.detection import evaluate_detection
from .evaluation.coco_detection import evaluate_coco_detection
from .evaluation.counting import evaluate_counting
from .evaluation.count_artifacts import (
    counting_inputs_metadata,
    evaluate_count_files,
    write_counting_report,
)
from .evaluation.cross_model import write_cross_model
from .evaluation.summary import write_summary
from .evaluation.ultralytics_native import evaluate_ultralytics_native
from .artifacts import finalize_reproduction_run, write_run_metadata
from .reproduction_evidence import collect_run_evidence
from .training.runner import run_prediction, run_predictions, run_training
from .progress import progress, status


EVALUATION_TASKS: Final = ("detection", "counting")
DEFAULT_EVALUATION_ROOT: Final = Path("results") / "reproduced"
DETECTION_BACKENDS: Final = ("config", "repository_ap_101", "pycocotools_coco_eval")
DATASET_SPLITS: Final = ("train", "val", "test")
SPLIT_SELECTIONS: Final = {
    "train": ("train",),
    "val": ("val",),
    "test": ("test",),
    "all": DATASET_SPLITS,
}
TASK_SELECTIONS: Final = {
    "detection": ("detection",),
    "counting": ("counting",),
    "both": ("detection", "counting"),
}


@dataclass(frozen=True)
class EvaluationInputs:
    """Validated arguments for one evaluation run resolved from a run directory."""

    run_dir: Path
    config: Path
    checkpoint: Path
    dataset_root: Path
    splits: tuple[str, ...]
    output_dir: Path
    tasks: tuple[str, ...]
    detection_backend: str = "config"
    ap_confidence: float | None = None
    native_score_threshold: float | None = None
    ultralytics_native: bool = False


def resolve_dataset_inputs(
    dataset_root: Path | None,
    source_root: Path | None,
    annotations: Path | None,
) -> tuple[Path, Path]:
    """Resolve either an extracted official download or explicit input paths."""

    if dataset_root is not None:
        if source_root is not None or annotations is not None:
            raise ValueError("--dataset-root cannot be combined with --source-root or --annotations")
        layout = discover_official_dataset(dataset_root)
        return layout.images_root, layout.annotations_path
    if source_root is None or annotations is None:
        raise ValueError("provide --dataset-root or both --source-root and --annotations")
    return Path(source_root), Path(annotations)


def prepare_dataset(
    source_root: Path | None,
    manifest_path: Path,
    *,
    dataset_root: Path | None = None,
    annotations: Path | None = None,
    format: str = "validate",
    output: Path | None = None,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> Path:
    """Validate an official COCO source or build the current manifest format."""
    if format not in {"validate", "yolo", "all"}:
        raise ValueError(f"unsupported dataset preparation format: {format}")
    if dataset_root is not None:
        source_root, annotations = resolve_dataset_inputs(dataset_root, source_root, annotations)
    if source_root is None:
        raise ValueError("source root is required for dataset preparation")
    status(f"dataset preparation started: format={format}")
    if annotations is not None:
        manifest = read_manifest(Path(manifest_path))
        validate_official_dataset(Path(source_root), Path(annotations), manifest)
        if format == "validate":
            result = Path(manifest_path)
            status(f"dataset validation complete: manifest={result}")
            return result
        if output is None:
            raise ValueError(f"{format} preparation requires --output")
        result = prepare_yolo_dataset(
            read_coco_detection(Path(annotations)), manifest, Path(output), Path(source_root), link_mode, overwrite
        )
        status(f"YOLO dataset preparation complete: output={result.output}", log_file=Path(result.output) / "console.log")
        return Path(output)
    if format != "validate":
        raise ValueError("--annotations is required for YOLO dataset preparation")
    layout = validate_dataset_root(Path(source_root))
    manifest = build_manifest(layout, provenance="validated-by-cabbage-detection-cli")
    write_manifest(manifest, Path(manifest_path))
    result = Path(manifest_path)
    status(f"dataset manifest complete: manifest={result}")
    return result


def evaluate_counts(
    predictions_path: Path,
    targets_path: Path,
    output_path: Path,
    iou_threshold: float = 0.5,
) -> Path:
    """Evaluate canonical dataset-level prediction and target records."""
    return evaluate_count_files(predictions_path, targets_path, output_path, iou_threshold)


def train(config_path: Path) -> Path:
    """Run the selected framework adapter using a strict configuration."""
    return run_training(load_config(Path(config_path)))


def _read_evidence_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} evidence JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} evidence JSON must contain an object")
    return {str(key): value for key, value in payload.items()}


def compare_models(
    root: Path, split: str, models: Sequence[str] | None = None, variant: str | None = None
) -> tuple[Path, Path]:
    """Write the cross-model table for one split of an evaluated results root."""
    status(f"model comparison started: root={root} split={split} variant={variant}")
    title = f"Reproduced model comparison ({split} split" + (f", {variant} variant)" if variant else ")")
    json_path, markdown_path = write_cross_model(root, split, title, models, variant)
    status(f"model comparison written: {markdown_path}")
    return json_path, markdown_path


def _resolve_results_dir(run_dir: Path, results_dir: Path | None) -> Path:
    """Default an evaluation output directory to the run's own results folder."""
    return Path(results_dir) if results_dir is not None else DEFAULT_EVALUATION_ROOT / run_dir.name


def collect_evidence(
    run_dir: Path,
    results_dir: Path | None,
    dataset_root: Path,
    *,
    split: str = "test",
    splits: Sequence[str] = DATASET_SPLITS,
) -> Path:
    """Build the evidence documents a promotion needs from existing artifacts."""
    run_dir = Path(run_dir)
    resolved_results = _resolve_results_dir(run_dir, results_dir)
    status(f"evidence collection started: run={run_dir} results={resolved_results}")
    evidence = collect_run_evidence(
        run_dir, resolved_results, dataset_root, split=split, splits=tuple(splits)
    )
    status(f"evidence collection complete: {evidence.dataset.parent}")
    return evidence.dataset.parent


def promote_run(
    run_dir: Path,
    results_dir: Path | None,
    dataset_root: Path,
    *,
    split: str = "test",
    splits: Sequence[str] = DATASET_SPLITS,
) -> Path:
    """Collect evidence for one run and promote it in a single auditable step."""
    run_dir = Path(run_dir)
    resolved_results = _resolve_results_dir(run_dir, results_dir)
    evidence = collect_run_evidence(
        run_dir, resolved_results, dataset_root, split=split, splits=tuple(splits)
    )
    status(f"promotion started: {evidence.model_name}")
    return finalize_reproduction(
        run_dir,
        evidence.predictions,
        evidence.metrics,
        evidence.dataset,
        evidence.pretrained,
        evidence.evaluation,
        evidence.deviations,
    )


def finalize_reproduction(
    run_dir: Path,
    predictions: Path,
    metrics: Path,
    dataset_json: Path,
    pretrained_json: Path,
    evaluation_json: Path,
    deviations: Sequence[str] = (),
) -> Path:
    """Attach explicit evidence and promote a successful candidate run."""
    # The run metadata already hashes console.log; finalization must not mutate
    # that evidence while adding the prediction/metrics hashes.
    status(f"finalization started: run={run_dir}")
    dataset = _read_evidence_mapping(dataset_json, "dataset")
    pretrained = _read_evidence_mapping(pretrained_json, "pretrained")
    evaluation = _read_evidence_mapping(evaluation_json, "evaluation")
    status("finalization committing metadata")
    return finalize_reproduction_run(
        run_dir,
        predictions,
        metrics,
        dataset=dataset,
        pretrained=pretrained,
        evaluation=evaluation,
        deviations=deviations,
    )


def predict(config_path: Path, source: Path, checkpoint_path: Path, output_path: Path) -> Path:
    """Run prediction and require adapters to return canonical records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console_path = output_path.parent / "console.log"
    status(
        f"prediction started: source={Path(source).name} checkpoint={Path(checkpoint_path).name}",
        log_file=console_path,
    )
    records = run_prediction(load_config(Path(config_path)), Path(source), Path(checkpoint_path))
    if not all(hasattr(record, "image_id") and hasattr(record, "detections") for record in records):
        raise ValueError("prediction adapter did not return canonical PredictionRecord values")
    from .evaluation.io import write_prediction_records

    result = write_prediction_records(output_path, records)
    status(f"prediction complete: records={len(records)} output={result}", log_file=console_path)
    return result


def _write_detection_report(
    config: ExperimentConfig,
    predictions: Sequence[PredictionRecord],
    targets: Sequence[TargetRecord],
    output_path: Path,
    backend: str = "config",
    export_confidence: float | None = None,
    native_score_threshold: float | None = None,
    console_path: Path | None = None,
) -> dict[str, object]:
    """Write ``detection.json`` and return the settings it was computed with."""
    if backend not in DETECTION_BACKENDS:
        raise ValueError(f"unsupported detection backend: {backend}")
    selected = config.evaluation.backend if backend == "config" else backend
    if selected == "pycocotools_coco_eval":
        report = evaluate_coco_detection(
            predictions, targets, max_detections=config.evaluation.max_detections
        )
    else:
        report = evaluate_detection(predictions, targets)
    settings = {
        "ap_iou_threshold": 0.5,
        "ap_iou_range": "0.50:0.95 step 0.05",
        "backend": report.backend,
        "backend_selection": backend,
        "configured_backend": config.evaluation.backend,
        "export_confidence_threshold": (
            config.confidence_threshold if export_confidence is None else export_confidence
        ),
        "native_score_threshold": native_score_threshold,
        "max_detections": config.evaluation.max_detections,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "detection",
        "input_schema": "canonical",
        "scope": {"image_count": len(targets)},
        "metrics": {
            "map50": report.map50,
            "map50_95": report.map50_95,
            "backend": report.backend,
        },
        "evaluation": settings,
        "units": {"map50": "fraction", "map50_95": "fraction"},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status(
        f"detection evaluation complete: backend={report.backend} "
        f"map50={report.map50:.6f} map50_95={report.map50_95:.6f}",
        log_file=console_path,
    )
    return settings


def generate_evaluation_records(
    config: ExperimentConfig,
    checkpoint_path: Path,
    dataset: EvaluationDataset,
    split: str,
    output_dir: Path,
    export_confidence: float | None = None,
    console_path: Path | None = None,
) -> tuple[Path, Path]:
    """Predict one split and write its canonical prediction/target records."""
    if export_confidence is not None:
        config = lower_export_confidence(config, export_confidence)
    image_ids = split_image_ids(config.split_manifest, split)
    targets = build_target_records(config, dataset.coco, image_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    if console_path is None:
        console_path = output_dir / "console.log"
    status(
        f"record generation started: split={split} images={len(image_ids)} "
        f"framework={config.framework} export_confidence={config.confidence_threshold}",
        log_file=console_path,
    )
    image_paths = [dataset.image_path(image_id) for image_id in image_ids]
    predictions = run_predictions(
        config, image_paths, Path(checkpoint_path), progress_description=f"predict[{split}]"
    )
    predictions_path = write_prediction_records(output_dir / "predictions.json", list(predictions))
    targets_path = write_target_records(output_dir / "targets.json", list(targets))
    status(
        f"record generation complete: predictions={len(predictions)} targets={len(targets)}",
        log_file=console_path,
    )
    return predictions_path, targets_path


def evaluate_canonical(
    config_path: Path,
    predictions_path: Path,
    targets_path: Path,
    output_dir: Path,
    tasks: Sequence[str] = EVALUATION_TASKS,
    detection_backend: str = "config",
    counting_confidence: float | None = None,
    export_confidence: float | None = None,
    native_score_threshold: float | None = None,
) -> Path:
    """Evaluate canonical records for the selected detection/counting tasks.

    Every artifact for one split lands directly in ``output_dir``: the task
    reports, one ``metadata.json`` shared by both tasks, and one
    ``console.log``. Results are staged in a temporary directory so a partial
    evaluation is never published.
    """
    selected = tuple(tasks)
    unknown = [task for task in selected if task not in EVALUATION_TASKS]
    if unknown:
        raise ValueError(f"unsupported evaluation task: {', '.join(sorted(set(unknown)))}")
    if not selected:
        raise ValueError("at least one evaluation task is required")
    config = load_config(Path(config_path))
    predictions_path = Path(predictions_path)
    targets_path = Path(targets_path)
    predictions = read_prediction_records(predictions_path)
    targets = read_target_records(targets_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"evaluation output is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    console_path = temporary / "console.log"
    try:
        evaluation: dict[str, object] = {"tasks": list(selected)}
        if "detection" in selected:
            evaluation["detection"] = _write_detection_report(
                config,
                predictions,
                targets,
                temporary / "detection.json",
                detection_backend,
                export_confidence=export_confidence,
                native_score_threshold=native_score_threshold,
                console_path=console_path,
            )
        if "counting" in selected:
            threshold = (
                config.confidence_threshold if counting_confidence is None else counting_confidence
            )
            counted = filter_by_confidence(predictions, threshold)
            counting_report = evaluate_counting(counted, targets, config.counting_iou_threshold)
            write_counting_report(
                counting_report,
                temporary / "counting.json",
                config.counting_iou_threshold,
                confidence_threshold=threshold,
                console_path=console_path,
            )
            evaluation["counting"] = {
                "iou_threshold": config.counting_iou_threshold,
                "confidence_threshold": threshold,
                "matching": "one_to_one_greedy_highest_iou",
                "aggregation": "micro",
            }
        metadata_path = write_run_metadata(
            temporary, config, command=sys.argv, filename="metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "input_schema": "canonical",
                "evaluation": evaluation,
                "deviations": (
                    ["native detector score threshold lowered for full-range AP"]
                    if native_score_threshold is not None
                    else []
                ),
                "inputs": counting_inputs_metadata(predictions_path, targets_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        for child in temporary.iterdir():
            child.replace(output_dir / child.name)
        status(
            f"evaluation complete: tasks={','.join(selected)}",
            log_file=output_dir / "console.log",
        )
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


evaluate = evaluate_canonical


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def audit_dataset(
    source_root: Path,
    annotations: Path,
    manifest: Path,
    generated_yolo: Path,
    output: Path,
    historical_yolo: Path | None = None,
) -> Path:
    status(f"dataset migration audit started: manifest={manifest}")
    report = audit_dataset_migration(source_root, annotations, manifest, generated_yolo, historical_yolo)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    if report.missing_ids or report.overlapping_ids or report.annotation_count_mismatches or report.maximum_yolo_round_trip_error > 1e-6:
        status(f"dataset migration audit failed: output={destination}", log_file=destination.parent / "console.log")
        raise ValueError("dataset migration audit failed; inspect the audit report")
    status(f"dataset migration audit complete: output={destination}", log_file=destination.parent / "console.log")
    return destination


def _resolve_evaluation_output(output_dir: Path | None, run_dir: Path) -> Path:
    """Derive the results directory from the run name unless one is given."""
    if output_dir is not None:
        return Path(output_dir)
    run_name = run_dir.resolve().name
    if not run_name:
        raise ValueError("cannot derive an output directory from the run path; pass --output-dir")
    return DEFAULT_EVALUATION_ROOT / run_name


def _resolve_evaluation_inputs(args: argparse.Namespace) -> EvaluationInputs:
    """Resolve the config, checkpoint, and dataset behind one evaluation run."""
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    config_path = run_artifact_path(run_dir, "config")
    checkpoint_path = run_artifact_path(run_dir, "best_checkpoint")
    for label, path in (("config.yaml", config_path), ("checkpoints/best.pt", checkpoint_path)):
        if not path.is_file():
            raise ValueError(f"run directory is missing {label}: {path}")
    config = load_config(config_path)
    if args.ultralytics_native and config.framework != "ultralytics":
        raise ValueError(
            "--ultralytics-native requires a run whose config uses framework: ultralytics; "
            f"this run uses {config.framework}"
        )
    tasks = TASK_SELECTIONS[args.tasks]
    ap_confidence = args.ap_confidence
    if ap_confidence is not None:
        if not 0 <= ap_confidence <= 1:
            raise ValueError("--ap-confidence must be between 0 and 1")
        if args.ultralytics_native:
            raise ValueError("--ap-confidence does not apply to --ultralytics-native")
        if ap_confidence > config.confidence_threshold:
            raise ValueError(
                "--ap-confidence must not exceed the configured confidence_threshold "
                f"({config.confidence_threshold})"
            )
    native_score_threshold = args.native_score_threshold
    if native_score_threshold is not None:
        if not 0 <= native_score_threshold <= 1:
            raise ValueError("--native-score-threshold must be between 0 and 1")
        if config.framework != "torchvision":
            raise ValueError(
                "--native-score-threshold applies to torchvision detectors; "
                "Ultralytics exposes its score gate through --ap-confidence"
            )
        if args.ultralytics_native:
            raise ValueError("--native-score-threshold does not apply to --ultralytics-native")
    if args.ultralytics_native and args.detection_backend != "config":
        raise ValueError("--detection-backend does not apply to --ultralytics-native")
    if args.ultralytics_native and "counting" in tasks:
        raise ValueError(
            "--ultralytics-native reports detection only; "
            "drop it or use --tasks detection for the native score"
        )
    dataset_root = Path(args.dataset)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root does not exist: {dataset_root}")
    output_dir = _resolve_evaluation_output(args.output_dir, run_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"evaluation output is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(f"evaluation output directory is not empty: {output_dir}")
    return EvaluationInputs(
        run_dir=run_dir,
        config=config_path,
        checkpoint=checkpoint_path,
        dataset_root=dataset_root,
        splits=SPLIT_SELECTIONS[args.split],
        output_dir=output_dir,
        tasks=tasks,
        detection_backend=args.detection_backend,
        ap_confidence=ap_confidence,
        native_score_threshold=args.native_score_threshold,
        ultralytics_native=bool(args.ultralytics_native),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cabbage-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-dataset")
    prepare_inputs = prepare.add_mutually_exclusive_group()
    prepare_inputs.add_argument(
        "--dataset-root", type=Path, help="outer directory of the extracted official Mendeley download"
    )
    prepare_inputs.add_argument(
        "--source-root", type=Path, help="explicit image root (compatibility input; pair with --annotations)"
    )
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--annotations", type=Path)
    prepare.add_argument("--format", choices=("validate", "yolo", "all"), default="validate")
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    prepare.add_argument("--overwrite", action="store_true")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize-reproduction")
    finalize_parser.add_argument("--run-dir", type=Path, required=True)
    finalize_parser.add_argument("--predictions", type=Path, required=True)
    finalize_parser.add_argument("--metrics", type=Path, required=True)
    finalize_parser.add_argument("--dataset-json", type=Path, required=True)
    finalize_parser.add_argument("--pretrained-json", type=Path, required=True)
    finalize_parser.add_argument("--evaluation-json", type=Path, required=True)
    finalize_parser.add_argument("--deviation", action="append", default=[])
    compare_parser = subparsers.add_parser("compare-models")
    compare_parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT,
        help=f"root holding one directory per evaluated model (default: {DEFAULT_EVALUATION_ROOT})",
    )
    compare_parser.add_argument(
        "--split",
        choices=DATASET_SPLITS,
        default="test",
        help="split to tabulate; splits are never pooled (default: test)",
    )
    compare_parser.add_argument(
        "--variant",
        help=(
            "read <model>-<variant> where that directory exists, e.g. detector512; the output is "
            "model_comparison_<split>_<variant>.{md,json} (default: no variant)"
        ),
    )
    compare_parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="restrict the table to this model; repeatable (default: every model found)",
    )
    evidence_parser = subparsers.add_parser("collect-evidence")
    evidence_parser.add_argument("--run", type=Path, required=True, help="training run directory to describe")
    evidence_parser.add_argument("--dataset", type=Path, required=True, help="extracted official dataset download root")
    evidence_parser.add_argument("--results", type=Path, help="evaluation output directory (default: results/reproduced/<run name>)")
    evidence_parser.add_argument("--split", choices=DATASET_SPLITS, default="test", help="split whose predictions are staged as promotion evidence (default: test)")
    promote_parser = subparsers.add_parser("promote-run")
    promote_parser.add_argument("--run", type=Path, required=True, help="training run directory to promote")
    promote_parser.add_argument("--dataset", type=Path, required=True, help="extracted official dataset download root")
    promote_parser.add_argument("--results", type=Path, help="evaluation output directory (default: results/reproduced/<run name>)")
    promote_parser.add_argument("--split", choices=DATASET_SPLITS, default="test", help="split whose predictions are staged as promotion evidence (default: test)")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--config", type=Path, required=True)
    predict_parser.add_argument("--source", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    unified_eval = subparsers.add_parser("evaluate")
    unified_eval.add_argument(
        "--run", type=Path, required=True, help="training run directory holding config.yaml and checkpoints/best.pt"
    )
    unified_eval.add_argument(
        "--dataset", type=Path, required=True, help="extracted official dataset download root"
    )
    unified_eval.add_argument(
        "--split",
        choices=tuple(SPLIT_SELECTIONS),
        default="all",
        help="dataset split to score, or all to score each split separately (default: all)",
    )
    unified_eval.add_argument(
        "--tasks", choices=tuple(TASK_SELECTIONS), default="both", help="reports to write (default: both)"
    )
    unified_eval.add_argument(
        "--detection-backend",
        choices=DETECTION_BACKENDS,
        default="config",
        help="AP backend; config uses evaluation.backend, the others force one for cross-model tables",
    )
    unified_eval.add_argument(
        "--ap-confidence",
        type=float,
        help=(
            "export detections down to this confidence so AP covers the full score range; "
            "counting still filters at the configured threshold"
        ),
    )
    unified_eval.add_argument(
        "--native-score-threshold",
        type=float,
        help=(
            "lower a torchvision detector's own score gate so AP covers the full score range; "
            "this deviates from the published model behaviour and is recorded as a deviation"
        ),
    )
    unified_eval.add_argument(
        "--ultralytics-native",
        action="store_true",
        help="score with the framework's own model.val instead of repository metrics",
    )
    unified_eval.add_argument(
        "--output-dir",
        type=Path,
        help="results directory (default: results/reproduced/<run name>)",
    )
    audit = subparsers.add_parser("audit-dataset-migration")
    audit_inputs = audit.add_mutually_exclusive_group()
    audit_inputs.add_argument(
        "--dataset-root", type=Path, help="outer directory of the extracted official Mendeley download"
    )
    audit_inputs.add_argument(
        "--source-root", type=Path, help="explicit image root (compatibility input; pair with --annotations)"
    )
    audit.add_argument("--annotations", type=Path, help="explicit COCO JSON (compatibility input)")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--generated-yolo", type=Path, required=True)
    audit.add_argument("--historical-yolo", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-dataset":
            if args.dataset_root is not None or args.annotations is not None:
                source_root, annotations = resolve_dataset_inputs(
                    args.dataset_root, args.source_root, args.annotations
                )
            else:
                source_root, annotations = args.source_root, None
            prepare_dataset(source_root, args.manifest, annotations=annotations, format=args.format, output=args.output, link_mode=args.link_mode, overwrite=args.overwrite)
        elif args.command == "train":
            train(args.config)
        elif args.command == "finalize-reproduction":
            finalize_reproduction(
                args.run_dir,
                args.predictions,
                args.metrics,
                args.dataset_json,
                args.pretrained_json,
                args.evaluation_json,
                args.deviation,
            )
        elif args.command == "compare-models":
            compare_models(args.results_root, args.split, args.models, args.variant)
        elif args.command == "collect-evidence":
            collect_evidence(args.run, args.results, args.dataset, split=args.split)
        elif args.command == "promote-run":
            promote_run(args.run, args.results, args.dataset, split=args.split)
        elif args.command == "predict":
            predict(args.config, args.source, args.checkpoint, args.output)
        elif args.command == "audit-dataset-migration":
            source_root, annotations = resolve_dataset_inputs(
                args.dataset_root, args.source_root, args.annotations
            )
            audit_dataset(source_root, annotations, args.manifest, args.generated_yolo, args.output, args.historical_yolo)
        elif args.command == "evaluate":
            inputs = _resolve_evaluation_inputs(args)
            config = load_config(inputs.config)
            output_root = Path(inputs.output_dir)
            # The official annotations are parsed once and reused by every
            # split instead of being re-read per split.
            dataset = (
                None
                if inputs.ultralytics_native
                else load_evaluation_dataset(inputs.dataset_root)
            )
            for split in inputs.splits:
                split_dir = output_root / split
                if inputs.ultralytics_native:
                    evaluate_ultralytics_native(
                        config, inputs.checkpoint, config.dataset.yaml, split, split_dir
                    )
                    continue
                records_dir = split_dir / "records"
                predictions, targets = generate_evaluation_records(
                    replace(config, native_score_threshold=inputs.native_score_threshold)
                    if inputs.native_score_threshold is not None
                    else config,
                    inputs.checkpoint,
                    dataset,
                    split,
                    records_dir,
                    export_confidence=inputs.ap_confidence,
                    console_path=split_dir / "console.log",
                )
                staged = split_dir / ".reports"
                evaluate_canonical(
                    inputs.config,
                    predictions,
                    targets,
                    staged,
                    tasks=inputs.tasks,
                    detection_backend=inputs.detection_backend,
                    export_confidence=inputs.ap_confidence,
                    native_score_threshold=inputs.native_score_threshold,
                )
                # ``evaluate_canonical`` publishes atomically into its own
                # directory; lift those files next to ``records/`` so a split
                # directory has no extra level to open. The split already has a
                # console log from record generation, so the staged one is
                # appended instead of replacing it.
                for child in staged.iterdir():
                    destination = split_dir / child.name
                    if child.name == "console.log" and destination.is_file():
                        with destination.open("a", encoding="utf-8") as stream:
                            stream.write(child.read_text(encoding="utf-8"))
                        child.unlink()
                        continue
                    child.replace(destination)
                staged.rmdir()
            if not inputs.ultralytics_native:
                json_path, markdown_path = write_summary(
                    output_root, inputs.splits, f"{inputs.run_dir.name} evaluation"
                )
                status(f"summary written: {markdown_path.name} and {json_path.name}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0
