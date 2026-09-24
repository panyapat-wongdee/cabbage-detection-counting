"""Collect one split's reports from several evaluated runs into one table.

``summary.py`` rolls the splits of a single run up into ``summary.json``. This
module is the other axis: it reads the per-model evaluation outputs under one
results root and renders the cross-model table for a chosen split.

A cross-model table is only meaningful when every row shares one metric
definition, so the collector refuses to mix AP backends, image counts, or
counting thresholds instead of quietly printing numbers that were measured
differently. Nothing here reads published values; every row is a number this
repository produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..release_artifacts import (
    MODEL_NAMES,
    TORCHVISION_MODELS,
    study_scope,
)
from .summary import COUNTING_COLUMNS, DETECTION_COLUMNS, render_table

MODEL_ORDER: dict[str, int] = {name: index for index, name in enumerate(MODEL_NAMES)}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _framework(model: str) -> str:
    return "torchvision" if model in TORCHVISION_MODELS else "ultralytics"


def _study_scope(model: str) -> str:
    return study_scope(model)


def discover_models(root: Path) -> tuple[str, ...]:
    """Return the known model directories under ``root`` in reporting order.

    Only directories named after a supported model are collected. Alternative
    scorings of the same run (for example a ``-native`` output directory) carry
    a different metric definition and are never folded into this table.
    """
    root = Path(root)
    found = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name in MODEL_ORDER and (path / "summary.json").is_file()
    ]
    return tuple(sorted(found, key=lambda name: MODEL_ORDER[name]))


def _collect_model(root: Path, model: str, split: str, directory: str | None = None) -> dict[str, Any]:
    directory = directory or model
    model_dir = root / directory
    summary = _read_json(model_dir / "summary.json")
    if summary is None:
        raise ValueError(f"{model}: no summary.json under {model_dir}")
    entry = summary.get("splits", {}).get(split)
    if not entry:
        raise ValueError(f"{model}: summary.json has no {split!r} split")

    record: dict[str, Any] = {
        "model": model,
        "framework": _framework(model),
        "study_scope": _study_scope(model),
        # The results directory the row was read from; it differs from the
        # model key only for a variant run such as ``fcos-detector512``.
        "source": directory,
    }
    detection = entry.get("detection")
    if detection is not None:
        record["detection"] = {
            "image_count": detection["image_count"],
            "backend": detection["backend"],
            **{name: detection[name] for name in DETECTION_COLUMNS},
        }
        report = _read_json(model_dir / split / "detection.json")
        if report is None:
            raise ValueError(f"{model}: {split}/detection.json is required to report its settings")
        record["detection"]["evaluation"] = report["evaluation"]
    counting = entry.get("counting")
    if counting is not None:
        record["counting"] = {
            "image_count": counting["image_count"],
            **{name: counting[name] for name in COUNTING_COLUMNS},
        }
        report = _read_json(model_dir / split / "counting.json")
        if report is None:
            raise ValueError(f"{model}: {split}/counting.json is required to report its settings")
        record["counting"]["evaluation"] = report["evaluation"]
    if "detection" not in record and "counting" not in record:
        raise ValueError(f"{model}: the {split!r} split holds neither report")

    metadata = _read_json(model_dir / split / "metadata.json") or {}
    record["deviations"] = list(metadata.get("deviations", ()))
    record["git_revision"] = metadata.get("git_revision")
    record["result_status"] = (
        metadata.get("config", {}).get("artifacts", {}).get("result_status")
    )
    return record


def _one_value(records: Sequence[dict[str, Any]], task: str, path: Sequence[str], label: str) -> Any:
    """Return the value shared by every record, or raise naming the outliers."""
    seen: dict[str, list[str]] = {}
    for record in records:
        if task not in record:
            continue
        value: Any = record[task]
        for key in path:
            value = value[key]
        seen.setdefault(json.dumps(value, sort_keys=True), []).append(record["model"])
    if not seen:
        return None
    if len(seen) > 1:
        detail = "; ".join(
            f"{json.loads(key)!r}: {', '.join(models)}" for key, models in sorted(seen.items())
        )
        raise ValueError(
            f"models disagree on {label}, so they cannot share one table ({detail})"
        )
    return json.loads(next(iter(seen)))


def collect_cross_model(
    root: Path, split: str, models: Sequence[str] | None = None, variant: str | None = None
) -> dict[str, Any]:
    """Read one split from every evaluated model under ``root``.

    With ``variant``, a model is read from ``<model>-<variant>`` when that
    directory exists and from ``<model>`` otherwise, so one table can hold a
    controlled variant of some models beside the unchanged runs of the rest.
    Every row names its ``source`` directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"results root does not exist: {root}")
    names = tuple(models) if models is not None else discover_models(root)
    if not names:
        raise ValueError(f"no evaluated model directories under {root}")
    unknown = [name for name in names if name not in MODEL_ORDER]
    if unknown:
        raise ValueError(f"unsupported model name(s): {', '.join(sorted(unknown))}")
    ordered = sorted(dict.fromkeys(names), key=lambda name: MODEL_ORDER[name])
    directories = {name: name for name in ordered}
    if variant is not None:
        directories.update(
            {name: f"{name}-{variant}" for name in ordered if (root / f"{name}-{variant}" / "summary.json").is_file()}
        )
        if all(directory == name for name, directory in directories.items()):
            raise ValueError(f"no <model>-{variant} directory under {root}")
    records = [_collect_model(root, name, split, directories[name]) for name in ordered]

    backend = _one_value(records, "detection", ("backend",), "the AP backend")
    detection_images = _one_value(records, "detection", ("image_count",), "the scored image count")
    counting_images = _one_value(records, "counting", ("image_count",), "the scored image count")
    if None not in (detection_images, counting_images) and detection_images != counting_images:
        raise ValueError(
            f"detection scored {detection_images} images and counting {counting_images}"
        )
    settings = {
        "detection": {
            "backend": backend,
            "ap_iou_threshold": _one_value(
                records, "detection", ("evaluation", "ap_iou_threshold"), "the AP IoU threshold"
            ),
            "ap_iou_range": _one_value(
                records, "detection", ("evaluation", "ap_iou_range"), "the AP IoU range"
            ),
            "export_confidence_threshold": _one_value(
                records,
                "detection",
                ("evaluation", "export_confidence_threshold"),
                "the AP export confidence",
            ),
        }
        if backend is not None
        else None,
        "counting": {
            "iou_threshold": _one_value(
                records, "counting", ("evaluation", "iou_threshold"), "the counting IoU threshold"
            ),
            "confidence_threshold": _one_value(
                records,
                "counting",
                ("evaluation", "confidence_threshold"),
                "the counting confidence threshold",
            ),
            "matching": _one_value(
                records, "counting", ("evaluation", "matching"), "the matching rule"
            ),
            "aggregation": _one_value(
                records, "counting", ("evaluation", "aggregation"), "the aggregation"
            ),
        }
        if any("counting" in record for record in records)
        else None,
    }
    revisions = sorted({record["git_revision"] for record in records if record["git_revision"]})
    return {
        "split": split,
        "variant": variant,
        "image_count": detection_images if detection_images is not None else counting_images,
        "settings": {key: value for key, value in settings.items() if value is not None},
        "git_revisions": revisions,
        "models": records,
        "notes": [
            "every value was produced by this repository; no published number appears here",
            "one AP backend and one counting threshold pair are enforced across the table",
            "study_scope records publication scope only; every row is a primary repository model",
        ],
    }


def _deviation_marks(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Assign one footnote marker per distinct deviation list."""
    marks: dict[str, str] = {}
    for record in records:
        if not record["deviations"]:
            continue
        key = json.dumps(record["deviations"], sort_keys=True)
        if key not in marks:
            marks[key] = "*" * (len(marks) + 1)
    return marks


def render_cross_model_markdown(payload: dict[str, Any], title: str) -> str:
    """Render the cross-model payload as a Markdown report."""
    records = payload["models"]
    split = payload["split"]
    lines = [f"# {title}", ""]
    lines += [
        f"Split `{split}`, {payload['image_count']} images, scored by this repository.",
        "",
    ]
    marks = _deviation_marks(records)

    def _name(record: dict[str, Any]) -> str:
        key = json.dumps(record["deviations"], sort_keys=True)
        suffix = marks.get(key, "")
        return f"{record['model']}{suffix}"

    detection_rows = [
        (
            _name(record),
            record["framework"],
            record["study_scope"],
            f"{record['detection']['map50']:.4f}",
            f"{record['detection']['map50_95']:.4f}",
        )
        for record in records
        if "detection" in record
    ]
    if detection_rows:
        lines += ["## Detection", ""]
        lines += render_table(("model", "framework", "scope", "mAP@50", "mAP@50:95"), detection_rows)
        lines.append("")

    counting_rows = [
        (
            _name(record),
            str(record["counting"]["actual_count"]),
            str(record["counting"]["predicted_count"]),
            str(record["counting"]["true_positives"]),
            str(record["counting"]["false_positives"]),
            str(record["counting"]["false_negatives"]),
            f"{record['counting']['count_error']:+.4f}",
            f"{record['counting']['fdr']:.4f}",
            f"{record['counting']['fnr']:.4f}",
            f"{record['counting']['f1']:.4f}",
        )
        for record in records
        if "counting" in record
    ]
    if counting_rows:
        lines += ["## Counting", ""]
        lines += render_table(
            (
                "model",
                "actual",
                "predicted",
                "TP",
                "FP",
                "FN",
                "count error",
                "FDR",
                "FNR",
                "F1",
            ),
            counting_rows,
        )
        lines.append("")

    lines += ["## Evaluation settings", ""]
    for task, values in payload["settings"].items():
        rendered = ", ".join(f"`{key}`: {value}" for key, value in values.items() if value is not None)
        lines.append(f"- {task}: {rendered}")
    if payload["git_revisions"]:
        lines.append(f"- code revision: {', '.join(payload['git_revisions'])}")
    lines.append("")

    if marks:
        lines += ["## Deviations", ""]
        for key, mark in marks.items():
            models = ", ".join(
                record["model"]
                for record in records
                if json.dumps(record["deviations"], sort_keys=True) == key
            )
            for deviation in json.loads(key):
                lines.append(f"- {mark} {models}: {deviation}")
        lines.append("")

    lines += ["## Notes", ""]
    lines += [f"- {note}" for note in payload["notes"]]
    lines.append("")
    return "\n".join(lines)


def write_cross_model(
    root: Path,
    split: str,
    title: str,
    models: Sequence[str] | None = None,
    variant: str | None = None,
) -> tuple[Path, Path]:
    """Write ``model_comparison_<split>[_<variant>].json`` and ``.md`` at the results root."""
    root = Path(root)
    payload = collect_cross_model(root, split, models, variant)
    stem = f"model_comparison_{split}" + (f"_{variant}" if variant else "")
    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    markdown_path = root / f"{stem}.md"
    markdown_path.write_text(render_cross_model_markdown(payload, title), encoding="utf-8", newline="\n")
    return json_path, markdown_path
