"""Roll per-split evaluation reports up into one readable summary.

An evaluation writes one directory per split. Reading a run should not require
opening each of them, so the split reports are also collected into
``summary.json`` and ``summary.md`` at the output root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

DETECTION_COLUMNS = ("map50", "map50_95")
COUNTING_COLUMNS = (
    "actual_count",
    "predicted_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "count_error",
    "fdr",
    "fnr",
    "f1",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_summary(output_dir: Path, splits: Sequence[str]) -> dict[str, Any]:
    """Read each split's reports and return one combined payload."""
    output_dir = Path(output_dir)
    splits_payload: dict[str, Any] = {}
    backends: set[str] = set()
    for split in splits:
        split_dir = output_dir / split
        entry: dict[str, Any] = {}
        detection = _read_json(split_dir / "detection.json")
        if detection is not None:
            entry["detection"] = {
                "image_count": detection["scope"]["image_count"],
                "backend": detection["metrics"]["backend"],
                **{name: detection["metrics"][name] for name in DETECTION_COLUMNS},
            }
            backends.add(str(detection["metrics"]["backend"]))
        counting = _read_json(split_dir / "counting.json")
        if counting is not None:
            entry["counting"] = {
                "image_count": counting["scope"]["image_count"],
                **{name: counting["metrics"][name] for name in COUNTING_COLUMNS},
                "macro": counting["metrics_macro"],
            }
        if entry:
            splits_payload[split] = entry
    return {
        "splits": splits_payload,
        "detection_backends": sorted(backends),
        "units": {
            "map50": "fraction",
            "map50_95": "fraction",
            "count_error": "signed_fraction_of_actual_count",
            "fdr": "fraction",
            "fnr": "fraction",
            "f1": "fraction",
        },
        "notes": [
            "counting metrics are micro-averaged over the split; macro holds the per-image mean",
            "splits are scored separately and are never pooled into one figure",
        ],
    }


def render_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render one Markdown table; shared with the cross-model report."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_summary_markdown(summary: dict[str, Any], title: str) -> str:
    """Render the combined payload as a short Markdown report."""
    lines = [f"# {title}", ""]
    splits = summary["splits"]
    if not splits:
        lines.append("No split reports were written.")
        return "\n".join(lines) + "\n"

    detection_rows = [
        (
            split,
            str(entry["detection"]["image_count"]),
            f"{entry['detection']['map50']:.4f}",
            f"{entry['detection']['map50_95']:.4f}",
            entry["detection"]["backend"],
        )
        for split, entry in splits.items()
        if "detection" in entry
    ]
    if detection_rows:
        lines += ["## Detection", ""]
        lines += render_table(("split", "images", "mAP@50", "mAP@50:95", "backend"), detection_rows)
        lines.append("")

    counting_rows = [
        (
            split,
            str(entry["counting"]["image_count"]),
            str(entry["counting"]["actual_count"]),
            str(entry["counting"]["predicted_count"]),
            str(entry["counting"]["true_positives"]),
            str(entry["counting"]["false_positives"]),
            str(entry["counting"]["false_negatives"]),
            f"{entry['counting']['count_error']:+.4f}",
            f"{entry['counting']['fdr']:.4f}",
            f"{entry['counting']['fnr']:.4f}",
            f"{entry['counting']['f1']:.4f}",
        )
        for split, entry in splits.items()
        if "counting" in entry
    ]
    if counting_rows:
        lines += ["## Counting", ""]
        lines += render_table(
            (
                "split",
                "images",
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
        lines += ["Counting metrics are micro-averaged over the split.", ""]

    lines += ["## Notes", ""]
    lines += [f"- {note}" for note in summary["notes"]]
    lines.append("")
    return "\n".join(lines)


def write_summary(output_dir: Path, splits: Sequence[str], title: str) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.md`` at the evaluation root."""
    output_dir = Path(output_dir)
    summary = collect_summary(output_dir, splits)
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(render_summary_markdown(summary, title), encoding="utf-8")
    return json_path, markdown_path
