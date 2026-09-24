"""README result tables, rendered from the generated comparison and complexity JSON.

The README carries copies of these tables; ``tests/test_public_docs.py`` checks
that the copies still equal what this module renders, so a re-scored run can
never leave a stale number behind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .figures import MODEL_LABELS, SCOPE_MARKS

RESULTS = Path("results/reproduced")

# Columns whose best value is marked bold, with how "best" is decided.
_BEST = {
    "map50": max,
    "map50_95": max,
    "count_error": "closest_to_zero",
    "fdr": min,
    "fnr": min,
    "f1": max,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _values(record: dict[str, Any]) -> dict[str, float]:
    detection, counting = record["detection"], record["counting"]
    return {
        "map50": detection["map50"],
        "map50_95": detection["map50_95"],
        "count_error": counting["count_error"],
        "fdr": counting["fdr"],
        "fnr": counting["fnr"],
        "f1": counting["f1"],
    }


def _formatted(key: str, value: float) -> str:
    return f"{value:+.4f}" if key == "count_error" else f"{value:.4f}"


def _best(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    best = {}
    for key, rule in _BEST.items():
        values = [_values(record)[key] for record in records]
        best[key] = min(values, key=abs) if rule == "closest_to_zero" else rule(values)
    return best


def _cells(record: dict[str, Any], best: dict[str, float] | None) -> list[str]:
    cells = []
    for key, value in _values(record).items():
        text = _formatted(key, value)
        # Bold compares the printed value, so a tie at four decimals is bold twice.
        if best is not None and text.lstrip("+-") == _formatted(key, best[key]).lstrip("+-"):
            text = f"**{text}**"
        cells.append(text)
    return cells


def _name(record: dict[str, Any]) -> str:
    deviation = "*" if record["deviations"] else ""
    return f"{MODEL_LABELS.get(record['model'], record['model'])}{deviation}{SCOPE_MARKS[record['study_scope']]}"


def _input(complexity: dict[str, Any]) -> str:
    height, width = complexity["run_detector_input"]
    return f"{height}×{width}"


def render_results_table(comparison_path: Path, complexity_path: Path) -> str:
    """Every model's test-split row, sorted by mAP@50:95, best values in bold.

    Params and GFLOPs describe the model as it was run: GFLOPs are counted at
    the input its detector actually received (``run_gflops``).
    """
    comparison = _read(comparison_path)
    complexity = {record["run"]: record for record in _read(complexity_path)["models"]}
    records = sorted(comparison["models"], key=lambda record: -record["detection"]["map50_95"])
    best = _best(records)
    lines = [
        "| model | framework | detector input | Params (M) | GFLOPs | mAP@50 | mAP@50:95 | count error | FDR | FNR | F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        cost = complexity[record["source"]]
        cells = [_name(record), record["framework"], _input(cost), f"{cost['params'] / 1e6:.2f}",
                 f"{cost['run_gflops']:.1f}", *_cells(record, best)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_input_size_table(
    default_path: Path, variant_path: Path, complexity_path: Path
) -> str:
    """Each varied model at its run's detector input and at the variant's, side by side."""
    default = {record["model"]: record for record in _read(default_path)["models"]}
    variant = [record for record in _read(variant_path)["models"] if record["source"] != record["model"]]
    complexity = {record["run"]: record for record in _read(complexity_path)["models"]}
    lines = [
        "| model | detector input | GFLOPs at that input | mAP@50 | mAP@50:95 | count error | FDR | FNR | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for changed in sorted(variant, key=lambda record: -default[record["model"]]["detection"]["map50_95"]):
        base = default[changed["model"]]
        for record, note in ((base, "torchvision default"), (changed, "configured image_size")):
            cost = complexity[record["source"]]
            cells = [_name(record), f"{_input(cost)} ({note})", f"{cost['run_gflops']:.1f}", *_cells(record, None)]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the README result tables.")
    parser.add_argument("table", choices=("results", "input-size"))
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--variant", default="detector512")
    args = parser.parse_args(argv)
    root = args.results_root
    if args.table == "results":
        print(render_results_table(root / "model_comparison_test.json", root / "model_complexity.json"))
    else:
        print(render_input_size_table(
            root / "model_comparison_test.json",
            root / f"model_comparison_test_{args.variant}.json",
            root / "model_complexity.json",
        ))
    return 0
