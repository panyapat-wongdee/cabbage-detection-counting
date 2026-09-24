"""Compare independently reproduced metrics with immutable published tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .counting import CountingReport
from .detection import DetectionReport


@dataclass(frozen=True)
class ComparisonReport:
    model: str
    metric_source: str
    reproduced: dict[str, float]
    published: dict[str, float]
    absolute_difference: dict[str, float]
    relative_difference: dict[str, float | None]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "metric_source": self.metric_source,
            "reproduced": self.reproduced,
            "published": self.published,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
        }


def _read_published(path: Path, model: str) -> Mapping[str, float]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row.get("model") == model]
    if not matches:
        raise ValueError(f"published metrics do not contain model {model!r}")
    row = matches[0]
    values: dict[str, float] = {}
    for key, value in row.items():
        if key in {"model", "source"} or value in {None, ""}:
            continue
        try:
            values[key] = float(value)
        except ValueError as error:
            raise ValueError(f"published metric {key!r} is not numeric") from error
    return values


def _compare(model: str, source: str, reproduced: dict[str, float], published_csv: Path) -> ComparisonReport:
    published = dict(_read_published(published_csv, model))
    missing = set(reproduced) - set(published)
    if missing:
        raise ValueError(f"published metrics missing: {', '.join(sorted(missing))}")
    if set(published) != set(reproduced):
        extra = set(published) - set(reproduced)
        raise ValueError(f"reproduced metrics missing: {', '.join(sorted(extra))}")
    absolute = {key: reproduced[key] - published[key] for key in reproduced}
    relative = {
        key: (absolute[key] / published[key] if published[key] != 0 else None)
        for key in reproduced
    }
    return ComparisonReport(model, source, reproduced, published, absolute, relative)


def compare_to_published(
    reproduced: DetectionReport | CountingReport,
    published_csv: Path,
    model_name: str | None = None,
) -> ComparisonReport:
    """Compare a report without modifying the published CSV.

    Counting values are converted from repository fractions to the percentage
    units used by the published table.
    """
    model = model_name or getattr(reproduced, "model_name", None)
    if not model:
        raise ValueError("model_name is required to compare a report")
    if isinstance(reproduced, DetectionReport):
        values = {"map50": reproduced.map50, "map50_95": reproduced.map50_95}
        return _compare(model, "detection", values, Path(published_csv))
    values = {
        "fdr_percent": reproduced.totals.fdr * 100,
        "fnr_percent": reproduced.totals.fnr * 100,
        "f1_percent": reproduced.totals.f1 * 100,
        "count_error_percent": reproduced.totals.count_error * 100,
    }
    return _compare(model, "counting", values, Path(published_csv))
