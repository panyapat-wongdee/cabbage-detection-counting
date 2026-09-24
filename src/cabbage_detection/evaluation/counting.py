"""Dataset-level counting evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .records import CountingMetrics, MatchSummary, PredictionRecord, TargetRecord
from ..metrics.counting import count_metrics, match_detections
from ..progress import progress


@dataclass(frozen=True)
class CountingReport:
    per_image: dict[str, CountingMetrics]
    totals: CountingMetrics
    model_name: str | None = None
    match_totals: MatchSummary | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly per-image and aggregate metrics."""
        return {
            "model": self.model_name,
            "per_image": {image_id: vars(metrics) for image_id, metrics in sorted(self.per_image.items())},
            "totals": vars(self.totals),
            "match_totals": vars(self.match_totals) if self.match_totals is not None else None,
        }


def evaluate_counting(
    predictions: Sequence[PredictionRecord],
    targets: Sequence[TargetRecord],
    iou_threshold: float,
) -> CountingReport:
    prediction_map = {record.image_id: record for record in predictions}
    target_map = {record.image_id: record for record in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("duplicate image_id in evaluation records")
    unknown = set(prediction_map) - set(target_map)
    if unknown:
        raise ValueError(f"unknown prediction image_id(s): {', '.join(sorted(unknown))}")
    summaries = []
    per_image = {}
    for image_id, target in progress(
        target_map.items(),
        description="evaluate counting",
        total=len(target_map),
    ):
        prediction = prediction_map.get(image_id, PredictionRecord(image_id, ()))
        summary = match_detections(prediction.detections, target.boxes, iou_threshold)
        summaries.append(summary)
        per_image[image_id] = count_metrics(summary)
    from ..evaluation.records import MatchSummary

    total = MatchSummary(
        true_positives=sum(item.true_positives for item in summaries),
        false_positives=sum(item.false_positives for item in summaries),
        false_negatives=sum(item.false_negatives for item in summaries),
        predicted_count=sum(item.predicted_count for item in summaries),
        actual_count=sum(item.actual_count for item in summaries),
    )
    return CountingReport(per_image=per_image, totals=count_metrics(total), match_totals=total)
