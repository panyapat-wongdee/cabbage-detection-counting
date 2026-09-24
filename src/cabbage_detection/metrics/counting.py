"""One-to-one detection matching and cabbage counting metrics."""

from __future__ import annotations

from collections.abc import Sequence

from cabbage_detection.evaluation.records import Box, CountingMetrics, Detection, MatchSummary

from .boxes import box_iou


def match_pairs(
    predictions: Sequence[Detection],
    targets: Sequence[Box],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Return ``(prediction_index, target_index)`` for each one-to-one match.

    Pairs are taken greedily in descending IoU order, so each prediction and
    each target is matched at most once. This is the rule every counting
    metric is built on; ``match_detections`` summarises its result.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    candidates = sorted(
        [
            (iou, prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
            for iou in [box_iou(prediction.box, target.coordinates)]
            if iou >= iou_threshold
        ],
        key=lambda item: (-item[0], item[1], item[2]),
    )
    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, prediction_index, target_index in candidates:
        if prediction_index not in matched_predictions and target_index not in matched_targets:
            matched_predictions.add(prediction_index)
            matched_targets.add(target_index)
            pairs.append((prediction_index, target_index))
    return pairs


def match_detections(
    predictions: Sequence[Detection],
    targets: Sequence[Box],
    iou_threshold: float,
) -> MatchSummary:
    """Greedily match highest-IoU prediction/target pairs once each."""
    true_positives = len(match_pairs(predictions, targets, iou_threshold))
    return MatchSummary(
        true_positives=true_positives,
        false_positives=len(predictions) - true_positives,
        false_negatives=len(targets) - true_positives,
        predicted_count=len(predictions),
        actual_count=len(targets),
    )


def count_metrics(summary: MatchSummary) -> CountingMetrics:
    """Calculate count error, FDR, FNR, and F1.

    Undefined zero-denominator cases are represented as ``0.0`` so an empty
    image contributes neutral metrics rather than NaN to serialized output.
    """
    count_error = (
        (summary.predicted_count - summary.actual_count) / summary.actual_count
        if summary.actual_count
        else 0.0
    )
    fdr = summary.false_positives / summary.predicted_count if summary.predicted_count else 0.0
    fnr = summary.false_negatives / summary.actual_count if summary.actual_count else 0.0
    denominator = 2 * summary.true_positives + summary.false_positives + summary.false_negatives
    f1 = 2 * summary.true_positives / denominator if denominator else 0.0
    return CountingMetrics(
        predicted_count=summary.predicted_count,
        actual_count=summary.actual_count,
        count_error=count_error,
        fdr=fdr,
        fnr=fnr,
        f1=f1,
    )
