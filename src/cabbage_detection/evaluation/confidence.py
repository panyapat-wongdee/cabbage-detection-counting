"""Confidence overrides for detection export and counting.

Detection AP needs the full score range, while counting is defined at the
configured operating point. Exporting records at a lower confidence and
filtering them back at the configured threshold is exact: NMS processes
detections in descending score order, so a detection below the configured
threshold can never suppress one above it, and the surviving set above the
threshold is therefore unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..config import ExperimentConfig
from .records import PredictionRecord


def lower_export_confidence(config: ExperimentConfig, confidence: float) -> ExperimentConfig:
    """Return a config that exports detections down to ``confidence``.

    Both the top-level threshold and the final post-processing stage are
    lowered so torchvision and Ultralytics adapters widen by the same amount.
    """
    if not 0 <= confidence <= 1:
        raise ValueError("export confidence must be between 0 and 1")
    if confidence > config.confidence_threshold:
        raise ValueError(
            "export confidence must not exceed the configured "
            f"confidence_threshold ({config.confidence_threshold})"
        )
    lowered = replace(config, confidence_threshold=confidence)
    if lowered.final_postprocess is not None:
        lowered = replace(
            lowered,
            final_postprocess=replace(lowered.final_postprocess, confidence_threshold=confidence),
        )
    return lowered


def filter_by_confidence(
    records: Sequence[PredictionRecord], threshold: float
) -> tuple[PredictionRecord, ...]:
    """Keep detections scoring strictly above ``threshold``.

    The strict comparison matches ``filter_torchvision_outputs`` and the
    Ultralytics NMS gate, so re-filtering an exported record set reproduces the
    detections a run at that threshold would have written.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")
    return tuple(
        replace(
            record,
            detections=tuple(
                detection for detection in record.detections if detection.score > threshold
            ),
        )
        for record in records
    )
