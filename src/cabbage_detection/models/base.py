"""Common adapter contract without importing heavyweight frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Sequence

from ..config import ExperimentConfig
from ..evaluation.records import PredictionRecord


class ModelAdapter(ABC):
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self.framework = config.framework

    @abstractmethod
    def train(self) -> Path:
        """Run native-framework training and return the output directory."""

    @abstractmethod
    def predict(self, source: Path, checkpoint: Path) -> list[PredictionRecord]:
        """Run native-framework prediction and return canonical records."""

    def predict_iter(
        self, sources: Sequence[Path], checkpoint: Path
    ) -> Iterator[PredictionRecord]:
        """Yield one record per source, loading the checkpoint once.

        The default reloads per source. Adapters override this so evaluating a
        split does not rebuild the detector for every image, while the
        per-source inference path stays shared so the two cannot diverge.
        Yielding lets a caller drive a progress bar without reaching into the
        adapter.
        """
        for source in sources:
            yield from self.predict(source, checkpoint)

    def predict_many(
        self, sources: Sequence[Path], checkpoint: Path
    ) -> list[PredictionRecord]:
        """Predict several sources, loading the checkpoint once."""
        return list(self.predict_iter(sources, checkpoint))
