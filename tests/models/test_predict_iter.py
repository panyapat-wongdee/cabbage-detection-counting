"""Batch prediction loads the checkpoint once and matches per-image records."""

from __future__ import annotations

from pathlib import Path

import pytest

from cabbage_detection.evaluation.records import Detection, PredictionRecord
from cabbage_detection.models.base import ModelAdapter


class _FakeAdapter(ModelAdapter):
    """Adapter that counts checkpoint loads without importing a framework."""

    def __init__(self) -> None:
        self.loads = 0
        self.predicted: list[Path] = []

    def train(self) -> Path:  # pragma: no cover - not exercised here
        raise NotImplementedError

    def _load(self, checkpoint: Path) -> None:
        self.loads += 1

    def predict(self, source: Path, checkpoint: Path) -> list[PredictionRecord]:
        self._load(checkpoint)
        return [self._one(source)]

    def _one(self, source: Path) -> PredictionRecord:
        self.predicted.append(Path(source))
        return PredictionRecord(Path(source).stem, (Detection((0, 0, 1, 1), 0.9),))


class _BatchingAdapter(_FakeAdapter):
    """The shape both real adapters use: load once, then yield per source."""

    def predict_iter(self, sources, checkpoint: Path):
        self._load(checkpoint)
        for source in sources:
            yield self._one(source)


def test_default_predict_iter_reloads_for_every_source():
    adapter = _FakeAdapter()
    records = adapter.predict_many([Path("a.png"), Path("b.png")], Path("best.pt"))
    assert [record.image_id for record in records] == ["a", "b"]
    assert adapter.loads == 2


def test_overridden_predict_iter_loads_the_checkpoint_once():
    adapter = _BatchingAdapter()
    records = adapter.predict_many([Path("a.png"), Path("b.png")], Path("best.pt"))
    assert [record.image_id for record in records] == ["a", "b"]
    assert adapter.loads == 1


def test_batching_produces_the_same_records_as_predicting_one_at_a_time():
    batched = _BatchingAdapter()
    single = _BatchingAdapter()
    sources = [Path("a.png"), Path("b.png"), Path("c.png")]

    many = batched.predict_many(sources, Path("best.pt"))
    one_by_one = [record for source in sources for record in single.predict(source, Path("best.pt"))]

    assert many == one_by_one


def test_predict_iter_is_lazy_so_a_caller_can_drive_a_progress_bar():
    adapter = _BatchingAdapter()
    stream = adapter.predict_iter([Path("a.png"), Path("b.png")], Path("best.pt"))
    assert adapter.predicted == []
    next(stream)
    assert [path.name for path in adapter.predicted] == ["a.png"]


def test_predict_many_accepts_no_sources():
    assert _BatchingAdapter().predict_many([], Path("best.pt")) == []


@pytest.mark.parametrize("adapter_name", ["torchvision_adapter", "ultralytics_adapter"])
def test_real_adapters_override_predict_iter(adapter_name: str):
    """A regression guard: losing the override silently restores per-image loads."""
    import importlib

    module = importlib.import_module(f"cabbage_detection.models.{adapter_name}")
    adapter_class = next(
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, ModelAdapter) and value is not ModelAdapter
    )
    assert "predict_iter" in vars(adapter_class)
