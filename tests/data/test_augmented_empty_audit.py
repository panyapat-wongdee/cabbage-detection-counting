from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cabbage_detection.data.augmented_empty_audit import probe_augmented_empty_targets


class _FakeDataset:
    image_ids = ("img-a", "img-b")
    samples = (
        ("img-a", Path("img-a.png"), ((0.0, 0.0, 10.0, 10.0),)),
        ("img-b", Path("img-b.png"), ()),
    )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        if index == 0:
            return None, {"boxes": torch.empty((0, 4))}
        return None, {"boxes": torch.empty((0, 4))}


def test_probe_reports_raw_and_transformed_empty_images(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "cabbage_detection.data.augmented_empty_audit.load_config",
        lambda path: SimpleNamespace(framework="torchvision"),
    )
    monkeypatch.setattr(
        "cabbage_detection.data.augmented_empty_audit.build_torchvision_loaders",
        lambda config: SimpleNamespace(train=SimpleNamespace(dataset=_FakeDataset())),
    )

    report = probe_augmented_empty_targets(tmp_path / "config.yaml", repeats=2, seed=42)

    assert report["raw_empty_images"] == 1
    assert report["transformed_empty_images"] == 4
    assert report["affected_image_ids"] == ["img-a", "img-b"]
    assert report["repeats"] == 2
    assert report["seed"] == 42
