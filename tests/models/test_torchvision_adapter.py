from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cabbage_detection.config import ExecutionConfig, load_config
from cabbage_detection.models.torchvision_adapter import TorchvisionAdapter


torch = pytest.importorskip("torch")


def test_torchvision_prediction_applies_repository_filter_and_nms_once(tmp_path: Path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (20, 20), color=(20, 30, 40)).save(image_path)
    config = replace(
        load_config(Path("configs/torchvision/faster_rcnn.yaml")),
        execution=ExecutionConfig(device="cpu", workers=0, seed=17),
    )

    class FakeModel(torch.nn.Module):
        def forward(self, images):
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 10.0, 10.0], [2.0, 2.0, 11.0, 11.0], [0.0, 0.0, 1.0, 1.0]]),
                    "scores": torch.tensor([0.9, 0.8, 0.1]),
                    "labels": torch.tensor([1, 1, 1]),
                }
            ]

    adapter = TorchvisionAdapter(config)
    monkeypatch.setattr(adapter, "build", lambda: FakeModel())
    checkpoint = tmp_path / "best.pt"
    torch.save({"model_state": FakeModel().state_dict()}, checkpoint)
    records = adapter.predict(image_path, checkpoint)
    assert len(records) == 1
    assert len(records[0].detections) == 1
    assert records[0].postprocessing.max_detections == 100
    assert records[0].postprocessing.confidence_owner == "repository"
    assert records[0].postprocessing.nms_owner == "repository"


def test_torchvision_train_reports_initialization_stages(tmp_path: Path, monkeypatch, capsys):
    import cabbage_detection.models.torchvision_adapter as adapter_module

    config = replace(
        load_config(Path("configs/torchvision/faster_rcnn.yaml")),
        output_dir=tmp_path / "run",
        execution=ExecutionConfig(device="cpu", workers=0, seed=17),
    )
    adapter = TorchvisionAdapter(config)
    model = object()
    monkeypatch.setattr(adapter, "build", lambda: model)
    monkeypatch.setattr(adapter_module, "build_detection_loaders", lambda config: "loaders")
    monkeypatch.setattr(
        adapter_module,
        "train_torchvision",
        lambda config, model, loaders, validation_evaluator: SimpleNamespace(checkpoint=tmp_path / "best.pt"),
    )

    result = adapter.train()

    assert result == tmp_path / "best.pt"
    terminal = capsys.readouterr().err
    assert "building model" in terminal
    assert "building data loaders" in terminal
    assert "starting trainer" in terminal
