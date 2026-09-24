import json
import shutil
from pathlib import Path

import pytest

from cabbage_detection.config import load_config
from cabbage_detection.models.complexity import measure_run, write_complexity


def test_torchvision_count_is_taken_at_image_size_and_cross_checked(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("torchinfo")
    from cabbage_detection.models.torchvision_models import build_torchvision_model

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    shutil.copy("configs/torchvision/ssdlite.yaml", run / "config.yaml")
    model = build_torchvision_model(load_config(run / "config.yaml"), weights=None)
    torch.save({"model_state": model.state_dict()}, run / "checkpoints" / "best.pt")

    record = measure_run(run)

    assert record.params == sum(parameter.numel() for parameter in model.parameters())
    assert record.flops_input == (512, 512)
    assert record.run_detector_input == (320, 320)
    assert record.gflops == pytest.approx(record.crosscheck_gflops, rel=0.02)

    output = write_complexity([record], tmp_path / "complexity.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["models"][0]["run_detector_input"] == [320, 320]
    assert b"\r" not in output.read_bytes()


def test_measure_run_requires_a_checkpoint(tmp_path: Path):
    shutil.copy("configs/torchvision/ssdlite.yaml", tmp_path / "config.yaml")
    with pytest.raises(FileNotFoundError):
        measure_run(tmp_path)
