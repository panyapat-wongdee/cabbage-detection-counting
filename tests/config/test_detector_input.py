from pathlib import Path

import pytest
import yaml

from cabbage_detection.config import load_config

EXPERIMENT = sorted(Path("configs/torchvision/detector_input_512").glob("*.yaml"))


def test_shipped_profiles_keep_the_framework_default_detector_input():
    for path in sorted(Path("configs").rglob("*.yaml")):
        if "detector_input_512" in path.parts or path.name == "publication-boundary.yaml":
            continue
        assert load_config(path).detector_input == "framework_default", path


def test_detector_input_is_written_to_the_resolved_config():
    config = load_config(Path("configs/torchvision/fcos.yaml"))
    assert config.to_dict()["model"]["detector_input"] == "framework_default"


def test_experiment_matrix_covers_every_torchvision_model():
    assert [path.stem for path in EXPERIMENT] == ["faster_rcnn", "fcos", "retinanet", "ssd", "ssdlite"]


@pytest.mark.parametrize("path", EXPERIMENT, ids=lambda path: path.stem)
def test_experiment_differs_from_its_parent_only_in_detector_input(path: Path):
    parent = yaml.safe_load(Path("configs/torchvision", path.name).read_text(encoding="utf-8"))
    child = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert child["model"].pop("detector_input") == "image_size"
    assert child["artifacts"].pop("output_dir") == f"runs/reproduced/{path.stem}-detector512"
    parent["artifacts"].pop("output_dir")
    assert child["provenance"].pop("model.detector_input") == "repository_protocol"
    assert child == parent
    assert load_config(path).detector_input == "image_size"


def test_ultralytics_rejects_a_detector_input_override(tmp_path: Path):
    raw = yaml.safe_load(Path("configs/ultralytics/yolov8n.yaml").read_text(encoding="utf-8"))
    raw["model"]["detector_input"] = "image_size"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="only to torchvision"):
        load_config(path)
