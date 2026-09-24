from pathlib import Path

import yaml

from cabbage_detection.config import load_config


def test_required_license_texts_are_present() -> None:
    assert "Apache License" in Path("LICENSE").read_text(encoding="utf-8")
    assert "Version 2.0, January 2004" in Path("LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in Path(
        "LICENSES/AGPL-3.0-only.txt"
    ).read_text(encoding="utf-8")
    assert "BSD 3-Clause License" in Path(
        "LICENSES/BSD-3-Clause-torchvision.txt"
    ).read_text(encoding="utf-8")


def test_notice_limits_the_apache_grant_to_original_repository_work() -> None:
    notice = Path("NOTICE").read_text(encoding="utf-8")
    for excluded in (
        "IEEE",
        "dataset",
        "pretrained weights",
        "Ultralytics",
    ):
        assert excluded.lower() in notice.lower()


def test_public_license_is_confirmed_without_pending_approval_language() -> None:
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("README.md", "NOTICE", "docs/licensing-and-attribution.md")
    ).lower()
    assert "subject to" not in combined
    assert "must be confirmed before a public release" not in combined
    assert "intended to be released" not in combined
    assert "apache-2.0" in combined


def test_citation_declares_revised_repository_license() -> None:
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    assert citation["license"] == "Apache-2.0"


def test_python_package_declares_apache_license() -> None:
    metadata = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'license = {file = "LICENSE"}' in metadata
    assert "License :: OSI Approved :: Apache Software License" in metadata


EXPECTED_MODELS = {
    "faster_rcnn",
    "ssd",
    "ssdlite",
    "retinanet",
    "fcos",
    "yolov8n",
    "yolov8m",
    "yolo11n",
    "yolo11m",
    "rt-detr-l",
}
EXTENSION_MODELS = {"yolo12n", "yolo12m", "yolo26n", "yolo26m"}


def test_weight_registry_covers_every_repository_model() -> None:
    registry = yaml.safe_load(Path("weights/provenance.yaml").read_text(encoding="utf-8"))
    assert set(registry) == {"models"}
    assert set(registry["models"]) == EXPECTED_MODELS | EXTENSION_MODELS
    for record in registry["models"].values():
        assert record["pretrained"]["redistributed"] is False
        assert record["pretrained"]["source_url"].startswith("https://")
        assert record["pretrained"]["training_dataset"] == "MS COCO"
        assert record["fine_tuned"]["distribution"] == "external_release_asset"


def test_weight_registry_assigns_the_approved_fine_tuned_terms() -> None:
    models = yaml.safe_load(
        Path("weights/provenance.yaml").read_text(encoding="utf-8")
    )["models"]
    for name in {"faster_rcnn", "ssd", "ssdlite", "retinanet", "fcos"}:
        assert models[name]["fine_tuned"]["license"] == "Apache-2.0"
        assert "BSD-3-Clause" in models[name]["required_notices"]
    for name in {"yolov8n", "yolov8m", "yolo11n", "yolo11m", "rt-detr-l"} | EXTENSION_MODELS:
        assert models[name]["fine_tuned"]["license"] == "AGPL-3.0-only"
        assert "AGPL-3.0-only" in models[name]["required_notices"]


def test_weight_registry_separates_paper_scope_and_supplementary_models() -> None:
    models = yaml.safe_load(
        Path("weights/provenance.yaml").read_text(encoding="utf-8")
    )["models"]
    assert models["ssdlite"]["study_scope"] == "supplementary_unreported"
    assert {
        name for name, record in models.items() if record["study_scope"] == "paper_model"
    } == EXPECTED_MODELS - {"ssdlite"}
    # Models added after the study are neither paper models nor the study's
    # unreported supplement; they are released in the same tag as the others.
    for name in EXTENSION_MODELS:
        assert models[name]["study_scope"] == "repository_extension"
        assert models[name]["fine_tuned"]["release_status"] == "released"
        assert models[name]["fine_tuned"]["release_tag"] == "reproduced-checkpoints-v1.0.0"


def test_weight_registry_matches_primary_configs() -> None:
    registry = yaml.safe_load(
        Path("weights/provenance.yaml").read_text(encoding="utf-8")
    )["models"]
    torchvision_identifiers = {
        "faster_rcnn": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        "ssd": "SSD300_VGG16_Weights.COCO_V1",
        "ssdlite": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
        "retinanet": "RetinaNet_ResNet50_FPN_Weights.COCO_V1",
        "fcos": "FCOS_ResNet50_FPN_Weights.COCO_V1",
    }
    ultralytics_identifiers = {
        "yolov8n": "yolov8n.pt",
        "yolov8m": "yolov8m.pt",
        "yolo11n": "yolo11n.pt",
        "yolo11m": "yolo11m.pt",
        "rt-detr-l": "rtdetr-l.pt",
    }
    for model_name, identifier in {
        **torchvision_identifiers,
        **ultralytics_identifiers,
    }.items():
        config = load_config(
            next(
                Path("configs").glob(
                    f"**/{model_name}.yaml"
                )
            )
        )
        assert config.model_name == model_name
        assert registry[model_name]["pretrained"]["identifier"] == identifier
        if config.framework == "torchvision":
            assert config.initialization.weights == "COCO_V1"
        else:
            assert config.initialization.weights == identifier
