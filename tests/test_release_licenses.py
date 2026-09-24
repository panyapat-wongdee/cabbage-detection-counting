from pathlib import Path

from scripts.audit_release_licenses import audit_release


def create_complete_license_fixture(root: Path) -> None:
    (root / "LICENSES").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (root / "NOTICE").write_text(
        "IEEE dataset pretrained weights Ultralytics\n", encoding="utf-8"
    )
    (root / "LICENSES" / "AGPL-3.0-only.txt").write_text(
        "GNU AFFERO GENERAL PUBLIC LICENSE\n", encoding="utf-8"
    )
    (root / "LICENSES" / "BSD-3-Clause-torchvision.txt").write_text(
        "BSD 3-Clause License\n", encoding="utf-8"
    )
    (root / "weights" / "provenance.yaml").write_text(
        "models:\n" + "".join(
            f"  {name}:\n    study_scope: {'supplementary_unreported' if name == 'ssdlite' else 'paper_model'}\n"
            "    pretrained:\n      source_url: https://example.invalid\n      training_dataset: MS COCO\n      redistributed: false\n"
            f"    fine_tuned:\n      license: {'Apache-2.0' if name in {'faster_rcnn', 'ssd', 'ssdlite', 'retinanet', 'fcos'} else 'AGPL-3.0-only'}\n      distribution: external_release_asset\n"
            f"    required_notices:\n      - {'BSD-3-Clause' if name in {'faster_rcnn', 'ssd', 'ssdlite', 'retinanet', 'fcos'} else 'AGPL-3.0-only'}\n"
            for name in (
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
                "yolo12n",
                "yolo12m",
                "yolo26n",
                "yolo26m",
            )
        ),
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "*.pt\n*.pth\n*.onnx\n*.safetensors\n", encoding="utf-8"
    )


def test_audit_rejects_tracked_model_and_paper_binaries(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked-files.txt"
    tracked.write_text("README.md\nmodel.pt\ndocs/My_paper.pdf\n", encoding="utf-8")
    findings = audit_release(tmp_path, tracked_manifest=tracked)
    assert "tracked model binary: model.pt" in findings
    assert "tracked publication PDF: docs/My_paper.pdf" in findings


def test_audit_rejects_tracked_release_outputs(tmp_path: Path) -> None:
    create_complete_license_fixture(tmp_path)
    tracked = tmp_path / "tracked-files.txt"
    tracked.write_text("dist/model.zip\ndist/releases/model/best.pt\n", encoding="utf-8")
    findings = audit_release(tmp_path, tracked_manifest=tracked)
    assert "tracked release archive: dist/model.zip" in findings
    assert "tracked model binary: dist/releases/model/best.pt" in findings
    assert "tracked generated release output: dist/releases/model/best.pt" in findings


def test_audit_separates_run_evidence_from_release_assets(tmp_path: Path) -> None:
    """Reproduced evidence is tracked on purpose; release assets never are.

    `runs/reproduced/` and `results/reproduced/` hold the small files a
    `reproduced_verified` claim rests on -- run metadata, configs, logs, curves,
    metric reports -- and those are committed so the claim can be audited from
    Git history. The heavy subtrees inside them are release assets or bulk
    regenerable output and must stay out, as must the staging trees.
    """
    tracked = tmp_path / "tracked-files.txt"
    tracked.write_text(
        "\n".join(
            (
                "runs/reproduced/README.md",
                "runs/reproduced/faster_rcnn/metadata.json",
                "runs/reproduced/faster_rcnn/logs/training_log.csv",
                "runs/reproduced/faster_rcnn/evaluation/dataset.json",
                "results/reproduced/README.md",
                "results/reproduced/faster_rcnn/test/detection.json",
                "runs/reproduced/faster_rcnn/checkpoints/best.pt",
                "runs/reproduced/yolov8n/framework/args.yaml",
                "results/reproduced/faster_rcnn/test/records/predictions.json",
                "dist/releases/model.zip",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    findings = audit_release(tmp_path, tracked_manifest=tracked)
    flagged = {
        item.split(": ", 1)[1]
        for item in findings
        if item.startswith("tracked generated release output: ")
    }
    assert flagged == {
        "runs/reproduced/faster_rcnn/checkpoints/best.pt",
        "runs/reproduced/yolov8n/framework/args.yaml",
        "results/reproduced/faster_rcnn/test/records/predictions.json",
        "dist/releases/model.zip",
    }


def test_audit_accepts_documentation_only_release(tmp_path: Path) -> None:
    create_complete_license_fixture(tmp_path)
    tracked = tmp_path / "tracked-files.txt"
    tracked.write_text(
        "LICENSE\nNOTICE\nLICENSES/AGPL-3.0-only.txt\n"
        "LICENSES/BSD-3-Clause-torchvision.txt\nweights/provenance.yaml\n",
        encoding="utf-8",
    )
    assert audit_release(tmp_path, tracked_manifest=tracked) == []
