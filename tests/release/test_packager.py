import hashlib
from pathlib import Path
import zipfile

import pytest

from cabbage_detection.release_artifacts import (
    ReleaseValidationError,
    audit_release_directory,
    build_release,
)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_archive_for_supplied_verified_models(release_fixture):
    root, manifest, output = release_fixture
    assets = build_release(root, manifest, output)
    archives = [path for path in assets if path.suffix == ".zip"]
    assert len(archives) == 1
    assert archives[0].name == "cabbage-faster_rcnn-checkpoints-reproduced-v1.0.0.zip"


def test_archive_contains_only_best_checkpoint_and_metadata(release_fixture):
    root, manifest, output = release_fixture
    archive = next(path for path in build_release(root, manifest, output) if path.suffix == ".zip")
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "faster_rcnn/augmented/best.pt" in names
    assert "faster_rcnn/CHECKPOINTS.json" in names
    assert "faster_rcnn/MODEL_CARD.md" in names
    assert "faster_rcnn/LICENSE" in names
    assert "faster_rcnn/THIRD_PARTY_NOTICES.md" in names
    assert not any("last.pt" in name or "args.yaml" in name or name.endswith(".csv") for name in names)


def test_repeated_build_is_byte_identical(release_fixture):
    root, manifest, output = release_fixture
    first = {p.name: file_digest(p) for p in build_release(root, manifest, output)}
    second = {p.name: file_digest(p) for p in build_release(root, manifest, output)}
    assert first == second


def test_audit_rejects_forbidden_member(release_fixture):
    root, manifest, output = release_fixture
    build_release(root, manifest, output)
    archive = next(output.glob("*.zip"))
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("model/last.pt", b"forbidden")
    with pytest.raises(ReleaseValidationError, match="forbidden archive member"):
        audit_release_directory(output, manifest)


def test_torchvision_archive_bundles_the_bsd_text_and_notices(release_fixture):
    """The notices must travel as text, not only as a pointer to the repository."""
    root, manifest, output = release_fixture
    archive = next(path for path in build_release(root, manifest, output) if path.suffix == ".zip")
    with zipfile.ZipFile(archive) as bundle:
        bsd = bundle.read("faster_rcnn/notices/BSD-3-Clause-torchvision.txt")
        assert bsd == (root / "LICENSES/BSD-3-Clause-torchvision.txt").read_bytes()
        assert b"Redistribution and use in source and binary forms" in bsd
        for name in ("MS-COCO.md", "cabbage-dataset.md"):
            assert bundle.read(f"faster_rcnn/notices/{name}") == (root / "weights/notices" / name).read_bytes()
        notices = bundle.read("faster_rcnn/THIRD_PARTY_NOTICES.md").decode("utf-8")
    assert "notices/BSD-3-Clause-torchvision.txt" in notices
    audit_release_directory(output, manifest)


def test_build_refuses_a_missing_notice(release_fixture):
    root, manifest, output = release_fixture
    (root / "LICENSES/BSD-3-Clause-torchvision.txt").unlink()
    with pytest.raises(ReleaseValidationError, match="BSD-3-Clause-torchvision"):
        build_release(root, manifest, output)


def test_audit_rejects_an_archive_without_its_notices(release_fixture):
    root, manifest, output = release_fixture
    build_release(root, manifest, output)
    archive = next(output.glob("*.zip"))
    with zipfile.ZipFile(archive) as bundle:
        kept = {
            info.filename: bundle.read(info.filename)
            for info in bundle.infolist()
            if "/notices/" not in info.filename
        }
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in kept.items():
            bundle.writestr(name, data)
    with pytest.raises(ReleaseValidationError, match="archive members differ"):
        audit_release_directory(output, manifest)


def test_release_metadata_uses_lf_on_every_platform(release_fixture):
    """`sha256sum -c` on Linux/macOS fails on a CRLF checksum file."""
    root, manifest, output = release_fixture
    build_release(root, manifest, output)
    for name in ("SHA256SUMS", "release-manifest.json"):
        assert b"\r" not in (output / name).read_bytes(), name


def test_a_variant_run_gets_its_own_archive_and_names_its_config(release_fixture):
    """fcos and fcos-detector512 are one model trained two ways; they must not collide."""
    import json
    import shutil
    from dataclasses import replace

    root, manifest, output = release_fixture
    base = manifest.checkpoints[0]
    run = f"{base.model_name}-detector512"
    shutil.copytree(root / "runs/reproduced" / base.model_name, root / "runs/reproduced" / run)
    metadata_path = root / "runs/reproduced" / run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["command"] = ["scripts/train.py", "--config", f"configs/torchvision/detector_input_512/{base.model_name}.yaml"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    variant = replace(
        base,
        run_id=run,
        source_path=f"runs/reproduced/{run}/checkpoints/best.pt",
        metadata_path=f"runs/reproduced/{run}/metadata.json",
    )
    manifest = manifest.with_checkpoints((base, variant))

    assets = build_release(root, manifest, output)

    names = sorted(path.name for path in assets if path.suffix == ".zip")
    assert names == [
        f"cabbage-{base.model_name}-checkpoints-reproduced-v1.0.0.zip",
        f"cabbage-{run}-checkpoints-reproduced-v1.0.0.zip",
    ]
    with zipfile.ZipFile(output / names[1]) as bundle:
        assert f"{run}/augmented/best.pt" in bundle.namelist()
        card = bundle.read(f"{run}/MODEL_CARD.md").decode("utf-8")
    assert f"configs/torchvision/detector_input_512/{base.model_name}.yaml" in card
    assert "not with the model's primary config" in card
    audit_release_directory(output, manifest)


def test_ultralytics_notice_names_the_published_source_and_the_training_revision():
    from cabbage_detection.release_artifacts import render_third_party_notices

    trained, published = "a" * 40, "b" * 40
    notice = render_third_party_notices("yolo11n", Path("."), trained, published)
    assert f"at commit `{published}`" in notice
    assert f"CHECKPOINTS.json (`{trained}`)" in notice
    assert "on request" not in notice
    with pytest.raises(ReleaseValidationError, match="published source commit"):
        render_third_party_notices("yolo11n", Path("."), trained, None)
