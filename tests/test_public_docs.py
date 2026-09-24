from pathlib import Path

import pytest
import yaml

# Run evidence, result tables and figures arrive in the results commit that
# follows a full re-run (scripts/reproduce_all.py); the code commit before it
# carries none, so the checks that compare the README against them wait.
EVIDENCE = Path("results/reproduced/model_comparison_test.json").is_file()
needs_evidence = pytest.mark.skipif(
    not EVIDENCE, reason="run evidence is added by the results commit"
)


def test_readme_contains_public_workflow_sections():
    readme = Path("README.md").read_text(encoding="utf-8").lower()
    for section in ("installation", "dataset", "evaluation", "citation", "limitations"):
        assert section in readme
    assert "no_augmentation" in readme
    assert "parameter-aligned revised augmented" in readme
    assert "citation-only" in readme


def test_published_results_are_not_public_artifacts():
    assert not Path("results/published").exists()
    assert Path("results/reproduced/README.md").exists()
    assert not Path("results/reproduced/detection_metrics.csv").exists()


def test_publication_boundary_documents_exist():
    assert Path("docs/publication-boundary.md").exists()
    assert Path("docs/publication-reference.md").exists()
    assert not Path("docs/project_source.md").exists()


def test_source_provenance_names_authoritative_inputs():
    text = Path("docs/source-provenance.md").read_text(encoding="utf-8")
    for required in ("Published paper", "private historical archive", "Dataset paper", "Unknown"):
        assert required in text
    assert "Ultralytics native augmentation" in text


def test_private_historical_experiments_are_not_public():
    assert not Path("original_experiments").exists()
    assert Path("docs/environment/recovered-research-requirements.txt").is_file()


def test_public_docs_do_not_reference_removed_private_paths():
    paths = [
        Path("README.md"),
        Path("AGENTS.md"),
        *(
            path
            for path in Path("docs").rglob("*.md")
            if "superpowers" not in path.parts
        ),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "original_experiments/" not in text
        assert "docs/project_source.md" not in text


def test_reproduction_protocol_explains_revised_counting_definition():
    text = Path("docs/reproduction-protocol.md").read_text(encoding="utf-8")
    assert "one-to-one" in text
    assert "may not reproduce the paper's pairwise counting values" in text


def test_configuration_reference_preserves_paths_and_unknowns():
    text = Path("docs/configuration.md").read_text(encoding="utf-8")
    for required in ("original_dataset", "prepared/fold1_yolo/data.yaml", "unknown", "reproduced"):
        assert required in text.lower()


def test_dataset_docs_name_official_version_license_and_layout():
    text = Path("docs/dataset.md").read_text(encoding="utf-8")
    for required in (
        "https://data.mendeley.com/datasets/5cp2dyjczk/2",
        "Version 2",
        "CC BY 4.0",
        "annotation.json",
        "<cultivar>/<location>/<YYYYMM>",
        "--dataset-root original_dataset",
    ):
        assert required in text


def test_external_download_roots_are_ignored():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "original_dataset/" in text
    assert "original_dataset" + "_real" not in text


def test_public_docs_distinguish_the_three_doi_records():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "10.1109/" + "KST65016.2025.11003298" in readme
    assert "10.1016/" + "j.dib.2024.110699" in readme
    assert "10.17632/" + "5cp2dyjczk.2" in readme
    assert "paper PDF is not distributed" in readme
    assert "dataset is not included" in readme.lower()


def test_readme_explains_the_multi_license_boundary():
    readme = Path("README.md").read_text(encoding="utf-8")
    for required in (
        "Apache-2.0",
        "AGPL-3.0",
        "pretrained weights",
        "IEEE",
        "CC BY 4.0",
        "weights/provenance.yaml",
    ):
        assert required.lower() in readme.lower()


def test_readme_does_not_present_ssdlite_as_a_paper_model():
    """SSDLite is reported like the other models but never called a paper model."""
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "ssdlite" in readme.lower()
    assert "supplementary_unreported" in readme
    assert "not one of the architectures compared in the paper" in readme
    # The paper compared nine architectures; the repository reports ten.
    assert "nine of them are the paper-scope models" in readme.lower()


def test_citation_file_contains_the_published_paper_metadata():
    citation = Path("CITATION.cff").read_text(encoding="utf-8")
    assert "A Comparative Study of Deep Learning Models for Cabbage Detection and Counting in Drone Imagery" in citation
    assert "10.1109/KST65016.2025.11003298" in citation
    assert "year: 2025" in citation


def test_public_community_documents_exist():
    assert Path("CONTRIBUTING.md").is_file()
    assert Path("SECURITY.md").is_file()


def test_citation_links_to_repository():
    citation = yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
    assert citation["repository-code"] == "https://github.com/panyapat-wongdee/cabbage-detection-counting"
    # The software version matches the release tag's version.
    assert citation["version"] == "1.0.0"
    assert str(citation["date-released"]) == "2026-09-24"


@needs_evidence
def test_readme_reproduced_table_matches_the_generated_comparison():
    """The README table is a copy of generated artifacts; it must not drift."""
    import json

    comparison = json.loads(
        Path("results/reproduced/model_comparison_test.json").read_text(encoding="utf-8")
    )["models"]
    # Keyed by run: a model can have several runs, e.g. fcos and fcos-detector512.
    complexity = {
        record["run"]: record
        for record in json.loads(
            Path("results/reproduced/model_complexity.json").read_text(encoding="utf-8")
        )["models"]
    }
    labels = {
        "faster_rcnn": "Faster R-CNN",
        "ssd": "SSD",
        "retinanet": "RetinaNet",
        "fcos": "FCOS",
        "yolov8n": "YOLOv8n",
        "yolov8m": "YOLOv8m",
        "yolo11n": "YOLO11n",
        "yolo11m": "YOLO11m",
        "rt-detr-l": "RT-DETR-L",
        "ssdlite": "SSDLite",
        "yolo12n": "YOLO12n",
        "yolo12m": "YOLO12m",
        "yolo26n": "YOLO26n",
        "yolo26m": "YOLO26m",
    }
    readme = Path("README.md").read_text(encoding="utf-8")
    # Only the headline section is a copy of the generated table; the
    # confidence-range discussion further down has its own unrelated table.
    section = readme.split("## Reproduced results", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| ")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]

    def cells(line: str) -> dict[str, str]:
        # Bold marks the best value; footnote markers carry the deviation and
        # publication-scope notes. Both are stripped so the check is numeric.
        values = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        return dict(zip(header, values))

    rows = [cells(line) for line in lines[1:]]
    by_label = {row["model"].rstrip("*†‡"): (row, raw) for row, raw in zip(rows, lines[1:])}
    assert set(by_label) == {labels[record["model"]] for record in comparison}

    for record in comparison:
        row, raw = by_label[labels[record["model"]]]
        cost = complexity[record["source"]]
        detection, counting = record["detection"], record["counting"]
        height, width = cost["run_detector_input"]
        expected = {
            "framework": record["framework"],
            "detector input": f"{height}×{width}",
            "Params (M)": f"{cost['params'] / 1e6:.2f}",
            # The model as it was run: GFLOPs at its own detector input.
            "GFLOPs": f"{cost['run_gflops']:.1f}",
            "mAP@50": f"{detection['map50']:.4f}",
            "mAP@50:95": f"{detection['map50_95']:.4f}",
            "count error": f"{counting['count_error']:+.4f}",
            "FDR": f"{counting['fdr']:.4f}",
            "FNR": f"{counting['fnr']:.4f}",
            "F1": f"{counting['f1']:.4f}",
        }
        for column, value in expected.items():
            assert row[column] == value, f"README {column} is stale for {record['model']}"
        # A model outside the publication must carry its visible marker:
        # † for the study's unreported supplement, ‡ for a later extension.
        mark = {"paper_model": None, "supplementary_unreported": "†", "repository_extension": "‡"}
        name_cell = raw.split("|")[1]
        if mark[record["study_scope"]]:
            assert mark[record["study_scope"]] in name_cell
        else:
            assert "†" not in name_cell and "‡" not in name_cell

    ordered = [float(row["mAP@50:95"]) for row in rows]
    assert ordered == sorted(ordered, reverse=True), "README table must be sorted by mAP@50:95"
    # Every complexity count was taken at one input, or GFLOPs are not comparable.
    assert len({tuple(record["flops_input"]) for record in complexity.values()}) == 1


def test_reproduced_results_keep_records_out_of_git_but_track_the_metrics():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "/results/reproduced/**/records/" in text
    assert "/results/reproduced/*\n" not in text
    if EVIDENCE:
        assert Path("results/reproduced/model_comparison_test.md").is_file()
        assert Path("results/reproduced/faster_rcnn/test/detection.json").is_file()


def test_no_tracked_file_carries_a_machine_local_path():
    """Published run evidence must not leak an operator's home directory."""
    import re
    import subprocess

    leak = re.compile(r"(/home/[a-z]|/Users/[A-Za-z]|[A-Za-z]:[\/](?:Users|home)[\/])")
    listed = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    skip = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".pt", ".pth", ".onnx"}
    offenders = []
    for name in listed:
        path = Path(name)
        if not name or path.suffix.lower() in skip or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # The test itself has to spell the patterns out.
        if path == Path("tests/test_public_docs.py"):
            continue
        if leak.search(text):
            offenders.append(name)
    assert not offenders, f"machine-local paths in tracked files: {offenders}"


def test_release_tag_and_title_are_consistent_across_documents():
    """The tag and title are stated in several places; none may drift."""
    from cabbage_detection.release_artifacts import RELEASE_TAG, RELEASE_TITLE

    notes = Path("docs/releases/reproduced-checkpoints-v1.0.0.md")
    publishing = Path("docs/releases/publishing.md")
    documents = [
        Path("README.md"),
        notes,
        publishing,
        Path("docs/release-checklist.md"),
        Path("weights/README.md"),
    ]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert RELEASE_TAG in text, f"{path} does not name the release tag"
        # Line wrapping may split the title, so compare on collapsed whitespace.
        assert RELEASE_TITLE in " ".join(text.split()), f"{path} does not name the release title"

    # publishing.md owns the maintainer procedure and the README links to it;
    # the notes document is the release body that downloaders read, so it must
    # not repeat maintainer steps.
    procedure = " ".join(publishing.read_text(encoding="utf-8").split())
    assert f'--title "{RELEASE_TITLE}"' in procedure, "publish command must set the title"
    assert "git push origin main" in procedure
    assert "docs/releases/publishing.md" in Path("README.md").read_text(encoding="utf-8")
    notes_text = notes.read_text(encoding="utf-8")
    assert "gh release create" not in notes_text, "publish command is duplicated in the notes"
    assert "git push" not in notes_text
    assert "prepare_checkpoint_release" not in notes_text


def test_release_title_survives_a_shell_argument():
    """A non-ASCII title mangles on a Windows console; keep it plain."""
    from cabbage_detection.release_artifacts import RELEASE_TITLE

    assert RELEASE_TITLE.isascii()
    assert '"' not in RELEASE_TITLE and "`" not in RELEASE_TITLE


def test_powershell_publish_command_expands_the_archive_glob():
    """PowerShell does not glob for a native command.

    Passing `*.zip` straight to `gh` hands it one literal string and the upload
    fails, so the documented PowerShell command must expand the archives first.
    The POSIX block may use the bare glob, because a POSIX shell expands it.
    """
    procedure = Path("docs/releases/publishing.md").read_text(encoding="utf-8")
    blocks = procedure.split("```")
    publish = [
        block
        for block in blocks
        if block.startswith("powershell") and "gh release create" in block
    ]
    assert publish, "publishing.md has no PowerShell publish command"
    for block in publish:
        assert "(Get-ChildItem" in block and ").FullName" in block, (
            "PowerShell publish command must expand the archive glob"
        )
        for line in block.splitlines():
            stripped = line.strip().rstrip("`").strip()
            assert not stripped.endswith("*.zip"), (
                f"bare glob passed to a native command: {stripped}"
            )


@needs_evidence
def test_readme_figures_exist_and_the_dataset_image_is_attributed():
    readme = Path("README.md").read_text(encoding="utf-8")
    for figure in ("docs/figures/counting_example.jpg", "docs/figures/model_comparison_test.png"):
        assert figure in readme
        assert Path(figure).is_file()
    # The example photograph is adapted from the CC BY 4.0 dataset record, which
    # requires attribution, a licence link, and an indication of changes.
    caption = readme.split("docs/figures/counting_example.jpg", 1)[1].split("\n## ", 1)[0]
    assert "10.17632/5cp2dyjczk.2" in caption
    assert "creativecommons.org/licenses/by/4.0" in caption
    assert "added by this repository" in caption


@needs_evidence
def test_readme_split_table_matches_the_manifest_and_scored_annotations():
    """Image counts come from the manifest; annotation counts from scored splits."""
    import csv
    import json
    from collections import Counter

    with Path("splits/fold1_recovered.csv").open(encoding="utf-8", newline="") as stream:
        images = Counter(row["split"] for row in csv.DictReader(stream))
    annotations = {
        split: json.loads(
            Path(f"results/reproduced/fcos/{split}/counting.json").read_text(encoding="utf-8")
        )["metrics"]["actual_count"]
        for split in ("train", "val", "test")
    }
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("### Split", 1)[1].split("\n## ", 1)[0]
    rows = {
        cells[0]: cells
        for cells in (
            [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            for line in section.splitlines()
            if line.startswith("| ") and not line.startswith("| split")
        )
    }
    total_images = sum(images.values())
    total_annotations = sum(annotations.values())
    for label, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        name, count, share, cabbages, per_image = rows[label]
        assert int(count) == images[split]
        assert share == f"{100 * images[split] / total_images:.1f}%"
        assert int(cabbages.replace(",", "")) == annotations[split]
        assert per_image == f"{annotations[split] / images[split]:.1f}"
    assert int(rows["total"][1]) == total_images
    assert int(rows["total"][3].replace(",", "")) == total_annotations


@needs_evidence
def test_readme_tables_equal_what_the_generator_renders():
    """Both result tables are copies; a re-scored run must not leave them stale."""
    from cabbage_detection.visualization.tables import (
        render_input_size_table,
        render_results_table,
    )

    root = Path("results/reproduced")
    readme = Path("README.md").read_text(encoding="utf-8")
    assert render_results_table(
        root / "model_comparison_test.json", root / "model_complexity.json"
    ) in readme
    assert render_input_size_table(
        root / "model_comparison_test.json",
        root / "model_comparison_test_detector512.json",
        root / "model_complexity.json",
    ) in readme
