import json
import hashlib
from pathlib import Path

import pytest

from cabbage_detection.data.manifests import SplitManifest, _digest, write_manifest
from cabbage_detection.cli import evaluate_canonical, evaluate_counts, main, predict, prepare_dataset
from cabbage_detection.evaluation.io import write_prediction_records, write_target_records
from cabbage_detection.evaluation.counting import evaluate_counting
from cabbage_detection.evaluation.records import Box, Detection, PredictionRecord, TargetRecord
from cabbage_detection.data.manifests import SplitManifest, write_manifest
from cabbage_detection.data.manifests import _digest


def _layout(tmp_path: Path) -> Path:
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        (tmp_path / split / "images" / f"{split}.png").write_bytes(b"image")
        (tmp_path / split / "labels" / f"{split}.txt").write_text("", encoding="utf-8")
    return tmp_path


def test_prepare_dataset_writes_manifest(tmp_path: Path):
    output = tmp_path / "manifest.csv"

    result = prepare_dataset(_layout(tmp_path / "dataset"), output)

    assert result == output
    assert output.exists()


def test_prepare_dataset_rejects_legacy_format(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported dataset preparation format"):
        prepare_dataset(_layout(tmp_path / "dataset"), tmp_path / "manifest.csv", format="legacy")


def test_prepare_dataset_validates_official_coco_source(tmp_path: Path):
    source = tmp_path / "images"
    source.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (source / name).write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "a.png", "width": 10, "height": 10},
            {"id": 2, "file_name": "b.png", "width": 10, "height": 10},
            {"id": 3, "file_name": "c.png", "width": 10, "height": 10},
        ],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
    }), encoding="utf-8")
    manifest = tmp_path / "fold_recovered.csv"
    rows = (("train", "a"), ("val", "b"), ("test", "c"))
    write_manifest(SplitManifest(rows, "fixture", _digest(rows)), manifest)
    result = prepare_dataset(source, manifest, annotations=annotations, format="validate")
    assert result == manifest


def _official_fixture(tmp_path: Path):
    download_root = tmp_path / "download"
    bundle = download_root / "An annotated image dataset of cabbages for instance segmentation"
    images = bundle / "images" / "OkinaSP" / "Kaizu" / "202010"
    images.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png"):
        (images / name).write_bytes(b"image")
    annotations = bundle / "annotation.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "path": "/OkinaSP/Kaizu/202010/a.png", "width": 10, "height": 10},
                    {"id": 2, "file_name": "b.png", "path": "/OkinaSP/Kaizu/202010/b.png", "width": 10, "height": 10},
                    {"id": 3, "file_name": "c.png", "path": "/OkinaSP/Kaizu/202010/c.png", "width": 10, "height": 10},
                ],
                "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]},
                                {"id": 2, "image_id": 2, "bbox": [1, 1, 2, 2]}],
            }
        ),
        encoding="utf-8",
    )
    rows = (("train", "a"), ("val", "b"), ("test", "c"))
    manifest = tmp_path / "fold.csv"
    write_manifest(SplitManifest(rows, "fixture", _digest(rows)), manifest)
    return download_root, images, annotations, manifest


def test_prepare_dataset_accepts_outer_dataset_root(tmp_path: Path):
    download_root, _, _, manifest = _official_fixture(tmp_path)

    result = main(
        [
            "prepare-dataset",
            "--dataset-root",
            str(download_root),
            "--manifest",
            str(manifest),
            "--format",
            "validate",
        ]
    )

    assert result == 0


def test_prepare_dataset_all_validates_then_builds_yolo(tmp_path: Path):
    download_root, _, _, manifest = _official_fixture(tmp_path)
    output = tmp_path / "prepared" / "fold1_yolo"

    result = prepare_dataset(
        None,
        manifest,
        dataset_root=download_root,
        format="all",
        output=output,
        link_mode="hardlink",
    )

    assert result == output
    assert (output / "data.yaml").is_file()
    report = json.loads((output / "preparation-report.json").read_text(encoding="utf-8"))
    assert report["split_counts"] == {"test": 1, "train": 1, "val": 1}


def test_prepare_dataset_cli_accepts_all(tmp_path: Path):
    download_root, _, _, manifest = _official_fixture(tmp_path)
    output = tmp_path / "prepared"

    assert main(
        [
            "prepare-dataset",
            "--dataset-root",
            str(download_root),
            "--manifest",
            str(manifest),
            "--format",
            "all",
            "--output",
            str(output),
        ]
    ) == 0
    assert (output / "data.yaml").is_file()


def test_prepare_dataset_all_requires_output(tmp_path: Path):
    download_root, _, _, manifest = _official_fixture(tmp_path)

    with pytest.raises(ValueError, match="all preparation requires --output"):
        prepare_dataset(None, manifest, dataset_root=download_root, format="all")


def test_prepare_dataset_all_does_not_create_output_when_validation_fails(tmp_path: Path):
    download_root, images, _, manifest = _official_fixture(tmp_path)
    next(images.glob("*.png")).unlink()
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="manifest images missing"):
        prepare_dataset(
            None,
            manifest,
            dataset_root=download_root,
            format="all",
            output=output,
        )

    assert not output.exists()


def test_prepare_dataset_rejects_mixed_discovered_and_explicit_inputs(tmp_path: Path):
    download_root, images, annotations, manifest = _official_fixture(tmp_path)

    with pytest.raises(SystemExit):
        main(
            [
                "prepare-dataset",
                "--dataset-root",
                str(download_root),
                "--source-root",
                str(images),
                "--annotations",
                str(annotations),
                "--manifest",
                str(manifest),
            ]
        )


def test_audit_dataset_accepts_outer_dataset_root(tmp_path: Path):
    download_root, _, _, manifest = _official_fixture(tmp_path)
    generated = tmp_path / "generated"
    output = tmp_path / "audit.json"
    assert main(
        [
            "prepare-dataset",
            "--dataset-root",
            str(download_root),
            "--manifest",
            str(manifest),
            "--format",
            "yolo",
            "--output",
            str(generated),
        ]
    ) == 0

    result = main(
        [
            "audit-dataset-migration",
            "--dataset-root",
            str(download_root),
            "--manifest",
            str(manifest),
            "--generated-yolo",
            str(generated),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert output.exists()


def test_evaluate_counts_rejects_legacy_single_image_lists(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "metrics.json"
    predictions.write_text(json.dumps([{"box": [0, 0, 1, 1], "score": 0.9}]), encoding="utf-8")
    targets.write_text(json.dumps([{"box": [0, 0, 1, 1]}]), encoding="utf-8")

    with pytest.raises(ValueError, match="schema-less"):
        evaluate_counts(predictions, targets, output, iou_threshold=0.5)
    assert not output.exists()


def test_evaluate_counts_writes_dataset_report_csv_and_plots(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "counting" / "metrics.json"
    prediction_records = [
        PredictionRecord("c", ()),
        PredictionRecord("a", (Detection((0, 0, 1, 1), 0.9),)),
        PredictionRecord("b", (Detection((0, 0, 1, 1), 0.9), Detection((0, 0, 1, 1), 0.8))),
    ]
    target_records = [
        TargetRecord("b", (Box((0, 0, 1, 1)),)),
        TargetRecord("c", (Box((0, 0, 1, 1)),)),
        TargetRecord("a", (Box((0, 0, 1, 1)),)),
    ]
    write_prediction_records(predictions, prediction_records)
    write_target_records(targets, target_records)

    result = evaluate_counts(predictions, targets, output, iou_threshold=0.5)

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert list(payload)[:4] == ["task", "input_schema", "scope", "metrics"]
    assert "schema_version" not in payload
    assert payload["task"] == "counting"
    assert payload["input_schema"] == "canonical"
    assert payload["scope"]["image_count"] == 3
    assert payload["metrics"]["predicted_count"] == 3
    assert payload["metrics"]["actual_count"] == 3
    assert payload["metrics"]["true_positives"] == 2
    assert payload["metrics"]["f1"] == pytest.approx(2 / 3)
    assert "per_image" not in payload
    evaluation_metadata = json.loads((output.parent / "metadata.json").read_text(encoding="utf-8"))
    assert "schema_version" not in evaluation_metadata
    csv_rows = (output.parent / "per_image.csv").read_text(encoding="utf-8").splitlines()
    assert [line.split(",")[0] for line in csv_rows[1:]] == ["a", "b", "c"]
    assert (output.parent / "metadata.json").exists()
    assert (output.parent / "plots" / "count_scatter.png").read_bytes().startswith(b"\x89PNG")
    assert (output.parent / "plots" / "count_error_histogram.png").read_bytes().startswith(b"\x89PNG")


def test_counting_plot_data_is_compact_and_histogram_is_bounded(tmp_path: Path):
    from cabbage_detection.evaluation.count_artifacts import count_histogram_bins, count_plot_data

    report = evaluate_counting(
        [PredictionRecord(str(index), (Detection((0, 0, 1, 1), 0.9),)) for index in range(3)],
        [TargetRecord(str(index), tuple(Box((0, 0, 1, 1)) for _ in range(index + 1))) for index in range(3)],
        iou_threshold=0.5,
    )
    data = count_plot_data(report)
    assert data["actual"] == [1, 2, 3]
    assert data["predicted"] == [1, 1, 1]
    assert len(data["point_sizes"]) == 3
    assert len(count_histogram_bins(list(range(-1000, 1001)))) <= 60


def test_counting_plot_data_handles_empty_dataset(tmp_path: Path):
    from cabbage_detection.evaluation.count_artifacts import count_plot_data

    report = evaluate_counting([], [], iou_threshold=0.5)
    assert count_plot_data(report) == {"actual": [], "predicted": [], "errors": [], "point_sizes": []}


def test_counting_artifact_writer_writes_report_artifacts_from_existing_report(tmp_path: Path):
    from cabbage_detection.evaluation.count_artifacts import write_counting_report

    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "counting" / "metrics.json"
    prediction_records = (PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),)),)
    target_records = (TargetRecord("image", (Box((0, 0, 1, 1)),)),)
    write_prediction_records(predictions, prediction_records)
    write_target_records(targets, target_records)
    report = evaluate_counting(prediction_records, target_records, iou_threshold=0.5)

    result = write_counting_report(report, output, iou_threshold=0.5)

    assert result == output
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["true_positives"] == 1
    assert (output.parent / "per_image.csv").is_file()
    assert (output.parent / "plots" / "count_scatter.png").is_file()
    assert (output.parent / "plots" / "count_error_histogram.png").is_file()


def test_evaluate_counts_rejects_mixed_canonical_and_legacy_inputs(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    predictions.write_text(json.dumps({"records": []}), encoding="utf-8")
    targets.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="schema-less"):
        evaluate_counts(predictions, targets, tmp_path / "metrics.json")


def test_evaluate_counts_rejects_versioned_records(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    predictions.write_text(json.dumps({"schema_version": 1, "records": []}), encoding="utf-8")
    targets.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema-less"):
        evaluate_counts(predictions, targets, tmp_path / "metrics.json")


def test_evaluate_counts_rejects_invalid_iou_before_writing(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    predictions.write_text(json.dumps({"records": []}), encoding="utf-8")
    targets.write_text(json.dumps({"records": []}), encoding="utf-8")
    output = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_counts(predictions, targets, output, iou_threshold=1.1)
    assert not output.exists()


def test_evaluate_counts_rejects_malformed_canonical_input(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    predictions.write_text(json.dumps({}), encoding="utf-8")
    targets.write_text(json.dumps({"records": []}), encoding="utf-8")
    output = tmp_path / "metrics.json"

    with pytest.raises(ValueError, match="schema-less"):
        evaluate_counts(predictions, targets, output)
    assert not output.exists()


def _config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model: {name: yolo11n, framework: ultralytics}
dataset: {format: yolo, root: dataset, split_manifest: splits.csv, yaml: dataset/data.yaml, labels_dirname: null, class_names: [Cabbage]}
initialization: {weights: yolo11n.pt, num_classes: 1}
training:
  image_size: [512, 512]
  optimizer: SGD
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  batch_size: 1
  epochs: 1
augmentation: {profile: reproduction_augmented, horizontal_flip: 0.5, vertical_flip: 0.5, translate: 0.1, scale: 0.1, degrees: 15.0, hsv_h: 0.1, hsv_s: 0.05, hsv_v: 0.05, mosaic: 0.0, normalize: true, border_value: 114}
scheduler: {name: constant, step_size: null, gamma: null, final_lr_factor: 1.0}
postprocess: {confidence_threshold: 0.5, iou_threshold: 0.5}
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}
artifacts: {output_dir: run}
""",
        encoding="utf-8",
    )
    return path


def test_predict_writes_canonical_records(tmp_path: Path, monkeypatch):
    records = [PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),))]
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("cabbage_detection.cli.run_prediction", lambda config, source, path: records)
    output = predict(
        _config_file(tmp_path),
        tmp_path / "image.png",
        checkpoint,
        tmp_path / "predictions.json",
    )
    assert output.exists()
    assert "schema_version" not in json.loads(output.read_text(encoding="utf-8"))


def test_predict_cli_requires_checkpoint(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "predict",
                "--config",
                str(_config_file(tmp_path)),
                "--source",
                str(tmp_path / "image.png"),
                "--output",
                str(tmp_path / "predictions.json"),
            ]
        )


def test_finalize_reproduction_cli_promotes_candidate(tmp_path: Path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"best")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "result_status": "reproduction_candidate",
                "command": ["train"],
                "git": {"revision": "a" * 40, "dirty": False},
                "config": {"model": {"name": "faster_rcnn"}},
                "environment": {"python_version": "3.11"},
                "outputs": {"best_checkpoint": {"path": "checkpoints/best.pt", "sha256": digest}},
            }
        ),
        encoding="utf-8",
    )
    predictions = run / "predictions.json"
    predictions.write_text('{"records": []}\n', encoding="utf-8")
    metrics = run / "metrics.json"
    metrics.write_text('{"map50": 0.5}\n', encoding="utf-8")
    evidence = {}
    for name, payload in {
        "dataset": {"doi": "10.0000/example-dataset.1", "version": "2", "split_manifest_sha256": "b" * 64, "annotation_sha256": "c" * 64},
        "pretrained": {"identifier": "COCO_V1", "source_url": "https://example.test/weights.pth", "sha256": "d" * 64},
        "evaluation": {"backend": "pycocotools.coco_eval", "iou_threshold": 0.5},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        evidence[name] = path

    assert main(
        [
            "finalize-reproduction",
            "--run-dir",
            str(run),
            "--predictions",
            str(predictions),
            "--metrics",
            str(metrics),
            "--dataset-json",
            str(evidence["dataset"]),
            "--pretrained-json",
            str(evidence["pretrained"]),
            "--evaluation-json",
            str(evidence["evaluation"]),
        ]
    ) == 0
    assert json.loads((run / "metadata.json").read_text(encoding="utf-8"))["result_status"] == "reproduced_verified"


def test_unified_evaluate_reads_canonical_records(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    write_prediction_records(predictions, [PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),))])
    write_target_records(targets, [TargetRecord("image", (Box((0, 0, 1, 1)),))])
    output = evaluate_canonical(_config_file(tmp_path), predictions, targets, tmp_path / "evaluation")
    payload = json.loads((output / "detection.json").read_text(encoding="utf-8"))
    assert "schema_version" not in payload
    assert payload["task"] == "detection"
    assert payload["metrics"]["map50"] == 1.0
    assert (output / "metadata.json").exists()


def test_evaluate_canonical_writes_detection_and_counting_artifacts(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output_dir = tmp_path / "evaluation"
    write_prediction_records(predictions, [PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),))])
    write_target_records(targets, [TargetRecord("image", (Box((0, 0, 1, 1)),))])

    result = evaluate_canonical(_config_file(tmp_path), predictions, targets, output_dir)

    assert result == output_dir
    detection = json.loads((output_dir / "detection.json").read_text(encoding="utf-8"))
    counting = json.loads((output_dir / "counting.json").read_text(encoding="utf-8"))
    assert detection["task"] == "detection"
    assert counting["task"] == "counting"
    assert (output_dir / "per_image.csv").is_file()
    assert (output_dir / "plots" / "count_scatter.png").is_file()


def test_evaluate_canonical_does_not_publish_partial_results(tmp_path: Path, monkeypatch):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output_dir = tmp_path / "evaluation"
    write_prediction_records(predictions, [PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),))])
    write_target_records(targets, [TargetRecord("image", (Box((0, 0, 1, 1)),))])

    def fail_writer(*args, **kwargs):
        raise RuntimeError("counting writer failed")

    monkeypatch.setattr("cabbage_detection.cli.write_counting_report", fail_writer)
    with pytest.raises(RuntimeError, match="counting writer failed"):
        evaluate_canonical(_config_file(tmp_path), predictions, targets, output_dir)

    assert not (output_dir / "detection.json").exists()
    assert not (output_dir / "counting.json").exists()


def test_unified_evaluate_uses_configured_coco_backend(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    write_prediction_records(predictions, [PredictionRecord("image", (Detection((0, 0, 10, 10), 0.9),))])
    write_target_records(targets, [TargetRecord("image", (Box((0, 0, 10, 10)),))])
    output = evaluate_canonical(
        Path("configs/torchvision/faster_rcnn.yaml"),
        predictions,
        targets,
        tmp_path / "evaluation",
    )
    payload = json.loads((output / "detection.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["backend"] == "pycocotools.coco_eval"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["evaluation"]["detection"]["backend"] == "pycocotools.coco_eval"


def _evaluation_run(tmp_path: Path, framework: str = "ultralytics") -> Path:
    """Build a run directory whose config points at the written split manifest."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    manifest = tmp_path / "splits.csv"
    rows = (("train", "img000"), ("val", "img001"), ("test", "img002"))
    write_manifest(SplitManifest(rows=rows, provenance="test-fixture", digest=_digest(rows)), manifest)
    source = _config_file(tmp_path).read_text(encoding="utf-8")
    source = source.replace("split_manifest: splits.csv", f"split_manifest: {manifest.as_posix()}")
    if framework == "torchvision":
        source = source.replace("{name: yolo11n, framework: ultralytics}", "{name: faster_rcnn, framework: torchvision}")
        source = source.replace("format: yolo", "format: coco")
        source = source.replace("yaml: dataset/data.yaml", "yaml: null")
        source = source.replace("confidence_owner: ultralytics, nms_owner: ultralytics", "confidence_owner: torchvision, nms_owner: torchvision")
        source = source.replace("weights: yolo11n.pt, num_classes: 1", "weights: COCO_V1, num_classes: 2")
    (run / "config.yaml").write_text(source, encoding="utf-8")
    return run


def _official_dataset(tmp_path: Path) -> Path:
    """Write a minimal extracted official download with one image per split."""
    bundle = tmp_path / "original_dataset" / "cabbage"
    (bundle / "images").mkdir(parents=True)
    images = []
    annotations = []
    for index in range(3):
        name = f"img{index:03d}"
        (bundle / "images" / f"{name}.png").write_bytes(b"image")
        images.append({"id": index, "file_name": f"images/{name}.png", "width": 100, "height": 50})
        annotations.append({"id": index, "image_id": index, "category_id": 1, "bbox": [10, 10, 20, 20]})
    (bundle / "annotation.json").write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )
    return tmp_path / "original_dataset"


def test_evaluate_cli_generates_records_and_reports_from_a_run(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    dataset = _official_dataset(tmp_path)
    output = tmp_path / "evaluation"
    sources: list[Path] = []

    def fake_prediction(config, source_paths, checkpoint, progress_description=None):
        sources.extend(Path(source) for source in source_paths)
        return [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in source_paths
        ]

    monkeypatch.setattr("cabbage_detection.cli.run_predictions", fake_prediction)
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert [path.name for path in sources] == ["img000.png", "img001.png", "img002.png"]
    for split in ("train", "val", "test"):
        assert (output / split / "records" / "predictions.json").is_file()
        assert (output / split / "records" / "targets.json").is_file()
        metrics = output / split / "detection.json"
        assert json.loads(metrics.read_text(encoding="utf-8"))["metrics"]["map50"] == 1.0
        assert (output / split / "counting.json").is_file()


@pytest.mark.parametrize(
    "tasks, written, absent",
    [("detection", "detection", "counting"), ("counting", "counting", "detection")],
)
def test_evaluate_cli_writes_only_the_selected_task(tmp_path: Path, monkeypatch, tasks, written, absent):
    run = _evaluation_run(tmp_path)
    dataset = _official_dataset(tmp_path)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        "cabbage_detection.cli.run_predictions",
        lambda config, sources, checkpoint, progress_description=None: [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in sources
        ],
    )
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(dataset),
            "--split",
            "test",
            "--tasks",
            tasks,
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert (output / "test" / f"{written}.json").is_file()
    assert not (output / "test" / f"{absent}.json").exists()


def test_evaluate_cli_scores_a_single_split_when_one_is_named(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        "cabbage_detection.cli.run_predictions",
        lambda config, sources, checkpoint, progress_description=None: [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in sources
        ],
    )
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(_official_dataset(tmp_path)),
            "--split",
            "val",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert (output / "val" / "detection.json").is_file()
    assert not (output / "train").exists()
    assert not (output / "test").exists()


def test_evaluate_cli_reports_a_split_missing_from_the_manifest(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    manifest = tmp_path / "splits.csv"
    rows = (("train", "img000"), ("test", "img002"))
    write_manifest(SplitManifest(rows=rows, provenance="test-fixture", digest=_digest(rows)), manifest)
    monkeypatch.setattr(
        "cabbage_detection.cli.run_predictions",
        lambda config, sources, checkpoint, progress_description=None: [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in sources
        ],
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(_official_dataset(tmp_path)),
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


def test_evaluate_cli_defaults_the_output_directory_to_the_run_name(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    dataset = _official_dataset(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        "cabbage_detection.cli.run_predictions",
        lambda config, sources, checkpoint, progress_description=None: [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in sources
        ],
    )
    assert main(
        ["evaluate", "--run", str(run), "--dataset", str(dataset), "--split", "test"]
    ) == 0
    assert (workspace / "results" / "reproduced" / run.name / "test" / "detection.json").is_file()


def test_evaluate_cli_rejects_a_non_empty_default_output_directory(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    workspace = tmp_path / "workspace"
    existing = workspace / "results" / "reproduced" / run.name
    existing.mkdir(parents=True)
    (existing / "previous.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(workspace)
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--run", str(run), "--dataset", str(_official_dataset(tmp_path))])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("config", "cabbage_detection.ap_101"),
        ("repository_ap_101", "cabbage_detection.ap_101"),
        ("pycocotools_coco_eval", "pycocotools.coco_eval"),
    ],
)
def test_evaluate_cli_can_force_a_detection_backend(tmp_path: Path, monkeypatch, backend, expected):
    run = _evaluation_run(tmp_path)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        "cabbage_detection.cli.run_predictions",
        lambda config, sources, checkpoint, progress_description=None: [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in sources
        ],
    )
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(_official_dataset(tmp_path)),
            "--split",
            "test",
            "--tasks",
            "detection",
            "--detection-backend",
            backend,
            "--output-dir",
            str(output),
        ]
    ) == 0
    payload = json.loads(
        (output / "test" / "detection.json").read_text(encoding="utf-8")
    )
    assert payload["metrics"]["backend"] == expected
    assert payload["evaluation"]["backend_selection"] == backend
    assert payload["evaluation"]["configured_backend"] == "repository_ap_101"


def test_evaluate_cli_rejects_a_forced_backend_with_native_validation(tmp_path: Path):
    run = _evaluation_run(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(_official_dataset(tmp_path)),
                "--split",
                "test",
                "--tasks",
                "detection",
                "--ultralytics-native",
                "--detection-backend",
                "repository_ap_101",
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


def test_ap_confidence_widens_detection_records_but_not_counting(tmp_path: Path, monkeypatch):
    """AP sees the full score range; counting stays at the configured 0.5."""
    run = _evaluation_run(tmp_path)
    output = tmp_path / "evaluation"
    seen: list[float] = []

    def fake_prediction(config, source_paths, checkpoint, progress_description=None):
        seen.append(config.confidence_threshold)
        return [
            PredictionRecord(
                Path(source).stem,
                (
                    Detection((10, 10, 30, 30), 0.9),
                    Detection((60, 10, 80, 30), 0.05),
                ),
            )
            for source in source_paths
        ]

    monkeypatch.setattr("cabbage_detection.cli.run_predictions", fake_prediction)
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(_official_dataset(tmp_path)),
            "--split",
            "test",
            "--ap-confidence",
            "0.001",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert seen == [0.001]
    records = json.loads(
        (output / "test" / "records" / "predictions.json").read_text(encoding="utf-8")
    )
    detection = json.loads(
        (output / "test" / "detection.json").read_text(encoding="utf-8")
    )
    counting = json.loads(
        (output / "test" / "counting.json").read_text(encoding="utf-8")
    )
    assert detection["evaluation"]["export_confidence_threshold"] == 0.001
    assert counting["evaluation"]["confidence_threshold"] == 0.5
    # The low-scoring box is exported and scored by AP, but not counted.
    exported = {record["image_id"]: record for record in records["records"]}
    assert len(exported["img002"]["detections"]) == 2
    assert counting["metrics"]["predicted_count"] == 1


def test_ap_confidence_must_not_exceed_the_configured_threshold(tmp_path: Path):
    run = _evaluation_run(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(_official_dataset(tmp_path)),
                "--ap-confidence",
                "0.9",
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


def test_counting_report_includes_a_macro_average_beside_the_micro_metrics(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        "cabbage_detection.cli.run_predictions",
        lambda config, sources, checkpoint, progress_description=None: [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in sources
        ],
    )
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(_official_dataset(tmp_path)),
            "--split",
            "test",
            "--tasks",
            "counting",
            "--output-dir",
            str(output),
        ]
    ) == 0
    payload = json.loads(
        (output / "test" / "counting.json").read_text(encoding="utf-8")
    )
    assert payload["evaluation"]["aggregation"] == "micro"
    assert set(payload["metrics_macro"]) == {"count_error", "fdr", "fnr", "f1"}


def test_native_score_threshold_is_rejected_for_an_ultralytics_run(tmp_path: Path):
    run = _evaluation_run(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(_official_dataset(tmp_path)),
                "--native-score-threshold",
                "0.001",
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


def test_native_score_threshold_is_recorded_as_a_deviation(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path, framework="torchvision")
    output = tmp_path / "evaluation"
    seen: list[float | None] = []

    def fake_prediction(config, source_paths, checkpoint, progress_description=None):
        seen.append(config.native_score_threshold)
        return [
            PredictionRecord(Path(source).stem, (Detection((10, 10, 30, 30), 0.9),))
            for source in source_paths
        ]

    monkeypatch.setattr("cabbage_detection.cli.run_predictions", fake_prediction)
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(_official_dataset(tmp_path)),
            "--split",
            "test",
            "--tasks",
            "detection",
            "--native-score-threshold",
            "0.001",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert seen == [0.001]
    metadata = json.loads(
        (output / "test" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["evaluation"]["detection"]["native_score_threshold"] == 0.001
    assert metadata["deviations"] == ["native detector score threshold lowered for full-range AP"]


def test_evaluate_cli_dispatches_native_validation_with_the_run_checkpoint(tmp_path: Path, monkeypatch):
    run = _evaluation_run(tmp_path)
    dataset = _official_dataset(tmp_path)
    output = tmp_path / "native-evaluation"
    calls: list[tuple[Path, Path, str, Path]] = []

    def fake_native(config, checkpoint_path, data_path, split, output_dir):
        calls.append((checkpoint_path, data_path, split, output_dir))
        return output / "native_metrics.json"

    monkeypatch.setattr("cabbage_detection.cli.evaluate_ultralytics_native", fake_native)
    assert main(
        [
            "evaluate",
            "--run",
            str(run),
            "--dataset",
            str(dataset),
            "--split",
            "test",
            "--tasks",
            "detection",
            "--ultralytics-native",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert calls == [(run / "checkpoints" / "best.pt", Path("dataset/data.yaml"), "test", output / "test")]


def test_evaluate_cli_rejects_native_validation_for_a_torchvision_run(tmp_path: Path):
    run = _evaluation_run(tmp_path, framework="torchvision")
    dataset = _official_dataset(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(dataset),
                "--tasks",
                "detection",
                "--ultralytics-native",
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


def test_evaluate_cli_rejects_counting_with_native_validation(tmp_path: Path):
    run = _evaluation_run(tmp_path)
    dataset = _official_dataset(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(dataset),
                "--ultralytics-native",
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("missing", ["config.yaml", "checkpoints/best.pt"])
def test_evaluate_cli_rejects_an_incomplete_run_directory(tmp_path: Path, missing):
    run = _evaluation_run(tmp_path)
    (run / missing).unlink()
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(_official_dataset(tmp_path)),
                "--output-dir",
                str(tmp_path / "evaluation"),
            ]
        )
    assert error.value.code == 2


def test_evaluate_cli_rejects_a_non_empty_output_directory(tmp_path: Path):
    run = _evaluation_run(tmp_path)
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "evaluate",
                "--run",
                str(run),
                "--dataset",
                str(_official_dataset(tmp_path)),
                "--output-dir",
                str(output),
            ]
        )
    assert error.value.code == 2


def test_unified_evaluate_requires_a_directory_output(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    write_prediction_records(predictions, [PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),))])
    write_target_records(targets, [TargetRecord("image", (Box((0, 0, 1, 1)),))])

    output = tmp_path / "evaluation" / "metrics.json"
    output.parent.mkdir()
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        evaluate_canonical(_config_file(tmp_path), predictions, targets, output)


def test_public_evaluation_surface_has_one_script():
    scripts = Path(__file__).parents[1] / "scripts"
    assert (scripts / "evaluate.py").is_file()
    assert not (scripts / "evaluate_counts.py").exists()
    assert not (scripts / "evaluate_ultralytics_native.py").exists()


def test_unified_counting_evaluation_writes_summary_and_metadata(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    write_prediction_records(predictions, [PredictionRecord("image", (Detection((0, 0, 1, 1), 0.9),))])
    write_target_records(targets, [TargetRecord("image", (Box((0, 0, 1, 1)),))])

    output = evaluate_canonical(_config_file(tmp_path), predictions, targets, tmp_path / "evaluation")
    payload = json.loads((output / "counting.json").read_text(encoding="utf-8"))
    assert list(payload)[:4] == ["task", "input_schema", "scope", "metrics"]
    assert payload["task"] == "counting"
    assert payload["metrics"]["true_positives"] == 1
    assert (output / "metadata.json").exists()
