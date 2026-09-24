# Reproduction protocol

This document describes the behavior implemented by this repository. It is
written as a repository guide, not as a copy of the associated publication.
Published article files, paper tables, paper figures, and publication metrics
are not distributed here; the citation-only record is in
[`publication-reference.md`](publication-reference.md).

## Scope

The supported model keys are `faster_rcnn`, `ssd`, `retinanet`, `fcos`,
`yolov8n`, `yolov8m`, `yolo11n`, `yolo11m`, `rt-detr-l`, and `ssdlite`, plus the
repository extensions `yolo12n`, `yolo12m`, `yolo26n`, and `yolo26m`. All share
one config style, one training protocol, one evaluation path, and one evidence
contract. `study_scope` records publication scope only:

- `paper_model`: the nine architectures compared in the cited study;
- `supplementary_unreported`: SSDLite, trained in the original work but not
  compared in the paper (marked † in reports);
- `repository_extension`: YOLO12 and YOLO26, added by this repository after the
  study and never part of it (marked ‡ in reports).

The extension profiles copy the YOLO11 profile of the same size setting for
setting; their provenance reads `repository_protocol` where the YOLO11 profile
reads `historical_code`, because no historical code exists for them. YOLO26's
detection head is end-to-end and applies no NMS, so its profiles set
`evaluation.nms_owner: none`, as RT-DETR-L's does, and the adapter refuses a
checkpoint whose head contradicts the configured owner. They were trained
with the Ultralytics version the other runs recorded (8.4.9, pinned in
`requirements.txt`) and are released in `reproduced-checkpoints-v1.0.0` with
the other models.

## Inputs

The cabbage dataset is an external input. Download Version 2 from the official
Mendeley record, preserve its original hierarchy, and keep it outside Git.
Follow the attribution and license requirements in [`dataset.md`](dataset.md).
The split manifest contains identifiers and digests, not image or annotation
bytes. Dataset preparation creates ignored framework-specific layouts when a
training adapter requires them.

Pretrained weights are also external. Obtain the exact artifact from the URL in
[`weights/provenance.yaml`](../weights/provenance.yaml), record its resolved URL
and SHA-256 in the run metadata, and do not commit the weight file.

## Configuration and execution

Each YAML file under `configs/` is strict and records model, framework, dataset
paths, preprocessing, optimizer, scheduler, post-processing, execution, and
checkpoint settings. Provenance values identify `historical_code`,
`repository_protocol`, `framework_default`, `repository_default`, or `unknown`.
They are not publication-result labels.

Run commands from the repository root. `scripts/train.py` and
`scripts/predict.py` select native torchvision or Ultralytics adapters without
pretending that their framework defaults are identical. `scripts/evaluate.py`
is the single evaluation entry point. It takes a training run directory and the
official dataset root, reads the run's recorded config and best checkpoint,
predicts the requested splits, and writes the canonical records beside the
reports. Results are written to `results/reproduced/<run name>` unless
`--output-dir` names another empty directory. `--split` defaults to `all` and
scores train, val, and test into separate subdirectories; split results are never pooled into one figure. `--tasks` selects `detection`, `counting`, or `both`. Ground truth is
built from the official COCO annotations for both frameworks and mapped into
the coordinate space of the adapter that produced the predictions: the
torchvision adapter returns resized boxes, the Ultralytics adapter returns
original image pixels. `--ultralytics-native` instead runs the separate native
`model.val` path labelled `ultralytics_native_val`; it requires an Ultralytics
run and reports detection only.

## Detector input size

Every profile sets `training.image_size: [512, 512]`, and every pipeline first
resizes each image to that size. The torchvision detectors then resize again
inside their own `GeneralizedRCNNTransform`, in training and in inference, so
their backbones do not receive 512×512:

| model | configured resize | backbone input in released runs | torchvision setting that decides it |
|---|---|---|---|
| Faster R-CNN, RetinaNet, FCOS | 512×512 | 800×800 | default `min_size=800` |
| SSD | 512×512 | 300×300 | `fixed_size=(300, 300)` of SSD300 |
| SSDLite | 512×512 | 320×320 | `fixed_size=(320, 320)` of SSDLite320 |
| YOLOv8, YOLO11, RT-DETR-L | 512×512 | 512×512 | none |

For SSD and SSDLite the builder also sets `min_size` and `max_size` to 512, but
`fixed_size` takes precedence, so those two assignments have no effect. They
are kept so the released runs rebuild identically. The behaviour was measured
with the torchvision version recorded in every run (0.20.1) and is pinned by
`tests/models/test_torchvision_models.py`.

This is what the reproduced results measure, and it is not changed in place.
The author of the published study confirms that its experiment used these
torchvision detectors unmodified, so the same internal resize applies to the
published results. The profiles in `configs/torchvision/detector_input_512/`
set `model.detector_input: image_size`, which forces the transform to keep
512×512 (and clears SSD300's 300-pixel anchor `steps`, so default boxes follow
the actual feature map). All five are trained and scored by
`scripts/reproduce_all.py` as a separate repository experiment, reported in the
README's *Controlled experiment: detector input size* and in
`results/reproduced/model_comparison_test_detector512.{md,json}`.

Parameter and FLOP counts (`scripts/measure_complexity.py`) are taken at a
common 512×512 detector input for every model, and each count records the
input the released run actually used as `run_detector_input`.

## Metrics

Torchvision training records `config.yaml`, `logs/console.log`,
`logs/training_log.csv`, `logs/best_epoch.json`,
`checkpoints/best.pt`, `checkpoints/last.pt`, and training/validation curves
under `plots/`. Per-epoch checkpoints are opt-in with
`checkpoint.retain_each_epoch: true`. Evaluation writes one directory per split holding
`detection.json`, `counting.json`, the sorted `per_image.csv`, counting plots,
one `metadata.json` shared by both tasks, and one `console.log`; the split
reports are rolled up into `summary.json` and `summary.md` at the output root.

`scripts/compare_models.py` rolls the other way, gathering one split from every
evaluated model under a results root into `model_comparison_<split>.{md,json}`.
It refuses to build the table unless the models share the AP backend, the scored
image count, and the counting IoU/confidence thresholds, and it labels each row
with its `study_scope`. Directories holding an alternative
scoring of a run, such as an `--ultralytics-native` output, are never folded in;
`--variant detector512` builds a separate table in which the `-detector512`
runs replace their models' primary runs.
Recorded run deviations become footnotes rather than being dropped.

Everything under `results/reproduced/` is tracked except `records/`, whose raw
prediction and target dumps are large and regenerable from the run checkpoint;
their SHA-256 values stay in each split's `metadata.json`.

Training, dataset preparation, and canonical evaluation expose live `tqdm`
progress bars for countable work. Human stage messages and bars use terminal
`stderr`, while machine-readable reports remain on `stdout`. Torchvision bars
show batch loss, running loss, learning rate, image counts, skipped images, and
CUDA memory when available; validation bars show processed images and
detections. Commands with an output directory append durable milestones to
`console.log`. Framework-owned Ultralytics progress remains unchanged.

Detection and counting are separate evaluation tasks. Detection uses the
repository's schema-less canonical prediction records and explicitly records confidence,
NMS ownership, native-NMS status, IoU, maximum detections, area filtering, and
the selected AP backend.
Canonical records use class ID `1` for cabbage; the Ultralytics adapter converts
its native zero-based cabbage class before records reach repository evaluators.
Historical-method torchvision profiles use COCOeval-compatible AP and separate
validation/final filtering settings. Ultralytics native validation and
canonical-record AP are reported under different backend names. Cross-framework comparison tables force one AP backend with
`--detection-backend`; the configured and selected backends are both recorded.
AP over the full score range uses `--ap-confidence`, which widens the export
without changing counting: NMS is score-ordered, so re-filtering the wider
export at the configured threshold reproduces the narrow export exactly.
`--native-score-threshold` additionally lowers a torchvision detector's own
score gate and is recorded as a deviation.
Counting uses one-to-one greedy highest-IoU matching at
`evaluation.counting_iou_threshold`, which defaults to the NMS
`postprocess.iou_threshold` when unset, at the configured
confidence threshold, and records actual count, predicted count, true positives, false
positives, false negatives, count error, FDR, FNR, and F1. The preserved
historical counting loops mark every prediction/target pair above the IoU
threshold, so the one-to-one repository metric may not reproduce the paper's pairwise counting values;
it remains the primary metric because it prevents
duplicate matches.

## Run evidence

Every run starts as `smoke` or `reproduction_candidate`. A run becomes
`reproduced_verified` only after its immutable metadata contains the code
revision, resolved config digest, dataset/split digest, pretrained provenance,
environment/device versions, seed, command, predictions, evaluation settings,
selected best-checkpoint digest, and documented deviations. Training run
evidence is written only to `runs/reproduced/<run-id>/`.

New repository metrics are not expected to match an external publication's
numbers. The repository reports what its revised implementation actually
measured and never substitutes an external table for run evidence.

The paper describes shared Albumentations augmentation and 512-by-512 resizing.
The repository records framework-native Ultralytics geometry separately from
the torchvision Albumentations path; this implementation difference is a
documented deviation until an equivalence audit is completed.

## Release rule

External releases may contain only `best.pt` or `best.pth` selected from verified
reproduction runs. Each archive includes a model card, checksum metadata, the
applicable license, and required third-party notices. Dataset content,
upstream pretrained assets, and local run directories remain outside the
release.
