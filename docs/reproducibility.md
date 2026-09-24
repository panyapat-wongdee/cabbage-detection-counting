# Reproducibility

The supported lightweight workflow is configuration loading, dataset-layout
validation, split-manifest handling, canonical record evaluation, and immutable
run metadata. It runs without the external dataset or pretrained weights when
synthetic inputs are used.

The official Mendeley Data Version 2 download is an external input. Keep its
`annotation.json` and nested `images/<cultivar>/<location>/<YYYYMM>/` hierarchy
unchanged, then pass the outer directory with `--dataset-root`. The loader and
YOLO preparation code consume the COCO `path` field and resolve each image
without requiring a second source-image copy.

Run commands from the project root. The Python entry points under `scripts/`
add the repository's `src/` directory automatically, so a separate
`PYTHONPATH` setting is not required.

The primary setup command validates that official COCO source for torchvision
and then materializes only the YOLO view:

```powershell
python scripts/prepare_dataset.py `
  --format all `
  --dataset-root original_dataset `
  --manifest splits/fold1_recovered.csv `
  --output prepared/fold1_yolo `
  --link-mode hardlink
```

Use `--format validate` or `--format yolo` separately for the corresponding
narrow operation.

The repository protocol uses the values encoded in the checked-in configs. Each
value is labeled as `historical_code`, `repository_protocol`, a framework
default, a repository default, or `unknown`; the label describes evidence for a
setting and never claims a publication result.

Execution configs expose device, worker count, seed, cache mode, and checkpoint
policy explicitly. Torchvision profiles select validation metrics where the
historical implementation supports it. Ultralytics profiles use the framework's
default checkpoint selection and record that provenance explicitly.

Do not call a result reproduced merely because a configuration exists. A
`reproduced_verified` result must include the resolved configuration, command,
code revision, environment/device information, dataset and split digests,
pretrained checkpoint provenance, predictions, evaluation settings, selected
best-checkpoint digest, and documented deviations under
`runs/reproduced/<run-id>/`.

Historical notebooks and outputs are private review inputs, not the supported
public command-line interface or release source.

The actual runtime environment is recorded in each run's immutable metadata.
The package's core dependencies and optional framework groups are declared in
`pyproject.toml`. For a first-time installation that will run both framework
families, use `python -m pip install -e ".[training,dev]"`; this convenience
extra is the exact union of the existing `torchvision` and `ultralytics`
extras. The framework-specific extras remain available for narrower installs.
`requirements.txt` remains the exact CUDA research lock, while
`requirements-dev.txt` contains only development tooling for compatibility with
older setup scripts.

SSDLite (MobileNetV3-Large) is a primary repository model and follows the same
protocol, evaluation, evidence, and release path as the other nine. Its
`study_scope` stays `supplementary_unreported` because it was not one of the
architectures compared in the cited study; that field records publication
scope, not reporting status. Its metrics are reported from its own verified run
artifact, like every other model's.

Detection mAP uses the backend recorded in the resolved configuration.
Repository-native profiles use the CPU-only 101-point interpolated AP
implementation; historical-method torchvision profiles use the
`pycocotools.coco_eval` backend and the explicit validation/final postprocess
settings. Framework adapters export canonical records before evaluation runs.

Unified entry points are available from the repository root:

```powershell
python scripts/train.py --config configs/ultralytics/yolov8n.yaml
python scripts/predict.py --config CONFIG --source SOURCE --checkpoint CHECKPOINT.pt --output PREDICTIONS.json
python scripts/evaluate.py --run RUN_DIR --dataset DATASET_ROOT
```

The evaluation command resolves the config and best checkpoint from the run
directory, so the same invocation works for both frameworks. `--output-dir` defaults to `results/reproduced/<run name>` and must be
empty, so a re-score needs an explicit directory rather than overwriting the
previous one. `--split` defaults to `all`, scoring train, val, and test separately into one subdirectory each;
splits are never pooled into a single figure. It writes detection and counting
reports together, or the subset selected with `--tasks`.
`--ultralytics-native` runs native detection validation instead and requires an
Ultralytics run.

Run the lightweight preflight profiles before a full experiment:

```powershell
python scripts/train.py --config configs/smoke/faster_rcnn.yaml
python scripts/train.py --config configs/smoke/yolov8n.yaml
```

Each profile runs one epoch over the configured recovered split with batch
size two, zero workers, and cache disabled. It verifies installation, dataset,
and training wiring; its metadata remains `result_status: smoke`, so it is not
a paper reproduction result and cannot pass `finalize_reproduction.py`. Run
directories are immutable. Select a new output directory or deliberately
remove/archive the previous local smoke output before repeating a run.

After training and evaluation, promote a candidate only with explicit evidence
files. The command verifies the checkpoint, prediction, and metrics hashes and
rejects incomplete dataset or pretrained-weight provenance:

```powershell
python scripts/finalize_reproduction.py `
  --run-dir runs/reproduced/RUN_ID `
  --predictions results/reproduced/RUN_ID/test/records/predictions.json `
  --metrics results/reproduced/RUN_ID/test/detection.json `
  --dataset-json evidence/dataset.json `
  --pretrained-json evidence/pretrained.json `
  --evaluation-json evidence/evaluation.json
```

`scripts/evaluate.py` writes, per split, the canonical prediction and target
records it generated under `<split>/records/` and the reports selected by
`--tasks` as `<split>/detection.json` and `<split>/counting.json`, defaulting
to both, alongside one `metadata.json` and one `console.log` per split and a
`summary.md`/`summary.json` pair at the output root. Retaining the
records makes a score re-auditable without re-running inference. The counting
report writes sorted per-image metrics and two counting plots beside the
aggregate JSON report. Versioned, list-based or malformed records are rejected
before any report is written. `--ultralytics-native` selects the separate
native detection path, labelled `ultralytics_native_val`; it does not produce
counting metrics.

Training writes `config.yaml`, append-only `logs/console.log` and
`logs/training_log.csv`, `logs/best_epoch.json`, grouped best/last checkpoints,
and loss/validation PNG curves inside the configured run directory, for both
torchvision and Ultralytics models. Ultralytics runs also keep the native
trainer's own artifacts under `framework/`. These files are generated
artifacts; a run is still only `reproduced_verified` after the immutable
provenance gate is complete. Current runs should be created in this grouped
layout; legacy artifact migration is not part of the public workflow.

Long-running commands show live `tqdm` progress bars and concise stage status
on terminal `stderr`. Training and evaluation commands also append those status
lines to the run or evaluation `console.log`; progress-bar redraws are not
copied into the file, and structured report data remains on `stdout` where a
command emits it. Ultralytics' native training bar remains framework-owned and
is not duplicated by the repository wrapper.

The train and predict commands fail explicitly when their optional framework is
not installed or when an adapter cannot produce canonical records. No command
is a reproduced result until a completed evidence artifact is stored under
`runs/reproduced/`.

The full field-by-field contract and profile matrix are documented in
[`configuration.md`](configuration.md). A config load, loader smoke, or
one-image prediction is not a reproduced experiment.
