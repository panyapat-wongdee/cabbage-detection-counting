# Reproduced evaluation results

Every number in this directory was produced by running this repository. None of
it is copied from the associated publication, whose DOI is recorded in
[`docs/publication-reference.md`](../../docs/publication-reference.md). This
directory intentionally contains no copy of the paper's metrics or tables, and
no comparison against them is published here.

## What is here

`scripts/evaluate.py` writes one directory per scored run:

```text
results/reproduced/
├── model_comparison_test.{md,json}   # all models, one split, one metric definition
├── model_comparison_test_detector512.{md,json}  # the same with <model>-detector512 runs
├── model_complexity.json             # parameters and GFLOPs at one input size
└── <model>/
    ├── summary.{md,json}             # every split and task for that model
    └── <split>/
        ├── detection.json            # mAP@50, mAP@50:95, backend, AP settings
        ├── counting.json             # micro and macro counting metrics, settings
        ├── per_image.csv             # one row per image, sorted
        ├── metadata.json             # resolved config, command, git revision,
        │                             # environment, input checksums, deviations
        ├── console.log
        ├── plots/{count_scatter.png,count_error_histogram.png}
        └── records/                  # predictions.json, targets.json (not tracked)
```

`records/` holds the raw per-image prediction and target dumps that the metrics
were computed from. They are large and regenerable from the run's checkpoint, so
they stay local; each split's `metadata.json` records their SHA-256 so a local
copy can be checked against the one that produced the tracked numbers.

Training run artifacts are written separately under `runs/reproduced/`; their
metadata, logs, curves, and promotion evidence are tracked there, while the
checkpoints are release assets.

## Cross-model table

[`model_comparison_test.md`](model_comparison_test.md) is generated, not typed:

```powershell
python scripts/compare_models.py --split test
```

It refuses to place models in one table unless they share the AP backend, the
scored image count, and the counting IoU/confidence thresholds, because rows
measured under different definitions are not comparable. Models whose run
recorded a deviation are marked with a footnote in the table.

Splits are always reported separately and are never pooled into one figure.

## Model complexity

[`model_complexity.json`](model_complexity.json) holds, for every run, the
parameter count, GFLOPs at a common 512×512 detector input (`gflops`), and
GFLOPs at the input the run's detector actually received (`run_gflops`, the
figure the README tables report):

```powershell
python -m pip install -e ".[analysis]"
python scripts/measure_complexity.py --run runs/reproduced/faster_rcnn --run runs/reproduced/yolo11m
```

Pass one `--run` per model; the file is rewritten with the models given. Each
record names its counter (`torchinfo` for torchvision, Ultralytics'
`model_info` on the fused model for Ultralytics), a `torch.utils.flop_counter`
cross-check, the RPN proposal count for Faster R-CNN, the
checkpoint SHA-256, and `run_detector_input`, the input the released run's
detector actually received. For the torchvision models that input differs from
512×512; see
[Detector input size](../../docs/reproduction-protocol.md#detector-input-size).

## Status of these numbers

Every run behind this directory has been promoted to `result_status:
reproduced_verified` in its `runs/reproduced/<model>/metadata.json`. That
promotion required the run's immutable metadata to record the resolved
configuration, launch command, code revision and dirty state, environment,
dataset DOI/version with annotation and split-manifest digests, base-weight
identifier/source URL/digest, evaluation settings, deviations, and the
re-verified SHA-256 of its checkpoints, predictions, and metrics. A
configuration load, one-image prediction, or wiring smoke is not a reproduced
result.

`reproduced_verified` is a statement about this repository's evidence chain, not
about the paper: it means these numbers trace to executed runs with complete
provenance. This repository does not restate or compare against the published
values.

Note that each split's `metadata.json` embeds the run's *configuration*, whose
`artifacts.result_status` field is the status the config requested at launch
(`reproduction_candidate`). The run's own status lives in the run metadata.

SSDLite (MobileNetV3-Large) is a primary repository model here, reported and
ranked like the other nine. Its `study_scope` is `supplementary_unreported`
because it is not one of the architectures in the associated publication; the
generated tables label it accordingly.
