# Experiment configuration

The model profiles under `configs/torchvision/` and `configs/ultralytics/` are
parsed by `cabbage_detection.config.load_config`. The separate
`configs/publication-boundary.yaml` is a publication policy read by its own
validator. These files are executable contracts, not claims that a training
run has already been reproduced.

## Profile families

- `configs/torchvision/*.yaml` discovers the official COCO bundle beneath the
  outer `original_dataset` download root and encodes the repository's augmented
  reproduction protocol.
- `configs/ultralytics/yolo12{n,m}.yaml` and `yolo26{n,m}.yaml` are repository
  extensions added after the study, trained with Ultralytics 8.4.9. Each equals the YOLO11
  profile of the same size; YOLO26 sets `evaluation.nms_owner: none` because
  its end-to-end head applies no NMS.
- `configs/torchvision/ssdlite.yaml` is an auxiliary profile recovered from the
  preserved notebook. SSDLite was not part of the nine-model comparison
  included in the associated publication and has no external-paper result entry here.
- `configs/torchvision/no_augmentation/*.yaml` keeps the corresponding
  torchvision model, training, scheduler, and evaluation values while setting
  all stochastic augmentation magnitudes to zero and retaining
  `normalize: true`. These are repository variants; the original torchvision
  run-specific config was not recovered separately.
- `configs/ultralytics/*.yaml` points at generated
  `prepared/fold1_yolo/data.yaml` and encodes the repository protocol exposed
  through the Ultralytics API.
- `configs/ultralytics/no_augmentation/*.yaml` preserves the recovered
  no-augmentation training values: all augmentation magnitudes are zero. These
  profiles must not be conflated with the paper's augmented description.
- `configs/torchvision/detector_input_512/*.yaml` is a controlled experiment,
  trained and reported separately from the reproduced results: each file equals its primary torchvision profile except
  `model.detector_input: image_size` and a `-detector512` output directory, so
  the detector itself receives 512×512. See
  [Detector input size](reproduction-protocol.md#detector-input-size).

The dataset and pretrained weights are external local inputs. The repository
does not commit them. The split manifest is identifier-only and records the
recovered 320/46/92 membership.

## Sections

- `model`: model name and framework (`torchvision` or `ultralytics`), and
  optional `detector_input` for torchvision models: `framework_default` (the
  default, used by every released run) keeps the detector transform's own
  resize, and `image_size` makes the detector receive `training.image_size`.
- `dataset`: format (`coco`, `pascal_voc`, or `yolo`), root, optional COCO
  annotations, split manifest, optional Ultralytics YAML, label directory name,
  and class names.
- `initialization`: pretrained weight identifier and framework class count
  (2 for torchvision background+cabbage, 1 for Ultralytics).
- `training`: image size, optimizer, learning rate, momentum, weight decay,
  batch size, and epochs.
- `augmentation`: profile, flip/geometry/HSV/mosaic magnitudes, normalization,
  and affine border value. `normalize` controls the optional Albumentations
  `A.Normalize` preprocessing step in every torchvision split; it is not a
  stochastic augmentation. Torchvision's detector transform still applies its
  own input normalization, so `normalize: true` is an additional preprocessing
  step in the recovered torchvision training and evaluation path. The
  no-augmentation profiles keep this preprocessing step enabled while setting
  stochastic augmentation probabilities and magnitudes to zero; the detector's
  internal normalization also remains enabled.
- `scheduler`: `step_lr` for the recovered torchvision scheduler or `constant`
  for the recovered Ultralytics scripts.
- `postprocess`: confidence and IoU thresholds.
- `evaluation`: maximum detections, the default area threshold, filtering/NMS
  ownership, the detection backend (`repository_ap_101` or
  `pycocotools_coco_eval`), and the optional `counting_iou_threshold`. Historical-method torchvision profiles select the
  COCO evaluator used by the preserved notebook. YOLOv8, YOLO11 and YOLO12
  profiles record native Ultralytics NMS; RT-DETR and YOLO26 profiles record
  `nms_owner: none` because their end-to-end heads do not use NMS, and the
  adapter refuses a checkpoint whose head contradicts the configured owner.
- Ultralytics paper-method scores are produced by
  `scripts/evaluate.py --ultralytics-native` and use the
  `ultralytics_native_val` artifact label;
  the YAML backend remains `repository_ap_101` for canonical-record
  evaluation.
- `evaluation.counting_iou_threshold` is the IoU used to match predictions to
  targets when counting. It is a different quantity from
  `postprocess.iou_threshold`, which owns NMS during prediction. Leaving it
  unset (`null`, the default in every shipped profile) keeps the historical
  behaviour of reusing the NMS value, so no existing result changes; set it to
  state the counting threshold independently.
- `native_nms_applied` is true for every torchvision profile, including those
  with `nms_owner: repository`, because torchvision detectors run NMS inside
  `postprocess_detections` before repository filtering sees an output. Those
  profiles therefore apply NMS twice: the detector's own threshold, then the
  repository threshold. The detector's score, NMS, and detections-per-image
  settings are recorded in each prediction's post-processing metadata.
- `overrides.native_score_threshold` is written only by
  `scripts/evaluate.py --native-score-threshold`. No shipped profile sets it,
  and using it is recorded as a deviation.
- `validation_postprocess` and `final_postprocess`: explicit confidence, IoU,
  area, ownership, and native-NMS settings for checkpoint selection and final
  prediction export. Historical profiles use strict score/area filtering,
  validation area `0`, and final area `100` for RetinaNet, SSD, SSDLite, and
  FCOS. Faster R-CNN final area remains `0` because the primary notebook uses
  that value; the copied notebook records a discrepancy.
- `native_training`: patience, cache behavior, train/validation shuffle,
  first-epoch warm-up, and empty-target policy. Historical torchvision profiles
  enable the notebook warm-up (`start_factor=0.001`) and explicitly record the
  current `include` policy until an input audit establishes whether empty images
  were skipped in the original run. Use `scripts/audit_augmented_empty_targets.py`
  to measure boxes removed by stochastic transforms separately from raw empty
  images; the probe does not modify dataset files.
  All profiles record `cache: true`. Ultralytics passes this value to its
  native trainer; torchvision datasets use it to retain decoded RGB images in
  a per-worker RAM cache while leaving stochastic transforms uncached. When
  workers are enabled, their processes stay persistent so the cache survives
  across epochs. Each worker and split owns a separate cache, so monitor RAM
  usage when increasing the worker count.
- `execution`: device, workers, and seed. The profiles use `cuda:0`, two
  workers, and seed `42` as repository defaults. These values are not labeled
  as historical settings. Ultralytics uses the same values as repository
  optimization settings; private historical scripts are not part of the public
  tree.
- `checkpoint`: metric, direction, and optional per-epoch retention. Torchvision
  profiles select validation `val_map_50_95`; every training run starts from
  the configured pretrained initialization. Prediction checkpoints are passed
  explicitly on the command line and are not stored in training config files.
  Ultralytics profiles omit the whole section because checkpoint selection is
  delegated to the framework.
- `artifacts`: `output_dir` selects the run directory, rooted under
  `runs/reproduced/` for every framework. `result_status` accepts
  exactly `reproduction_candidate` or `smoke` for training configs. Smoke runs
  are preflight checks and cannot be promoted; only the completed evidence gate
  can produce `reproduced_verified`.
- Completed runs write `config.yaml`, `logs/console.log`,
  `logs/training_log.csv`, `logs/best_epoch.json`,
  `checkpoints/best.pt`, `checkpoints/last.pt`, and plots under `plots/` for
  both torchvision and Ultralytics models. The per-epoch CSV uses
  `train_loss_mean` for the arithmetic mean of batch losses and leaves
  unavailable validation metrics empty. `plots/training_loss.png` plots
  `val_loss` alongside `train_loss_mean` (when validation runs that epoch) for
  direct train/validation comparison; for torchvision this is an extra
  `model.train()` + `torch.no_grad()` forward pass over the validation set
  each epoch (roughly doubling validation time), while for Ultralytics it is
  read directly from the native `val/*loss` columns in `results.csv`. Set
  `checkpoint.retain_each_epoch:
  true` to retain full per-epoch checkpoints under `checkpoints/epochs/` (torchvision
  only); the default is false. Ultralytics runs additionally keep the native
  trainer's own output under `framework/` (`args.yaml`, `results.csv`, native
  plots, and `weights/`), with `logs/best_epoch.json` reporting Ultralytics'
  own fitness-based checkpoint selection as `selection_metric:
  ultralytics_fitness`.
- Training and evaluation commands use `tqdm` for terminal-only progress bars
  on `stderr`. Durable stage messages are appended to the run/evaluation
  `console.log`; bar redraws are intentionally not persisted, and structured
  report output on `stdout` remains machine-readable.
- `provenance`: source labels (`historical_code`, `repository_protocol`,
  `framework_default`, `repository_default`, or `unknown`) for settings that
  affect scientific behavior. Ultralytics checkpoint selection is explicitly
  delegated to the framework default.

Validate every profile from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -c "from pathlib import Path; from cabbage_detection.config import load_config; [load_config(path) for folder in ('torchvision', 'ultralytics') for path in Path('configs', folder).rglob('*.yaml')]"
```

Full training additionally requires the external data, matching pretrained
weights, optional framework packages, and a compatible CUDA device when the
selected config requests one. A loaded config or a smoke test is not a
reproduced result; only completed artifacts under `runs/reproduced/` are.
