# Reproduced checkpoints v1.0.0

Best checkpoints from nineteen training runs executed by this repository:
fourteen models, and five of them trained a second time at a different
detector input size. The release contains no part of the published paper: no
tables, figures, or transcribed metrics.

## What `reproduced_verified` means here

Every run in this release carries `result_status: reproduced_verified` in its
`runs/reproduced/<run>/metadata.json`, which is tracked in Git. Promotion
required that metadata to record the resolved configuration and its digest, the
launch command, the code revision and dirty state, the environment, the dataset
DOI and record version with the annotation and split-manifest digests, the base
weight's identifier, official source URL and digest, the evaluation settings and
any deviations, and the re-verified SHA-256 of the checkpoints, predictions, and
metrics.

That is a claim about this repository's evidence chain: these checkpoints trace
to executed runs with complete provenance. It is **not** a claim that they
reproduce the published paper's values, which this repository does not restate
or compare against.

## Contents

- one archive per run, `cabbage-<run>-checkpoints-reproduced-v1.0.0.zip`,
  each holding only that run's `best.pt`;
- `CHECKPOINTS.json`, `MODEL_CARD.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`
  inside each archive, with the full notice texts under `notices/`: the
  torchvision BSD-3-Clause licence or the Ultralytics notice, plus the MS COCO
  provenance and dataset attribution;
- `release-manifest.json` and `SHA256SUMS` beside them.

One archive per run is what keeps the licence boundary intact; see below.

## Models

Nine are the architectures compared in the associated publication:
Faster R-CNN, SSD, RetinaNet, FCOS, YOLOv8n, YOLOv8m, YOLO11n, YOLO11m, and
RT-DETR-L. The tenth, SSDLite (MobileNetV3-Large), is released on the same terms
and from the same protocol, but it was not compared in the paper, so its
`study_scope` is `supplementary_unreported`. YOLO12n, YOLO12m, YOLO26n, and
YOLO26m were added by this repository after the study (`repository_extension`).
That field records publication scope only.

### Input-size variants

`faster_rcnn-detector512`, `fcos-detector512`, `retinanet-detector512`,
`ssd-detector512`, and `ssdlite-detector512` are the same torchvision models
retrained with one change: the detector keeps the configured 512×512 input
instead of torchvision's internal resize (800×800 for Faster R-CNN, RetinaNet and
FCOS; 300×300 for SSD; 320×320 for SSDLite). They are a controlled repository
experiment, not a reproduction of the paper. A variant checkpoint only behaves
as scored when used with its own config,
`configs/torchvision/detector_input_512/<model>.yaml`, which its
`MODEL_CARD.md` names.

## Licensing

The checkpoints do not share one licence, which is why they are not distributed
as a single archive:

- checkpoints fine-tuned from torchvision COCO_V1 weights (Faster R-CNN, SSD,
  RetinaNet, FCOS, SSDLite) carry Apache-2.0 for the authors' contribution and
  retain the upstream BSD-3-Clause notice and MS COCO provenance;
- checkpoints fine-tuned from Ultralytics weights (YOLOv8, YOLO11, YOLO12,
  YOLO26, RT-DETR) are `AGPL-3.0-only`. Each archive's `THIRD_PARTY_NOTICES.md` names the
  corresponding source: this repository at the commit the run recorded, and the
  upstream Ultralytics source.

The training data is the external cabbage dataset, DOI `10.17632/5cp2dyjczk.2`,
CC BY 4.0. It is not redistributed here and must be obtained from its official
record. None of these terms relicense one another.

## Using a checkpoint

Each archive unpacks to `<run>/augmented/best.pt`. From a checkout of the
repository, predict one image with the config its `MODEL_CARD.md` names:

```powershell
python scripts/predict.py `
  --config configs/torchvision/fcos.yaml `
  --checkpoint fcos/augmented/best.pt `
  --source path/to/image.png `
  --output predictions.json
```

Use `configs/ultralytics/<model>.yaml` for YOLO and RT-DETR checkpoints and
`configs/torchvision/detector_input_512/<model>.yaml` for a `-detector512`
checkpoint.
Ultralytics boxes are in original image pixels; torchvision boxes are in the
resized `training.image_size` space of the config.

## Verifying a download

Each archive's SHA-256 is in `SHA256SUMS`, and the digest of the `best.pt`
inside it is in `release-manifest.json` and in the run's tracked
`metadata.json`. A download can therefore be checked against evidence that lives
in Git history rather than against the release itself.

## Provenance

Released as tag `reproduced-checkpoints-v1.0.0`, titled `Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants`. The tag names the artifact kind rather
than the repository version, so a source release can be tagged independently.

These archives are built by the repository's own scripts from a manifest
generated out of the promoted runs, never assembled by hand. The build refuses a
run that is not `reproduced_verified` and re-hashes every checkpoint against the
digest its run recorded. The procedure is in the repository's
[`docs/releases/publishing.md`](https://github.com/panyapat-wongdee/cabbage-detection-counting/blob/main/docs/releases/publishing.md).
