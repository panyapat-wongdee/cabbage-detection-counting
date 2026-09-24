# Dataset preparation

This project uses the external cabbage UAV dataset published by Yokoyama, Matsui, and Tanaka:

> *An instance segmentation dataset of cabbages over the whole growing season for UAV imagery*, Data in Brief, 2024. DOI: `10.1016/j.dib.2024.110699`.

The dataset is not created by this repository and its images/annotations are not distributed here, apart from one attributed example image adapted for the README figure (see [licensing and attribution](licensing-and-attribution.md)). Obtain **Version 2** from the [official Mendeley Data record](https://data.mendeley.com/datasets/5cp2dyjczk/2), review its terms, and keep the downloaded files outside Git-tracked source paths. The record DOI is `10.17632/5cp2dyjczk.2` and the dataset is released under **CC BY 4.0**.

Keep the publication and data-record citations separate: the article DOI is
`10.1016/j.dib.2024.110699`, while the downloadable Version 2 dataset record
is *An annotated image dataset of cabbages for instance segmentation*, DOI
`10.17632/5cp2dyjczk.2`. The repository does not redistribute either the
dataset bytes or the dataset-paper PDF.

The outer extracted download should remain unchanged under the canonical local directory `original_dataset/`. The recovered membership is tracked under `splits/fold1_recovered.csv`, and generated framework artifacts are written under the ignored `prepared/` directory.

The unmodified Version 2 hierarchy is discovered from the unique bundle that contains `annotation.json` and its sibling `images/` directory:

```text
original_dataset/
`-- An annotated image dataset of cabbages for instance segmentation/
    |-- annotation.json
    `-- images/
        `-- <cultivar>/<location>/<YYYYMM>/*.png
```

The COCO `images[].path` field supplies the nested relative path. The preparation code resolves it beneath `images/` and never flattens or rewrites the official source. `file_name` remains the basename and is supported as a compatibility fallback for older flat COCO fixtures.
The generated Ultralytics layout is:

```text
<prepared-root>/
|-- train/images/
|-- train/labels/
|-- val/images/
|-- val/labels/
`-- test/images/
    `-- test/labels/
```

Each image must have a label file with the same stem. The recovered repository split contains 320 training, 46 validation, and 92 test images. The split manifest records identifiers and provenance only; it does not contain image bytes.

For historical-method parity, compare the official COCO bundle with the
recovered Pascal export using `scripts/audit_historical_inputs.py`. This
read-only audit reports image digests, converted bounding boxes, split counts,
and empty-image counts. An unrun or mismatching audit prevents a claim that the
two input layouts are equivalent.

Preferred preparation after installation validates the official COCO source
used by torchvision and then creates only the ignored Ultralytics view:

```powershell
python scripts/prepare_dataset.py `
  --format all `
  --dataset-root original_dataset `
  --manifest splits/fold1_recovered.csv `
  --output prepared/fold1_yolo `
  --link-mode hardlink
```

The narrow operations remain available when needed. To validate only:

```powershell
python scripts/prepare_dataset.py `
  --format validate `
  --dataset-root original_dataset `
  --manifest splits/fold1_recovered.csv
```

To generate only the ignored Ultralytics view without duplicating image bytes,
use hard links (or explicitly choose `copy` when links are unavailable):

```powershell
python scripts/prepare_dataset.py `
  --dataset-root original_dataset `
  --manifest splits/fold1_recovered.csv `
  --format yolo `
  --output prepared/fold1_yolo `
  --link-mode hardlink
```

Framework-specific conversions must write to an explicit output directory and preserve the source dataset attribution. No command in this repository downloads data implicitly. The explicit `--source-root` plus `--annotations` input pair remains available for synthetic test fixtures.
