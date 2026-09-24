# Weight provenance and release policy

This directory contains metadata only. Git carries no model binaries:
pretrained weights are never redistributed, and the fine-tuned checkpoints are
published only as assets of the `reproduced-checkpoints-v1.0.0` release.

`provenance.yaml` is the machine-readable registry for every repository model
key. The nine models marked `study_scope: paper_model` are the models in scope
for a fresh reproduction; `ssdlite` is marked `supplementary_unreported` because
it was trained and compared but omitted from the paper; `yolo12n`, `yolo12m`,
`yolo26n`, and `yolo26m` are marked `repository_extension` because they were
added after the study; all are released in the same tag as the others. These labels identify architecture scope.

The registry separates the following concepts:

- the framework source license;
- the terms and notices applicable to the upstream pretrained artifact;
- the training-data provenance and attribution;
- the license for the authors' releasable fine-tuned contribution; and
- architecture and paper citations.

The five torchvision records use Apache-2.0 for the authors' releasable
contribution, while retaining the torchvision BSD-3-Clause notice and COCO
provenance. The nine Ultralytics records use `AGPL-3.0-only` for fine-tuned
outputs under Ultralytics' default terms. Apache-2.0 does not override terms
that apply to an Ultralytics-derived model or combined application.

## External release sidecar contract

Every actual external checkpoint release must include a versioned metadata
sidecar next to the checkpoint. A registry record moves from
`release_status: not_released` to `released` only when its checkpoint is an
asset of a published release, named by `release_tag`; never invent sidecar
values.

| Field | Required value and validation |
|---|---|
| `model_name` | One of the exact registry keys |
| `base_identifier` | Exact identifier copied from the selected registry record |
| `base_resolved_url` | Absolute HTTPS URL captured by the approved run, not only a short filename |
| `base_sha256` | SHA-256 of the actual pretrained input, exactly 64 lowercase hexadecimal characters |
| `fine_tuned_sha256` | SHA-256 of the released checkpoint, exactly 64 lowercase hexadecimal characters |
| `framework_version` | Exact runtime version captured by the run |
| `config_sha256` | SHA-256 of the resolved training config, exactly 64 lowercase hexadecimal characters |
| `git_commit` | Full 40-character Git commit used for training |
| `dataset_doi` | Exactly `10.17632/5cp2dyjczk.2` |
| `license` | `Apache-2.0` for a torchvision record or `AGPL-3.0-only` for an Ultralytics record |

Do not use the local ignored `yolo26n.pt` as provenance for this study; it is
not one of the registry entries.

## Reproduced checkpoint release

Published as tag `reproduced-checkpoints-v1.0.0`, titled `Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants`. See
[`docs/releases/reproduced-checkpoints-v1.0.0.md`](../docs/releases/reproduced-checkpoints-v1.0.0.md).

The release manifest is generated from the promoted runs by
`scripts/build_release_manifest.py`, not written by hand, so it cannot state a
digest the runs do not already record. Every record must be a newly trained
`reproduced_verified` run and must include only its `best` file. The release
sidecar records the exact run metadata, configuration digest, repository
commit, pretrained source and digest, and dataset DOI.

Generated release archives also include the portable notices in
[`weights/notices/`](notices/), and torchvision archives the BSD-3-Clause text
from [`LICENSES/`](../LICENSES/), all under each archive's `notices/`. Dataset files and upstream pretrained assets
remain external.
