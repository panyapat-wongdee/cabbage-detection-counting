# Research source provenance

This repository distinguishes historical research facts, recovered implementation details, and new repository behavior.

| Fact category | Authoritative source | Policy |
|---|---|---|
| Published methodology | **Published paper** (external IEEE record cited below) | Use the DOI as a citation and implement a fresh protocol; publication text and tables are not copied here. |
| Omitted implementation details | **private historical archive** | Preserve behavior only where it is represented by the current source/configuration; do not infer missing values. |
| Dataset origin, structure, and attribution | **Dataset paper** (external record cited below) | Treat images/annotations as external and follow their terms. |
| Framework-native behavior | Ultralytics/torchvision APIs | Record values supplied by a framework default as `framework_default`; do not present them as paper-specific settings. |
| Curated project context | `docs/reproduction-protocol.md` | This repository guide describes implementation behavior without reproducing publication content. |
| New package defaults and interfaces | Current `src/`, `configs/`, and tests | Identify as repository behavior, not historical fact. |

The IEEE paper record is DOI `10.1109/KST65016.2025.11003298`. The
dataset article record is DOI `10.1016/j.dib.2024.110699`; its downloadable
Mendeley data record is DOI `10.17632/5cp2dyjczk.2` and remains external under
CC BY 4.0.

The SSDLite notebook path is implementation evidence for a model that is
`supplementary_unreported` in study scope but otherwise follows the same
repository protocol as the paper-scope models.

The current workspace contains the official COCO bundle and the recovered
manifest (458 images: 320 train, 46 validation, 92 test; no COCO image has zero
annotations). The Pascal export referenced by the historical notebook is not
present, so COCO/Pascal input equivalence remains unverified. Use
`scripts/audit_historical_inputs.py` when that export is available.

## Recovered environment evidence

`docs/environment/recovered-research-requirements.txt` records the sanitized
direct environment requirements recovered from the private archive, including
PyTorch 2.5.1+cu118, torchvision 0.20.1+cu118, Ultralytics 8.4.9,
Albumentations 1.4.21, and pycocotools 2.0.8. The paper did not specify these
exact versions. `requirements.txt` curates the direct research/runtime
dependencies from this snapshot; it is not a second complete environment
export.

## Explicitly unknown until recovered

- exact historical random seed;
- exact historical data-loader worker count for every framework;
- exact historical library versions used for every run;
- whether every framework applied confidence filtering and NMS internally before export.

The released configs intentionally use seed `42` and two workers as repository
optimization settings. Ultralytics checkpoint selection is left to the
framework default and is labeled `framework_default`.

The primary augmented configs use **Ultralytics native augmentation** for the
Ultralytics family and Albumentations for torchvision. Matching parameter
magnitudes do not prove that the two preprocessing pipelines are identical.

An unknown value must remain `Unknown` or a null provenance field in configuration. A repository default may be added only when it is labeled as a new default and does not replace an externally cited publication claim.
