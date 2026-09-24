# AGENTS.md

## Purpose

This repository contains the source code and reproducibility materials for the research project:

**A Comparative Study of Deep Learning Models for Cabbage Detection and Counting in Drone Imagery**

The repository is intended to be:

1. a faithful, reproducible companion to the published paper;
2. a clean public GitHub repository;
3. a portfolio-quality project that can be linked from a resume or professional profile.

When working in this repository, prioritize **research fidelity, reproducibility, clarity, and maintainability** over clever abstractions or unnecessary framework complexity.

---

## Repository Context

`AGENTS.md` is the root-level instruction file for Codex and other coding agents.

Project/reference documents are stored under `docs/`.

Before making changes that affect experiments, metrics, model behavior, dataset handling, or reported results, read the relevant files in this order:

1. `docs/reproduction-protocol.md`
2. `docs/publication-reference.md`
3. existing source code, experiment scripts, configs, logs, and environment files

Private paper and dataset-paper files may be consulted outside the tracked
public tree when rights review requires them. Never copy their text, tables,
figures, PDFs, or generated excerpts into this repository.

Use `docs/reproduction-protocol.md` as the concise public working reference.
Treat the revised executable source/configuration and verified run metadata as
authoritative for repository behavior. The publication is a citation-only
external reference, not a public repository content source.

Do not assume that information absent from these sources is known.

---

## Source-of-Truth Priority

When sources disagree, use the following priority:

1. **Published paper** for the methodology and results that were actually reported.
2. **Recovered research provenance and current source/configuration** for implementation details not fully described in the paper.
3. **Dataset paper** for dataset provenance, structure, annotation format, and attribution.
4. **`docs/reproduction-protocol.md` and `docs/publication-reference.md`** as curated repository references.
5. New implementation choices made for this public repository.

If two authoritative sources conflict:

- do not silently choose one;
- document the discrepancy;
- preserve the behavior needed for reproducibility where practical;
- clearly distinguish the published description from the released implementation.

Never fabricate a missing experimental detail.

---

## Important Research Facts

The study compares nine object detection models:

- Faster R-CNN (ResNet50-FPN)
- SSD (VGG16)
- RetinaNet (ResNet50-FPN)
- FCOS (ResNet50-FPN)
- YOLOv8n
- YOLOv8m
- YOLO11n
- YOLO11m
- RT-DETR-L

The public cabbage UAV dataset is external to this repository.

The repository must not imply that the dataset was created by the authors of this code repository.

For exact dataset details, training parameters, evaluation definitions, and published results, consult the current docs/configs and the external paper records instead of duplicating or guessing values here.

---

## Published Results vs Reproduced Results

Always distinguish between:

### Published results

Values reported in the paper.

### Reproduced results

Values obtained by actually running code from this repository.

Never label copied paper numbers as reproduced results.

Never overwrite published values with reproduction results.

Prefer a structure such as:

```text
runs/
└── reproduced/        # training run evidence (checkpoints stay local)
results/
└── reproduced/         # evaluation outputs (metrics, predictions, plots)
```

This repository is citation-only: published values are cited by DOI, never
copied into a `results/published/` tree.

If reproduction differs from the paper, preserve both and investigate the reason.

Do not modify code solely to force reproduced metrics to match published numbers.

---

## Repository Layout

Prefer conventional locations for public repository files.

A reasonable target structure is:

```text
.
├── AGENTS.md
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml / requirements.txt
├── configs/
├── src/
│   └── cabbage_detection/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── metrics/
│       └── visualization/
├── scripts/
├── tests/
├── runs/
│   └── reproduced/
├── results/
│   └── reproduced/
└── docs/
    ├── reproduction-protocol.md
    ├── publication-reference.md
    └── ...           # no paper or dataset-paper PDFs
```

This is a target, not a mandatory historical structure.

Do not reorganize working code merely to match this tree.

Standard repository files such as `README.md`, `LICENSE`, `CITATION.cff`, dependency files, configs, and source code should remain in their conventional locations. The `docs/` rule applies to project/reference documentation, design notes, and related supporting material; paper and dataset-paper PDFs are never committed.

---

## Historical Research Provenance

The private historical archive is not part of the public repository. Use the
sanitized dependency snapshot under `docs/environment/` and the current
source/configuration as the public evidence boundary. Do not copy private
notebooks, local paths, generated outputs, or historical checkpoints into this
tree.

---

## Implementation Principles

Prefer simple, explicit Python.

Use:

- `pathlib` instead of machine-specific hard-coded paths;
- type hints where they improve clarity;
- docstrings for public functions/classes;
- small reusable functions;
- configuration-driven experiments;
- explicit device selection;
- structured logging for long-running tasks;
- deterministic/reproducible options where practical;
- clear separation between data, models, training, evaluation, metrics, and visualization.

Avoid:

- unnecessary abstraction;
- duplicated training/evaluation logic;
- hidden global state;
- absolute local paths;
- silent fallback behavior;
- undocumented constants;
- excessive notebook-only logic;
- framework wrappers that obscure important differences between torchvision and Ultralytics.

Keep commands runnable from the repository root whenever practical.

---

## Framework Boundaries

The project uses more than one model ecosystem.

### torchvision family

Typically includes:

- Faster R-CNN
- SSD
- RetinaNet
- FCOS

### Ultralytics family

Typically includes:

- YOLOv8
- YOLO11
- RT-DETR

Shared experiment semantics should be consistent where possible, but do not pretend framework APIs are identical.

If a framework performs confidence filtering, NMS, resizing, augmentation, or metric computation internally, verify whether equivalent repository logic would duplicate that operation.

Do not introduce a unified abstraction if it changes model behavior or makes evaluation harder to audit.

---

## Configuration

Prefer configuration files for experiment parameters rather than duplicated constants across scripts.

Important settings should be visible and inspectable, including where applicable:

- model name;
- backbone or model size;
- input image size;
- optimizer;
- learning rate;
- momentum;
- weight decay;
- batch size;
- epochs;
- augmentation settings;
- confidence threshold;
- IoU/NMS threshold;
- dataset paths;
- split definitions;
- random seed;
- checkpoint path;
- output directory.

If a value was not specified in the paper and cannot be recovered from original code, mark it as unknown/TODO instead of inventing a historical value.

New repository defaults are allowed, but they must be identified as new defaults rather than published settings.

---

## Dataset Rules

The dataset is externally hosted and should normally not be committed to this repository.

Provide:

- citation;
- official dataset/DOI information;
- download instructions;
- expected directory structure;
- preparation/conversion scripts when needed.

Do not commit:

- the full image dataset;
- generated copies of the full dataset;
- large derived archives.

Keep dataset provenance intact.

If annotations are converted to a new format, document the transformation.

If a cleaned or corrected annotation set is ever created, keep it separate from the benchmark dataset used to reproduce the paper.

Do not silently correct known annotation issues when reproducing the published experiment.

---

## Dataset Splits

Exact train/validation/test membership is scientifically important.

If original split files exist, preserve them.

Do not regenerate a random split and call it the published split.

If the original split cannot be recovered:

1. state that clearly;
2. create a reproducible new split using an explicit seed;
3. label it as a repository/reproduction split, not the original published split.

Whenever practical, keep split definitions in version-controlled text/JSON/CSV files.

---

## Training

Do not change training semantics merely to make the code cleaner.

Before changing:

- optimizer behavior;
- learning-rate schedule;
- image preprocessing;
- augmentation;
- pretrained weights;
- batch behavior;
- epoch count;
- checkpoint selection;
- image resolution;

verify the paper and original implementation.

Model-specific differences are acceptable when they reflect the original frameworks.

If a common trainer would hide or alter those differences, prefer separate adapters or explicit model-specific paths.

---

## Evaluation

Detection evaluation and cabbage-counting evaluation are separate concerns.

Keep them separate in code and reporting.

Detection metrics include:

- mAP@50;
- mAP@50:95.

Counting evaluation includes concepts such as:

- actual count;
- predicted count;
- TP;
- FP;
- FN;
- count error;
- False Discovery Rate (FDR);
- False Negative Rate (FNR);
- F1-score.

Metric implementations must be auditable.

Pay special attention to:

- confidence filtering;
- IoU threshold;
- prediction-to-ground-truth matching;
- duplicate matches;
- NMS;
- framework-specific post-processing;
- handling of empty predictions/targets.

Do not silently rely on incompatible metric definitions across frameworks.

---

## Tests

Add tests for logic that can change scientific results.

Prioritize tests for:

- bounding-box conversion;
- annotation parsing;
- dataset split loading;
- IoU calculations;
- TP/FP/FN matching;
- count error;
- FDR;
- FNR;
- F1-score;
- confidence filtering;
- NMS-related repository logic;
- config parsing.

Use small synthetic examples for metric tests so expected outcomes are obvious.

Avoid tests that require downloading the full dataset or large pretrained weights unless explicitly marked as integration tests.

---

## Verification Before Claiming Success

Before saying a task is complete:

1. run the most relevant tests;
2. run formatting/lint/type checks if configured;
3. verify changed CLI commands with `--help` or a small smoke test where practical;
4. inspect changed configs and paths for machine-specific values;
5. confirm no large datasets, checkpoints, credentials, or temporary files were added accidentally.

When a full training run is impractical, say so explicitly.

Do not claim that training or reproduction succeeded unless it was actually executed.

---

## Reproducibility

Prefer reproducible behavior where possible.

Record or expose:

- random seeds;
- software versions;
- CUDA/device information;
- model checkpoint provenance;
- config used for each run;
- output directory;
- metrics artifact;
- command used to launch the experiment.

Do not falsely imply bit-for-bit determinism where CUDA/framework operations prevent it.

When adding experiment outputs, make it possible to identify which config/code version produced them.

---

## Dependencies

Avoid unnecessary dependencies.

When adding one:

- explain why it is needed;
- prefer established packages;
- use a version constraint when compatibility matters;
- avoid introducing two libraries that solve the same narrow problem without a strong reason.

Do not arbitrarily upgrade PyTorch, torchvision, Ultralytics, Albumentations, CUDA-related packages, or metric libraries when reproducibility could be affected.

If modernizing the environment, document the modernization separately from the historical environment.

---

## Public Repository Quality

This repository will be visible to researchers, engineers, and recruiters.

Code and documentation should therefore be:

- professional;
- concise;
- easy to navigate;
- technically accurate;
- free of unnecessary generated clutter.

The README should eventually make it easy to understand:

- the research problem;
- the compared models;
- the dataset;
- the main published result;
- installation;
- dataset preparation;
- training;
- evaluation;
- inference;
- result reproduction;
- citation.

Do not turn the README into a copy of the paper.

Use tables, commands, figures, and short explanations where they communicate the project more effectively.

---

## Documentation

Project-context documents belong under `docs/`, except for `AGENTS.md`.

When implementation behavior changes, update the relevant documentation in the same change when practical.

Do not duplicate large blocks of experimental facts across many files.

Prefer:

- `docs/reproduction-protocol.md` for canonical implementation behavior;
- `docs/publication-reference.md` for citation-only publication context;
- README for user-facing quick start and project overview;
- focused documents under `docs/` for deeper explanations.

If a generated document contradicts the paper, fix the document rather than silently redefining the experiment.

---

## Citation and Attribution

Preserve proper attribution for:

- the published research paper;
- the external cabbage dataset;
- third-party libraries;
- pretrained model sources.

Do not imply ownership of external data or third-party models.

Keep `CITATION.cff` consistent with the published paper.

Do not assume the repository code, paper, dataset, and pretrained weights share the same license.

---

## Git Hygiene

Never commit:

- credentials;
- tokens;
- API keys;
- private paths;
- `.env` secrets;
- local caches;
- notebook checkpoints;
- temporary training artifacts;
- large dataset files;
- large model checkpoints unless intentionally released.

Maintain a useful `.gitignore`.

Prefer small, coherent changes.

Do not mix unrelated refactors with scientific behavior changes.

When modifying experiment logic, make the change easy to review in isolation.

---

## Codex Working Procedure

For non-trivial tasks:

1. inspect the relevant existing files first;
2. read `docs/reproduction-protocol.md` when research behavior is involved;
3. inspect the external paper record when required; private historical inputs are not public source files;
4. state important assumptions;
5. make the smallest change that solves the task;
6. preserve existing behavior unless the task explicitly changes it;
7. add or update tests for result-sensitive logic;
8. run relevant verification;
9. summarize what changed and what was not verified.

Do not rewrite large portions of the repository without first understanding how they are used.

When asked to refactor, separate behavior-preserving refactors from deliberate methodology changes.

---

## Do Not Guess

Do not invent historical values for items such as:

- exact Python version;
- exact PyTorch/torchvision version;
- exact Ultralytics version;
- exact Albumentations version;
- exact CUDA/cuDNN versions;
- exact random seed;
- exact split filenames;
- exact checkpoint-selection policy;
- exact worker count;
- undocumented scheduler settings;
- undocumented model overrides;
- training runtime.

Search the repository and project documents first.

If the information cannot be recovered, report it as unknown.

---

## Do Not Overclaim

Do not claim:

- exact reproduction without executing the experiment;
- real-time performance unless measured;
- deployment readiness unless demonstrated;
- production usage unless documented;
- ownership of the external dataset;
- improvements over the paper without controlled evidence.

Use precise language such as:

- "reported in the paper";
- "implemented in this repository";
- "reproduced in this run";
- "not yet verified";
- "repository default, not a published setting".

---

## Definition of Done

A change is complete when, as applicable:

- the requested behavior is implemented;
- research semantics remain faithful or deviations are documented;
- relevant tests pass;
- commands/configs are usable from the repository root;
- no machine-specific paths or secrets were introduced;
- documentation is updated where needed;
- published and reproduced results remain clearly separated;
- verification actually performed is reported accurately.

For changes that affect scientific results, "code compiles" is not sufficient.

The result must also be understandable, reviewable, and reproducible.
