# Public release checklist

- [x] Confirm Apache-2.0 for the revised repository source; keep external dataset, paper, dependency, and model-output terms separate.
- [x] Keep this repository citation-only for publication content; reserve `runs/reproduced/` for training runs and `results/reproduced/` for evaluation outputs.
- [x] Add offline CPU tests for configuration, split/data preparation, counting, detection mAP, artifact lifecycle, and comparison reports.
- [x] Execute and archive a run for all fourteen repository models and the five `-detector512` variants, and score every split.
- [x] Promote every run to `reproduced_verified`; each promotion re-hashed its checkpoints, predictions, and metrics and recorded dataset, base-weight, and evaluation evidence.
- [x] Keep the split manifest byte-identical across platforms so its recorded digest stays reproducible (`.gitattributes`, `eol=lf`).
- [x] Pin the torchvision/Ultralytics execution environment: `requirements.txt` holds the exact CUDA lock (torch 2.5.1+cu118, torchvision 0.20.1+cu118, Ultralytics 8.4.9, Albumentations 1.4.21), and every run's `metadata.json` records those same versions. `pyproject.toml` extras stay ranged for contributors.
- [x] Track provenance notes only after the sensitive-path scan is clean; run metadata records repository-relative input paths, never an operator's home directory.
- [x] Do not add paper PDFs or dataset-paper PDFs; publish title/DOI citations only.
- [x] Confirm dataset DOI, attribution, and redistribution terms against the Mendeley record and DataCite metadata (Version 2, CC BY 4.0; checked 2026-09-24; recorded in `docs/licensing-and-attribution.md`).
- [x] Keep the official download under ignored `original_dataset/` and track only `splits/fold1_recovered.csv`.
- [x] Validate the recovered 320/46/92 membership and 17,621 canonical annotations.
- [x] Generate `prepared/fold1_yolo/` from COCO labels with hard links and record the preparation report.
- [x] Audit generated labels against the historical YOLO export before removing local historical copies.
- [x] Obtain explicit approval before deleting `cabbages_YOLO/` or historical `data/fold1_*` directories.
- [x] Confirm pretrained and fine-tuned weight terms separately against the torchvision pre-trained model notice and the Ultralytics licence statement (checked 2026-09-24; recorded in `docs/licensing-and-attribution.md`).
- [x] Register newly reproduced best checkpoints with size and SHA-256 in a versioned release manifest, generated from the promoted runs by `scripts/build_release_manifest.py`.
- [ ] Build and audit per-run release archives under ignored `dist/releases/` (one per run, audit PASS).
- [x] Keep all pretrained weights outside Git and record exact official source URL and SHA-256 for each external fine-tuned release (`weights/provenance.yaml` URLs; base-weight digests in each run's `evaluation/pretrained.json` and the release manifest).
- [x] Include torchvision BSD-3-Clause notice and MS COCO provenance with every torchvision fine-tuned release.
- [x] Include AGPL-3.0-only and corresponding source with every Ultralytics fine-tuned release; the notice names this repository at the run's commit and the upstream Ultralytics source.
- [x] Include CC BY 4.0 dataset attribution and the data-record DOI in every archive's notices.
- [x] Keep SSDLite's `study_scope: supplementary_unreported` while reporting and releasing it like the other nine; never present it as a paper model or as a published comparison.
- [x] Run `pytest -q` in the current environment; repeat in a clean public snapshot before publication.
- [x] Inspect torchvision `config.yaml`, `logs/console.log`, `logs/training_log.csv`, `logs/best_epoch.json`, `checkpoints/best.pt`, `checkpoints/last.pt`, and plot files for every candidate run (all five present; 500 epochs logged; `best_epoch.json` equals the arg-max of `val_map50_95`; both checkpoint digests match the run metadata; resolved configs differ from `configs/torchvision/` only by filled-in defaults).
- [x] Run all public CLI commands with `--help` and exercise synthetic inputs through the test suite.
- [x] Run `git diff --check`.
- [x] Run the publication-boundary and tracked-file hygiene gates; repeat them on the final public snapshot.
- [x] Confirm publication citation and reproduction evidence are separate.
- [x] Confirm README does not claim unexecuted reproduction.
- [x] Track the reproduced metric reports, per-image rows, plots, and per-split provenance; keep `records/` local.
- [x] Generate `results/reproduced/model_comparison_test.{md,json}` with one AP backend and regenerate it whenever a run is re-scored.
- [x] Confirm the README reproduced table matches the generated comparison artifact digit for digit.
- [ ] Upload the built archives with tag `reproduced-checkpoints-v1.0.0` and title `Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants` (manual, deliberate step; procedure in `docs/releases/publishing.md`).
- [x] Inspect the final clean clone before pushing public (fresh clone from GitHub into a fresh virtual environment, CPU torch, `pytest -q`: 430 passed, 6 skipped for absent dataset/weights; working tree clean afterwards).
- [x] Re-run every experiment from a clean clone of the code commit with `scripts/reproduce_all.py`, and commit the evidence, tables, and figures as the results commit.
- [ ] Make the GitHub repository public; the release's AGPL corresponding-source notice needs it.
- [ ] After it is public: enable private vulnerability reporting (named in `SECURITY.md`) and add rulesets blocking force-push/deletion on `main` and updates/deletion of `reproduced-checkpoints-*` tags.

The release is one tag, `reproduced-checkpoints-v1.0.0`, titled
`Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants`, with one archive per run represented in the manifest: the
fourteen models and the five `-detector512` variants. Each archive contains only
verified `best` checkpoints plus sidecar, model card, license, and third-party
notices. SSDLite is included on the same terms as the other torchvision models
and keeps its `supplementary_unreported` study scope.
