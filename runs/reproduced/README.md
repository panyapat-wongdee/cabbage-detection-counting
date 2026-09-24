# Reproduced training runs

This directory is tracked in Git for the evidence a `reproduced_verified`
claim rests on: each run's `metadata.json`, `config.yaml`, `logs/`, `plots/`,
and `evaluation/` promotion documents. The heavy outputs stay local and are
ignored: `checkpoints/` (published as release assets instead), Ultralytics'
native `framework/` directory, and staged prediction dumps.

This directory is reserved for **training** artifacts produced by an actual
run of the revised repository implementation, for both torchvision and
Ultralytics models. Each run must contain immutable `metadata.json`, the
resolved configuration, command, code revision and dirty state, environment
details, dataset/split and pretrained-weight provenance, and selected
best-checkpoint metadata.

Run directories begin as `smoke` or `reproduction_candidate`. Only a complete
run that passes the evidence validator may use `result_status:
reproduced_verified`. A configuration load, one-image prediction, or wiring
smoke is not a reproduced result.

A completed run groups `config.yaml`, `metadata.json`, `checkpoints/`,
`logs/`, and `plots/`. Ultralytics runs additionally keep the native
trainer's own output under `framework/`.

Evaluation outputs (detection/counting metrics, predictions, plots) are
written separately under [`results/reproduced/`](../../results/reproduced/README.md),
not here.

The associated publication is available through the DOI in
[`docs/publication-reference.md`](../../docs/publication-reference.md). This
directory intentionally contains no copy of its metrics or tables.

Runs are named after their model (`fcos`) or, for a controlled variant, after
the model and variant (`fcos-detector512`). SSDLite (MobileNetV3-Large) is a
primary repository model with the same run, evidence, and release treatment as
the nine paper models. Its `study_scope` stays
`supplementary_unreported` because it is not one of the architectures in the
associated publication.
