# Public publication boundary

This repository is an author-maintained reimplementation and reproduction
repository. Its public footprint for the associated IEEE publication is
citation-only. A repository license never relicenses the IEEE article, its
figures, tables, text, or other copyrighted components.

## Allowed publication references

The repository may include the paper title, authors, venue, year, DOI, a link to
the IEEE record, and a short independently written statement explaining the
association. It may also describe the repository's own code, configurations,
new run artifacts, and newly measured results.

The external dataset may be identified by its title, authors, DOI, download
link, and CC BY 4.0 attribution requirements. Dataset images and annotations
remain outside Git, apart from one attributed example figure
(`docs/figures/counting_example.jpg`) documented in
[`licensing-and-attribution.md`](licensing-and-attribution.md).

## Prohibited publication artifacts

Do not add an IEEE Version of Record, accepted manuscript, proof, screenshot,
extracted text, substantial excerpt, copied caption, copied figure, copied
table, reconstructed publication table, or transcribed publication metrics.
Keep dataset bytes and upstream pretrained assets outside this repository.

The terms `paper_model`, `supplementary_unreported`, and
`repository_extension` describe study scope;
they do not assert that any result in this repository was reported by IEEE, and
they say nothing about how fully a model is supported here. SSDLite is
`supplementary_unreported` because it was omitted from the paper, while being a
primary repository model in every other respect. YOLO12 and YOLO26 are
`repository_extension` because this repository added them after the study.
The label is never rewritten to match a later decision about what this
repository reports.

## Result status

- `smoke`: wiring or one-batch validation only;
- `reproduction_candidate`: a new run awaiting evidence review;
- `reproduced_verified`: a new run with complete immutable provenance,
  predictions, metrics, and checkpoint verification.

Only `reproduced_verified` runs may be presented as reproduced results or supply
best checkpoints to an external release. The repository does not promise that
new metrics will equal any metric in the associated publication.

## Manual review requirement

Automated checks can reject known paths, extensions, and publication-artifact identifiers but
cannot decide whether prose or artwork is too close to a copyrighted source.
Before a public release, an independent reviewer must inspect the complete
tracked tree and confirm that all source code, documentation, figures, and
weights are independently releasable. The exact IEEE agreement and any
coauthor, university, employer, sponsor, framework, dataset, and model-weight
terms must also be reviewed. This document is not legal advice.
