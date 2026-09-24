# Contributing

Install the lightweight development profile from the repository root:

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

Install `.[torchvision,dev]` or `.[ultralytics,dev]` when working on a
framework adapter. The CUDA-pinned `requirements.txt` is reserved for actual
research reproduction environments.

Changes that affect data handling, model behavior, metrics, configuration, or
reported results must preserve the research provenance rules in
`docs/reproduction-protocol.md`. Add focused tests for result-sensitive logic,
record the config and evidence contract, and run the relevant audits before
opening a pull request.

Do not commit datasets, pretrained weights, fine-tuned checkpoints, private
research inputs, or credentials. Training runs belong under
`runs/reproduced/<run>/` and evaluation outputs under
`results/reproduced/<run>/`; `.gitignore` keeps their heavy parts
(`checkpoints/`, `framework/`, `records/`) out of Git while the small evidence
files are tracked on purpose. `scripts/audit_release_licenses.py` and the
public-hygiene workflow reject anything that slips through.
