# Building and publishing a checkpoint release

Maintainer procedure. Downloaders need only the release body,
[`reproduced-checkpoints-v1.0.0.md`](reproduced-checkpoints-v1.0.0.md).

## Build

The manifest is generated from the promoted runs rather than written by hand,
so it cannot state a digest the runs do not already record:

```powershell
python scripts/build_release_manifest.py `
  --output weights/releases/reproduced-checkpoints-v1.0.0.json

python scripts/prepare_checkpoint_release.py `
  --manifest weights/releases/reproduced-checkpoints-v1.0.0.json `
  --output-dir dist/releases/reproduced-checkpoints-v1.0.0 `
  --release-notes docs/releases/reproduced-checkpoints-v1.0.0.md
```

Generation refuses a run that is not `reproduced_verified` and re-hashes each
checkpoint against its run metadata. `prepare_checkpoint_release.py` names the
current `HEAD` (or `--source-commit`) as the AGPL corresponding source in every
Ultralytics archive, so build from a commit that is already pushed. The build writes one archive per run,
with `SHA256SUMS` and the manifest beside them, and audits the result. `dist/`
is not tracked. What each archive holds, and why the checkpoints ship one
archive per model rather than as one bundle, is described in the release body.

## Publish

Upload is a manual step, so that publishing stays a deliberate act. The release
is tagged `reproduced-checkpoints-v1.0.0` and titled `Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants`; the tag names the artifact kind rather
than the repository version, so a later source release can be tagged separately.
Push first, or the AGPL corresponding-source notice points at a commit GitHub
cannot serve:

```powershell
git push origin main

gh release create reproduced-checkpoints-v1.0.0 `
  (Get-ChildItem dist/releases/reproduced-checkpoints-v1.0.0/*.zip).FullName `
  dist/releases/reproduced-checkpoints-v1.0.0/SHA256SUMS `
  dist/releases/reproduced-checkpoints-v1.0.0/release-manifest.json `
  --title "Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants" `
  --notes-file dist/releases/reproduced-checkpoints-v1.0.0/RELEASE_NOTES.md
```

`Get-ChildItem` is doing real work here. PowerShell does not expand wildcards
for a native command, so passing `*.zip` directly hands `gh` one literal string
and the upload fails; expanding it first passes every archive as its own
argument. `RELEASE_NOTES.md` is deliberately not uploaded as an asset, because
`--notes-file` already puts its text in the release body.

From a POSIX shell the glob expands on its own:

```bash
gh release create reproduced-checkpoints-v1.0.0 \
  dist/releases/reproduced-checkpoints-v1.0.0/*.zip \
  dist/releases/reproduced-checkpoints-v1.0.0/SHA256SUMS \
  dist/releases/reproduced-checkpoints-v1.0.0/release-manifest.json \
  --title "Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants" \
  --notes-file dist/releases/reproduced-checkpoints-v1.0.0/RELEASE_NOTES.md
```

`gh` itself must be on `PATH`. The Windows installer adds it machine-wide, but a
terminal opened before the install keeps its old copy of `PATH`, so `gh` reads
as an unknown command until the terminal is reopened.

## Updating the release body

When the notes document changes after publication, update the body in place
without touching the tag or the assets:

```powershell
gh release edit reproduced-checkpoints-v1.0.0 `
  --notes-file docs/releases/reproduced-checkpoints-v1.0.0.md
```

## Adding runs to the published release

The release is one tag holding every released run. After new runs are
promoted, rebuild the manifest and the archives with the commands above,
commit and push, move the tag to the commit that holds the new manifest, and
replace the assets in place:

```powershell
git tag -f reproduced-checkpoints-v1.0.0
git push --force origin refs/tags/reproduced-checkpoints-v1.0.0
```

Moving a published tag rewrites what it names, so do it only before anyone
depends on the old target, as was the case for v1.0.0 while the repository was
private. Then replace the assets and the release body:

```powershell
gh release upload reproduced-checkpoints-v1.0.0 `
  (Get-ChildItem dist/releases/reproduced-checkpoints-v1.0.0/*.zip).FullName `
  dist/releases/reproduced-checkpoints-v1.0.0/SHA256SUMS `
  dist/releases/reproduced-checkpoints-v1.0.0/release-manifest.json `
  --clobber
gh release edit reproduced-checkpoints-v1.0.0 `
  --title "Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants" `
  --notes-file docs/releases/reproduced-checkpoints-v1.0.0.md
```
