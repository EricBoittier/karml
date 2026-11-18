# Releasing (development alpha)

KARML versions are derived from git tags by
[`versioningit`](https://versioningit.readthedocs.io/) (see `[tool.versioningit]` in
`pyproject.toml`). A clean, tagged commit produces an exact version; anything after a
tag gets a dev/local suffix.

## Fast install (what end users get)

The alpha is distributed as a Python package, not a git clone:

```bash
pip install karml            # or: uv pip install karml
```

The wheel/sdist are ~11 MB (they exclude large repo data such as `karml/models/EF/data/`,
trajectories, and `setup/charmm.tar.xz`). pyCHARMM and a GPU are optional — see the README
and [Getting started](getting-started.md).

## Cut an alpha release

1. Make sure `main` is green and the working tree is clean.
2. Choose a PEP 440 pre-release version and tag it (annotated):
   ```bash
   git tag -a v0.1.0a1 -m "karml 0.1.0a1 (alpha)"
   git push origin v0.1.0a1
   ```
   `versioningit` turns `v0.1.0a1` into version `0.1.0a1`.
3. Build and check the artifacts:
   ```bash
   uv build                       # writes dist/*.whl and dist/*.tar.gz
   uv run --with twine twine check dist/*
   ```
4. (Optional) Publish to PyPI / TestPyPI:
   ```bash
   uv run --with twine twine upload --repository testpypi dist/*
   ```
5. Create the GitHub Release from the tag and attach `dist/*` (and, if wanted,
   `setup/charmm.tar.xz` as a release asset rather than tracking it in git).

## Keeping the git repo small

`git clone` is large because of binaries committed to history. To reduce it:

- Users (no rewrite needed): `git clone --filter=blob:none <url>` or `git clone --depth 1 <url>`.
- Maintainers (destructive, rewrites history, requires force-push and everyone re-clones):
  run `scripts/slim_repo_history.sh --analyze` then `--rewrite`. Do this only when active
  branches/PRs can be coordinated, and reclaim Git LFS storage afterwards (GitHub does not
  garbage-collect LFS objects automatically).

Do **not** commit new large/regenerable artifacts; `.gitignore` already covers
`node_modules/`, `*.dcd`, `*.traj`, `*.parquet`, `karml/models/EF/data/`, etc.

## Citable release and Zenodo (first release)

1. Complete `CITATION.cff` (authors, ORCID iDs, title; add `version:` and `date-released:` for the tag). Without a
   `.zenodo.json`, Zenodo takes its metadata from `CITATION.cff`, so placeholders there become the Zenodo record.
2. Resolve the `[maintainer to confirm]` line in `THIRD_PARTY_NOTICES.md` (CHARMM redistribution statement). pyCHARMM (GPL-3.0-or-later) is not packaged: `pyproject.toml` builds `karml*` only, so the wheel/sdist are MIT-only; the pyCHARMM sources stay in the repository tree under their own licence.
3. Enable the repository at <https://zenodo.org/account/settings/github/> (toggle on) **before** publishing the GitHub
   release; Zenodo archives a release only after the toggle is on.
4. Tag and publish (see above). First tag: `v0.1.0` (decided; no PyPI upload for it). `versioningit` derives the version from the
   tag, so nothing in the tree is bumped. Use `v0.1.0a1` instead if the aim is a PyPI pre-release that plain
   `pip install karml` does not select.
5. The DOI exists only after the GitHub release has been archived; add it to `CITATION.cff` and the README badge in a
   follow-up commit (do not move the tag).

See also the [GPU cluster runbook](gpu-cluster-runbook.md).

The scripted version of these steps is `release.sh` (checks open PRs, CI on main, placeholders in `CITATION.cff`, tags,
pushes, creates the GitHub release, and prints the Zenodo DOI and the patches for `refs.bib`, `CITATION.cff` and the README badge).
