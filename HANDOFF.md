# Agent handoff

You are in [EricBoittier/karml](https://github.com/EricBoittier/karml), not
`EricBoittier/mmml`. History starts at `87a7c51`. Do not import the mmml
history, and do not push this repo back onto `mmml`.

`devtools/AGENTS.md` and `devtools/CLAUDE.md` still describe the old tree.
This file wins where they disagree. In particular, do not add tutorial pages
or an `examples/` tree as part of this cut.

## Already done

`87a7c51` is a single snapshot, about 133 MB, 3,037 files. Left behind on
purpose:

- `examples/`
- the extracted `setup/charmm/` tree (`setup/charmm.tar.xz` stayed; `setup/install.sh` unpacks it)
- `docs/images/`, `docs/plot-style-gallery-assets/`, `docs/robustness-report-assets/`
- regenerable checkpoints, including `pet-omol-s-v1.0.0.pt` and the PhysNet epoch JSON files
- `devtools/scratch/`

GitHub warned that `setup/charmm.tar.xz` is 75 MB. It is under the 100 MB
hard limit. Leave the tarball in git.

GitHub Actions on this repo will fail. MkDocs still points at removed
figures, and some tests still point at `examples/`. Do not patch those
references. Delete the tests and tutorial pages that exist only for them.

Do not run molecular dynamics, Packmol box builds, or `karml md-system` in
the agent session. CHARMM and GPU JAX are not available there. Verify with
unit tests.

## The cut to make

Two engines, one model code.

Training and inference call the same modules in `karml/models`. A checkpoint
written by training is what PyCHARMM MLpot and JAX-MD both evaluate. Outside
models such as PET load through the metatomic adapter inside the MLpot
callback. They do not get their own MD program.

### Keep

- `karml/models/` (PhysNet, DCMNet, efield, KerNN, MBD, hybrid MM). This is
  the only implementation of several of these models.
- Training commands that call those modules (`physnet-train` and the related
  trainers). Do not split training into another package.
- `karml/interfaces/pycharmmInterface/mlpot/` and `md-system` on PyCHARMM.
- JAX-MD: `karml/cli/run/jaxmd_runner.py`, `lambda_jaxmd.py`,
  `karml/cli/run/md_pbc_suite/jaxmd.py`, `karml/interfaces/jaxmdInterface/`.
- ORCA: `karml/interfaces/orca_external/` (server, client, and external wrapper).
- The metatomic loader used by MLpot (`metatomic_mlpot` and the calculator
  adapter). Keep `metatomic-pbc-md` off the CLI; that command is the
  CHARMM-free side program.
- `docs/handoff.md` and `karml/cli/run/md_handoff.py`. That is the
  PyCHARMM ↔ JAX-MD state handoff. Remove ASE from that doc. Do not delete it.
- Docstrings on the models and both engines.
- Tests of the models, MLpot, JAX-MD, and ORCA.
- `setup/charmm.tar.xz` and the small overlays in `setup/nbonds` and `setup/image`.

In `karml_calculator` (now `karml/interfaces/pycharmmInterface/karml_calculator.py`),
do not remove `stop_gradient` on the MIC lattice shift that wraps monomer B,
and do not replace that wrap with smooth MIC.

### Remove

- ASE as a third MD engine: `karml/cli/run/ase_runner.py`,
  `ase_minimize_log.py`, `karml/cli/run/md_pbc_suite/ase.py`,
  `karml/interfaces/aseInterface/`.
- `karml/interfaces/openmmInterface/`, `blenderInterface/`, `openfeInterface/`.
- GUI: `karml/gui/`, `karml/cli/gui.py`.
- MCP: `karml/mcp/` and the `karml-mcp` entry point.
- The `metatomic-pbc-md` command (`karml/cli/misc/metatomic_pbc_md.py` and its
  registry entry).
- `docs/cli/` (generated command reference, 75 pages).
- Campaign notes, not the engine handoff: `docs/HANDOFF-npt-des-hybrid.md`,
  `docs/des-lj-scales-handoff.md`, `docs/md-cg-unification-handoff.md`,
  `docs/q0-charge-retraining-handoff.md`, `docs/npt-campaign-status-2026-08-02.md`,
  `docs/npt-state-2026-08-02.md`, `docs/npt-density-dh-dg.md`,
  `docs/npt-jaxmd-charmm-comparison.md`, `docs/q0-charge-aware-water-validation-report.md`,
  `docs/q0-distillation-findings.md`, `docs/simulation-robustness-report.md`.
- Tutorial markdown that only walks through removed figures or `examples/`.
  Do not rewrite those pages in this cut. Docstrings stay.
- Test files whose only job is to open `examples/`. There were 43 of these
  in the parent tree. Leave the rest of `tests/unit` alone unless a test
  fails because it imports a module you removed.

After deletions, drop the matching entries from `karml/cli/registry.py`,
`pyproject.toml` scripts and package data, and MkDocs nav. A green unit-test
run of what remains is the check. Do not try to make the old docs site build.

## Docs to write later

Not this task. When tutorials come back, write a few pages only:

1. Install, including the CHARMM tarball.
2. Train one in-repo model.
3. Run `md-system` with that checkpoint on PyCHARMM and on JAX-MD.
4. Point ORCA at the server.
5. Load a metatomic checkpoint into the MLpot callback.

## Do not

- Delete or split `karml/models`.
- Remove JAX-MD or ORCA.
- Reintroduce `examples/`, figure galleries, extracted `setup/charmm/`, or
  checkpoints.
- Run MD or Packmol here.
- Force-push `main`. The snapshot commit stays.
