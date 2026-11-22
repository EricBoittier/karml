<h1 align="center">karml</h1>

[![CI](https://github.com/EricBoittier/karml/workflows/CI/badge.svg)](https://github.com/EricBoittier/karml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/EricBoittier/karml/branch/main/graph/badge.svg)](https://codecov.io/gh/EricBoittier/karml/branch/main)
[![Docs](https://readthedocs.org/projects/karml/badge/?version=latest)](https://karml.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

**karml** (pronounced Karamell) — machine-learned potentials in molecular-mechanics MD.

karml combines CHARMM/OpenMM workflows with JAX-based neural models for electrostatics and force prediction, for building and running hybrid ML/MM condensed-phase simulations.

## Status

KARML is in **alpha**. The core calculator, CLI, and `md-system` YAML workflow are usable today, but interfaces are still settling and may change without notice ahead of a first tagged release. Feedback and issues are welcome — see [Getting Help](#getting-help).

## Quick Installation

Requires **Python 3.13**. Prefer [`uv`](https://docs.astral.sh/uv/).

### Using `uv` (recommended)

```bash
git clone https://github.com/EricBoittier/karml.git
cd karml
uv sync

# Optional extras
uv sync --extra cli      # shell tab completion (argcomplete)
uv sync --extra md-cpu   # Vesin NL + MDAnalysis (CPU MD smokes)
uv sync --extra dev      # pytest, MkDocs, ruff
uv sync --extra metatomic  # TorchScript AtomisticModel ASE + CHARMM MLpot
make install-gpu         # JAX CUDA 13 + CuPy (GPU nodes; SM 7.5+)
# make install-gpu-cuda12  # older GPUs / CUDA 12
```

For PyCHARMM / Packmol (native libs, not installed by `uv`):

```bash
make install-native   # builds libcharmm + packmol under setup/charmm
make doctor           # karml doctor — env / CHARMM readiness
```

Or from a fresh clone: `make install-full` (`uv sync` + native build).
GPU cluster install, precision settings, job scripts and expected speed: [`docs/gpu-cluster-runbook.md`](docs/gpu-cluster-runbook.md).

### As a `uv` tool (CLI only, no clone needed)

Installs the `karml` command into its own isolated environment. Pick one GPU
extra; JAX's CUDA wheels bring their own CUDA/cuDNN, so only an NVIDIA driver
is needed on the machine.

```bash
uv tool install -p 3.13 "karml[gpu] @ git+https://github.com/EricBoittier/karml"         # CUDA 13 (SM 7.5+)
uv tool install -p 3.13 "karml[gpu-cuda12] @ git+https://github.com/EricBoittier/karml"  # CUDA 12 (older GPUs/drivers)
uv tool install -p 3.13 "karml @ git+https://github.com/EricBoittier/karml"              # CPU only
# from a local checkout instead: uv tool install -p 3.13 ".[gpu]"
```

The `git+` form clones the full repository history (about 2 minutes today).
Pass `-p 3.13` so uv does not try an incompatible default Python. Add
`,jax-pme-solver` to the extras for the jax-pme long-range backend. The CUDA 12
and CUDA 13 extras are mutually exclusive. PyCHARMM still needs the native
`libcharmm` (see above; point `CHARMM_LIB_DIR` at it); everything else works
without it.

### Jupyter kernel (required for the example notebooks)

Register the project venv as its own kernel **once**, and select it in the
notebook (`Kernel → Change Kernel → karml-venv`):

```bash
.venv/bin/python -m ipykernel install --user --name karml-venv --display-name "karml venv"
```

Without this, Jupyter's default `python3` kernel may start a different
interpreter: the kernelspec `uv` installs uses a bare `"python"` in its `argv`,
so it resolves against `PATH` and picks up an active conda environment instead of
`.venv`. The symptom is an immediate

```
TypeError: 'type' object is not subscriptable
```

on the first `import karml...`, because that interpreter is too old to parse
`tuple[float, ...]` annotations. It looks like broken code but is purely kernel
selection. Check with `import sys; print(sys.executable)` — it must point inside
`.venv`.

### Using Conda / micromamba

```bash
# Conda
conda env create -f setup/environment.yml
conda activate karml

# Or Makefile micromamba targets (preferred on clusters)
make micromamba-create
make micromamba-create-gpu        # CUDA 12 env file
make micromamba-create-gpu-cuda13
```

GPU env files: `setup/environment-gpu.yml`, `setup/environment-gpu-cuda13.yml`.

### Using Docker

Dockerfile and Compose live under [`devtools/docker/`](devtools/docker/):

```bash
cd devtools/docker
docker compose up -d karml-cpu
docker compose exec karml-cpu bash
# GPU: docker compose up -d karml-gpu
```

### CLI tab completion

```bash
uv sync --extra cli
eval "$(register-python-argcomplete karml)"
# or: eval "$(karml completion bash)"
```

## CLI quick start

```bash
karml -h                 # compact top-level help
karml commands           # all subcommands by category
karml examples           # copy-paste invocations
karml configure          # interactive YAML / Snakemake wizard
karml env                # checkpoints + CHARMM paths
karml md-system --help   # condensed-phase MD flags
karml doctor             # environment health check
```

## Paper weights

The checkpoints behind the ML/MM two-body liquids manuscript are in the private
Hugging Face model repo
[EricBoi/mlmm-twobody-weights](https://huggingface.co/EricBoi/mlmm-twobody-weights)
(MIT, same as karml). The figure repo pins them on branch `weights-manifest`
(`weights/MANIFEST.json`: epoch, EMA or raw weights, sha256, and which figure
or table uses each file). Master of that repo is the figure reproducer and
does not carry the manifest.

```bash
git clone https://github.com/EricBoittier/mlmm-twobody-figures
cd mlmm-twobody-figures
git checkout weights-manifest
hf auth login          # read access; the model repo stays private until publication
python scripts/fetch_weights.py --list
python scripts/fetch_weights.py                 # all models -> weights/hf/, sha256-checked
python scripts/fetch_weights.py --only dcm_student acetone_student
```

The checksum-verified upload was 35 models and 22.6 MB. The repo now has 36
models, 524 files, and 22.66 MB: the same set plus `dcmnet_n3_desp_dimer`
(dESP dimer fine-tune, fig:dcmnet c, 0.11 MB). `MANIFEST.json` on the Hub
matches `weights-manifest`, and a clean re-download matched every sha256.
Teachers and parameter files the paper cites but does not redistribute are
listed under `third_party_not_included`: PET-OMol-L, PET-MAD-xs, the released
QM9 DCMnet starts, and CHARMM/CGenFF.

Each folder has a portable JSON (`entry_points` in the manifest) plus, where
the training run saved one, an Orbax `epoch-*` directory. Point karml at the
JSON:

```bash
karml physnet-evaluate \
  --checkpoint weights/hf/dcm_student/dcm_omol_l_A_v2_best.json \
  --data path/to/valid.npz
karml md-system --backend pycharmm --ml-potential-mode physnet \
  --checkpoint weights/hf/acetone_student/aco_omol_l_A_v2_best.json \
  --residue ACO --n-molecules 2 --setup pycharmm_minimize
```

| task | checkpoint |
|---|---|
| DCM liquid / static ML/MM | `dcm_student/dcm_omol_l_A_v2_best.json` |
| Acetone liquid, IR, static ML/MM | `acetone_student/aco_omol_l_A_v2_best.json` |
| Water/methanol production | `watmeoh_mdaug_ft_ep59/wm_omol_l_A_cap1_mdaug_ft_ep59.json` |
| Water/methanol/octanol K_ow | `wmo_ft3_al5g3/wmo_ft3_al5g3.json` |

DCMnet charge models (`dcmnet_n3_cgenff`, `dcmnet_n4_qm`, `dcmnet_n3_desp_dimer`)
are `best_params.msgpack` plus `model_config.json`, not PhysNet JSON.

`python scripts/fetch_weights.py --list` prints every model with its epoch and
the manuscript labels that use it.

## Distilling a PhysNet for another molecule

`karml pet-physnet-distill` labels a pool you specify. It does not default to
acetone.

```bash
# Bundled acetone recipe (ACO PDB, DMC dimers, carbonyl stretch, COM windows)
karml pet-physnet-distill --example acetone --preset smoke \
  --checkpoint pet-mad.pt --out-dir ./acetone_pet_distill

# Any other monomer. COM windows are name:lo:hi in Å. Stretch is i,j (j moves).
karml pet-physnet-distill --monomer monomer.xyz \
  --com-windows com_repulsive:3.2:4.0 com_well:4.0:5.5 com_long:7:10 \
  --stretch-bonds 0,1 \
  --dataset dimers.extxyz \
  --checkpoint pet-mad.pt --out-dir ./pet_distill --preset md

# Clusters cut from periodic liquid frames
karml pet-physnet-distill --from-box-extxyz runs/seed_*/traj.extxyz \
  --atoms-per-monomer 5 \
  --reference-monomer-xyz monomer.xyz \
  --checkpoint pet-mad.pt --out-dir ./box_clusters
```

Then `karml physnet-train --config <out-dir>/physnet-train.yaml`. Details:
[`docs/metatomic.md`](docs/metatomic.md).

## License

MIT License, Copyright (c) 2025, Eric Boittier. See [LICENSE](LICENSE) for details.

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
