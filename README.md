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

## License

MIT License, Copyright (c) 2025, Eric Boittier. See [LICENSE](LICENSE) for details.

## Acknowledgements

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
