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
As a `uv` tool (CLI only, no clone needed, only an NVIDIA driver if required)
```bash
uv tool install -p 3.13 "karml[gpu] @ git+https://github.com/EricBoittier/karml"         # CUDA 13 (SM 7.5+)
uv tool install -p 3.13 "karml[gpu-cuda12] @ git+https://github.com/EricBoittier/karml"  # CUDA 12 (older GPUs/drivers)
uv tool install -p 3.13 "karml @ git+https://github.com/EricBoittier/karml"              # CPU only
# from a local checkout instead: uv tool install -p 3.13 ".[gpu]"
```
Using `uv` (recommended):
```bash
git clone https://github.com/EricBoittier/karml.git
cd karml
uv sync
make install-gpu         # JAX CUDA 13 + CuPy 
# make install-gpu-cuda12  # older GPUs / CUDA 12
```
For PyCHARMM / Packmol (native libs, not installed by `uv`):
```bash
make install-native   # builds libcharmm + packmol under setup/charmm
make doctor           # karml doctor — env / CHARMM readiness
```
Jupyter kernel (required for the example notebooks)
Register the project venv as its own kernel **once**, and select it in the
notebook (`Kernel → Change Kernel → karml-venv`):
```bash
.venv/bin/python -m ipykernel install --user --name karml-venv --display-name "karml venv"
```
Using Conda / micromamba
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
Dockerfile and Compose live under [`devtools/docker/`](devtools/docker/).

## Acknowledgements
Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
MIT License, Copyright (c) 2026, Eric Boittier. See [LICENSE](LICENSE) for details.
