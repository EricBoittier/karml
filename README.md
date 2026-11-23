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
KARML is in **alpha**. The core calculator, CLI, and `md-system` YAML workflow are usable today, but interfaces are still settling and may change without notice ahead of a first tagged release. Feedback and issues are welcome — see the [FAQ](#faq).
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

## FAQ

### Q1. How can I distil a pair potential from a foundation model?

```bash
# The .pt is the teacher. pet-box-dataset keeps a few frames; pet-physnet-distill
# cuts monomers and close dimers and writes the student NPZ. Molecule size is
# the reference structure (this TIP3 pdb is 3 atoms), not a separate flag.
# Ethanol is the same command with an ethanol monomer file; this tree ships TIP3.
# Two molecules and a few steps stays short once the .pt is already on disk.
# The yaml warm-starts DESdimers and names the run acetone_pet_student.
uv sync --extra metatomic
python -c "from upet import save_upet; save_upet(model='pet-mad', size='xs', version='1.5.0', output='pet-mad-xs-v1.5.0.pt')"

karml pet-box-dataset --checkpoint pet-mad-xs-v1.5.0.pt \
  --residue TIP3 --monomer-xyz karml/data/charmm/tip3.pdb \
  --n-molecules 2 --box-size 10 --seeds 0 \
  --fire-steps 5 --md-steps 10 --out-dir runs/tip3_box

karml pet-physnet-distill --checkpoint pet-mad-xs-v1.5.0.pt \
  --from-box-extxyz runs/tip3_box/seed_*/traj.extxyz \
  --reference-monomer-xyz karml/data/charmm/tip3.pdb \
  --out-dir runs/tip3_pairs

# Lower num_epochs in that yaml for a first fit; 200 is the written default.
karml physnet-train --config runs/tip3_pairs/physnet-train.yaml
```

### Q2. How do I evaluate the checkpoint?

```bash
# Pair curves of the teacher (.pt): water–water and ethanol–ethanol.
karml pet-interaction-pes --checkpoint pet-mad-xs-v1.5.0.pt \
  --no-include-acetone --output-dir runs/pair_plots

# Student against the held-out NPZ.
karml physnet-evaluate \
  --checkpoint ./ckpts/acetone_pet_student \
  --data runs/tip3_pairs/valid.npz \
  -o runs/tip3_pairs/eval

# Small-cluster stability: minimize, finite-difference forces, bond scan, vibrations, kick.
karml mode-check --composition TIP3:2 --checkpoint ./ckpts/acetone_pet_student \
  --output-dir runs/mode_tip3_2 --checks minimize,fd,bond-scan,vibrations,kick

# Same question on a small ethanol–water box, then density / RDF / MSD.
karml md-system --backend jaxmd --setup pbc_nve \
  --ml-potential-mode physnet --checkpoint ./ckpts/acetone_pet_student \
  --composition ETOH:4,TIP3:8 --ps 0.05 --output-dir runs/etoh_water_nve
karml analyze-liquid --campaign-dir runs/etoh_water_nve -o runs/etoh_water_nve/analysis
```

### Q3. How do I run a simulation with my potential?

```bash
# JAX-MD, no CHARMM library. Packmol (make install-native) builds the box.
# Checkpoint directory is ckpt_dir in the training yaml.
karml md-system --backend jaxmd --setup pbc_nvt \
  --ml-potential-mode physnet --checkpoint ./ckpts/acetone_pet_student \
  --composition ETOH:4,TIP3:8 --temperature 300 --ps 0.2 \
  --output-dir runs/etoh_water
```

### Q4. How do I do a mixed MM+ML simulation?

```bash
# Close pairs stay on the learned potential. Past the switch, LJ and Coulomb are MM.
# --jaxmd-molpair-mm is the fast periodic MM path. The PyCHARMM line needs make install-native.
karml md-system --backend jaxmd --setup pbc_npt \
  --ml-potential-mode physnet --checkpoint ./ckpts/acetone_pet_student \
  --composition ETOH:4,TIP3:8 --temperature 300 --pressure 1 \
  --mm-switch-on 6.0 --mm-switch-width 5.0 --ps 0.2 \
  --output-dir runs/etoh_water_hybrid \
  --extra-args --jaxmd-molpair-mm

karml md-system --backend pycharmm --mlpot-pbc --setup pbc_nvt \
  --ml-potential-mode physnet --checkpoint ./ckpts/acetone_pet_student \
  --composition ETOH:4,TIP3:8 --temperature 300 --ps 0.2 \
  --mm-switch-on 6.0 --mm-switch-width 5.0 \
  --output-dir runs/etoh_water_charmm
```

## Acknowledgements
Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
MIT License, Copyright (c) 2026, Eric Boittier. See [LICENSE](LICENSE) for details.
