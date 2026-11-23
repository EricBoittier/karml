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

Ethanol is `ETOH` and water is `TIP3`. Those names show up in the liquid, not in the distill command. `karml env` prints checkpoint paths.

### Q1. How can I distil a pair potential from a foundation model?

The teacher is an AtomisticModel `.pt`. You supply the frames. `pet-physnet-distill` cuts monomers and close dimers out of a periodic trajectory and asks the teacher for labels. One file is one molecule size: ethanol and water are two trajectories. `monomer.xyz` is that molecule in vacuum (the energy zero, and the atom count). The command writes `physnet-train.yaml` next to the NPZ. That file trains a PhysNet from scratch: there is no warm-start checkpoint. Three epochs only checks that the file runs; raise `num_epochs` for a real fit. The weights land in `./ckpts/pairs`.

```yaml
# runs/pairs/physnet-train.yaml
data: runs/pairs/train.npz
valid_data: runs/pairs/valid.npz
ckpt_dir: ./ckpts/pairs
tag: pairs
charges: false
use_energy_bias: true
seed: 0
batch_size: 1
# Three epochs only checks the file. Raise this for a real fit.
num_epochs: 3
learning_rate: 0.001
objective: valid_forces_mae
energy_weight: 1.0
forces_weight: 52.91
features: 32
max_degree: 0
num_basis_functions: 16
num_iterations: 2
n_res: 2
cutoff: 8.0
max_atomic_number: 8
```

```bash
# Any AtomisticModel .pt. This export is one teacher you can download.
uv sync --extra metatomic
python -c "from upet import save_upet; save_upet(model='pet-mad', size='xs', version='1.5.0', output='teacher.pt')"

# One trajectory, one molecule size. monomer.xyz is the vacuum energy zero.
karml pet-physnet-distill --checkpoint teacher.pt \
  --from-box-extxyz traj.extxyz \
  --reference-monomer-xyz monomer.xyz \
  --out-dir runs/pairs
# Fits the yaml above. Raise num_epochs there for a real student.
karml physnet-train --config runs/pairs/physnet-train.yaml
```

More monomer geometries, still one molecule size: displace along Cartesian modes. `--calculator metatomic` finite-differences the teacher `.pt` (the same `--checkpoint` flag as `dimer-scan`). A unit mode moves by `--amplitude` Å. A PySCF Hessian or any stored displacements work the same way.

```bash
# Modes from the teacher.
karml normal-mode-sample --calculator metatomic --checkpoint teacher.pt \
  --structure monomer.xyz -o monomer_modes.npz \
  --amplitude 0.05 --include-equilibrium

# Any (n, n_atoms, 3) displacements in Å, or a multi-frame extxyz.
karml normal-mode-sample --displacements modes.npz -o sampled.npz --amplitude 0.05

# PySCF Hessian from `karml pyscf-dft --harmonic`.
karml normal-mode-sample -i monomer.h5 -o monomer_modes.npz --amplitude 0.05
```

### Q2. How do I evaluate the checkpoint?

Pair curves are the teacher on water and ethanol. The student is the PhysNet from Q1: score it on the held-out NPZ, then on a water dimer (minimize, finite-difference forces, a bond scan, vibrations). In the liquid file, `ml_potential_mode: physnet` loads that student directory. `metatomic` loads the teacher `.pt`. The box is four ethanols and eight waters. `analyze-liquid` reads density, RDF, and MSD. `ps: 0.01` at `dt_fs: 0.5` is 20 steps.

```yaml
# runs/etoh_water_nve.yaml
setup: pbc_nve
backend: jaxmd
# physnet: student directory from Q1. metatomic: the teacher .pt.
ml_potential_mode: physnet
checkpoint: ./ckpts/pairs
composition: "ETOH:4,TIP3:8"
box_size: 18.0
dt_fs: 0.5
ps: 0.01
temperature: 300
seed: 1
output_dir: runs/etoh_water_nve
include_mm: false
steps_per_recording: 5
jaxmd_minimize_steps: 0
jaxmd_pbc_minimize_steps: 0
calculator_pre_minimize: false
```

```bash
# Teacher pair curves. --checkpoint is the same .pt that labeled the NPZ.
karml pet-interaction-pes --checkpoint teacher.pt --no-include-acetone --output-dir runs/pairs/plots

# Student against the held-out labels.
karml physnet-evaluate --checkpoint ./ckpts/pairs --data runs/pairs/valid.npz -o runs/pairs/eval

# Minimize, finite-difference forces, a bond scan, vibrations.
karml mode-check --composition TIP3:2 --checkpoint ./ckpts/pairs \
  --output-dir runs/mode_tip3 --checks minimize,fd,bond-scan,vibrations

# Short NVE. The mixture and the student directory are in the yaml above.
karml md-system --config runs/etoh_water_nve.yaml
# Density, RDF, MSD from that trajectory.
karml analyze-liquid --campaign-dir runs/etoh_water_nve -o runs/etoh_water_nve/analysis
```

### Q3. How do I run a simulation with my potential?

Same potential and same mixture, held at 300 K. Packmol packs the four ethanols and eight waters into the cube. `steps_per_recording: 5` writes a frame from this short run.

```yaml
# runs/etoh_water.yaml
setup: pbc_nvt
backend: jaxmd
ml_potential_mode: physnet
checkpoint: ./ckpts/pairs
composition: "ETOH:4,TIP3:8"
box_size: 18.0
temperature: 300
dt_fs: 0.5
ps: 0.01
seed: 1
output_dir: runs/etoh_water
include_mm: false
steps_per_recording: 5
jaxmd_minimize_steps: 0
jaxmd_pbc_minimize_steps: 0
calculator_pre_minimize: false
```

```bash
karml md-system --config runs/etoh_water.yaml
```

### Q4. How do I do a mixed MM+ML simulation?

Close pairs stay on the PhysNet potential. Past `mm_switch_on`, Lennard-Jones and Coulomb are classical, over `mm_switch_width`. The box has to be large enough for that switch: with `mm_switch_on: 6` and `mm_switch_width: 5` the cube here is 36 Å. `checkpoint` is the portable JSON that `physnet-train` writes under `./ckpts/pairs` (`params_<tag>_*.json`). `extra_args` turns on the fast periodic MM path. The second file is that split inside CHARMM.

```yaml
# runs/etoh_water_hybrid.yaml
setup: pbc_npt
backend: jaxmd
ml_potential_mode: physnet
checkpoint: ./ckpts/pairs/params_pairs.json
composition: "ETOH:4,TIP3:8"
box_size: 36.0
temperature: 300
pressure: 1.0
dt_fs: 0.5
ps: 0.01
mm_switch_on: 6.0
mm_switch_width: 5.0
seed: 1
output_dir: runs/etoh_water_hybrid
steps_per_recording: 5
jaxmd_minimize_steps: 0
jaxmd_pbc_minimize_steps: 0
calculator_pre_minimize: false
extra_args:
  - --jaxmd-molpair-mm
```

```yaml
# runs/etoh_water_charmm.yaml
setup: pbc_nvt
backend: pycharmm
mlpot_pbc: true
ml_potential_mode: physnet
checkpoint: ./ckpts/pairs/params_pairs.json
composition: "ETOH:4,TIP3:8"
box_size: 36.0
temperature: 300
dt_fs: 0.5
ps: 0.01
mm_switch_on: 6.0
mm_switch_width: 5.0
seed: 1
output_dir: runs/etoh_water_charmm
ps_heat: 0.01
ps_equi: 0.01
```

```bash
karml md-system --config runs/etoh_water_hybrid.yaml
karml md-system --config runs/etoh_water_charmm.yaml
```

## Acknowledgements
Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
MIT License, Copyright (c) 2026, Eric Boittier. See [LICENSE](LICENSE) for details.
