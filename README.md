<h1 align="center">karml</h1>

[![CI](https://github.com/EricBoittier/karml/workflows/CI/badge.svg)](https://github.com/EricBoittier/karml/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/EricBoittier/karml/branch/main/graph/badge.svg)](https://codecov.io/gh/EricBoittier/karml/branch/main)
[![Docs](https://readthedocs.org/projects/karml/badge/?version=latest)](https://karml.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
**karml** (pronounced Karamell) — machine-learned potentials in molecular-mechanics MD.
karml combines CHARMM/OpenMM workflows with JAX-based neural models for electrostatics and force prediction, for building and running hybrid ML/MM condensed-phase simulations.

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

## Newer stack

NpT density replicas use JAX 0.12.1, Flax 0.12.10, and JAX-MD [`karml-npt-a279379`](https://github.com/EricBoittier/jax-md/tree/karml-npt-a279379) (`a279379b55110a9a9cb29957b9775ad7da8dbe90`). That JAX-MD tree is not on PyPI. One force evaluation per NpT step. Measurements are in [`docs/newer-jax-stack.md`](docs/newer-jax-stack.md).

```bash
uv pip install "jax[cuda12]==0.12.1" "flax==0.12.10" \
  "jax-md @ git+https://github.com/EricBoittier/jax-md.git@karml-npt-a279379"
```

## FAQ

- [Q1. Make a molecule](#q1-how-do-i-make-a-molecule)
- [Q2. Distil a pair potential](#q2-how-can-i-distil-a-pair-potential-from-a-foundation-model)
- [Q3. Evaluate the checkpoint](#q3-how-do-i-evaluate-the-checkpoint)
- [Q4. Run a simulation](#q4-how-do-i-run-a-simulation-with-my-potential)
- [Q5. Mixed MM+ML](#q5-how-do-i-do-a-mixed-mmml-simulation)
- [Q6. QM calculation](#q6-how-do-i-run-a-qm-calculation)
- [Q7. Diffusion Monte Carlo](#q7-how-do-i-run-diffusion-monte-carlo)
- [Q8. DCMNet](#q8-how-do-i-use-dcmnet)
- [Q9. Precision](#q9-which-precision-choices-change-the-numbers)

Ethanol is `ETOH` and water is `TIP3`. Those names are used in the liquid, not in the distill command. `karml env` prints checkpoint paths.

### Q1. How do I make a molecule?

A molecule here is a CGenFF residue name. `make-res` writes one monomer (PDB, PSF, xyz) and needs the CHARMM build. `make-box` packs that PDB. With `--solvent`, `--n` is the number of solvent residues around it. The same names are `ETOH` and `TIP3` in the liquid files below.

```bash
karml make-res --list-residues 
karml make-res --res ETOH
# Eight waters around the ethanol just written.
karml make-box --n 8 --box-size 18 --solvent TIP3
```

A name missing from that list is built at [CHARMM-GUI](https://www.charmm-gui.org/). Any CHARMM topology is fine when its atom types are valid parameter names. `KARML_CGENFF_EXTRA_RTF` is that topology, and `KARML_CGENFF_EXTRA_PRM` is the matching parameter file when those names are not already in the bundled set. `make-res --res` then uses the `RESI` name from the file.

### Q2. How can I distil a pair potential from a foundation model?

The teacher is an AtomisticModel `.pt`. The short example exports PET-MAD xs 1.5.0. The paper's teacher is PET-OMol-L 1.0.0 (`pet-omol`, size `l`), checkpoint [`pet-omol-l-v1.0.0.ckpt`](https://huggingface.co/lab-cosmo/upet/blob/main/models/pet-omol-l-v1.0.0.ckpt) in `lab-cosmo/upet`. Frames come from a periodic trajectory. `pet-physnet-distill` cuts monomers and close dimers out of that trajectory and labels them with the teacher. One file is one molecule size: ethanol and water are two trajectories. `monomer.xyz` is that molecule in vacuum (the energy zero, and the atom count). `physnet-train.yaml` is written next to the NPZ and trains a PhysNet from scratch, with no warm-start checkpoint. `num_epochs: 3` only checks that the file runs; a real fit uses more epochs. Weights are written to `./ckpts/pairs`.

```bash
uv sync --extra metatomic
# Short example: PET-MAD xs 1.5.0.
python -c "from upet import save_upet; save_upet(model='pet-mad', size='xs', version='1.5.0', output='teacher.pt')"
# Paper teacher: PET-OMol-L 1.0.0 from lab-cosmo/upet.
python -c "from upet import save_upet; save_upet(model='pet-omol', size='l', version='1.0.0', output='pet-omol-l-v1.0.0.pt')"

# One trajectory, one molecule size. monomer.xyz is the vacuum energy zero.
karml pet-physnet-distill --checkpoint teacher.pt \
  --from-box-extxyz traj.extxyz \
  --reference-monomer-xyz monomer.xyz \
  --out-dir runs/pairs
# Trains the yaml below.
karml physnet-train --config runs/pairs/physnet-train.yaml
```

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
# Three epochs checks the file. A real fit needs more.
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

Further monomer geometries, still one molecule size, are displacements along Cartesian modes. `--calculator metatomic` finite-differences the teacher `.pt` (the same `--checkpoint` flag as `dimer-scan`). A unit mode is scaled by `--amplitude` in Å. A PySCF Hessian or any stored displacements are read the same way.

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

### Q3. How do I evaluate the checkpoint?

Pair curves are the teacher on water and ethanol. The student is the PhysNet from Q2, scored on the held-out NPZ and then on a water dimer (minimize, finite-difference forces, a bond scan, vibrations). In the liquid file, `ml_potential_mode: physnet` loads that student directory. `metatomic` loads the teacher `.pt`. The box holds four ethanols and eight waters. `analyze-liquid` reads density, RDF, and MSD from the trajectory. `ps: 0.01` at `dt_fs: 0.5` is 20 steps.

```bash
# Teacher pair curves. --checkpoint is the same .pt that labeled the NPZ.
karml pet-interaction-pes --checkpoint teacher.pt --no-include-acetone --output-dir runs/pairs/plots

# Student against the held-out labels.
karml physnet-evaluate --checkpoint ./ckpts/pairs --data runs/pairs/valid.npz -o runs/pairs/eval

# Minimize, finite-difference forces, a bond scan, vibrations.
karml mode-check --composition TIP3:2 --checkpoint ./ckpts/pairs \
  --output-dir runs/mode_tip3 --checks minimize,fd,bond-scan,vibrations

# Short NVE. The mixture and the student directory are in the yaml below.
karml md-system --config runs/etoh_water_nve.yaml
```

```yaml
# runs/etoh_water_nve.yaml
setup: pbc_nve
backend: jaxmd
# physnet: student directory from Q2. metatomic: the teacher .pt.
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
# Density, RDF, MSD from that trajectory.
karml analyze-liquid --campaign-dir runs/etoh_water_nve -o runs/etoh_water_nve/analysis
```

### Q4. How do I run a simulation with my potential?

Same potential and same mixture, at 300 K. The four ethanols and eight waters are packed into the cube with Packmol. `steps_per_recording: 5` writes a frame from this short run.

```bash
karml md-system --config runs/etoh_water.yaml
```

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

### Q5. How do I do a mixed MM+ML simulation?

Close pairs stay on the PhysNet potential. Past `mm_switch_on`, Lennard-Jones and Coulomb are classical, over `mm_switch_width`. With `mm_switch_on: 6` and `mm_switch_width: 5` the cube is 36 Å, so the switch fits in the box. `checkpoint` is the portable JSON written by `physnet-train` under `./ckpts/pairs` (`params_<tag>_*.json`). `extra_args` turns on the fast periodic MM path.

```bash
karml md-system --config runs/etoh_water_hybrid.yaml
```

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

The same split is run inside PyCHARMM. `backend: pycharmm` and `mlpot_pbc: true` load that potential in CHARMM. `ps_heat` and `ps_equi` are the heat and equilibration lengths.

```bash
karml md-system --config runs/etoh_water_charmm.yaml
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

### Q6. How do I run a QM calculation?

`pyscf-dft` is one molecule: energy, nuclear gradient, Hessian, and the harmonic modes read by `normal-mode-sample -i results.h5`. `pyscf-mp2` is the post-HF energy and gradient on the same `--mol` input. `pyscf-evaluate` labels every geometry in an NPZ, including the file from the teacher mode sample. The quantum GPU extra is required (`uv sync --extra quantum-gpu`).

```bash
# Energy, forces, Hessian, normal modes. Writes results.npz and results.h5.
karml pyscf-dft --mol monomer.xyz --energy --gradient --hessian --harmonic --output results

# MP2 on the same molecule.
karml pyscf-mp2 --mol monomer.xyz --energy --gradient --output mp2

# Label a batch. R, Z, N in; E, F, and dipoles out.
karml pyscf-evaluate -i monomer_modes.npz -o labeled.npz
```

### Q7. How do I run diffusion Monte Carlo?

`karml dmc` estimates a vibrational ground-state energy on a PhysNet or KerNN surface. Walkers share one batched JAX energy evaluation. The input frame is minimised, displaced by `--random-sigma` (0.02 Å), then diffused for `--nstep` imaginary-time steps. Averaging starts after `--eqstep`. `--natm` is the atom count of that frame. The short check uses an acetone dimer (20 atoms); the geometry is in the repo. The `.pot` and `.log` files are described in `docs/dmc.md`.

```bash
karml dmc \
  --natm 20 --nwalker 64 --stepsize 5e-4 --nstep 200 --eqstep 50 \
  --alpha 1200.0 --max-batch 64 --seed 0 \
  --checkpoint ./ckpts/pairs \
  --input karml/generate/dmc/examples/acetone_dmc.extxyz \
  --output-dir runs/dmc_acetone_smoke
```

### Q8. How do I use DCMNet?

DCMNet puts several charge sites on each atom (`--n-dcm`, default 3). The sites are fit to the electrostatic potential. Labels start from geometries (`R`, `Z`, `N`), such as `monomer_modes.npz` from the teacher mode sample. `pyscf-evaluate --esp` adds the energy, forces, dipole, and the potential on a grid around each structure, in one NPZ. `fix-and-split` converts PySCF units (Hartree, Hartree/Bohr, Debye, Bohr grid) into the training units and writes the two splits read by `train-joint`.

```bash
# Geometries in, QM labels out. --esp is the grid DCMNet fits.
karml pyscf-evaluate -i monomer_modes.npz -o labeled.npz --esp

# Same samples stay paired. Writes energies_forces_dipoles_*.npz and grids_esp_*.npz.
karml fix-and-split --efd labeled.npz -o splits

karml train-joint \
  --train-efd splits/energies_forces_dipoles_train.npz \
  --train-esp splits/grids_esp_train.npz \
  --valid-efd splits/energies_forces_dipoles_valid.npz \
  --valid-esp splits/grids_esp_valid.npz \
  --n-dcm 3

# Joint checkpoint: energy and forces, plus the distributed charges next to CGenFF.
karml compare-charmm-ml --checkpoint ./ckpts/joint \
  --valid-efd splits/energies_forces_dipoles_valid.npz \
  --valid-esp splits/grids_esp_valid.npz \
  --pdb monomer.pdb --n-samples 20 --out-dir runs/charmm_ml
```

### Q9. Which precision choices change the numbers?

The default environment is a float32 network, with JAX x64 switched on in-process. The float32 trace is taken with x64 off, so the network stays float32 while the state and the box stay float64. A float64 network (`KARML_ML_DTYPE=float64`) and the fast paths (molecule-pair MM, the analytic NpT virial) differ from that default. The Triton MM pair force (`KARML_MM_TRITON=1`) is nondeterministic at about 1e-3 kcal/mol.

## Acknowledgements
Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
MIT License, Copyright (c) 2026, Eric Boittier. See [LICENSE](LICENSE) for details.
