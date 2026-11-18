# Running karml on a GPU cluster: install, precision, job scripts, speed

Measured on one development cluster and one HPC centre in November 2026. Items marked **[tested]** were executed;
**[unverified]** were not. Timings were measured with the pre-rename development tree (`mmml`, `MMML_*` variables) and
re-checked on the ported tree for two cases (section 6); in this repository the variables are `KARML_*`.

## 1. Clean install [tested]

Prerequisites: gcc/gfortran >= 11, cmake >= 3.22, OpenMPI >= 4.1 with Fortran wrappers, FFTW3 (double and single),
`uv`, git, NVIDIA driver >= 580 for the CUDA 13 wheels (`--extra gpu`) or CUDA 12 wheels (`--extra gpu-cuda12`).

```bash
git clone https://github.com/EricBoittier/karml && cd karml
uv sync --extra gpu-cuda12 --extra md-cpu     # ONE call with all extras: a second `uv sync --extra X` removes the first
uv pip install pip                            # the CHARMM cmake wants pip in the venv; `uv sync` removes it again
unset CHARMM_HOME CHARMM_LIB_DIR              # a login profile that exports these redirects the build
export CHARMM_BUILD_DIR=/tmp/$USER/karml-charmm-build    # node-local disk (network file systems break cmake probes)
export CHARMM_PYTHON=$PWD/.venv/bin/python
make install-native                           # about 2.5 min on 12 cores; extracts the tarball, ends with `karml doctor`
```

`make install-native` calls `scripts/prepare_charmm_source.sh`, which extracts `setup/charmm.tar.xz` without the macOS
AppleDouble (`._*`) files it contains and applies `setup/charmm_tarball_fixes.patch` (library output name, the restart
lattice fix in `dynio.F90`, pyCHARMM cmake and python files). In a batch script do not use `bash -l` unless your site needs
`module`: a login shell re-sources the profile (and any `CHARMM_HOME` in it).

Failures met and fixes:

1. AppleDouble files in the tarball break `add_custom_target(_docu_target)`: handled by `prepare_charmm_source.sh`.
2. The shipped tarball was older than the tree the studies ran: 15 files, handled by the patch.
3. Foreign `CHARMM_HOME`: the build script now prints a warning; unset it.
4. A uv venv has no pip, so pyCHARMM is skipped: `uv pip install pip`.
5. `libmpi.so.40` not found in batch scripts on sites that need `module load`: `#!/bin/bash -l`.
6. `import vesin`/MDAnalysis fail on a base sync: `--extra md-cpu`.
7. `karml doctor` reports "JAX CPU only" after a base sync: expected without `--extra gpu*`.

Smoke tests [tested, CPU]: `karml md-system --backend jaxmd --setup pbc_nve --composition ACO:8 ... --ps 0.1` and
`karml md-system --backend pycharmm --mlpot-pbc --setup pbc_nve --md-stages nve --ps-nve 0.05 ...`.

## 2. Environment and precision

| setting | value | why |
|---|---|---|
| `JAX_DEFAULT_MATMUL_PRECISION` | `highest` | Ampere and later default to TF32 matrix products: energies shift by up to 1e-2 kcal/mol per molecule, forces by 0.1 kcal/mol/A. Required for conservation, NpT and ESP work. Cost 3-11 % |
| `NVIDIA_TF32_OVERRIDE=0`, `TF_ENABLE_TF32=0` | set | libraries that read the NVIDIA switch |
| `JAX_ENABLE_X64=1`, `KARML_ML_DTYPE=float32` | NpT production | float32 network, float64 state and box |
| `JAX_ENABLE_X64=1`, `KARML_ML_DTYPE=float64` | parity / conservation | float64 everywhere; consumer GPUs have 1/64 FP64 rate, use A100/H100/H200 |
| `KARML_NPT_VIRIAL` | unset: `fd64` (default) | nine float64 whole-hybrid evaluations per step; the analytic virial below is opt-in |
| `KARML_NPT_VIRIAL=analytic KARML_ML_LATTICE_VIRIAL=1` | opt-in | forces and strain virial in one PhysNet pass; validated to 1e-11 atm in float64, 0.02-0.06 atm in float32 against the float64 reference |
| `KARML_ML_CHUNK_REMAT=1` | opt-in | recompute per-chunk activations in the reverse pass: lower peak memory, more FLOPs |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | set | otherwise 75 % of the GPU per process (breaks packing) |
| `JAX_COMPILATION_CACHE_DIR` | persistent dir per GPU model and jax version | default `~/.cache/karml/jax-compilation-cache` for md-system (both backends) and the trainer; `KARML_NO_JAX_COMPILATION_CACHE=1` disables |
| `HDF5_USE_FILE_LOCKING=FALSE`, `PYTHONHASHSEED=0` | set | network file systems; reproducible hashes |
| `KARML_NO_CHARMM_MPI=1` | JAX-MD runs | no MPI re-launch |
| `--steps-per-recording 250` | production | default is 25 for NpT; each record does host work |

The defaults of the code are unchanged by the opt-ins: `fd64` virial, no remat.

## 3. Job scripts

One JAX-MD NpT replica on one GPU (Slurm):

```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=4 --mem=48G -t 1-00:00:00
source $KARML/.venv/bin/activate
export JAX_DEFAULT_MATMUL_PRECISION=highest NVIDIA_TF32_OVERRIDE=0 TF_ENABLE_TF32=0 JAX_ENABLE_X64=1 KARML_ML_DTYPE=float32 \
       XLA_PYTHON_CLIENT_PREALLOCATE=false HDF5_USE_FILE_LOCKING=FALSE PYTHONHASHSEED=0 KARML_NO_CHARMM_MPI=1
# opt-in speed path (not the default): export KARML_NPT_VIRIAL=analytic KARML_ML_LATTICE_VIRIAL=1
for v in $(env | grep -E '^(SLURM_MEM|PMI|PMIX|OMPI)' | cut -d= -f1); do unset $v; done
karml md-system --backend jaxmd --setup pbc_npt --ml-potential-mode physnet --checkpoint $CKPT \
  --from-psf model.psf --from-crd start.crd --skip-cluster-build --box-size $L --composition DCM:439 \
  --no-mc-density-equalize --density-prep-mode off --no-calculator-pre-minimize --no-charmm-pre-minimize \
  --jaxmd-minimize-steps 0 --jaxmd-pbc-minimize-steps 0 --dt-fs 0.5 --temperature 300 --pressure 1 \
  --mm-switch-on 6.0 --mm-switch-width 5.0 --steps-per-recording 250 --ps 100 --seed 1 --output-dir out \
  --extra-args --jaxmd-molpair-mm --ml-dimer-cap-auto 1.15
```

`--jaxmd-molpair-mm --ml-dimer-cap-auto 1.15` select the fast hybrid path (molecule-pair Verlet lists, device-side dimer
candidate lists). Per-record timing: `KARML_JAXMD_TIMING_LOG=$OUT/rec_timing.json`. Start from an equilibrated frame;
the first ~100 ps relax the density. Start frames with very large forces can abort with "Molecule extent ... exceeds ...":
use `--jaxmd-minimize-steps 100-200` or a relaxed start.

Packing replicas on one GPU needs CUDA MPS (plain time-slicing is slower than one process): measured gain +8-11 % on an
A100 and +22-26 % on an H200 (3-4 processes, about 4.7 GB each; slow path, float64; not measured with the fast path).

```bash
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d; sleep 3
for i in 1 2 3 4; do SEED=$i OUT=run_r$i bash one_replica.sh & done; wait
echo quit | nvidia-cuda-mps-control
```

CHARMM + MLpot: `karml md-system --backend pycharmm --mlpot-pbc --setup pbc_nve ...`, about 19x slower than JAX-MD in a
2080 Ti comparison (different code revisions, NVE vs NVT; see caveats in the study); use it for CHARMM-only features and
cross-engine checks.

## 4. Expected speed (ms/step at 0.5 fs; ns/day = 43.2 / ms; 36 A boxes: DCM:439 = 2195 atoms, ACO:379 = 3790 atoms)

| GPU | system | NVT `highest` | NpT `fd64` (default) | NpT analytic | NpT analytic + lattice (opt-in) |
|---|---|---|---|---|---|
| H200 | ACO | 4.81 | 13.14 | 11.71 | 8.40 |
| H200 | DCM | 2.41 | 5.79 | 5.41 | 4.46 |
| RTX 3090 Ti | ACO | 11.72 | 75.13 | 33.20 | 20.07 |
| RTX 3090 Ti | DCM | 4.57 | 29.46 | 11.54 | 7.97 |

Steady-state slope of per-record wall clock (250 steps per record, at least 12 records). The speed-up of the opt-in path
depends on the FP64 rate of the GPU (3.7x on a 1/64-FP64 card, 1.3-1.6x on an H200). Compile cache effect: process wall
140 s cold and 67 s warm (small NVT test), about 390 s cold and 50 s warm (large NpT).

## 5. Known failures

* TF32 is the default on Ampere and later (including RTX 5090): set `JAX_DEFAULT_MATMUL_PRECISION=highest`.
* `libmpi.so.40` not found inside batch scripts: use a login shell and load the MPI module.
* Strict-FIFO schedulers: a head job that fits no node blocks everything; check node memory before `--mem`.
* PET-OMol-L full-box single points do not fit in GPU memory with mlip.cpp (about 150 GB for 2195 atoms).
* Float32 `fd` strain virial is noisy (1.6-3 atm error); use `fd64` or the analytic path.
* Runs made before the JAX-MD time-unit fix (commit "pass dt and NHC/barostat tau in jax-md time units") have wrong
  ps labels; ms/step timings are unaffected.

## 6. GPU smoke of the ported tree [tested]

One RTX 2080 Ti, clean environment, `KARML_*` variables only, DCM:439 steady-state: NVT `highest` 7.94 ms/step
(development tree 7.98), NpT analytic + lattice virial 15.57 (development tree 15.56). The persistent compile cache
wrote entries without `JAX_COMPILATION_CACHE_DIR` set. The float64 parity test was run on the development tree only; the
port is textually identical apart from the names.

## 7. Not covered

Multi-node and DOMDEC runs, CHARMM + MLpot timings on A100/H200, packing with the fast path, trainer cold/warm compile
times, H100.
