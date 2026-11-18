# Optional newer JAX / flax / jax-md stack

**Status: experimental, not the default.** `pyproject.toml` keeps `jax`/`jaxlib` below 0.11.2 and the released
`jax-md` from PyPI. Use this page only to reproduce the measurements below or to try the newer stack; production
work should use the pinned stack until the long NpT density comparison listed under "Open check" is finished.

## What it is

| piece | default pin | newer stack |
|---|---|---|
| jax / jaxlib | `>=0.10.0,<0.11.2` | 0.12.1 |
| flax | `>=0.12.1` | `>=0.12.10` (0.12.1 fails on jax 0.12 with `jax.core.Effect`) |
| jax-md | PyPI 0.2.29 (NpT: two force evaluations per step) | git commit `a279379` (25 Oct 2026, NpT: one force evaluation per step; not on PyPI) |

jax 0.11.2 renamed `jax.experimental.hijax.HiPrimitive` to `HiPrim`; flax 0.12.10 and later follow the rename, which
is why the default cap exists for older flax.

## Install (on top of a normal `uv sync`, one environment)

```bash
uv sync --extra gpu-cuda12 --extra md-cpu           # default stack first (CUDA 13: --extra gpu)
uv pip install "jax[cuda12]==0.12.1" "flax>=0.12.10" \
    "jax-md @ git+https://github.com/jax-md/jax-md@a279379"
export KARML_NEWER_JAX_STACK=1                       # makes the unit tests that assert the default cap skip
```

A later `uv sync` restores the default pins. Nothing in `pyproject.toml` changes.

## Measured (one H200 and one RTX 4090; acetone, 379 molecules / 3790 atoms; analytic NpT virial)

| stack | NVE ms/step | NpT ms/step (H200) | cold-start wall NpT (s) |
|---|---|---|---|
| default (jax 0.11.1, PyPI jax-md) | 4.5 | 11.6 | about 390 |
| jax 0.12.1, PyPI jax-md | 4.5 | 11.0 | about 146 |
| jax 0.12.1, jax-md `a279379` | 4.5 | 6.6 | about 125 |

Float64 energies, forces and virial agree with the default stack to 1e-13 kcal/mol; the unit tests agree except for the
two that assert the default cap. 64-bit NpT over 1.5 ps from an identical start agrees to round-off with the expected
chaotic growth (max |dE_pot| 8e-7 eV), but the 64-bit NpT speed-up is only about 1.15x (48 vs 55 ms/step) because the
float64 force evaluation dominates.

## Open check

A 20 ps or longer NpT density and pressure comparison of the two stacks on the real hybrid box has not finished. Until it
has, treat the newer stack as unvalidated for production.
