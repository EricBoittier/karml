"""KARML_ML_CHUNK_REMAT: opt-in jax.checkpoint around the per-chunk apply; values and gradients unchanged."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from karml.interfaces.pycharmmInterface.mlpot_gpu import run_chunked_model_apply


def _run(R):
    Z = jnp.ones(R.shape[:3], dtype=jnp.int32)
    N = jnp.full(R.shape[:2], R.shape[2], dtype=jnp.int32)

    def apply_one_chunk(Rc, Zc, Nc):
        e = jnp.sum(jnp.sin(Rc) * jnp.cos(Rc[..., ::-1]) + 0.1 * Rc**2, axis=(-1, -2))
        return e, jnp.zeros_like(Rc)

    def total(Rall):
        out = run_chunked_model_apply(
            R_chunks=Rall, Z_chunks=Z, N_chunks=N, n_chunks=Rall.shape[0],
            effective_batch_size=Rall.shape[0] * Rall.shape[1], chunk_size=Rall.shape[1],
            max_atoms=Rall.shape[2], n_gpus=1, apply_one_chunk=apply_one_chunk, has_aux=False,
        )
        return jnp.sum(out[0])

    return jax.value_and_grad(total)(R)


def test_remat_same_value_and_grad(monkeypatch):
    R = jnp.asarray(np.random.default_rng(0).normal(size=(3, 2, 4, 3)))
    monkeypatch.delenv("KARML_ML_CHUNK_REMAT", raising=False)
    v0, g0 = _run(R)
    monkeypatch.setenv("KARML_ML_CHUNK_REMAT", "1")
    v1, g1 = _run(R)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v0), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g0), rtol=1e-5, atol=1e-6)


def test_remat_is_off_by_default(monkeypatch):
    import inspect

    from karml.interfaces.pycharmmInterface import mlpot_gpu

    assert 'os.environ.get("KARML_ML_CHUNK_REMAT", "0")' in inspect.getsource(mlpot_gpu)
