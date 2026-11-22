"""Fixed-point derivative of the induced-dipole energy."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _x64():
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _system():
    R = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.98, 0.05, 0.0],
            [3.4, 0.2, 0.1],
            [4.3, 0.1, 0.15],
        ],
        dtype=jnp.float64,
    )
    alpha = jnp.asarray([1.4, 0.5, 1.4, 0.5])
    q = jnp.asarray([-0.7, 0.7, -0.7, 0.7])
    mol = jnp.asarray([0, 0, 1, 1])
    return R, alpha, q, mol


def _fd(energy, R, eps=1.0e-6):
    g = np.zeros(R.shape, dtype=np.float64)
    for i in range(R.shape[0]):
        for k in range(R.shape[1]):
            step = jnp.zeros_like(R).at[i, k].set(eps)
            g[i, k] = float(energy(R + step) - energy(R - step)) / (2.0 * eps)
    return g


def test_dense_fixed_point_gradient_matches_finite_difference():
    from karml.md.polarization import _jacobi_mu, _T_dense, _dense_fields, _upol_dense

    R, alpha, q, mol = _system()
    n_iter = 30
    a = 0.2

    def energy(pos):
        return _upol_dense(pos, alpha, q, mol, a, None, None, n_iter)

    E0, d, w3, w5 = _dense_fields(R, alpha, q, mol, a, None, None, None, 1.0)
    mu = _jacobi_mu(E0, lambda m: _T_dense(d, w3, w5, m), alpha, n_iter)
    residual = mu - alpha[:, None] * (E0 + _T_dense(d, w3, w5, mu))
    assert float(jnp.linalg.norm(residual)) < 1e-8

    got = np.asarray(jax.grad(energy)(R))
    ref = _fd(energy, R)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)


def test_sparse_fixed_point_gradient_matches_dense():
    from karml.md.polarization import _upol_dense, _upol_sparse

    R, alpha, q, mol = _system()
    n = R.shape[0]
    # Every other atom, padded the way the neighbour list does (index == n is a hole).
    idx = jnp.asarray([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=jnp.int32)
    L = jnp.asarray(20.0)
    taper = (9.0, 12.0)
    n_iter = 25
    a = 0.2
    # Positions sit well inside the taper, so the sparse list is the dense interaction.
    dense = jax.grad(lambda pos: _upol_dense(pos, alpha, q, mol, a, L, taper, n_iter))(R)
    sparse = jax.grad(
        lambda pos: _upol_sparse(pos, alpha, q, mol, a, L, taper, n_iter, idx)
    )(R)
    np.testing.assert_allclose(sparse, dense, rtol=1e-6, atol=1e-8)


def test_dense_gradient_jaxpr_does_not_unroll_the_jacobi_loop():
    from karml.md.polarization import _upol_dense

    R, alpha, q, mol = _system()

    def energy(pos):
        return _upol_dense(pos, alpha, q, mol, a=0.2, L=None, taper=None, n_iter=12)

    eqns = jax.make_jaxpr(jax.grad(energy))(R).jaxpr.eqns
    # One forward Jacobi scan and one adjoint scan. A reverse-mode trace of the
    # iteration would transpose that scan and keep a residual per iteration.
    assert sum(e.primitive.name == "scan" for e in eqns) == 2
