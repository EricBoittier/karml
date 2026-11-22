"""Compile boundaries for the shared MD energy: outlined ML, analytic NpT virial."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jax_md")

from jax_md import quantity, space  # noqa: E402

from karml.md.npt_virial import analytic_npt_energy, strain_box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _frac(seed=0, n=4):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((n, 3)), dtype=jnp.float64)


def _reference_energy(energy_of_real):
    """Full autodiff through the strained box, the quantity.pressure convention."""

    def energy_fn(frac, box=None, perturbation=1.0, **dyn):
        box_eff = strain_box(box, perturbation)
        return energy_of_real(space.transform(box_eff, frac), box_eff, dyn)

    return energy_fn


def test_analytic_virial_matches_autodiff_for_a_box_dependent_pair():
    n = 4

    def energy_of_real(real, box, dyn):
        lengths = jnp.diagonal(box)
        d = real[:, None, :] - real[None, :, :]
        d = d - lengths * jnp.round(d / lengths)
        r2 = jnp.sum(d * d, -1) + jnp.eye(n, dtype=real.dtype)
        return 0.5 * jnp.sum(jnp.where(jnp.eye(n, dtype=bool), 0.0, 1.0 / r2))

    analytic = analytic_npt_energy(
        energy_of_real, energy_of_real
    )
    reference = _reference_energy(energy_of_real)
    frac = _frac(0)
    box = jnp.eye(3) * 10.0
    ke = 0.25
    p_an = float(quantity.pressure(analytic, frac, box, kinetic_energy=ke))
    p_ad = float(quantity.pressure(reference, frac, box, kinetic_energy=ke))
    assert p_an == pytest.approx(p_ad, rel=1e-8, abs=1e-10)
    # The explicit box channel is part of the answer: dropping it disagrees.
    pos_only = analytic_npt_energy(energy_of_real, lambda real, box, dyn: jnp.zeros(()))
    p_pos = float(quantity.pressure(pos_only, frac, box, kinetic_energy=ke))
    assert abs(p_pos - p_ad) > 1e-4 * max(1.0, abs(p_ad))


def test_analytic_virial_matches_autodiff_when_energy_ignores_the_box():
    origin = jnp.asarray(_frac(1), dtype=jnp.float64)

    def energy_of_real(real, box, dyn):
        return 0.5 * jnp.sum((real - origin) ** 2)

    analytic = analytic_npt_energy(energy_of_real, lambda real, box, dyn: jnp.zeros(()))
    reference = _reference_energy(energy_of_real)
    frac = _frac(2)
    box = jnp.diag(jnp.asarray([8.0, 9.0, 10.0]))
    f_an = jax.grad(lambda r: analytic(r, box=box, perturbation=1.0))(frac)
    f_ad = jax.grad(lambda r: reference(r, box=box, perturbation=1.0))(frac)
    np.testing.assert_allclose(f_an, f_ad, rtol=1e-8, atol=1e-10)
    p_an = float(quantity.pressure(analytic, frac, box, kinetic_energy=0.1))
    p_ad = float(quantity.pressure(reference, frac, box, kinetic_energy=0.1))
    assert p_an == pytest.approx(p_ad, rel=1e-8, abs=1e-10)


def test_monomer_network_stays_a_call_under_grad():
    """The intramolecular network is its own module, not inlined into grad."""
    from karml.interfaces.jaxmdInterface.hybrid_energy import make_monomer_energy_fn

    class _Model:
        def apply(self, params, atomic_numbers, positions, dst_idx, src_idx, compute_forces=False):
            return {"energy": jnp.sum(positions**2) * params}

    R = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]],
        dtype=jnp.float64,
    )
    Z = jnp.asarray([8, 1, 1, 8, 1, 1], dtype=jnp.int32)
    monomers = [jnp.asarray([0, 1, 2]), jnp.asarray([3, 4, 5])]
    disp, _ = space.free()
    fn = make_monomer_energy_fn(_Model(), jnp.asarray(2.0), Z, monomers, disp)
    closed = jax.make_jaxpr(jax.jit(jax.grad(fn)))(R)
    eqns = closed.jaxpr.eqns
    assert len(eqns) == 1
    assert eqns[0].primitive.name == "jit"
    assert eqns[0].params["name"] == "total_monomer_energy"


def test_npt_pressure_calls_the_network_instead_of_inlining_it():
    """Barostat derivative is three outlined calls, not the network body."""
    from karml.interfaces.jaxmdInterface.hybrid_energy import make_monomer_energy_fn

    class _Model:
        def apply(self, params, atomic_numbers, positions, dst_idx, src_idx, compute_forces=False):
            return {"energy": jnp.sum(jnp.sin(positions * params))}

    Z = jnp.asarray([8, 1, 1], dtype=jnp.int32)
    disp, _ = space.free()
    ml = make_monomer_energy_fn(_Model(), jnp.asarray(1.5), Z, [jnp.asarray([0, 1, 2])], disp)

    def full(real, box, dyn):
        return ml(real) + 0.5 * jnp.sum(real**2)

    def box_only(real, box, dyn):
        return 0.5 * jnp.sum((real - jnp.diagonal(box)) ** 2)

    fn = analytic_npt_energy(full, box_only)
    frac = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.2, 0.2], [0.3, 0.5, 0.1]])
    box = jnp.eye(3) * 10.0
    closed = jax.make_jaxpr(lambda r: quantity.pressure(fn, r, box, kinetic_energy=0.2))(frac)
    names = [e.params.get("name") for e in closed.jaxpr.eqns if e.primitive.name == "jit"]
    assert {"_energy", "_force_and_real", "_box_grad"} <= set(names)
    assert not any(e.primitive.name == "sin" for e in closed.jaxpr.eqns)


def test_box_energy_omits_intramolecular_ml():
    from karml.md.energy.registry import EnergyContext, HybridEnergy, TermFns
    from karml.md.system import MolecularSystem

    class _ML:
        name = "ml_intra"

        def neighbor_request(self, system):
            return None

        def make(self, system, ctx):
            return TermFns(jax_energy_fn=lambda R, **kwargs: jnp.sum(R))

    class _MM:
        name = "mm_nonbonded"

        def neighbor_request(self, system):
            return None

        def make(self, system, ctx):
            return TermFns(jax_energy_fn=lambda R, **kwargs: jnp.sum(R**2))

    system = MolecularSystem(
        R=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        Z=np.array([1, 1]),
        box=np.eye(3),
        mol_id=np.array([0, 1]),
    )
    energy = HybridEnergy([_ML(), _MM()], system, EnergyContext())
    R = jnp.asarray(system.R)
    assert float(energy.as_jax_box_energy_fn()(R)) == pytest.approx(float(jnp.sum(R**2)))
    assert float(energy.as_jax_energy_fn()(R)) == pytest.approx(float(jnp.sum(R) + jnp.sum(R**2)))
