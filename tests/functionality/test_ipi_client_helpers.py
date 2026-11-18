"""CPU-only tests of the pure helpers of karml.interfaces.ipi_client (no JAX, no i-PI, no CHARMM needed)."""
import numpy as np

from karml.interfaces.ipi_client import verlet_list_valid, wrap_molecules


def test_wrap_molecules_keeps_molecules_whole_and_centres_in_box():
    rng = np.random.default_rng(0)
    L, nmol, apm = 20.0, 7, 5
    base = rng.uniform(0, L, (nmol, 1, 3)) + rng.normal(0, 0.5, (nmol, apm, 3))
    shifted = base + L * rng.integers(-3, 4, (nmol, 1, 3))
    w = wrap_molecules(shifted.reshape(-1, 3), L, nmol).reshape(nmol, apm, 3)
    c = w.mean(1)
    assert np.all(c >= 0) and np.all(c < L)
    # intramolecular geometry unchanged
    assert np.allclose(w - w[:, :1], shifted - shifted[:, :1])
    # a lattice-vector shift of the input gives the same wrapped output
    w2 = wrap_molecules((shifted + L * np.array([1, -2, 0])).reshape(-1, 3), L, nmol)
    assert np.allclose(w2.reshape(nmol, apm, 3), w, atol=1e-10)


def test_verlet_list_validity_criterion():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 30, (50, 3))
    assert not verlet_list_valid(ref, None, 30.0, 30.0, 0.25, 11.25)
    assert not verlet_list_valid(ref, ref, 30.0, 30.0, 0.0, 11.25)           # skin 0: always rebuild
    assert verlet_list_valid(ref, ref, 30.0, 30.0, 0.25, 11.25)              # unchanged
    step = np.zeros_like(ref); step[3] = [0.10, 0.0, 0.0]
    assert verlet_list_valid(ref + step, ref, 30.0, 30.0, 0.25, 11.25)       # 2*0.10 <= 0.25
    step[3] = [0.13, 0.0, 0.0]
    assert not verlet_list_valid(ref + step, ref, 30.0, 30.0, 0.25, 11.25)   # 2*0.13 > 0.25
    # a box change of 1e-3 (r_list*1e-3 = 0.011 A) with positions scaled affinely stays valid; 3e-2 does not
    assert verlet_list_valid(ref * 1.001, ref, 30.03, 30.0, 0.25, 11.25)
    assert not verlet_list_valid(ref * 1.03, ref, 30.9, 30.0, 0.25, 11.25)
