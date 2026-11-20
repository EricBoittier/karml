"""vesin-jax half-lists match the NumPy Vesin oracle."""

import numpy as np
import pytest

from karml.interfaces.pycharmmInterface.nl_backend import (
    build_mm_pairs_with_backend,
    resolve_mm_nl_backend,
)
from karml.interfaces.pycharmmInterface.nl_reference import (
    have_vesin,
    have_vesin_jax,
    vesin_jax_mic_pair_arrays,
    vesin_mic_pair_arrays,
)


pytestmark = pytest.mark.skipif(
    not (have_vesin() and have_vesin_jax()),
    reason="vesin and vesin-jax are both required",
)


def _liquid(n_monomers=8, atoms_per=10, box=24.0, cutoff=6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_monomers * atoms_per
    positions = rng.random((n, 3)) * box
    offsets = np.arange(0, n + 1, atoms_per, dtype=np.int32)
    monomer_id = np.repeat(np.arange(n_monomers, dtype=np.int32), atoms_per)
    return positions, np.eye(3) * box, offsets, monomer_id, cutoff


def test_vesin_jax_pair_set_matches_vesin():
    positions, cell, offsets, monomer_id, cutoff = _liquid()
    i, j = vesin_mic_pair_arrays(positions, cell, cutoff, monomer_id, monomer_offsets=offsets)
    ji, jj = vesin_jax_mic_pair_arrays(positions, cell, cutoff, monomer_id, monomer_offsets=offsets)
    assert set(zip(i.tolist(), j.tolist())) == set(zip(ji.tolist(), jj.tolist()))


def test_vesin_jax_backend_matches_vesin_backend():
    positions, cell, offsets, _mid, cutoff = _liquid(seed=1)
    left = build_mm_pairs_with_backend(
        "vesin", positions, cell, cutoff=cutoff, monomer_offsets=offsets, mm_r_min=4.0
    )
    right = build_mm_pairs_with_backend(
        "vesin_jax", positions, cell, cutoff=cutoff, monomer_offsets=offsets, mm_r_min=4.0
    )
    def pairs(result):
        pi, pj, mask, n_valid, _cap, name = result
        assert int(mask.sum()) == n_valid
        return name, set(zip(pi[mask].tolist(), pj[mask].tolist()))
    lname, lp = pairs(left)
    rname, rp = pairs(right)
    assert lname == "vesin"
    assert rname == "vesin_jax"
    assert lp == rp


def test_resolve_accepts_vesin_jax():
    assert resolve_mm_nl_backend("vesin_jax") == "vesin_jax"
