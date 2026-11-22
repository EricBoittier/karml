"""Homodimer geometry pool is molecule-agnostic; acetone is only a recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from karml.distill.geometry_pool import (
    PoolConfig,
    apply_preset,
    build_geometry_pool,
    load_frames,
    preset_count_overrides,
)


def _water() -> Atoms:
    return Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )


def _water_cfg(**overrides) -> PoolConfig:
    base = dict(
        seed=0,
        n_monomer_noise=1,
        n_monomer_large_noise=0,
        n_bond_stretch=1,
        n_dimer_orientations=1,
        n_com_per_region=1,
        n_far_field=1,
        include_dataset=False,
        com_windows_A=(("com_well", 2.6, 3.2),),
        stretch_bonds=((0, 1),),
    )
    base.update(overrides)
    return PoolConfig(**base)


def test_pool_uses_the_given_monomer_not_a_bundled_molecule() -> None:
    geos = build_geometry_pool(_water(), _water_cfg())
    assert all(int(g.numbers[0]) == 8 for g in geos if g.kind == "monomer")
    assert any(g.atoms_per_monomer == (3, 3) and g.source == "com_well" for g in geos)
    eq = next(g for g in geos if g.source == "pdb_eq")
    stretched = next(g for g in geos if g.source == "bond_stretch")
    d0 = float(np.linalg.norm(eq.positions[1] - eq.positions[0]))
    d1 = float(np.linalg.norm(stretched.positions[1] - stretched.positions[0]))
    assert d1 != pytest.approx(d0)
    assert stretched.atoms_per_monomer == (3,)


def test_pool_is_reproducible_for_the_same_seed() -> None:
    a = build_geometry_pool(_water(), _water_cfg())
    b = build_geometry_pool(_water(), _water_cfg())
    assert len(a) == len(b)
    for ga, gb in zip(a, b):
        assert np.allclose(ga.positions, gb.positions)


def test_dataset_frames_follow_the_reference_size() -> None:
    water = _water()
    pos = np.asarray(water.get_positions(), dtype=np.float64)
    dimer = Atoms(
        numbers=[8, 1, 1, 8, 1, 1],
        positions=np.vstack([pos, pos + np.array([4.0, 0.0, 0.0])]),
    )
    other = Atoms(numbers=[6], positions=[[0.0, 0.0, 0.0]])
    geos = build_geometry_pool(
        water,
        _water_cfg(
            n_monomer_noise=0,
            n_bond_stretch=0,
            n_dimer_orientations=0,
            n_far_field=0,
            include_dataset=True,
        ),
        dataset=[dimer, other],
    )
    sources = [g.source for g in geos]
    assert sources.count("dataset") == 1
    assert sources.count("dataset_noise") == 1
    dimer_geo = next(g for g in geos if g.source == "dataset")
    assert dimer_geo.kind == "dimer"
    assert dimer_geo.r_com_A == pytest.approx(4.0)


def test_md_preset_only_changes_counts() -> None:
    smoke = _water_cfg()
    md = apply_preset(smoke, "md")
    assert md.n_dimer_orientations > smoke.n_dimer_orientations
    assert md.com_windows_A == smoke.com_windows_A
    assert md.stretch_bonds == smoke.stretch_bonds
    assert preset_count_overrides("smoke") == {}
    with pytest.raises(ValueError):
        preset_count_overrides("acetone")


def test_load_frames_reads_comment_separated_element_xyz(tmp_path: Path) -> None:
    path = tmp_path / "h2.xyz"
    path.write_text("# a\nH 0 0 0\nH 0.7 0 0\n\n# b\nH 0 0 0\nH 0.8 0 0\n")
    frames = load_frames(path, n_atoms=2)
    assert len(frames) == 2
    assert frames[0].get_atomic_numbers().tolist() == [1, 1]
    assert frames[1].get_positions()[1, 0] == pytest.approx(0.8)
