"""Acetone recipe for :func:`karml.distill.geometry_pool.build_geometry_pool`.

Bundled monomer PDB, DMC dimer frames, the carbonyl C–O stretch (atom order
of ``aco_monomer.pdb``), and the four acetone–acetone COM windows. Sampling
counts and the clash filter live on the general pool config.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from ase import Atoms

from karml.distill.geometry_pool import (
    BOND_STRETCH_FRACTION,
    CARTESIAN_NOISE_SIGMA_A,
    FAR_FIELD_COM_A,
    LARGE_NOISE_SIGMA_A,
    MIN_PAIR_DISTANCE_A,
    Geometry,
    PoolConfig,
    build_geometry_pool,
    load_frames,
    load_structure,
    preset_count_overrides,
)
from karml.paths import bundled_file, default_aco_template_pdb

ATOMS_PER_ACETONE = 10
DIMER_ATOMS = 2 * ATOMS_PER_ACETONE

# Carbonyl C and O in ``aco_monomer.pdb`` (0-based). Stretch moves O away from C.
CARBONYL_O_INDEX = 0
CARBONYL_C_INDEX = 1

# Four intermolecular COM windows (Å). Acetone–acetone contact is ~4–5 Å.
COM_REPULSIVE_A = (3.6, 4.4)
COM_WELL_A = (4.4, 5.6)
COM_SHOULDER_A = (5.6, 7.0)
COM_LONG_A = (7.0, 10.0)

ACO_DMC_EXTXYZ = bundled_file("generate", "dmc", "examples", "acetone_dmc.extxyz")


@dataclass(frozen=True)
class AcetonePoolConfig:
    """Acetone counts, COM windows, and optional extra extxyz paths."""

    seed: int = 0
    n_monomer_noise: int = 8
    n_monomer_large_noise: int = 4
    n_bond_stretch: int = 4
    n_dimer_orientations: int = 4
    n_com_per_region: int = 2
    n_far_field: int = 2
    n_global_rotations_extra: int = 0
    cartesian_noise_sigma_A: float = CARTESIAN_NOISE_SIGMA_A
    large_noise_sigma_A: float = LARGE_NOISE_SIGMA_A
    bond_stretch_fraction: float = BOND_STRETCH_FRACTION
    min_pair_distance_A: float = MIN_PAIR_DISTANCE_A
    far_field_com_A: float = FAR_FIELD_COM_A
    com_repulsive_A: tuple[float, float] = COM_REPULSIVE_A
    com_well_A: tuple[float, float] = COM_WELL_A
    com_shoulder_A: tuple[float, float] = COM_SHOULDER_A
    com_long_A: tuple[float, float] = COM_LONG_A
    include_dataset: bool = True
    extra_extxyz: tuple[Path, ...] = field(default_factory=tuple)


def pool_config_for_preset(preset: str, *, seed: int = 0) -> AcetonePoolConfig:
    return replace(
        AcetonePoolConfig(seed=int(seed)),
        **preset_count_overrides(preset),
    )


def load_acetone_monomer(*, path: Path | None = None) -> Atoms:
    pdb = Path(path) if path is not None else default_aco_template_pdb()
    atoms = load_structure(pdb)
    if len(atoms) != ATOMS_PER_ACETONE:
        raise ValueError(f"{pdb} has {len(atoms)} atoms, expected {ATOMS_PER_ACETONE}")
    return atoms


def load_dataset_dimers(*, path: Path | None = None) -> list[Atoms]:
    extxyz = Path(path) if path is not None else ACO_DMC_EXTXYZ
    frames = load_frames(extxyz, n_atoms=DIMER_ATOMS)
    if not frames:
        raise ValueError(f"no {DIMER_ATOMS}-atom dimer frames in {extxyz}")
    return frames


def _pool_config(config: AcetonePoolConfig) -> PoolConfig:
    shared = {f.name for f in fields(PoolConfig)} & {f.name for f in fields(AcetonePoolConfig)}
    kwargs = {name: getattr(config, name) for name in shared}
    kwargs["com_windows_A"] = (
        ("com_repulsive", *config.com_repulsive_A),
        ("com_well", *config.com_well_A),
        ("com_shoulder", *config.com_shoulder_A),
        ("com_long", *config.com_long_A),
    )
    kwargs["stretch_bonds"] = ((CARBONYL_C_INDEX, CARBONYL_O_INDEX),)
    kwargs["stretch_source"] = "co_stretch"
    kwargs["dataset_source"] = "dmc_extxyz"
    return PoolConfig(**kwargs)


def build_acetone_pool(config: AcetonePoolConfig | None = None) -> list[Geometry]:
    """Acetone monomers and homodimers (DMC frames + synthetic pool)."""
    cfg = config if config is not None else AcetonePoolConfig()
    dataset = load_dataset_dimers() if cfg.include_dataset else ()
    return build_geometry_pool(load_acetone_monomer(), _pool_config(cfg), dataset=dataset)


__all__ = [
    "ACO_DMC_EXTXYZ",
    "ATOMS_PER_ACETONE",
    "AcetonePoolConfig",
    "CARBONYL_C_INDEX",
    "CARBONYL_O_INDEX",
    "COM_LONG_A",
    "COM_REPULSIVE_A",
    "COM_SHOULDER_A",
    "COM_WELL_A",
    "DIMER_ATOMS",
    "build_acetone_pool",
    "load_acetone_monomer",
    "load_dataset_dimers",
    "pool_config_for_preset",
]
