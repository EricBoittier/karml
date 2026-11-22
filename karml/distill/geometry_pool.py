"""Unlabeled monomer/dimer geometry pools for teacher → student distillation.

Mixes one reference monomer with synthetic augmentations aimed at MD
stability: Cartesian noise, bond stretches, random relative orientations,
and caller-supplied COM-distance windows. Dataset frames and extra extxyz
files are optional.

Molecule recipes (paths, which bonds to stretch, COM windows) live in
``karml.distill.examples``. This module builds homodimers of whatever
monomer it is given. The reference geometry is tagged ``pdb_eq`` so
``label_geometries`` can take ``E_ref`` from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read as ase_read
from ase.io.extxyz import XYZError

from karml.interfaces.pycharmmInterface.cutoffs import DEFAULT_MM_SWITCH_ON

# Far-field replica: beyond the ML/MM handoff so teacher E_int should vanish.
FAR_FIELD_COM_A = float(DEFAULT_MM_SWITCH_ON) + 6.0

MIN_PAIR_DISTANCE_A = 0.75
CARTESIAN_NOISE_SIGMA_A = 0.05
LARGE_NOISE_SIGMA_A = 0.15
BOND_STRETCH_FRACTION = 0.08

PRESET_SMOKE = "smoke"
PRESET_MD = "md"
POOL_PRESETS = (PRESET_SMOKE, PRESET_MD)

# ``source`` of the reference monomer. ``label_geometries`` reads E_ref from it.
SOURCE_REFERENCE = "pdb_eq"

_MD_COUNTS: dict[str, int] = {
    "n_monomer_noise": 48,
    "n_monomer_large_noise": 24,
    "n_bond_stretch": 12,
    "n_dimer_orientations": 12,
    "n_com_per_region": 4,
    "n_far_field": 8,
    "n_global_rotations_extra": 1,
}


@dataclass(frozen=True)
class PoolConfig:
    """Counts and scales for a homodimer geometry mixture.

    ``com_windows_A`` is ``(source_name, lo_Å, hi_Å)`` per intermolecular
    window. ``stretch_bonds`` is ``(i, j)`` pairs: atom ``j`` is scaled away
    from atom ``i``. Both are empty until a molecule recipe fills them.
    """

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
    com_windows_A: tuple[tuple[str, float, float], ...] = ()
    stretch_bonds: tuple[tuple[int, int], ...] = ()
    include_dataset: bool = True
    extra_extxyz: tuple[Path, ...] = field(default_factory=tuple)
    reference_source: str = SOURCE_REFERENCE
    stretch_source: str = "bond_stretch"
    dataset_source: str = "dataset"


@dataclass
class Geometry:
    """One unlabeled structure. Positions Å, no PBC."""

    numbers: np.ndarray
    positions: np.ndarray
    kind: str
    source: str
    r_com_A: float | None
    atoms_per_monomer: tuple[int, ...]
    # Provenance for structures cut from MD (``box_cluster_pool``); None for
    # a synthetic pool. ``group_seed`` is the trajectory seed (independent
    # trajectory), ``group_file`` the input-file index (fallback group when
    # frames carry no seed), ``group_frame`` the frame index in the pool's
    # frame list, ``group_step``/``group_phase`` from frame info.
    group_seed: int | None = None
    group_file: int | None = None
    group_frame: int | None = None
    group_step: int | None = None
    group_phase: str | None = None


def preset_count_overrides(preset: str) -> dict[str, int]:
    """Sample counts for ``smoke`` (library defaults) or the larger ``md`` set."""
    name = str(preset).strip().lower()
    if name == PRESET_SMOKE:
        return {}
    if name == PRESET_MD:
        return dict(_MD_COUNTS)
    raise ValueError(f"unknown pool preset {preset!r}; use {POOL_PRESETS}")


def apply_preset(config: PoolConfig, preset: str) -> PoolConfig:
    """Return ``config`` with the sample counts for ``preset`` (seed unchanged)."""
    return replace(config, **preset_count_overrides(preset))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) via a unit quaternion."""
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def min_pair_distance(positions: np.ndarray) -> float:
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        return float("inf")
    dmin = np.inf
    for i in range(n):
        delta = pos[i + 1 :] - pos[i]
        d = np.linalg.norm(delta, axis=1)
        if d.size:
            dmin = min(dmin, float(np.min(d)))
    return float(dmin)


def load_structure(path: Path | str) -> Atoms:
    """First frame of an ASE-readable structure file."""
    atoms = ase_read(str(path))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"no frames in {path}")
        atoms = atoms[0]
    return atoms


def _parse_element_xyz_blocks(text: str, *, n_atoms: int | None) -> list[Atoms]:
    """Parse ``# comment`` separated element-XYZ blocks (not ASE extxyz)."""
    frames: list[Atoms] = []
    zs: list[int] = []
    pos: list[list[float]] = []

    def _flush_exact() -> None:
        nonlocal zs, pos
        if n_atoms is not None and len(zs) == n_atoms:
            frames.append(
                Atoms(
                    numbers=np.asarray(zs, dtype=np.int32),
                    positions=np.asarray(pos, dtype=np.float64),
                )
            )
        zs, pos = [], []

    def _flush_open() -> None:
        nonlocal zs, pos
        if zs:
            frames.append(
                Atoms(
                    numbers=np.asarray(zs, dtype=np.int32),
                    positions=np.asarray(pos, dtype=np.float64),
                )
            )
        zs, pos = [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if n_atoms is None:
                _flush_open()
            else:
                _flush_exact()
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        z = atomic_numbers.get(parts[0])
        if z is None:
            z = atomic_numbers.get(parts[0].capitalize())
        if z is None:
            continue
        zs.append(int(z))
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if n_atoms is not None and len(zs) == n_atoms:
            _flush_exact()
    if n_atoms is None:
        _flush_open()
    elif len(zs) == n_atoms:
        _flush_exact()
    return frames


def load_frames(path: Path | str, *, n_atoms: int | None = None) -> list[Atoms]:
    """Read an ASE trajectory, or comment-separated element XYZ if ASE cannot."""
    src = Path(path)
    try:
        frames = ase_read(str(src), index=":")
        if not isinstance(frames, list):
            frames = [frames]
    except (XYZError, ValueError, OSError):
        frames = _parse_element_xyz_blocks(src.read_text(), n_atoms=n_atoms)
    if n_atoms is None:
        return list(frames)
    return [frame for frame in frames if len(frame) == int(n_atoms)]


def _center(positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    return pos - pos.mean(axis=0)


def _geometry(
    numbers: np.ndarray,
    positions: np.ndarray,
    *,
    kind: str,
    source: str,
    r_com_A: float | None,
    atoms_per_monomer: tuple[int, ...],
    min_pair_distance_A: float,
) -> Geometry | None:
    if min_pair_distance(positions) < float(min_pair_distance_A):
        return None
    return Geometry(
        numbers=np.asarray(numbers, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.float64),
        kind=kind,
        source=source,
        r_com_A=None if r_com_A is None else float(r_com_A),
        atoms_per_monomer=tuple(int(n) for n in atoms_per_monomer),
    )


def assemble_dimer(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    *,
    com_distance_A: float,
    rotation_b: np.ndarray,
) -> np.ndarray:
    """Place A at the origin and B along +x at ``com_distance_A`` after rotating B."""
    a = _center(pos_a)
    b = _center(pos_b) @ np.asarray(rotation_b, dtype=np.float64).T
    b = b + np.array([float(com_distance_A), 0.0, 0.0], dtype=np.float64)
    return np.concatenate([a, b], axis=0)


def _stretch_bond(pos: np.ndarray, i: int, j: int, scale: float) -> np.ndarray:
    """Scale atom ``j`` away from atom ``i`` along the current bond."""
    out = np.asarray(pos, dtype=np.float64).copy()
    bond = out[j] - out[i]
    out[j] = out[i] + bond * float(scale)
    return out


def _com_distance(pos: np.ndarray, n_mono: int) -> float:
    return float(np.linalg.norm(pos[:n_mono].mean(0) - pos[n_mono:].mean(0)))


def build_geometry_pool(
    monomer: Atoms,
    config: PoolConfig | None = None,
    *,
    dataset: Sequence[Atoms] | None = None,
) -> list[Geometry]:
    """Return unlabeled monomers and homodimers (optional dataset + synthetic).

    Dataset and extra-extxyz frames are kept when they have ``n`` atoms
    (monomer) or ``2n`` atoms (dimer), where ``n`` is the reference monomer.
    Other sizes are skipped. Bond stretches run only when ``stretch_bonds``
    is set; COM scans run only when ``com_windows_A`` is set.
    """
    cfg = config if config is not None else PoolConfig()
    rng = np.random.default_rng(int(cfg.seed))
    z_m = np.asarray(monomer.get_atomic_numbers(), dtype=np.int32)
    r_eq = np.asarray(monomer.get_positions(), dtype=np.float64)
    n_mono = int(r_eq.shape[0])
    if n_mono < 1 or z_m.shape[0] != n_mono:
        raise ValueError("monomer has no atoms")
    for i, j in cfg.stretch_bonds:
        if i == j or not (0 <= int(i) < n_mono and 0 <= int(j) < n_mono):
            raise ValueError(f"stretch bond {(i, j)} is outside monomer atom count {n_mono}")
    per_mono = (n_mono,)
    per_dimer = (n_mono, n_mono)
    geos: list[Geometry] = []
    rejected = 0

    def _add(geo: Geometry | None) -> None:
        nonlocal rejected
        if geo is None:
            rejected += 1
            return
        geos.append(geo)

    _add(
        _geometry(
            z_m,
            r_eq,
            kind="monomer",
            source=cfg.reference_source,
            r_com_A=None,
            atoms_per_monomer=per_mono,
            min_pair_distance_A=cfg.min_pair_distance_A,
        )
    )

    def _noise_monomers(n: int, sigma: float, source: str) -> None:
        for _ in range(int(n)):
            rattled = r_eq + rng.normal(0.0, float(sigma), size=r_eq.shape)
            _add(
                _geometry(
                    z_m,
                    rattled,
                    kind="monomer",
                    source=source,
                    r_com_A=None,
                    atoms_per_monomer=per_mono,
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    _noise_monomers(cfg.n_monomer_noise, cfg.cartesian_noise_sigma_A, "noise")
    _noise_monomers(cfg.n_monomer_large_noise, cfg.large_noise_sigma_A, "noise_large")

    if cfg.stretch_bonds:
        for _ in range(int(cfg.n_bond_stretch)):
            scale = 1.0 + float(cfg.bond_stretch_fraction) * float(rng.uniform(-1.0, 1.0))
            stretched = r_eq
            for i, j in cfg.stretch_bonds:
                stretched = _stretch_bond(stretched, int(i), int(j), scale)
            _add(
                _geometry(
                    z_m,
                    stretched,
                    kind="monomer",
                    source=cfg.stretch_source,
                    r_com_A=None,
                    atoms_per_monomer=per_mono,
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    z_d = np.concatenate([z_m, z_m])
    if cfg.com_windows_A:
        for _ in range(int(cfg.n_dimer_orientations)):
            rot_b = random_rotation(rng)
            pos_a = r_eq + rng.normal(0.0, cfg.cartesian_noise_sigma_A, size=r_eq.shape)
            pos_b = r_eq + rng.normal(0.0, cfg.cartesian_noise_sigma_A, size=r_eq.shape)
            for source, lo, hi in cfg.com_windows_A:
                for _j in range(int(cfg.n_com_per_region)):
                    com = float(rng.uniform(float(lo), float(hi)))
                    dimer = assemble_dimer(pos_a, pos_b, com_distance_A=com, rotation_b=rot_b)
                    _add(
                        _geometry(
                            z_d,
                            dimer,
                            kind="dimer",
                            source=str(source),
                            r_com_A=com,
                            atoms_per_monomer=per_dimer,
                            min_pair_distance_A=cfg.min_pair_distance_A,
                        )
                    )

    for _ in range(int(cfg.n_far_field)):
        rot_b = random_rotation(rng)
        dimer = assemble_dimer(
            r_eq, r_eq, com_distance_A=cfg.far_field_com_A, rotation_b=rot_b
        )
        _add(
            _geometry(
                z_d,
                dimer,
                kind="dimer",
                source="far_field",
                r_com_A=float(cfg.far_field_com_A),
                atoms_per_monomer=per_dimer,
                min_pair_distance_A=cfg.min_pair_distance_A,
            )
        )

    if cfg.include_dataset:
        for frame in dataset or ():
            n = len(frame)
            pos = np.asarray(frame.get_positions(), dtype=np.float64)
            numbers = np.asarray(frame.get_atomic_numbers(), dtype=np.int32)
            if n == n_mono:
                kind, per, r_com = "monomer", per_mono, None
            elif n == 2 * n_mono:
                kind, per, r_com = "dimer", per_dimer, _com_distance(pos, n_mono)
            else:
                continue
            _add(
                _geometry(
                    numbers,
                    pos,
                    kind=kind,
                    source=cfg.dataset_source,
                    r_com_A=r_com,
                    atoms_per_monomer=per,
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )
            jitter = pos + rng.normal(0.0, cfg.cartesian_noise_sigma_A, size=pos.shape)
            _add(
                _geometry(
                    numbers,
                    jitter,
                    kind=kind,
                    source=f"{cfg.dataset_source}_noise",
                    r_com_A=r_com,
                    atoms_per_monomer=per,
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    for extra in cfg.extra_extxyz:
        frames = ase_read(str(extra), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        for frame in frames:
            n = len(frame)
            pos = np.asarray(frame.get_positions(), dtype=np.float64)
            if n == n_mono:
                kind, per, r_com = "monomer", per_mono, None
            elif n == 2 * n_mono:
                kind, per, r_com = "dimer", per_dimer, _com_distance(pos, n_mono)
            else:
                continue
            _add(
                _geometry(
                    np.asarray(frame.get_atomic_numbers(), dtype=np.int32),
                    pos,
                    kind=kind,
                    source=f"extra:{Path(extra).name}",
                    r_com_A=r_com,
                    atoms_per_monomer=per,
                    min_pair_distance_A=cfg.min_pair_distance_A,
                )
            )

    if int(cfg.n_global_rotations_extra) > 0:
        extras: list[Geometry] = []
        for geo in list(geos):
            for _ in range(int(cfg.n_global_rotations_extra)):
                rot = random_rotation(rng)
                extras.append(
                    replace(
                        geo,
                        numbers=geo.numbers.copy(),
                        positions=geo.positions @ rot.T,
                        source=geo.source + "+rot",
                    )
                )
        geos.extend(extras)

    if not geos:
        raise RuntimeError(f"geometry pool is empty after clash filter (rejected={rejected})")
    return geos
