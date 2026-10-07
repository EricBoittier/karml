"""Cluster construction and CHARMM prep shared by JAX-MD and PyCHARMM.

Geometry, PSF, and calculator setup live here so those engines do not import
the removed ASE molecular-dynamics program.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from ase import Atoms

import karml.interfaces.pycharmmInterface.import_pycharmm as pyci
from karml.cli.base import resolve_checkpoint_paths
from karml.interfaces.pycharmmInterface.import_pycharmm import reset_block, reset_block_no_internal
from karml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
)
from karml.interfaces.pycharmmInterface.karml_calculator import CutoffParameters, setup_calculator
from karml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    _get_actual_psf_charges,
)
from karml.interfaces.pycharmmInterface.mlpot.cli_common import parse_composition
from karml.utils.geometry_checks import assert_no_intermonomer_atom_overlap
from karml.interfaces.pycharmmInterface.utils import get_Z_from_psf
from karml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster
from karml.paths import default_meoh_template_pdb

_parse_composition = parse_composition

# PyCHARMM loads libcharmm as an import side effect. Keep this module
# importable on hosts without that optional native runtime and cross the boundary
# only when a CHARMM-backed builder or minimizer is actually used.
param = psf = read = settings = coor = gen = ic = charmm_minimize = None


def _load_pycharmm_modules() -> None:
    """Load and cache PyCHARMM modules for CHARMM-backed operations."""
    global param, psf, read, settings, coor, gen, ic, charmm_minimize

    if read is not None:
        return
    from karml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    if not ensure_pycharmm_loaded():
        raise RuntimeError(
            "This operation requires the optional CHARMM runtime. Set CHARMM_LIB_DIR "
            "to the directory containing libcharmm before using CHARMM-backed builders "
            "or minimization."
        )
    import pycharmm as _pycharmm
    import pycharmm.coor as _coor
    import pycharmm.generate as _gen
    import pycharmm.ic as _ic
    import pycharmm.minimize as _minimize
    import pycharmm.param as _param
    import pycharmm.psf as _psf
    import pycharmm.read as _read
    import pycharmm.settings as _settings

    param, psf, read, settings = _param, _psf, _read, _settings
    coor, gen, ic, charmm_minimize = _coor, _gen, _ic, _minimize
    pyci.pycharmm = _pycharmm
    pyci.read = _read
    pyci.settings = _settings
    pyci.psf = _psf

reset_block()
reset_block_no_internal()


def _has_resolved_geometry(coords: np.ndarray, min_span: float = 1.0e-4) -> bool:
    """Return True if a residue has more than origin/identical coordinates.

    The span test catches an unresolved IC table, which collapses every atom of
    a polyatomic residue onto the same point. A **monoatomic** residue has zero
    span by definition -- there is no internal geometry to resolve -- so the
    same test would reject a perfectly good single point and make every
    single-atom residue unbuildable (AR1/KR1/XE1, and the monoatomic ions
    CLA/POT/SOD/LIT). For one atom, finiteness is the whole check.
    """
    if coords.size == 0 or not np.all(np.isfinite(coords)):
        return False
    if coords.shape[0] == 1:
        return True
    return bool(np.max(np.ptp(coords, axis=0)) > min_span)


def _read_cgenff_toppar(*, enable_drude: bool = False) -> None:
    from karml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_toppar

    read_cgenff_toppar(enable_drude=enable_drude)


def _reset_pycharmm_system() -> None:
    _load_pycharmm_modules()
    pyci.pycharmm.lingo.charmm_script("DELETE ATOM SELE ALL END")
    reset_block()
    reset_block_no_internal()
    reset_block()


def _make_res_minimize(nbxmod: int, nstep: int = 1000) -> None:
    """Run the same nonbonded/minimization recipe used by make-res."""
    from karml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

    pyci.pycharmm_quiet()
    apply_vacuum_nbonds(nbxmod=nbxmod)
    charmm_minimize.run_abnr(nstep=nstep, tolenr=1e-3, tolgrd=1e-3)


def _generate_residue_with_make_res_recipe(
    residue: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Generate one residue in PyCHARMM using make-res style coordinate relaxation."""
    from karml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    import pycharmm
    pycharmm.settings.set_bomb_level(-5)
    read.sequence_string(residue)
    from karml.interfaces.pycharmmInterface.heme_library import (
        heme_reference_positions,
        is_heme_library_residue,
        segment_terminal_patches,
    )

    gen.new_segment(seg_name="TMP", setup_ic=True, **segment_terminal_patches())
    from karml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill

    ic_prm_fill(replace_all=True)
    ic.build()

    initial = np.array(coor.get_positions().to_numpy(dtype=float), dtype=float, copy=True)
    atom_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    if initial.shape[0] != len(atom_names):
        raise RuntimeError(
            f"PSF atom-name count mismatch while generating {residue}: "
            f"{len(atom_names)} vs positions {initial.shape[0]}"
        )

    # The heme stream IC table has zero bond lengths, so ic.build places nothing.
    # A random cloud plus live CHARMM electrostatics is a 10^10 kcal/mol/Å gradient.
    if is_heme_library_residue(residue):
        placed = heme_reference_positions(atom_names)
        if placed is not None:
            coor.set_positions(pd.DataFrame(placed, columns=["x", "y", "z"]))
            z = np.asarray(get_Z_from_psf(), dtype=int)
            return placed, atom_names, z

    from karml.interfaces.pycharmmInterface.heme_electronic import is_protein_ion

    if is_protein_ion(residue) and initial.shape[0] == 1:
        # A sodium has no internal geometry. The heme build seats it on a
        # carboxylate after the cluster is assembled.
        coords = np.zeros((1, 3), dtype=float)
        coor.set_positions(pd.DataFrame(coords, columns=["x", "y", "z"]))
        z = np.asarray(get_Z_from_psf(), dtype=int)
        return coords, atom_names, z

    # make-res deliberately escapes incomplete IC tables by randomizing coordinates
    # before two CHARMM minimization passes.
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(residue.upper())) + 1729
    rng = np.random.default_rng(seed)
    xyz = pd.DataFrame(2.0 * rng.random(initial.shape), columns=["x", "y", "z"])
    coor.set_positions(xyz)
    _make_res_minimize(nbxmod=1)

    xyz = np.array(coor.get_positions().to_numpy(dtype=float), dtype=float, copy=True)
    xyz *= rng.random(xyz.shape)
    coor.set_positions(pd.DataFrame(xyz, columns=["x", "y", "z"]))
    _make_res_minimize(nbxmod=5)

    coords = np.array(coor.get_positions().to_numpy(dtype=float), dtype=float, copy=True)
    if not _has_resolved_geometry(coords):
        raise RuntimeError(f"PyCHARMM make-res coordinate generation failed for residue {residue!r}")
    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(z.shape[0]) != len(atom_names):
        raise RuntimeError(
            f"PSF Z count mismatch for {residue}: {z.shape[0]} vs {len(atom_names)} atom names"
        )
    return coords, atom_names, z


def _tmark() -> float:
    return time.perf_counter()


def _tlog(msg: str, log_lines: list[str] | None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_lines is not None:
        log_lines.append(line)


def _cubic_box_length(positions: np.ndarray, ml_cutoff: float, pad: float = 10.0) -> float:
    """Side length (Å) so extent + 2*(cutoff+pad) fits; MIC-safe vs model cutoff."""
    r_cut = float(ml_cutoff) + 2.0
    d = np.ptp(positions, axis=0)
    span = float(np.max(d))
    return max(span + 2.0 * (r_cut + pad), 50.0)


def _enforce_min_com_separation(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    min_com_distance: float,
    max_passes: int = 10,
) -> np.ndarray:
    """Push monomer COMs apart so all pair distances are >= min_com_distance."""
    n_molecules = int(len(monomer_offsets) - 1)
    if min_com_distance <= 0.0 or n_molecules <= 1:
        return positions
    pos = np.asarray(positions, dtype=float).copy()
    for _ in range(max_passes):
        coms = np.zeros((n_molecules, 3), dtype=float)
        for i in range(n_molecules):
            s = int(monomer_offsets[i])
            e = int(monomer_offsets[i + 1])
            coms[i] = pos[s:e].mean(axis=0)
        moved = False
        for i in range(n_molecules):
            for j in range(i + 1, n_molecules):
                dvec = coms[j] - coms[i]
                dist = float(np.linalg.norm(dvec))
                if dist < 1e-12:
                    dvec = np.array([1.0, 0.0, 0.0], dtype=float)
                    dist = 1e-12
                if dist < min_com_distance:
                    direction = dvec / dist
                    delta = 0.5 * (min_com_distance - dist) * direction
                    si, ei = int(monomer_offsets[i]), int(monomer_offsets[i + 1])
                    sj, ej = int(monomer_offsets[j]), int(monomer_offsets[j + 1])
                    pos[si:ei] -= delta
                    pos[sj:ej] += delta
                    coms[i] -= delta
                    coms[j] += delta
                    moved = True
        if not moved:
            break
    return pos


def _randomize_monomer_com_positions(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    *,
    spacing: float,
    min_com_distance: float,
    seed: int,
) -> np.ndarray:
    """Place molecule centers at reproducible random 3D positions."""
    n_molecules = int(len(monomer_offsets) - 1)
    if n_molecules <= 1:
        return positions

    rng = np.random.default_rng(int(seed))
    min_dist = float(max(0.0, min_com_distance))
    side = max(float(spacing), min_dist, 1.0) * max(1.0, n_molecules ** (1.0 / 3.0))
    targets: list[np.ndarray] = []
    max_attempts = 5000

    while True:
        targets.clear()
        for _i in range(n_molecules):
            for _attempt in range(max_attempts):
                candidate = rng.uniform(0.0, side, size=3)
                if all(float(np.linalg.norm(candidate - prev)) >= min_dist for prev in targets):
                    targets.append(candidate)
                    break
            else:
                break
        if len(targets) == n_molecules:
            break
        side *= 1.25

    target_arr = np.asarray(targets, dtype=float)
    target_arr -= target_arr.mean(axis=0)
    randomized = np.asarray(positions, dtype=float).copy()
    for i, target in enumerate(target_arr):
        s = int(monomer_offsets[i])
        e = int(monomer_offsets[i + 1])
        com = randomized[s:e].mean(axis=0)
        randomized[s:e] += target - com
    return randomized


def _load_packmol_sphere_positions(
    pdb_path: str | Path,
    atoms_per_list: list[int],
    psf_atom_names: list[str],
) -> np.ndarray:
    from karml.interfaces.pycharmmInterface.packmol_placement import (
        assign_packmol_pdb_to_psf_order,
    )

    return assign_packmol_pdb_to_psf_order(
        pdb_path,
        psf_atom_names,
        atoms_per_list,
    )


def _residue_geometries_for_composition(
    composition: list[tuple[str, int]],
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """Relaxed monomer coords, atom names, and Z per residue type (SD-only cluster recipe)."""
    from karml.cli.run.md_pbc_suite.cluster import relax_monomer_geometry_for_cluster

    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for residue, _count in composition:
        if residue not in residue_geometries:
            residue_geometries[residue] = relax_monomer_geometry_for_cluster(residue)
    return residue_geometries


def _residue_geometries_for_packmol(
    composition: list[tuple[str, int]],
    *,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, list[str], np.ndarray]]:
    """Minimized monomer geometries (PSF order) for Packmol input PDBs."""
    from karml.cli.run.md_pbc_suite.cluster import build_minimized_monomer_for_packmol

    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for residue, _count in composition:
        if residue not in residue_geometries:
            residue_geometries[residue] = build_minimized_monomer_for_packmol(
                residue,
                nstep_sd=charmm_sd_steps,
                nstep_abnr=charmm_abnr_steps,
                tolenr=charmm_tolenr,
                tolgrd=charmm_tolgrd,
                verbose=verbose,
            )
    return residue_geometries


def _build_cluster_psf_topology_only(
    composition: list[tuple[str, int]],
    *,
    expected_atoms: int,
    atoms_per_list: list[int],
    residue_labels: list[str],
) -> np.ndarray:
    """Build cluster PSF from composition without make-res monomer geometry work."""
    if int(sum(atoms_per_list)) != int(expected_atoms):
        raise RuntimeError(
            f"Handoff layout atom count ({int(sum(atoms_per_list))}) does not match "
            f"expected_atoms={int(expected_atoms)}"
        )
    if len(atoms_per_list) != len(residue_labels):
        raise RuntimeError(
            "Handoff residue label count does not match per-monomer atom layout"
        )

    sequence_items: list[str] = []
    for residue, count in composition:
        sequence_items.extend([residue] * int(count))
    sequence = " ".join(sequence_items)
    from karml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(sequence)
    from karml.interfaces.pycharmmInterface.heme_library import segment_terminal_patches

    gen.new_segment(seg_name="CLST", setup_ic=True, **segment_terminal_patches())
    from karml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill

    ic_prm_fill(replace_all=True)
    ic.build()

    z = np.asarray(get_Z_from_psf(), dtype=int)
    if int(len(z)) != int(expected_atoms):
        raise RuntimeError(
            f"Composition-derived PSF atom count ({len(z)}) does not match handoff "
            f"geometry ({int(expected_atoms)}). Composition={composition}"
        )
    return z


def _build_cluster_psf_from_composition(
    composition: list[tuple[str, int]],
    *,
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    """Build CHARMM PSF for a mixed cluster; return Z, PSF atom names, atoms/monomer, residue order."""
    if residue_geometries is None:
        residue_geometries = _residue_geometries_for_composition(composition)
    else:
        for residue, _count in composition:
            if residue not in residue_geometries:
                from karml.cli.run.md_pbc_suite.cluster import relax_monomer_geometry_for_cluster

                residue_geometries[residue] = relax_monomer_geometry_for_cluster(residue)

    sequence_items: list[str] = []
    for residue, count in composition:
        sequence_items.extend([residue] * int(count))
    sequence = " ".join(sequence_items)
    from karml.interfaces.pycharmmInterface.mlpot.setup import prepare_charmm_vacuum

    _reset_pycharmm_system()
    prepare_charmm_vacuum()
    _read_cgenff_toppar()
    read.sequence_string(sequence)
    from karml.interfaces.pycharmmInterface.heme_library import segment_terminal_patches

    gen.new_segment(seg_name="CLST", setup_ic=True, **segment_terminal_patches())
    from karml.interfaces.pycharmmInterface.nbonds_config import ic_prm_fill

    ic_prm_fill(replace_all=True)
    ic.build()

    z = np.asarray(get_Z_from_psf(), dtype=int)
    atom_names = [str(x) for x in np.asarray(psf.get_atype(), dtype=str)]
    atoms_per_list: list[int] = []
    ordered_residue_names: list[str] = []
    for residue, count in composition:
        n_atoms_res = int(residue_geometries[residue][0].shape[0])
        for _ in range(int(count)):
            atoms_per_list.append(int(n_atoms_res))
            ordered_residue_names.append(residue)
    expected_atoms = int(np.sum(np.asarray(atoms_per_list, dtype=int)))
    if expected_atoms != int(len(atom_names)):
        raise RuntimeError(
            f"Composition-derived atom count ({expected_atoms}) does not match PSF "
            f"({len(atom_names)}). Composition={composition}"
        )

    offsets = np.zeros(len(atoms_per_list) + 1, dtype=int)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    for i, residue in enumerate(ordered_residue_names):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        residue_coords, residue_atom_names, _residue_z = residue_geometries[residue]
        local_atom_names = atom_names[s:e]
        if local_atom_names != residue_atom_names:
            raise RuntimeError(
                f"Atom order mismatch for {residue}: PSF has {local_atom_names}, "
                f"make-res produced {residue_atom_names}"
            )
        if int(residue_coords.shape[0]) != e - s:
            raise RuntimeError(f"Atom count mismatch for {residue} at monomer {i}")
    return z, atom_names, atoms_per_list, ordered_residue_names


def _build_cluster_from_composition_packmol(
    *,
    composition: list[tuple[str, int]],
    placement: str = "cube",
    center: tuple[float, float, float],
    cube_side: float | None = None,
    radius: float | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    scratch_dir: str | Path | None = None,
    verbose: bool = True,
    reuse_packmol_cache: bool = True,
    packmol_cache_dir: str | Path | None = None,
    force_rebuild_packmol_cache: bool = False,
    sim_cell_side: float | None = None,
    box_sizing_source: str | None = None,
    packmol_padding_A: float | None = None,
    spacing: float | None = None,
    prep_gate_settings: dict[str, Any] | None = None,
    quiet: bool = False,
    geometry_store: Any | None = None,
    monomer_pdb_templates: dict[str, Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    from karml.cli.run.md_pbc_suite.cluster import build_packmol_composition_cluster

    return build_packmol_composition_cluster(
        composition=composition,
        placement=placement,
        center=center,
        cube_side=cube_side,
        radius=radius,
        tolerance=tolerance,
        seed=seed,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
        scratch_dir=scratch_dir,
        verbose=verbose,
        reuse_packmol_cache=reuse_packmol_cache,
        packmol_cache_dir=packmol_cache_dir,
        force_rebuild_packmol_cache=force_rebuild_packmol_cache,
        sim_cell_side=sim_cell_side,
        box_sizing_source=box_sizing_source,
        packmol_padding_A=packmol_padding_A,
        spacing=spacing,
        prep_gate_settings=prep_gate_settings,
        quiet=quiet,
        geometry_store=geometry_store,
        monomer_pdb_templates=monomer_pdb_templates,
    )


def _build_cluster_from_composition_pyxtal(
    *,
    composition: list[tuple[str, int]],
    space_group: int = 14,
    dimension: int = 3,
    factor: float = 1.0,
    unit_stoichiometry: list[int] | None = None,
    supercell_reps: tuple[int, int, int] | None = None,
    seed: int | None = None,
    max_attempts: int = 20,
    charmm_sd_steps: int = 50,
    charmm_abnr_steps: int = 100,
    charmm_tolenr: float = 1e-3,
    charmm_tolgrd: float = 1e-3,
    scratch_dir: str | Path | None = None,
    verbose: bool = True,
    optimize_ase: bool = False,
    optimize_ase_emt: bool = False,
    trim_to_composition: bool = True,
    geometry_store: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    from karml.cli.run.md_pbc_suite.cluster import build_pyxtal_composition_cluster

    return build_pyxtal_composition_cluster(
        composition=composition,
        space_group=space_group,
        dimension=dimension,
        factor=factor,
        unit_stoichiometry=unit_stoichiometry,
        supercell_reps=supercell_reps,
        seed=seed,
        max_attempts=max_attempts,
        charmm_sd_steps=charmm_sd_steps,
        charmm_abnr_steps=charmm_abnr_steps,
        charmm_tolenr=charmm_tolenr,
        charmm_tolgrd=charmm_tolgrd,
        scratch_dir=scratch_dir,
        verbose=verbose,
        optimize_ase=optimize_ase,
        optimize_ase_emt=optimize_ase_emt,
        trim_to_composition=trim_to_composition,
        geometry_store=geometry_store,
    )


def resolve_cluster_packmol(args: argparse.Namespace) -> bool:
    from karml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_use

    return resolve_packmol_use(
        composition=getattr(args, "composition", None),
        packmol=getattr(args, "packmol", None),
        pyxtal=getattr(args, "pyxtal", None),
        builder=getattr(args, "builder", None),
    )


def resolve_cluster_pyxtal(args: argparse.Namespace) -> bool:
    from karml.interfaces.pyxtal_placement import resolve_pyxtal_use

    return resolve_pyxtal_use(
        composition=getattr(args, "composition", None),
        pyxtal=getattr(args, "pyxtal", None),
        builder=getattr(args, "builder", None),
    )


def resolve_cluster_packmol_sphere(args: argparse.Namespace) -> bool:
    """Backward-compatible alias: True when any Packmol placement is active."""
    return resolve_cluster_packmol(args)


def packmol_sphere_center_from_args(args: argparse.Namespace) -> tuple[float, float, float]:
    center = getattr(args, "packmol_center", None)
    if center is not None:
        if len(center) != 3:
            raise ValueError("--packmol-center requires three floats: CX CY CZ")
        return (float(center[0]), float(center[1]), float(center[2]))
    return (0.0, 0.0, 0.0)


def certified_box_geometry_requested(args: argparse.Namespace | None) -> bool:
    """True when suite should load liquid-box / mini PSF+CRD instead of Packmol."""
    if args is None:
        return False
    return (
        getattr(args, "from_psf", None) is not None
        and getattr(args, "from_crd", None) is not None
    )


def _residue_labels_from_loaded_psf(atoms_per_list: list[int]) -> list[str]:
    import pycharmm.psf as psf

    if hasattr(psf, "get_res"):
        res_all = [str(x).strip().upper() for x in psf.get_res()]
        labels: list[str] = []
        offset = 0
        for n in atoms_per_list:
            labels.append(res_all[offset] if 0 <= offset < len(res_all) else "UNK")
            offset += int(n)
        return labels
    return ["UNK"] * len(atoms_per_list)


def _maybe_apply_certified_box_json(args: argparse.Namespace, crd_path: Path) -> float:
    """Prefer sibling ``box.json`` cubic side over YAML ``--box-size`` / density auto.

    Returns the certified side (Å). Raises if ``box.json`` is missing or lacks a
    positive ``box_side_A``. Density-auto sizing can change the cell
    (e.g. 28.17 → 27.75 Å at ρ=1.36).
    """
    import json

    box_json = Path(crd_path).expanduser().resolve().parent / "box.json"
    if not box_json.is_file():
        # Allow an explicit CLI/YAML --box-size when mini/handoff CRDs lack box.json.
        prev = getattr(args, "box_size", None)
        if prev is not None and float(prev) > 0.0:
            side_f = float(prev)
            args.box_auto = None
            args.target_density_g_cm3 = None
            args.bulk_density_fraction = None
            if not getattr(args, "quiet", False):
                print(
                    f"Certified box: L={side_f:.3f} Å from --box-size "
                    f"(no sibling box.json at {box_json})",
                    flush=True,
                )
            return side_f
        raise FileNotFoundError(
            f"Certified --from-crd requires sibling box.json next to {crd_path} "
            f"(expected {box_json}). Pass an explicit --box-size matching the "
            "liquid-box REPORT if box.json is unavailable."
        )
    try:
        data = json.loads(box_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read certified box.json {box_json}: {exc}") from exc
    side = data.get("box_side_A")
    if side is None:
        side = data.get("final_cubic_side_A")
    if side is None:
        raise ValueError(
            f"Certified box.json {box_json} has no box_side_A / final_cubic_side_A"
        )
    side_f = float(side)
    if side_f <= 0.0:
        raise ValueError(f"Certified box_side_A must be positive, got {side_f}")
    prev = getattr(args, "box_size", None)
    args.box_size = side_f
    # Density-auto must not replace the certified cell if box_size is cleared later.
    args.box_auto = None
    args.target_density_g_cm3 = None
    args.bulk_density_fraction = None
    rho = data.get("density_g_cm3")
    if not getattr(args, "quiet", False):
        rho_txt = f", ρ={float(rho):.4f} g/cm³" if rho is not None else ""
        if prev is None:
            print(
                f"Certified box: L={side_f:.3f} Å{rho_txt} from {box_json}",
                flush=True,
            )
        elif abs(float(prev) - side_f) > 1.0e-3:
            print(
                f"Certified box: overriding --box-size {float(prev):.3f} → "
                f"{side_f:.3f} Å{rho_txt} from {box_json}",
                flush=True,
            )
        else:
            print(
                f"Certified box: L={side_f:.3f} Å{rho_txt} (matches --box-size)",
                flush=True,
            )
    return side_f


def cluster_geometry_from_certified_artifacts(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int] | None]:
    """Load PSF+CRD (liquid-box / mini) for ASE/JAX-MD; skip Packmol rebuild."""
    from karml.interfaces.pycharmmInterface.mlpot.setup import load_cluster_from_artifacts
    from karml.interfaces.pycharmmInterface.mlpot.trimer_scan import (
        atoms_per_monomer_from_psf,
    )

    crd = Path(getattr(args, "from_crd")).expanduser().resolve()
    certified_side = _maybe_apply_certified_box_json(args, crd)
    setattr(args, "_certified_box_side_A", float(certified_side))
    z, r0, _n_mol, _tag = load_cluster_from_artifacts(args)
    atoms_per_list = [int(x) for x in atoms_per_monomer_from_psf()]
    if sum(atoms_per_list) != int(len(z)):
        raise ValueError(
            f"PSF resid atom counts sum to {sum(atoms_per_list)} but Z has {len(z)} atoms"
        )
    if getattr(args, "composition", None):
        composition = _parse_composition(args.composition)
        residue_labels = [res for res, cnt in composition for _ in range(int(cnt))]
        composition_summary = {res: int(cnt) for res, cnt in composition}
        if len(residue_labels) != len(atoms_per_list):
            residue_labels = _residue_labels_from_loaded_psf(atoms_per_list)
            composition_summary = None
    else:
        residue_labels = _residue_labels_from_loaded_psf(atoms_per_list)
        composition_summary = None
    if not getattr(args, "quiet", False):
        print(
            f"Loading certified geometry from {Path(args.from_psf).name} + "
            f"{crd.name} ({len(z)} atoms, {len(atoms_per_list)} monomers); "
            "skipped Packmol/cluster build",
            flush=True,
        )
    return (
        np.asarray(z, dtype=int),
        np.asarray(r0, dtype=float),
        atoms_per_list,
        residue_labels,
        composition_summary,
    )


def resolve_cluster_geometry(
    args: argparse.Namespace,
    handoff: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int] | None]:
    """Build or load cluster geometry; skip Packmol when continuing from handoff."""
    if handoff is not None:
        from karml.cli.run.md_handoff import cluster_geometry_from_handoff

        z, r0, atoms_per_list, residue_labels, composition_summary = (
            cluster_geometry_from_handoff(
                handoff,
                composition=getattr(args, "composition", None),
                n_molecules=int(getattr(args, "n_molecules", 1) or 1),
            )
        )
        if not getattr(args, "quiet", False):
            print(
                f"Continuing from handoff ({len(z)} atoms); skipped Packmol/cluster build",
                flush=True,
            )
        return z, r0, atoms_per_list, residue_labels, composition_summary
    if certified_box_geometry_requested(args):
        return cluster_geometry_from_certified_artifacts(args)
    if getattr(args, "from_psf", None) or getattr(args, "from_crd", None):
        raise ValueError(
            "--from-psf and --from-crd must be provided together to load a "
            "certified liquid-box (or mini) geometry on ase/jaxmd."
        )
    from karml.interfaces.pycharmmInterface.mlpot.composition_spec import (
        apply_from_pdb_alias,
        composition_mode,
        parse_composition_entries,
    )

    apply_from_pdb_alias(args)
    if getattr(args, "composition", None):
        entries = parse_composition_entries(str(args.composition))
        if composition_mode(entries) == "full_system_pdb":
            from karml.interfaces.pycharmmInterface.mlpot.setup import load_cluster_from_pdb

            z, r0, n_mol, _tag = load_cluster_from_pdb(args)
            atoms_per_list = list(getattr(args, "_cluster_atoms_per_list", []) or [])
            residue_labels = list(getattr(args, "_cluster_residue_labels", []) or [])
            composition_summary = dict(
                getattr(args, "_cluster_composition_summary", {}) or {}
            )
            if not atoms_per_list:
                atoms_per_list = [int(len(z) // max(n_mol, 1))] * int(n_mol)
            if not residue_labels:
                residue_labels = ["UNK"] * int(n_mol)
            return z, r0, atoms_per_list, residue_labels, composition_summary or None
    return build_initial_cluster_from_args(args)


def build_initial_cluster_from_args(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str], dict[str, int] | None]:
    """Build cluster geometry; returns z, r0, atoms_per_list, residue_labels, composition_summary."""
    from karml.interfaces.pycharmmInterface.packmol_placement import (
        packmol_center_for_cold_start,
        resolve_packmol_cube_side_from_args,
        resolve_packmol_placement_mode,
        resolve_packmol_sphere_radius,
    )
    from karml.interfaces.pyxtal_placement import parse_supercell_reps

    use_pyxtal = resolve_cluster_pyxtal(args)
    use_packmol = resolve_cluster_packmol(args)
    if use_pyxtal and not args.composition:
        raise ValueError("PyXtal placement requires --composition (e.g. MEOH:8).")
    if use_packmol and not args.composition:
        raise ValueError(
            "Packmol placement requires --composition (e.g. MEOH:30) "
            "and --box-size (Å) for cube packing."
        )

    if args.composition:
        from karml.interfaces.pycharmmInterface.mlpot.composition_spec import (
            resolve_composition_plan,
        )

        _entries, mode, composition, monomer_pdb_templates = resolve_composition_plan(
            str(args.composition),
            builder=getattr(args, "builder", None),
            packmol=getattr(args, "packmol", None),
            pyxtal=getattr(args, "pyxtal", None) if use_pyxtal else False,
        )
        if mode == "full_system_pdb":
            raise RuntimeError(
                "full-system PDB composition should be handled before "
                "build_initial_cluster_from_args"
            )
        composition_summary = {res: int(cnt) for res, cnt in composition}
        if use_pyxtal:
            supercell_reps = None
            if getattr(args, "pyxtal_supercell", None):
                supercell_reps = parse_supercell_reps(str(args.pyxtal_supercell))
            z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition_pyxtal(
                composition=composition,
                space_group=int(getattr(args, "pyxtal_spg", 14)),
                dimension=int(getattr(args, "pyxtal_dim", 3)),
                factor=float(getattr(args, "pyxtal_factor", 1.0)),
                unit_stoichiometry=getattr(args, "pyxtal_stoichiometry", None),
                supercell_reps=supercell_reps,
                seed=int(getattr(args, "seed", 0)),
                max_attempts=int(getattr(args, "pyxtal_attempts", 20)),
                charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 50)),
                charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
                charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                scratch_dir=(
                    Path(args.output_dir) / "pyxtal_cluster"
                    if getattr(args, "output_dir", None) is not None
                    else None
                ),
                verbose=not getattr(args, "quiet", False),
                optimize_ase=bool(getattr(args, "optimize_pyxtal", False)),
                optimize_ase_emt=bool(getattr(args, "optimize_pyxtal_emt", False)),
                trim_to_composition=bool(getattr(args, "pyxtal_trim", True)),
            )
            print(
                f"Cluster built with PyXtal: spg={int(getattr(args, 'pyxtal_spg', 14))} "
                f"dim={int(getattr(args, 'pyxtal_dim', 3))}"
            )
        elif use_packmol:
            from karml.interfaces.pycharmmInterface.packmol_cache import (
                packmol_prep_settings_from_namespace,
            )

            placement = resolve_packmol_placement_mode(
                packmol_placement=getattr(args, "packmol_placement", None),
                packmol_sphere=getattr(args, "packmol_sphere", None),
            )
            center = packmol_center_for_cold_start(args)
            tolerance = float(getattr(args, "packmol_tolerance", 2.0))
            cube_side: float | None = None
            radius: float | None = None
            if placement == "sphere":
                radius = resolve_packmol_sphere_radius(
                    getattr(args, "packmol_radius", None),
                    getattr(args, "flat_bottom_radius", None),
                )
            else:
                cube_side = resolve_packmol_cube_side_from_args(args)
            z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition_packmol(
                composition=composition,
                placement=placement,
                center=center,
                cube_side=cube_side,
                radius=radius,
                tolerance=tolerance,
                seed=int(getattr(args, "seed", 0)),
                charmm_sd_steps=int(getattr(args, "charmm_sd_steps", 50)),
                charmm_abnr_steps=int(getattr(args, "charmm_abnr_steps", 100)),
                charmm_tolenr=float(getattr(args, "charmm_tolenr", 1e-3)),
                charmm_tolgrd=float(getattr(args, "charmm_tolgrd", 1e-3)),
                scratch_dir=(
                    Path(args.output_dir) / "packmol_cluster"
                    if getattr(args, "output_dir", None) is not None
                    else None
                ),
                verbose=not getattr(args, "quiet", False),
                reuse_packmol_cache=bool(getattr(args, "reuse_packmol_cache", True)),
                packmol_cache_dir=getattr(args, "packmol_cache_dir", None),
                force_rebuild_packmol_cache=bool(
                    getattr(args, "rebuild_packmol", False)
                ),
                sim_cell_side=getattr(args, "_cold_start_sim_cell_side_A", None),
                box_sizing_source=getattr(args, "_cold_start_box_sizing_source", None),
                packmol_padding_A=getattr(args, "_cold_start_packmol_padding_A", None),
                spacing=float(getattr(args, "spacing", 5.0)),
                prep_gate_settings=packmol_prep_settings_from_namespace(args),
                quiet=bool(getattr(args, "quiet", False)),
                monomer_pdb_templates=monomer_pdb_templates,
            )
        else:
            placement = resolve_packmol_placement_mode(
                packmol_placement=getattr(args, "packmol_placement", None),
                packmol_sphere=getattr(args, "packmol_sphere", None),
            )
            if mode == "packmol_pdb":
                raise ValueError(
                    "PDB composition tokens require Packmol; remove --no-packmol"
                )
            center = packmol_center_for_cold_start(args)
            cube_side: float | None = None
            radius: float | None = None
            if placement == "sphere":
                radius = resolve_packmol_sphere_radius(
                    getattr(args, "packmol_radius", None),
                    getattr(args, "flat_bottom_radius", None),
                )
            elif getattr(args, "builder", None) != "gas":
                cube_side = resolve_packmol_cube_side_from_args(args)
            z, r0, atoms_per_list, residue_labels = _build_cluster_from_composition(
                composition=composition,
                spacing=args.spacing,
                placement=placement,
                center=center,
                cube_side=cube_side,
                radius=radius,
                seed=int(getattr(args, "seed", 0)),
            )
        return z, r0, atoms_per_list, residue_labels, composition_summary

    z, r0 = _build_psf_ordered_cluster(
        "MEOH",
        args.n_molecules,
        args.spacing,
        template_pdb=args.template_pdb.expanduser().resolve(),
    )
    n_atoms_tmp = len(z)
    atoms_per_uniform = n_atoms_tmp // args.n_molecules
    atoms_per_list = [int(atoms_per_uniform)] * int(args.n_molecules)
    residue_labels = ["MEOH"] * int(args.n_molecules)
    composition_summary = {"MEOH": int(args.n_molecules)}
    return z, r0, atoms_per_list, residue_labels, composition_summary


def _build_cluster_from_composition(
    *,
    composition: list[tuple[str, int]],
    spacing: float,
    placement: str = "cube",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cube_side: float | None = None,
    radius: float | None = None,
    seed: int | None = None,
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    # Build monomer templates once; do not call make-res again after the cluster PSF
    # (_generate_residue_with_make_res_recipe resets CHARMM and deletes the cluster).
    if residue_geometries is None:
        residue_geometries = _residue_geometries_for_composition(composition)
    z, atom_names, atoms_per_list, ordered_residue_names = _build_cluster_psf_from_composition(
        composition,
        residue_geometries=residue_geometries,
    )

    n_molecules = len(atoms_per_list)
    offsets = np.zeros(n_molecules + 1, dtype=int)
    offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    expected_atoms = int(offsets[-1])
    shifted = coor.get_positions().to_numpy(dtype=float).copy()
    if int(shifted.shape[0]) != expected_atoms:
        raise RuntimeError(
            f"CHARMM coordinate count ({shifted.shape[0]}) != cluster PSF atom count "
            f"({expected_atoms}). The cluster PSF may have been cleared after build."
        )
    from karml.interfaces.pycharmmInterface.grid_placement import (
        grid_centers_cube,
        grid_centers_sphere,
        resolve_grid_placement_mode,
    )

    grid_mode = resolve_grid_placement_mode(placement=placement)
    if grid_mode == "sphere":
        if radius is None:
            raise ValueError("Spherical grid placement requires a positive radius.")
        centers = grid_centers_sphere(
            n_molecules,
            center=center,
            radius=float(radius),
            seed=seed,
        )
    else:
        centers = grid_centers_cube(
            n_molecules,
            center=center,
            side=cube_side,
            spacing=spacing,
            seed=seed,
        )

    # Random SO(3) per monomer when a seed is provided (evenly spaced COMs,
    # randomized orientations — preferred for dilute PBC smokes).
    rng = np.random.default_rng(int(seed)) if seed is not None else None
    if rng is not None:
        from karml.utils.geometry_checks import _oriented_repack_template

    for i in range(n_molecules):
        s = int(offsets[i])
        e = int(offsets[i + 1])
        residue = ordered_residue_names[i]
        residue_coords, _residue_atom_names, _residue_z = residue_geometries[residue]
        block = np.asarray(residue_coords, dtype=float).copy()
        com0 = block.mean(axis=0)
        internal = block - com0
        if rng is not None:
            internal = _oriented_repack_template(
                internal, random_rotations=True, rng=rng
            )
        shifted[s:e] = internal + centers[i]
    from karml.interfaces.pycharmmInterface.heme_electronic import seat_heme_counterions

    shifted = seat_heme_counterions(
        shifted, atom_names, ordered_residue_names, atoms_per_list
    )
    coor.set_positions(pd.DataFrame(shifted, columns=["x", "y", "z"]))
    return z, shifted, atoms_per_list, ordered_residue_names


def _factory_karml(
    *,
    z: np.ndarray,
    r: np.ndarray,
    n_mol: int,
    atoms_per: int | list[int],
    base_ckpt_dir: Path,
    ml_cut: float,
    mm_sw: float,
    mm_cut: float,
    cell_scalar: float | None,
    verbose: bool,
    jax_md_capacity_multiplier: float,
    jax_md_capacity_growth_factor: float,
    jax_md_max_overflow_retries: int,
    jax_md_overflow_fallback_to_cell_list: bool,
    jax_md_update_interval: int,
    jax_md_skin_distance: float,
    max_pairs: int,
    do_ml: bool = True,
    do_ml_dimer: bool = True,
    do_mm: bool = True,
    timings: dict[str, float] | None = None,
    flat_bottom_radius: float | None = None,
    flat_bottom_force_const: float = 1.0,
    flat_bottom_mode: str = "system",
    min_com_restraint_distance: float | None = None,
    min_com_restraint_force_const: float = 1.0,
    defer_xla_gpu_warmup: bool = False,
    ml_batch_size: Optional[int] = None,
    ml_gpu_count: int = 1,
    ml_max_active_dimers: Optional[int] = None,
    ml_compute_dtype: Optional[str] = None,
    electrostatics_damping_sigma: float | None = None,
    at_codes_override: np.ndarray | None = None,
    mbd_checkpoint: str | Path | None = None,
    mbd_weight: float = 1.0,
    lr_solver: str | None = None,
    ewald_include_self: bool = True,
    ewald_include_intra: bool = True,
    mm_charge_mode: str | None = None,
    mm_charge_correction: bool = False,
    mm_latent_charge_template: str | Path | None = None,
    backprop: bool = False,
    hybrid_hamiltonian: str = "handoff",
    shared_cutoff: float | None = None,
    ml_potential_mode: str | None = None,
    jax_mm_spoof_psf: str | Path | None = None,
    bonded_intra_damp_onset: float | None = None,
    bonded_intra_damp_cutoff: float = 15.0,
):
    _load_pycharmm_modules()
    if at_codes_override is not None:
        at_codes = np.asarray(at_codes_override, dtype=int)
    else:
        at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    cell_arg = float(cell_scalar) if cell_scalar is not None else False
    t0 = _tmark()
    if isinstance(atoms_per, int):
        max_atoms_per = int(atoms_per)
    else:
        max_atoms_per = int(max(atoms_per))
    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per,
        N_MONOMERS=n_mol,
        ml_switch_width=ml_cut,
        mm_switch_on=mm_sw,
        mm_switch_width=mm_cut,
        doML=do_ml,
        doMM=do_mm,
        doML_dimer=do_ml_dimer,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max_atoms_per * 2,
        cell=cell_arg,
        ep_scale=np.ones(n_types),
        sig_scale=np.ones(n_types),
        at_codes_override=at_codes,
        verbose=verbose,
        max_pairs=max_pairs,
        jax_md_capacity_multiplier=jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=jax_md_overflow_fallback_to_cell_list,
        jax_md_update_interval=jax_md_update_interval,
        jax_md_skin_distance=jax_md_skin_distance,
        flat_bottom_radius=flat_bottom_radius,
        flat_bottom_force_const=flat_bottom_force_const,
        flat_bottom_mode=flat_bottom_mode,
        min_com_restraint_distance=min_com_restraint_distance,
        min_com_restraint_force_const=min_com_restraint_force_const,
        defer_xla_gpu_warmup=defer_xla_gpu_warmup,
        ml_batch_size=ml_batch_size,
        ml_gpu_count=int(ml_gpu_count or 1),
        ml_max_active_dimers=ml_max_active_dimers,
        ml_compute_dtype=ml_compute_dtype,
        electrostatics_damping_sigma=electrostatics_damping_sigma,
        mbd_checkpoint=mbd_checkpoint,
        mbd_weight=mbd_weight,
        lr_solver=lr_solver,
        ewald_include_self=bool(ewald_include_self),
        ewald_include_intra=bool(ewald_include_intra),
        mm_charge_mode=mm_charge_mode,
        mm_charge_correction=mm_charge_correction,
        mm_latent_charge_template=mm_latent_charge_template,
        ml_potential_mode=ml_potential_mode,
        jax_mm_spoof_psf=jax_mm_spoof_psf,
        bonded_intra_damp_onset_kcal=bonded_intra_damp_onset,
        bonded_intra_damp_cutoff_kcal=(
            15.0 if bonded_intra_damp_cutoff is None else float(bonded_intra_damp_cutoff)
        ),
        hybrid_hamiltonian=hybrid_hamiltonian,
        shared_cutoff=shared_cutoff,
    )
    t1 = _tmark()
    cutoff = CutoffParameters(
        ml_switch_width=ml_cut,
        mm_switch_on=mm_sw,
        mm_switch_width=mm_cut,
        hybrid_hamiltonian=hybrid_hamiltonian,
        shared_cutoff=shared_cutoff,
    )
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=r,
        n_monomers=n_mol,
        cutoff_params=cutoff,
        doML=do_ml,
        doMM=do_mm,
        doML_dimer=do_ml_dimer,
        backprop=bool(backprop),
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=verbose,
    )
    t2 = _tmark()
    if timings is not None:
        timings["karml_setup_calculator_s"] = t1 - t0
        timings["karml_factory_call_s"] = t2 - t1
    if len(calc_result) == 3:
        karml_calc, _, _ = calc_result
    else:
        karml_calc, _ = calc_result
    return karml_calc


def _validate_psf_charges(
    *,
    monomer_offsets: np.ndarray,
    residue_labels: list[str],
    total_atoms: int,
    log_lines: list[str] | None = None,
) -> dict:
    # Import locally: certified-box / artifact load never calls _load_pycharmm_modules(),
    # so the module-level ``psf`` cache stays None even though CHARMM already has a PSF.
    import pycharmm.psf as psf_mod

    charges = _get_actual_psf_charges(total_atoms)[:total_atoms]
    atom_types = np.asarray(psf_mod.get_atype(), dtype=str)[:total_atoms]
    if charges.shape[0] != total_atoms:
        raise RuntimeError(
            f"PSF charge count mismatch: expected {total_atoms}, got {charges.shape[0]}"
        )
    if atom_types.shape[0] != total_atoms:
        raise RuntimeError(
            f"PSF atom-type count mismatch: expected {total_atoms}, got {atom_types.shape[0]}"
        )
    if not np.all(np.isfinite(charges)):
        bad = np.where(~np.isfinite(charges))[0][:10].tolist()
        raise RuntimeError(f"PSF charges contain non-finite values at atom indices {bad}")

    residue_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    residue_summaries: dict[str, dict] = {}
    for i, residue in enumerate(residue_labels):
        s = int(monomer_offsets[i])
        e = int(monomer_offsets[i + 1])
        local_types = atom_types[s:e]
        local_charges = charges[s:e]
        if residue in residue_reference:
            ref_types, ref_charges = residue_reference[residue]
            if list(local_types) != list(ref_types):
                raise RuntimeError(
                    f"PSF atom-type order differs between {residue} copies: "
                    f"{list(local_types)} != {list(ref_types)}"
                )
            if not np.allclose(local_charges, ref_charges, atol=1e-8, rtol=0.0):
                raise RuntimeError(
                    f"PSF charges differ between {residue} copies: "
                    f"{local_charges.tolist()} != {ref_charges.tolist()}"
                )
        else:
            residue_reference[residue] = (local_types.copy(), local_charges.copy())
            residue_summaries[residue] = {
                "n_atoms": int(e - s),
                "charge_sum_e": float(np.sum(local_charges)),
                "atom_types": [str(x) for x in local_types],
                "charges_e": [float(x) for x in local_charges],
            }

    total_charge = float(np.sum(charges))
    summary = {
        "total_charge_e": total_charge,
        "n_charged_atoms": int(np.count_nonzero(np.abs(charges) > 1e-12)),
        "min_charge_e": float(np.min(charges)) if charges.size else 0.0,
        "max_charge_e": float(np.max(charges)) if charges.size else 0.0,
        "residues": residue_summaries,
    }
    _tlog(
        "PSF charge validation: "
        f"total={total_charge:.6f} e, charged_atoms={summary['n_charged_atoms']}, "
        + ", ".join(
            f"{res}:{data['charge_sum_e']:.6f} e"
            for res, data in residue_summaries.items()
        ),
        log_lines,
    )
    return summary


def _numpy_wrap_monomers_primary_cell(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    cell_matrix: np.ndarray,
) -> np.ndarray:
    from karml.utils.geometry_checks import wrap_monomers_primary_cell

    return wrap_monomers_primary_cell(positions, monomer_offsets, cell_matrix)


def _run_charmm_minimize(
    atoms: Atoms,
    *,
    nstep_sd: int,
    nstep_abnr: int,
    tolenr: float,
    tolgrd: float,
    nbxmod: int = 5,
    timings: dict[str, float] | None = None,
    cubic_box_side_A: float | None = None,
    quiet: bool = True,
    show_energy: bool = False,
) -> None:
    """Run optional CHARMM minimization and write updated coords back to ASE atoms.

    When ``cubic_box_side_A`` is set, CHARMM crystal / IMAGE PBC is rebuilt for a
    cubic cell of that edge length (matches the MD periodic box for JAX-MD / ASE).

    By default CHARMM runs at PRNLev 0 (no per-step ENERGY / nonbond banners). Set
    ``show_energy=True`` for a compact energy table before and after minimization.
    """
    if nstep_sd <= 0 and nstep_abnr <= 0:
        return
    # Certified-box / artifact load imports pycharmm elsewhere and leaves these
    # module-level caches as None until a CHARMM-backed builder runs.
    _load_pycharmm_modules()
    t0 = _tmark()
    reset_block()
    coor.set_positions(pd.DataFrame(atoms.get_positions(), columns=["x", "y", "z"]))
    try:
        if quiet:
            pyci.pycharmm_quiet()
        from karml.interfaces.pycharmmInterface.nbonds_config import apply_vacuum_nbonds

        use_pbc_charmm = cubic_box_side_A is not None and float(cubic_box_side_A) > 0.0
        if use_pbc_charmm:
            from karml.interfaces.pycharmmInterface.mlpot.pbc_env import (
                apply_pbc_nbonds,
                prepare_charmm_pbc,
            )

            L = float(cubic_box_side_A)
            prepare_charmm_pbc(L)
            cuts = apply_pbc_nbonds(nbxmod=nbxmod, cubic_box_side_A=L)
            if cuts.was_capped and not quiet:
                print(cuts.summary_line(), flush=True)
        else:
            apply_vacuum_nbonds(nbxmod=nbxmod)
        if show_energy:
            print("CHARMM energy before minimization:")
            pyci.pycharmm_quiet()
            pyci.safe_energy_show()
        if nstep_sd > 0:
            pyci.pycharmm_quiet()
            charmm_minimize.run_sd(nstep=nstep_sd, tolenr=tolenr, tolgrd=tolgrd)
        if nstep_abnr > 0:
            pyci.pycharmm_quiet()
            charmm_minimize.run_abnr(nstep=nstep_abnr, tolenr=tolenr, tolgrd=tolgrd)
        minimized_positions = coor.get_positions().to_numpy(dtype=float)
        coor.set_positions(pd.DataFrame(minimized_positions, columns=["x", "y", "z"]))
        if show_energy:
            print("CHARMM energy after minimization:")
            pyci.pycharmm_quiet()
            pyci.safe_energy_show()
    finally:
        pyci.pycharmm_quiet()
    atoms.set_positions(coor.get_positions().to_numpy(dtype=float))
    if timings is not None:
        timings["charmm_min_wall_s"] = _tmark() - t0


def _check_or_charmm_overlap_rescue(
    atoms: Atoms,
    monomer_offsets: np.ndarray,
    *,
    min_distance: float,
    context: str,
    nstep_sd: int,
    nstep_abnr: int,
    tolenr: float,
    tolgrd: float,
    nbxmod: int = 5,
    timings: dict[str, float] | None = None,
) -> float:
    """Check for inter-monomer overlaps, using CHARMM minimization as first rescue."""
    cell = atoms.cell.array if atoms.pbc.any() else None
    try:
        return assert_no_intermonomer_atom_overlap(
            atoms.get_positions(),
            monomer_offsets,
            min_distance=min_distance,
            cell=cell,
            context=context,
        )
    except RuntimeError as exc:
        print(f"{context}: overlap detected; attempting CHARMM SD/ABNR rescue before aborting.")
        if timings is not None:
            timings["overlap_rescue_attempted"] = float(timings.get("overlap_rescue_attempted", 0.0) + 1.0)
        _run_charmm_minimize(
            atoms,
            nstep_sd=nstep_sd,
            nstep_abnr=nstep_abnr,
            tolenr=tolenr,
            tolgrd=tolgrd,
            nbxmod=nbxmod,
            timings=timings,
            quiet=True,
        )
        try:
            return assert_no_intermonomer_atom_overlap(
                atoms.get_positions(),
                monomer_offsets,
                min_distance=min_distance,
                cell=cell,
                context=f"{context} after CHARMM overlap rescue",
            )
        except RuntimeError as rescue_exc:
            raise RuntimeError(f"{exc}; CHARMM overlap rescue failed: {rescue_exc}") from rescue_exc


def _save_cutoff_plot(
    *,
    out_dir: Path,
    ml_cutoff: float,
    mm_switch_on: float,
    mm_cutoff: float,
    log_lines: list[str] | None = None,
) -> Path:
    cutoff = CutoffParameters(
        ml_cutoff=ml_cutoff,
        mm_switch_on=mm_switch_on,
        mm_cutoff=mm_cutoff,
        complementary_handoff=True,
    )
    _ = cutoff.plot_cutoff_parameters(save_dir=out_dir)
    out_path = out_dir / (
        f"cutoffs_schematic_{float(ml_cutoff):.2f}_{float(mm_switch_on):.2f}_"
        f"{float(mm_cutoff):.2f}_complementary.png"
    )
    _tlog(f"cutoff plot: {out_path}", log_lines)
    return out_path

