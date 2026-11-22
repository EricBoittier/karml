"""PET teacher → PhysNet student NPZ (monomer/dimer pool or liquid-box clusters).

CHARMM-free. Labels come from an ASE teacher (metatomic `.pt` or any calculator
passed in tests). Does not run MD or PhysNet training.

The geometry pool is whatever monomer you pass. ``--example acetone`` is the
bundled ACO recipe (PDB, DMC dimers, carbonyl stretch, acetone COM windows).

Example::

    JAX_PLATFORMS=cpu KARML_METATOMIC_DEVICE=cpu \\
      karml pet-physnet-distill \\
      --checkpoint /tmp/karml-metatomic-models/pet-mad-xs-v1.5.0.pt \\
      --monomer monomer.xyz --com-windows com_well:4.0:6.0 \\
      --out-dir ./pet_distill --preset smoke

    karml pet-physnet-distill --example acetone --preset smoke \\
      --out-dir ./acetone_pet_distill --geometries-only

Labels go through one batched TorchScript forward per ``--max-atoms-per-batch``
chunk (``--teacher-backend torchscript``); ``ase`` is the per-structure path.
Other PETs: ``python -c "from upet import save_upet; save_upet(model='pet-omol',
size='m', version='1.0.0', output='pet-omol-m-v1.0.0.pt')"``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from karml.distill.geometry_pool import (
    POOL_PRESETS,
    PRESET_SMOKE,
    PoolConfig,
    apply_preset,
    build_geometry_pool,
    load_frames,
    load_structure,
)

_EXAMPLES = ("acetone",)
from karml.distill.npz_export import SPLIT_MODES, SPLIT_SAMPLE, SPLIT_SEED, write_distill_npz
from karml.distill.teacher_label import (
    ENERGY_MODE_MLMM,
    ENERGY_MODE_TOTAL,
    ENERGY_MODES,
    label_geometries,
)

_REPO = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="karml pet-physnet-distill",
        description=(
            "Build a monomer/dimer geometry pool, label it with a metatomic PET "
            "teacher, and write PhysNet-train NPZ in eV. Pass --monomer, or "
            "--example acetone for the bundled ACO recipe."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Teacher AtomisticModel (.pt). Required unless --geometries-only.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Train/valid NPZ + report.json")
    p.add_argument("--preset", choices=POOL_PRESETS, default=PRESET_SMOKE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--example",
        choices=_EXAMPLES,
        default=None,
        help=(
            "Bundled recipe. acetone: ACO monomer PDB, DMC dimer frames, "
            "carbonyl C–O stretch, acetone–acetone COM windows."
        ),
    )
    p.add_argument(
        "--monomer",
        type=Path,
        default=None,
        help="Reference monomer (PDB, XYZ, or extxyz). Required unless --example or --from-box-extxyz.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Extra frames: n-atom monomers or 2n-atom homodimers (n = the reference monomer).",
    )
    p.add_argument(
        "--no-dataset",
        action="store_true",
        help="Skip dataset frames (including the acetone DMC set when --example acetone).",
    )
    p.add_argument(
        "--com-windows",
        nargs="*",
        default=None,
        metavar="NAME:LO:HI",
        help="Intermolecular COM windows in Å, e.g. com_well:4.4:5.6 com_long:7:10. Overrides the example.",
    )
    p.add_argument(
        "--stretch-bonds",
        nargs="*",
        default=None,
        metavar="I,J",
        help="Bonds to scale, 0-based i,j (j moves away from i). Example: 1,0. Overrides the example.",
    )
    p.add_argument(
        "--energy-mode",
        choices=ENERGY_MODES,
        default=ENERGY_MODE_MLMM,
        help=(
            "mlmm (default): monomer E-E_ref, dimer E_AB-2E_ref, full forces; matches "
            "PhysNet MLpot, which forms E_int=P(AB)-P(A)-P(B) itself. interaction: "
            "dimer E=E_int (not MLpot-consistent). total: raw teacher energies."
        ),
    )
    p.add_argument(
        "--geometries-only",
        action="store_true",
        help="Write unlabeled R/Z/N NPZ (no teacher). For pool inspection.",
    )
    p.add_argument(
        "--extra-extxyz",
        type=Path,
        nargs="*",
        default=(),
        help="Additional ASE frames with n or 2n atoms (n = the reference monomer)",
    )
    p.add_argument(
        "--from-box-extxyz",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Periodic MD frames (extxyz with cell, e.g. metatomic-pbc-md "
            "--traj-every). Uses monomers and COM-close dimers cut out by "
            "minimum image instead of the synthetic pool. Needs --atoms-per-monomer."
        ),
    )
    p.add_argument("--atoms-per-monomer", type=int, default=None)
    p.add_argument(
        "--reference-monomer-xyz",
        type=Path,
        default=None,
        help="Gas-phase monomer for E_ref in interaction mode (box pool only)",
    )
    p.add_argument("--frame-stride", type=int, default=1, help="Use every Nth box frame")
    p.add_argument(
        "--dimer-com-cutoff",
        type=float,
        default=7.5,
        help="Å centroid distance for box dimers (MLpot sparse ML range: on + ml width)",
    )
    p.add_argument(
        "--dimer-r-bins",
        type=str,
        default="0,3.5,4.5,5.25,6.0,7.5",
        help="Å bin edges for an even dimer draw per frame",
    )
    p.add_argument("--max-monomers-per-frame", type=int, default=8)
    p.add_argument("--max-dimers-per-frame", type=int, default=24)
    p.add_argument(
        "--include-dimer-fragments",
        action="store_true",
        help="Also store each dimer's A and B as monomer samples (matched triples for MLpot E_int)",
    )
    p.add_argument(
        "--valid-fraction",
        type=float,
        default=0.15,
        help="Valid share: of samples (--split sample) or of trajectories (--split seed)",
    )
    p.add_argument(
        "--split",
        choices=SPLIT_MODES,
        default=None,
        help=(
            "seed: whole trajectories (frame info seed, else input file) go to "
            "train or valid, so AB/A/B triples and repeated monomers never "
            "straddle the split; default with --from-box-extxyz. sample: "
            "per-sample permutation; default for the synthetic pool"
        ),
    )
    p.add_argument(
        "--teacher-backend",
        choices=("torchscript", "ase"),
        default="torchscript",
        help=(
            "torchscript: batched AtomisticModel forward over many structures "
            "(default); ase: one MetatomicCalculator call per structure"
        ),
    )
    p.add_argument(
        "--max-atoms-per-batch",
        type=int,
        default=4096,
        help="torchscript backend: atom budget per forward (lower for larger PETs)",
    )
    p.add_argument(
        "--max-systems-per-batch",
        type=int,
        default=512,
        help="torchscript backend: structure budget per forward",
    )
    p.add_argument(
        "--student-yaml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write physnet-train.yaml next to the NPZ (warm-start DESdimers)",
    )
    return p


def _write_student_yaml(path: Path, *, train: Path, valid: Path, seed: int, tag: str) -> None:
    ckpt = _REPO / "examples" / "ckpts_json" / "DESdimers_params.json"
    if not ckpt.is_file():
        ckpt = Path("examples/ckpts_json/DESdimers_params.json")
    text = f"""# PhysNet student on PET labels (eV / eV/Å).
# Warm-start architecture from DESdimers; labels *are* the teacher, so
# --distill is off (do not pass the .pt as --teacher-checkpoint).
data: {train.resolve()}
valid_data: {valid.resolve()}
ckpt_dir: ./ckpts/{tag}
tag: {tag}
physnet_checkpoint: {ckpt}
match_checkpoint_architecture: true
distill: false
charges: false
use_energy_bias: true
seed: {int(seed)}
batch_size: 8
num_epochs: 200
learning_rate: 0.0003
early_stop_patience: 40
objective: valid_forces_mae
energy_weight: 1.0
forces_weight: 52.91
# MAE in kcal/mol for logs only; NPZ stays eV.
conversion:
  energy: 23.060549
  forces: 23.060549
"""
    path.write_text(text)


def _parse_com_windows(items: list[str] | None) -> tuple[tuple[str, float, float], ...] | None:
    if not items:
        return None
    windows: list[tuple[str, float, float]] = []
    for item in items:
        for piece in item.split(","):
            piece = piece.strip()
            if not piece:
                continue
            parts = piece.split(":")
            if len(parts) != 3:
                raise SystemExit(
                    f"COM window {piece!r} must be name:lo:hi (Å), e.g. com_well:4.4:5.6"
                )
            name, lo, hi = parts
            windows.append((name.strip(), float(lo), float(hi)))
    return tuple(windows) or None


def _parse_stretch_bonds(items: list[str] | None) -> tuple[tuple[int, int], ...] | None:
    if not items:
        return None
    bonds: list[tuple[int, int]] = []
    for item in items:
        for piece in item.replace(";", " ").split():
            piece = piece.strip()
            if not piece:
                continue
            bits = piece.split(",")
            if len(bits) != 2:
                raise SystemExit(f"stretch bond {piece!r} must be i,j (0-based indices)")
            bonds.append((int(bits[0]), int(bits[1])))
    return tuple(bonds) or None


def _student_tag(example: str | None, monomer: Path | None) -> str:
    if monomer is not None:
        return f"{monomer.stem.replace(' ', '_')}_pet_student"
    if example == "acetone":
        return "acetone_pet_student"
    return "pet_student"


def _synthetic_pool(args: argparse.Namespace):
    """Homodimer pool from --monomer and/or a bundled --example recipe."""
    if args.no_dataset and args.dataset is not None:
        raise SystemExit("--no-dataset conflicts with --dataset")
    if args.example is None and args.monomer is None:
        raise SystemExit("pass --monomer, --example acetone, or --from-box-extxyz")

    monomer = None
    dataset = None
    windows = None
    bonds = None
    include_dataset = False
    stretch_source = "bond_stretch"
    dataset_source = "dataset"

    if args.example == "acetone":
        from karml.distill.examples.acetone_pool import (
            CARBONYL_C_INDEX,
            CARBONYL_O_INDEX,
            COM_LONG_A,
            COM_REPULSIVE_A,
            COM_SHOULDER_A,
            COM_WELL_A,
            load_acetone_monomer,
            load_dataset_dimers,
        )

        monomer = load_acetone_monomer()
        windows = (
            ("com_repulsive", *COM_REPULSIVE_A),
            ("com_well", *COM_WELL_A),
            ("com_shoulder", *COM_SHOULDER_A),
            ("com_long", *COM_LONG_A),
        )
        bonds = ((CARBONYL_C_INDEX, CARBONYL_O_INDEX),)
        stretch_source = "co_stretch"
        dataset_source = "dmc_extxyz"
        include_dataset = not args.no_dataset
        if include_dataset:
            dataset = load_dataset_dimers()

    if args.monomer is not None:
        monomer = load_structure(args.monomer)
    if args.dataset is not None:
        dataset = load_frames(args.dataset)
        include_dataset = True
    parsed_windows = _parse_com_windows(args.com_windows)
    if parsed_windows is not None:
        windows = parsed_windows
    parsed_bonds = _parse_stretch_bonds(args.stretch_bonds)
    if parsed_bonds is not None:
        bonds = parsed_bonds

    if monomer is None:
        raise SystemExit("pass --monomer, --example acetone, or --from-box-extxyz")
    cfg = apply_preset(
        PoolConfig(
            seed=int(args.seed),
            com_windows_A=tuple(windows or ()),
            stretch_bonds=tuple(bonds or ()),
            include_dataset=include_dataset,
            extra_extxyz=tuple(Path(p) for p in (args.extra_extxyz or ())),
            stretch_source=stretch_source,
            dataset_source=dataset_source,
        ),
        args.preset,
    )
    geos = build_geometry_pool(monomer, cfg, dataset=dataset if include_dataset else None)
    n_mono = len(monomer)
    meta = {
        "example": args.example,
        "monomer": None if args.monomer is None else str(Path(args.monomer).resolve()),
        "dataset": None if args.dataset is None else str(Path(args.dataset).resolve()),
        "n_monomer_atoms": n_mono,
    }
    return geos, 2 * n_mono, _student_tag(args.example, args.monomer), meta


def _box_pool(args: argparse.Namespace):
    from ase.io import read as ase_read

    from karml.distill.box_clusters import BoxClusterConfig, box_cluster_pool

    if args.atoms_per_monomer is None:
        raise SystemExit("--from-box-extxyz needs --atoms-per-monomer")
    stride = max(int(args.frame_stride), 1)
    frames = []
    frame_files: list[int] = []
    for i_file, path in enumerate(args.from_box_extxyz):
        chunk = ase_read(str(path), index=f"::{stride}")
        frames.extend(chunk)
        frame_files.extend([i_file] * len(chunk))
    ref = None
    if args.reference_monomer_xyz is not None:
        ref = ase_read(str(args.reference_monomer_xyz))
    elif str(args.energy_mode) != ENERGY_MODE_TOTAL:
        raise SystemExit(
            "mlmm/interaction labels need --reference-monomer-xyz (gas-phase monomer "
            "for E_ref), or pass --energy-mode total"
        )
    cfg = BoxClusterConfig(
        atoms_per_monomer=int(args.atoms_per_monomer),
        dimer_com_cutoff_A=float(args.dimer_com_cutoff),
        dimer_r_bins_A=tuple(float(x) for x in str(args.dimer_r_bins).split(",") if x.strip()),
        max_monomers_per_frame=int(args.max_monomers_per_frame),
        max_dimers_per_frame=int(args.max_dimers_per_frame),
        seed=int(args.seed),
    )
    stats: dict = {}
    geos = box_cluster_pool(
        frames, cfg, reference_monomer=ref, stats=stats, frame_files=frame_files
    )
    stats["n_box_frames"] = len(frames)
    seeds = sorted({int(f.info["seed"]) for f in frames if f.info.get("seed") is not None})
    stats["n_box_seeds"] = len(seeds)
    stats["n_box_frames_without_seed"] = sum(1 for f in frames if f.info.get("seed") is None)
    return geos, stats


def run(args: argparse.Namespace) -> dict:
    box_stats = None
    pool_meta: dict | None = None
    tag = "pet_student"
    if args.from_box_extxyz:
        geos, box_stats = _box_pool(args)
        pad_atoms = 2 * int(args.atoms_per_monomer)
        if args.reference_monomer_xyz is not None:
            tag = _student_tag(None, Path(args.reference_monomer_xyz))
    else:
        geos, pad_atoms, tag, pool_meta = _synthetic_pool(args)
    split = args.split or (SPLIT_SEED if args.from_box_extxyz else SPLIT_SAMPLE)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.geometries_only:
        from karml.distill.teacher_label import LabeledSample

        zeros = [
            LabeledSample(
                geometry=g,
                energy_eV=0.0,
                forces_ev_per_angstrom=g.positions * 0.0,
                energy_total_eV=0.0,
                energy_int_eV=None if g.kind == "monomer" else 0.0,
            )
            for g in geos
        ]
        paths = write_distill_npz(
            zeros,
            out_dir,
            pad_atoms=pad_atoms,
            valid_fraction=float(args.valid_fraction),
            seed=int(args.seed),
            split=split,
            metadata={
                "geometries_only": True,
                "preset": args.preset,
                "seed": int(args.seed),
                **(pool_meta or {}),
            },
        )
        return {"n_geometries": len(geos), "paths": {k: str(v) for k, v in paths.items()}}

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --geometries-only")
    if args.teacher_backend == "torchscript":
        from karml.distill.batched_teacher import BatchedMetatomicTeacher

        evaluator = BatchedMetatomicTeacher(
            args.checkpoint,
            max_atoms_per_batch=int(args.max_atoms_per_batch),
            max_systems_per_batch=int(args.max_systems_per_batch),
        )
    else:
        from karml.interfaces.calculators.metatomic import load_metatomic_calculator

        evaluator = load_metatomic_calculator(args.checkpoint)
    t_label = time.perf_counter()
    labeled = label_geometries(
        evaluator,
        geos,
        energy_mode=str(args.energy_mode),
        include_dimer_fragments=bool(args.include_dimer_fragments),
    )
    label_s = time.perf_counter() - t_label
    teacher = Path(args.checkpoint).resolve()
    metadata = {
        "teacher": str(teacher),
        "teacher_size_bytes": int(teacher.stat().st_size),
        "teacher_backend": str(args.teacher_backend),
        "include_dimer_fragments": bool(args.include_dimer_fragments),
        "label_s": float(label_s),
        "energy_mode": str(args.energy_mode),
        "preset": str(args.preset),
        "seed": int(args.seed),
        "n_geometries": len(geos),
    }
    if pool_meta is not None:
        metadata.update(pool_meta)
    if box_stats is not None:
        metadata["box_extxyz"] = [str(Path(p).resolve()) for p in args.from_box_extxyz]
        metadata.update(box_stats)
    paths = write_distill_npz(
        labeled,
        out_dir,
        pad_atoms=pad_atoms,
        valid_fraction=float(args.valid_fraction),
        seed=int(args.seed),
        split=split,
        metadata=metadata,
    )
    if args.student_yaml:
        _write_student_yaml(
            out_dir / "physnet-train.yaml",
            train=paths["train"],
            valid=paths["valid"],
            seed=int(args.seed),
            tag=tag,
        )
        paths["student_yaml"] = out_dir / "physnet-train.yaml"
    return {
        "n_geometries": len(geos),
        "n_labeled": len(labeled),
        "paths": {k: str(v) for k, v in paths.items()},
        "metadata": metadata,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
