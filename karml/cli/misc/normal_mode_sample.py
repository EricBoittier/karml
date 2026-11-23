#!/usr/bin/env python
"""Sample geometries along Cartesian displacements.

The displacements can come from a PySCF Hessian, from any stored vectors, or
from finite-difference vibrations of an ASE calculator. The teacher is the
same ``--calculator metatomic --checkpoint teacher.pt`` pair used by
``dimer-scan``.

Output NPZ: ``R`` (n_samples, n_atoms, 3) Å, ``Z``, ``N``, plus the
``displacements`` that were scaled (Å) and ``freq_wavenumber`` when known.

Examples::

    karml normal-mode-sample --calculator metatomic --checkpoint teacher.pt \\
        --structure monomer.xyz -o monomer_modes.npz --amplitude 0.05

    karml normal-mode-sample --displacements modes.npz -o sampled.npz --amplitude 0.05

    karml normal-mode-sample -i out/04_results.h5 -o sampled.npz --amplitude 0.1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator


BOHR_TO_ANG = 0.529177
_GEOMETRY_SUFFIXES = {".xyz", ".extxyz", ".pdb"}
_CALCULATORS = (
    "physnet",
    "spookynet",
    "mbd",
    "multipoles",
    "efield",
    "kernnn",
    "metatomic",
    "xtb",
    "dftb3-d4",
    "pyscf",
)


def _load_h5(path: Path) -> dict:
    """Load R, Z, harmonic/norm_mode, harmonic/freq_wavenumber from h5."""
    import h5py

    data = {}
    with h5py.File(path, "r") as f:
        data["R"] = np.asarray(f["R"])
        data["Z"] = np.asarray(f["Z"])
        if "harmonic" not in f:
            raise ValueError(
                f"No 'harmonic' group in {path}. Run pyscf-dft with --harmonic --hessian."
            )
        grp = f["harmonic"]
        data["norm_mode"] = np.asarray(grp["norm_mode"])
        data["freq_wavenumber"] = np.asarray(grp["freq_wavenumber"])

    return data


def _as_geometry(R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(R, dtype=float)
    numbers = np.asarray(Z)
    if positions.ndim == 3:
        positions = positions[0]
    if numbers.ndim == 2:
        numbers = numbers[0]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"R must be (n_atoms, 3), got {positions.shape}")
    if numbers.shape != (positions.shape[0],):
        raise ValueError(
            f"Z length {numbers.shape} does not match {positions.shape[0]} atoms"
        )
    return positions, numbers


def _scale_to_angstrom(vectors: np.ndarray, units: str) -> np.ndarray:
    name = units.strip().lower()
    if name in {"angstrom", "ang", "å", "a"}:
        return np.asarray(vectors, dtype=float)
    if name in {"bohr", "bohrs", "au"}:
        return np.asarray(vectors, dtype=float) * BOHR_TO_ANG
    raise ValueError(f"displacement units must be angstrom or bohr, got {units!r}")


def sample_displacements(
    R_eq: np.ndarray,
    Z: np.ndarray,
    displacements: np.ndarray,
    freq_wavenumber: np.ndarray | None,
    *,
    amplitudes: list[float],
    freq_min: float = 50.0,
    include_equilibrium: bool = False,
    samples_per_mode: int = 2,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale stored Cartesian displacements and add them to ``R_eq``.

    ``displacements`` is ``(n, n_atoms, 3)`` in Å. ``amplitude`` multiplies
    that vector, so a unit-length mode moves the coordinates by ``amplitude``
    Å in the Euclidean norm. Frequencies, when present, drop modes at or
    below ``freq_min`` (cm^-1).
    """
    vectors = np.asarray(displacements, dtype=float)
    if vectors.ndim != 3 or vectors.shape[1:] != R_eq.shape:
        raise ValueError(
            f"displacements must be (n, {R_eq.shape[0]}, 3), got {vectors.shape}"
        )
    if freq_wavenumber is None:
        mode_indices = np.arange(vectors.shape[0])
    else:
        freq_real = np.real(np.asarray(freq_wavenumber))
        if freq_real.shape[0] != vectors.shape[0]:
            raise ValueError(
                f"freq_wavenumber length {freq_real.shape[0]} != {vectors.shape[0]} displacements"
            )
        mode_indices = np.where(freq_real > freq_min)[0]

    if len(mode_indices) == 0:
        raise ValueError(
            f"No displacements with freq > {freq_min} cm^-1. "
            "Try lowering --freq-min or check the mode file."
        )

    samples: list[np.ndarray] = []
    if include_equilibrium:
        samples.append(R_eq.copy())

    n_modes = len(mode_indices)

    for k in mode_indices:
        if max_samples is not None and len(samples) >= max_samples:
            break
        mode_k = vectors[k]
        for amp in amplitudes:
            if max_samples is not None and len(samples) >= max_samples:
                break
            if samples_per_mode >= 2:
                samples.append(R_eq + amp * mode_k)
                if max_samples is not None and len(samples) >= max_samples:
                    break
                samples.append(R_eq - amp * mode_k)
            else:
                samples.append(R_eq + amp * mode_k)

    if max_samples is not None and len(samples) < max_samples and n_modes >= 2:
        scale = 1.0 / np.sqrt(2)
        for i in range(n_modes):
            if len(samples) >= max_samples:
                break
            mode_i = vectors[mode_indices[i]]
            for j in range(i + 1, n_modes):
                if len(samples) >= max_samples:
                    break
                mode_j = vectors[mode_indices[j]]
                for amp in amplitudes:
                    if len(samples) >= max_samples:
                        break
                    samples.append(R_eq + scale * amp * (mode_i + mode_j))
                    if len(samples) >= max_samples:
                        break
                    samples.append(R_eq + scale * amp * (mode_i - mode_j))

    return np.stack(samples, axis=0), Z


def sample_normal_modes(
    R_eq: np.ndarray,
    Z: np.ndarray,
    norm_mode: np.ndarray,
    freq_wavenumber: np.ndarray,
    *,
    amplitudes: list[float],
    freq_min: float = 50.0,
    include_equilibrium: bool = False,
    samples_per_mode: int = 2,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample PySCF normal modes. ``norm_mode`` is in Bohr."""
    return sample_displacements(
        R_eq,
        Z,
        np.asarray(norm_mode, dtype=float) * BOHR_TO_ANG,
        freq_wavenumber,
        amplitudes=amplitudes,
        freq_min=freq_min,
        include_equilibrium=include_equilibrium,
        samples_per_mode=samples_per_mode,
        max_samples=max_samples,
    )


def displacements_from_calculator(
    atoms,
    calculator: Calculator,
    *,
    delta: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference Cartesian modes of ``calculator`` on ``atoms``.

    Each mode is scaled to unit Euclidean length, so ``--amplitude`` is the
    displacement length in Å. Frequencies are cm^-1.
    """
    from ase.vibrations import Vibrations

    vibrating = atoms.copy()
    vibrating.calc = calculator
    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(
            vibrating,
            name=str(Path(tmp) / "vib"),
            delta=float(delta),
            nfree=2,
        )
        vib.run()
        data = vib.get_vibrations()
        freqs = np.real(np.asarray(data.get_frequencies(), dtype=np.complex128))
        modes = np.asarray(data.get_modes(), dtype=float)

    kept_modes: list[np.ndarray] = []
    kept_freq: list[float] = []
    for freq, mode in zip(freqs, modes, strict=True):
        norm = float(np.linalg.norm(mode))
        if not np.isfinite(norm) or norm < 1e-12 or not np.isfinite(freq):
            continue
        kept_modes.append(mode / norm)
        kept_freq.append(float(freq))
    if not kept_modes:
        raise ValueError("calculator vibrations produced no finite modes")
    return np.stack(kept_modes), np.asarray(kept_freq, dtype=float)


def _read_geometry(path: Path):
    from ase.io import read as ase_read

    atoms = ase_read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    return atoms


def _displacements_from_frames(path: Path, reference) -> dict:
    from ase.io import read as ase_read

    frames = ase_read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if reference is None:
        reference = frames[0]
        frames = frames[1:]
    R_eq = np.asarray(reference.get_positions(), dtype=float)
    Z = np.asarray(reference.get_atomic_numbers())
    vectors = []
    for frame in frames:
        if len(frame) != len(reference):
            raise ValueError(
                f"{path} mixes molecule sizes ({len(frame)} vs {len(reference)})"
            )
        shift = np.asarray(frame.get_positions(), dtype=float) - R_eq
        if float(np.linalg.norm(shift)) < 1e-8:
            continue
        vectors.append(shift)
    if not vectors:
        raise ValueError(
            f"{path} has no displaced frames. Pass --calculator to compute modes, "
            "or a file with more than the equilibrium geometry."
        )
    return {
        "R": R_eq,
        "Z": Z,
        "displacements": np.stack(vectors),
        "freq_wavenumber": None,
    }


def _array_from_npz(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    return None


def _load_displacement_npz(path: Path, structure: Path | None) -> dict:
    with np.load(path, allow_pickle=False) as data:
        names = ("displacements", "displacement", "modes", "mode", "norm_mode")
        key = next((name for name in names if name in data.files), None)
        if key is None:
            raise ValueError(
                f"{path} has no displacements, modes, or norm_mode array"
            )
        vectors = np.asarray(data[key])
        units = "bohr" if key == "norm_mode" else "angstrom"
        if "units" in data.files:
            raw_units = data["units"]
            units = str(raw_units.item() if getattr(raw_units, "shape", ()) == () else raw_units)
        vectors = _scale_to_angstrom(vectors, units)
        if vectors.ndim == 2:
            vectors = vectors.reshape(1, *vectors.shape)
        freq = _array_from_npz(data, ("freq_wavenumber", "frequencies_cm", "frequencies"))
        positions = _array_from_npz(data, ("R", "positions"))
        numbers = _array_from_npz(data, ("Z", "numbers"))
    if positions is None or numbers is None:
        if structure is None:
            raise ValueError(f"{path} has no R/Z. Pass --structure for the equilibrium geometry.")
        atoms = _read_geometry(structure)
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
    R_eq, Z = _as_geometry(positions, numbers)
    return {
        "R": R_eq,
        "Z": Z,
        "displacements": np.asarray(vectors, dtype=float),
        "freq_wavenumber": None if freq is None else np.real(np.asarray(freq)),
    }


def _load_pyscf_h5(path: Path) -> dict:
    data = _load_h5(path)
    R_eq, Z = _as_geometry(data["R"], data["Z"])
    return {
        "R": R_eq,
        "Z": Z,
        "displacements": np.asarray(data["norm_mode"], dtype=float) * BOHR_TO_ANG,
        "freq_wavenumber": np.real(np.asarray(data["freq_wavenumber"])),
    }


def _calculator_from_args(args: argparse.Namespace) -> Calculator:
    """Build an ASE calculator with the dimer-scan calculator flags."""
    from karml.dimer_scan.calculators import calculator_factory
    from karml.dimer_scan.config import DimerScanConfig

    field = tuple(args.electric_field) if args.electric_field is not None else None
    config = DimerScanConfig(
        residues=("TIP3", "TIP3"),
        calculator=args.calculator,
        distances_angstrom=(1.0,),
        checkpoint=args.checkpoint,
        charge=args.charge,
        spin=args.spin,
        method=args.method,
        basis=args.basis,
        xc=args.xc,
        calculator_config=args.calculator_config,
        electric_field_au=field,
        slako_dir=args.slako_dir,
        workdir=args.calculator_workdir,
        executable=args.calculator_executable,
    )
    return calculator_factory(config)()


def _modes_from_teacher(args: argparse.Namespace) -> dict:
    structure = args.structure
    if structure is None and args.input is not None and args.input.suffix.lower() in _GEOMETRY_SUFFIXES:
        structure = args.input
    if structure is None:
        raise ValueError("--calculator needs --structure (xyz, extxyz, or pdb)")
    if not structure.is_file():
        raise FileNotFoundError(f"structure not found: {structure}")
    atoms = _read_geometry(structure)
    displacements, freq = displacements_from_calculator(
        atoms, _calculator_from_args(args), delta=float(args.delta)
    )
    return {
        "R": np.asarray(atoms.get_positions(), dtype=float),
        "Z": np.asarray(atoms.get_atomic_numbers()),
        "displacements": displacements,
        "freq_wavenumber": freq,
    }


def _load_request(args: argparse.Namespace) -> dict:
    if args.calculator is not None:
        if args.displacements is not None:
            raise ValueError("pass either --calculator or --displacements")
        if args.input is not None and args.input.suffix.lower() in {".h5", ".hdf5"}:
            raise ValueError("a Hessian .h5 already has modes; drop --calculator")
        return _modes_from_teacher(args)

    if args.displacements is not None:
        return _load_displacement_npz(args.displacements, args.structure)

    if args.input is None:
        raise ValueError(
            "pass -i (PySCF .h5 or a geometry), --displacements, "
            "or --calculator with --checkpoint and --structure"
        )
    path = args.input
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _load_pyscf_h5(path)
    if suffix == ".npz":
        return _load_displacement_npz(path, args.structure)
    if suffix in _GEOMETRY_SUFFIXES:
        return _displacements_from_frames(path, None if args.structure is None else _read_geometry(args.structure))
    raise ValueError(
        f"unrecognized input {path.name}. Use a PySCF .h5, an NPZ of displacements, "
        "a multi-frame extxyz, or --calculator."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karml normal-mode-sample",
        description=(
            "Sample geometries along Cartesian displacements. "
            "Modes can be a PySCF Hessian, any stored displacements, "
            "or finite-difference vibrations of --calculator."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="PySCF .h5, displacement .npz, or a multi-frame extxyz",
    )
    parser.add_argument(
        "--displacements",
        type=Path,
        default=None,
        help="NPZ of Cartesian displacements (n, n_atoms, 3), in Å unless units=bohr",
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=None,
        help="Equilibrium geometry (xyz, extxyz, pdb) for --calculator or a displacement file",
    )
    parser.add_argument(
        "--calculator",
        choices=_CALCULATORS,
        default=None,
        help="ASE calculator that finite-differences the modes (metatomic = the teacher .pt)",
    )
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint, e.g. the teacher .pt")
    parser.add_argument("--calculator-config", type=Path, help="Calculator model config JSON")
    parser.add_argument("--method", help="Calculator method (for example pyscf: dft or hf)")
    parser.add_argument("--basis", help="PySCF basis (default: def2-svp)")
    parser.add_argument("--xc", help="PySCF DFT functional (default: pbe0)")
    parser.add_argument("--charge", type=float)
    parser.add_argument("--spin", type=float, help="Spin multiplicity when the calculator needs it")
    parser.add_argument(
        "--electric-field",
        nargs=3,
        type=float,
        metavar=("EX", "EY", "EZ"),
        help="EField vector in atomic units",
    )
    parser.add_argument("--slako-dir", type=Path, help="DFTB+ 3ob-3-1 directory")
    parser.add_argument("--calculator-workdir", type=Path, help="External calculator scratch directory")
    parser.add_argument("--calculator-executable", help="External calculator executable")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Finite-difference step for --calculator, in Å (default: 0.01)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("sampled.npz"),
        help="Output NPZ path (default: sampled.npz)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Scale applied to each displacement. Unit modes move by this many Å (default: 0.1)",
    )
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=None,
        help="List of amplitudes (overrides --amplitude)",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=50.0,
        help="Drop modes at or below this frequency in cm^-1 when frequencies are known (default: 50)",
    )
    parser.add_argument(
        "--include-equilibrium",
        action="store_true",
        help="Add the equilibrium geometry as the first sample",
    )
    parser.add_argument(
        "--samples-per-mode",
        type=int,
        choices=[1, 2],
        default=2,
        help="2 for +/- amplitude (default), 1 for + only",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of structures to generate (default: no limit)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run normal-mode-sample CLI."""
    args = parse_args(argv)
    t0 = time.perf_counter()

    try:
        loaded = _load_request(args)
        amplitudes = args.amplitudes if args.amplitudes is not None else [args.amplitude]
        R_samples, Z_out = sample_displacements(
            loaded["R"],
            loaded["Z"],
            loaded["displacements"],
            loaded["freq_wavenumber"],
            amplitudes=amplitudes,
            freq_min=args.freq_min,
            include_equilibrium=args.include_equilibrium,
            samples_per_mode=args.samples_per_mode,
            max_samples=args.max_samples,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "R": R_samples,
        "Z": Z_out,
        "N": np.array(len(Z_out)),
        "displacements": np.asarray(loaded["displacements"], dtype=float),
        "units": np.array("angstrom"),
    }
    if loaded["freq_wavenumber"] is not None:
        payload["freq_wavenumber"] = np.asarray(loaded["freq_wavenumber"])
    np.savez_compressed(args.output, **payload)

    elapsed = time.perf_counter() - t0
    print(f"Saved {R_samples.shape[0]} geometries to {args.output}")
    print(f"Elapsed: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
