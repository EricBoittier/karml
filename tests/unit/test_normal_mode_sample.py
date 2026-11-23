"""Displacements for karml normal-mode-sample: stored vectors and a calculator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write as ase_write

from karml.cli.misc.normal_mode_sample import (
    BOHR_TO_ANG,
    displacements_from_calculator,
    main,
    sample_displacements,
    sample_normal_modes,
)


class _Harmonic(Calculator):
    """Isotropic spring about a reference geometry. Forces are -k (R - R0)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, reference: np.ndarray, k: float = 5.0):
        super().__init__()
        self.reference = np.asarray(reference, dtype=float)
        self.k = float(k)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        from ase.calculators.calculator import all_changes

        super().calculate(
            atoms,
            properties or ("energy", "forces"),
            all_changes if system_changes is None else system_changes,
        )
        shift = self.atoms.get_positions() - self.reference
        self.results = {
            "energy": 0.5 * self.k * float(np.sum(shift**2)),
            "forces": -self.k * shift,
        }


def test_sample_displacements_scales_any_vector() -> None:
    R = np.zeros((2, 3))
    Z = np.array([1, 1])
    disp = np.zeros((1, 2, 3))
    disp[0, 0, 0] = 1.0
    samples, _ = sample_displacements(
        R, Z, disp, None, amplitudes=[0.2], freq_min=0.0, samples_per_mode=2
    )
    assert samples.shape == (2, 2, 3)
    assert samples[0, 0, 0] == pytest.approx(0.2)
    assert samples[1, 0, 0] == pytest.approx(-0.2)


def test_pyscf_modes_are_converted_from_bohr() -> None:
    R = np.zeros((1, 3))
    Z = np.array([8])
    mode = np.zeros((1, 1, 3))
    mode[0, 0, 0] = 1.0
    freq = np.array([1000.0])
    samples, _ = sample_normal_modes(
        R, Z, mode, freq, amplitudes=[1.0], samples_per_mode=1
    )
    assert samples[0, 0, 0] == pytest.approx(BOHR_TO_ANG)


def test_calculator_modes_are_unit_displacements() -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    modes, freq = displacements_from_calculator(
        atoms, _Harmonic(atoms.get_positions()), delta=0.01
    )
    assert modes.shape[1:] == (2, 3)
    assert freq.shape == (modes.shape[0],)
    assert np.linalg.norm(modes, axis=(1, 2)) == pytest.approx(1.0)


def test_cli_reads_pyscf_hessian(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "hessian.h5"
    mode = np.zeros((1, 1, 3))
    mode[0, 0, 0] = 1.0
    with h5py.File(path, "w") as handle:
        handle["R"] = np.zeros((1, 3))
        handle["Z"] = np.array([8])
        handle.create_group("harmonic")
        handle["harmonic/norm_mode"] = mode
        handle["harmonic/freq_wavenumber"] = np.array([1200.0])
    out = tmp_path / "from_h5.npz"
    code = main(["-i", str(path), "-o", str(out), "--amplitude", "1", "--samples-per-mode", "1"])
    assert code == 0
    saved = np.load(out)
    assert saved["R"][0, 0, 0] == pytest.approx(BOHR_TO_ANG)


def test_cli_applies_stored_displacements(tmp_path: Path) -> None:
    disp = tmp_path / "modes.npz"
    np.savez(
        disp,
        R=np.zeros((2, 3)),
        Z=np.array([1, 1]),
        displacements=np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
    )
    out = tmp_path / "sampled.npz"
    code = main(
        [
            "--displacements",
            str(disp),
            "-o",
            str(out),
            "--amplitude",
            "0.25",
            "--freq-min",
            "0",
            "--samples-per-mode",
            "1",
        ]
    )
    assert code == 0
    saved = np.load(out)
    assert saved["R"].shape == (1, 2, 3)
    assert saved["R"][0, 0, 1] == pytest.approx(0.25)


def test_cli_reads_multiframe_extxyz(tmp_path: Path) -> None:
    eq = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    moved = eq.copy()
    moved.positions[1, 0] += 0.1
    path = tmp_path / "frames.extxyz"
    ase_write(path, [eq, moved])
    out = tmp_path / "sampled.npz"
    code = main(
        ["-i", str(path), "-o", str(out), "--amplitude", "2", "--samples-per-mode", "1", "--freq-min", "0"]
    )
    assert code == 0
    saved = np.load(out)
    assert saved["R"][0, 1, 0] == pytest.approx(0.74 + 0.2)


def test_cli_calculator_uses_familiar_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    xyz = tmp_path / "monomer.xyz"
    ase_write(xyz, atoms)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"unused")

    def _fake_calculator(args):
        assert args.calculator == "metatomic"
        assert args.checkpoint == checkpoint
        return _Harmonic(atoms.get_positions())

    monkeypatch.setattr(
        "karml.cli.misc.normal_mode_sample._calculator_from_args",
        _fake_calculator,
    )
    out = tmp_path / "from_teacher.npz"
    code = main(
        [
            "--calculator",
            "metatomic",
            "--checkpoint",
            str(checkpoint),
            "--structure",
            str(xyz),
            "-o",
            str(out),
            "--amplitude",
            "0.05",
            "--include-equilibrium",
            "--max-samples",
            "3",
            "--freq-min",
            "0",
        ]
    )
    assert code == 0
    saved = np.load(out)
    assert saved["R"].shape[0] == 3
    assert saved["displacements"].shape[1:] == (2, 3)
    assert np.linalg.norm(saved["R"][0] - atoms.positions) == pytest.approx(0.0)
