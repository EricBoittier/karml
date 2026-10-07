"""YAML ML/MM cuts. No CHARMM."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


def test_ml_cut_rejects_mm_region(tmp_path: Path) -> None:
    from karml.interfaces.pycharmmInterface.ml_cut import ml_cut_from_args

    cut = tmp_path / "cut.yaml"
    cut.write_text(
        "charge: 0\nspin_multiplicity: 1\nml_atoms:\n  - resname: HEME\n",
        encoding="utf-8",
    )
    args = Namespace(
        ml_cut=str(cut),
        mm_region="his93",
        _cluster_atom_names=["FE"],
        _cluster_atom_resnames=["HEME"],
        _cluster_atom_resids=[1],
        _cluster_atom_segids=["HEM"],
    )
    with pytest.raises(ValueError, match="replaces --mm-region"):
        ml_cut_from_args(args)
