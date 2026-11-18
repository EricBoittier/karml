"""The analytic ML lattice virial reaches the CHARMM CPT path only through its own explicit, experimental opt-in."""
from __future__ import annotations

import pytest

from karml.cli.run.md_system import perf_switch_summary
from karml.interfaces.pycharmmInterface.mlpot import hybrid_mlpot


def test_jaxmd_switch_does_not_enable_charmm_path(monkeypatch):
    monkeypatch.setenv("KARML_ML_LATTICE_VIRIAL", "1")
    monkeypatch.delenv("KARML_CHARMM_LATTICE_VIRIAL", raising=False)
    assert hybrid_mlpot._lattice_virial_enabled() is False


def test_charmm_opt_in_warns_experimental(monkeypatch):
    monkeypatch.setenv("KARML_CHARMM_LATTICE_VIRIAL", "1")
    monkeypatch.setattr(hybrid_mlpot, "_warned_experimental", False)
    with pytest.warns(UserWarning, match="EXPERIMENTAL"):
        assert hybrid_mlpot._lattice_virial_enabled() is True


def test_defaults_are_off_and_logged():
    s = perf_switch_summary({})
    assert "ML lattice virial=off" in s and "CHARMM-path lattice virial=off" in s and "ML chunk remat=off" in s
    assert "default (fd64" in s


def test_log_reports_active_switches():
    s = perf_switch_summary({"KARML_NPT_VIRIAL": "analytic", "KARML_ML_LATTICE_VIRIAL": "1", "KARML_ML_CHUNK_REMAT": "1"})
    assert "strain virial=analytic" in s and "ML lattice virial=ON" in s and "ML chunk remat=ON" in s
