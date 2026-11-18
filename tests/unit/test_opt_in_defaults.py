"""Opt-in performance switches must stay OFF by default (reported results used fd64 virial, no remat)."""
from __future__ import annotations

import inspect


def test_lattice_virial_is_opt_in_in_calculator():
    from karml.interfaces.pycharmmInterface import karml_calculator as kc

    src = inspect.getsource(kc)
    assert "ml_lattice_virial: bool = False" in src


def test_modeloutput_lattice_g_defaults_to_none():
    from karml.interfaces.pycharmmInterface.calculator_utils import ModelOutput

    assert ModelOutput._field_defaults["lattice_G"] is None


def test_runner_reads_lattice_virial_env_with_off_default():
    from karml.cli.run import jaxmd_runner

    src = inspect.getsource(jaxmd_runner)
    assert '(os.environ.get("KARML_ML_LATTICE_VIRIAL") or "0")' in src
