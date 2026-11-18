"""ML lattice-shift strain virial (``ml_lattice_virial``): ``autodiff(E)(box)`` with the ML cell detached plus the analytic
``ModelOutput.lattice_G`` equals plain ``autodiff(E)(box)`` through the PhysNet batch, to float64 round-off.

Heavy (a real 379/439-molecule hybrid box in float64 on CPU, ~2 min): runs only when KARML_TEST_LATTICE_BOX points to a directory
with model.psf/model.crd/box_length.txt/student.json (see scripts/check_lattice_virial.py,
which is the same check and also covers atom-wise wrapped (split-molecule) frames). Recorded result (18 Nov 2026): ACO 379 and DCM 439,
box as (3,) and (3,3), re-wrapped frame: max relative difference 7e-15, E and F identical to 6e-15.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

BOX = os.environ.get("KARML_TEST_LATTICE_BOX")
pytestmark = pytest.mark.skipif(not BOX, reason="set KARML_TEST_LATTICE_BOX=<dir with model.psf model.crd box_length.txt student.json residue.txt natom.txt>")


def test_lattice_virial_matches_autodiff(tmp_path):
    d = BOX
    L = float(open(os.path.join(d, "box_length.txt")).read())
    res_n = open(os.path.join(d, "residue.txt")).read().split()  # e.g. "ACO 379"
    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "check_lattice_virial.py")
    if not os.path.exists(script):
        pytest.skip("check_lattice_virial.py not shipped in this checkout")
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", KARML_ML_DTYPE="float64")
    out = tmp_path / "check.json"
    cmd = [sys.executable, script, str(out), "--", "md-system", "--backend", "jaxmd", "--setup", "pbc_nvt", "--ml-potential-mode", "physnet",
           "--from-psf", f"{d}/model.psf", "--from-crd", f"{d}/model.crd", "--skip-cluster-build", "--box-size", str(L),
           "--composition", f"{res_n[0]}:{res_n[1]}", "--no-mc-density-equalize", "--density-prep-mode", "off", "--checkpoint", f"{d}/student.json",
           "--ps", "0.05", "--no-pre-minimize", "--no-calculator-pre-minimize", "--no-charmm-pre-minimize", "--jaxmd-minimize-steps", "0",
           "--jaxmd-pbc-minimize-steps", "0", "--output-dir", str(tmp_path), "--extra-args", "--ml-dimer-cap-auto", "1.15"]
    subprocess.run(cmd, check=True, env=env, timeout=3600)
    import json
    r = json.load(open(out))
    for k, v in r.items():
        if isinstance(v, dict):
            assert v["dEdbox_rel_err"] < 1e-9, (k, v["dEdbox_rel_err"])
            assert v["F_maxdiff"] < 1e-9 * max(v["F_max"], 1.0)
