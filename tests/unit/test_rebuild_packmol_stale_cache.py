"""rebuild_packmol.sh drops a CMake cache that was generated from another checkout instead of failing on it."""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebuild_packmol.sh"


def test_stale_cache_from_another_checkout_is_reconfigured(tmp_path):
    build = tmp_path / "build"
    (build / "CMakeFiles").mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/somewhere/else/packmol\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("cmake", "gfortran"):
        stub = bindir / name
        stub.write_text("#!/bin/sh\n" + ('[ "$1" = "-S" ] && [ -f "$PB/CMakeCache.txt" ] && echo STALE_CACHE_KEPT >&2 && exit 1\n' if name == "cmake" else "") + "exit 0\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", PACKMOL_BUILD_DIR=str(build), PB=str(build), FC=str(bindir / "gfortran"))
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=120)
    assert "STALE_CACHE_KEPT" not in r.stderr, r.stderr
    assert "reconfiguring" in r.stdout
    assert not (build / "CMakeCache.txt").exists()
