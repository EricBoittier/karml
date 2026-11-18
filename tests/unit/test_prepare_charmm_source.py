"""scripts/prepare_charmm_source.sh: extract without AppleDouble files, apply fixes, idempotent."""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_charmm_source.sh"
pytestmark = pytest.mark.skipif(not shutil.which("patch") or not shutil.which("bash"), reason="needs patch and bash")


def _make_tar(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "charmm"
    (src / "doc").mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("set(x 1)\n")
    (src / "doc" / "abpo.info").write_text("doc\n")
    (src / "doc" / "._abpo.info").write_text("appledouble\n")
    tar = tmp_path / "charmm.tar.xz"
    with tarfile.open(tar, "w:xz") as t:
        t.add(src, arcname="charmm")
    return tar


def _run(tmp_path, tar, patch, *args):
    env = dict(os.environ, CHARMM_TAR=str(tar), CHARMM_PATCH=str(patch), CHARMM_DEST=str(tmp_path / "out" / "charmm"))
    (tmp_path / "out").mkdir(exist_ok=True)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)


def test_extract_strips_appledouble_and_patch_is_idempotent(tmp_path):
    tar = _make_tar(tmp_path)
    patch = tmp_path / "fix.patch"
    patch.write_text(
        "--- a/CMakeLists.txt\n+++ b/CMakeLists.txt\n@@ -1 +1,2 @@\n set(x 1)\n+set(y 2)\n"
    )
    r = _run(tmp_path, tar, patch)
    assert r.returncode == 0, r.stderr
    dest = tmp_path / "out" / "charmm"
    assert (dest / "CMakeLists.txt").read_text() == "set(x 1)\nset(y 2)\n"
    assert not list(dest.rglob("._*"))
    assert (dest / "doc" / "abpo.info").exists()
    r2 = _run(tmp_path, tar, patch)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    assert (dest / "CMakeLists.txt").read_text() == "set(x 1)\nset(y 2)\n"


def test_shipped_patch_is_relative_and_clean():
    text = (ROOT / "setup" / "charmm_tarball_fixes.patch").read_text()
    assert "/mm" + "home" not in text and "boitt" + "ier" not in text
    assert "--- a/CMakeLists.txt" in text and "--- a/source/dynamc/dynio.F90" in text
