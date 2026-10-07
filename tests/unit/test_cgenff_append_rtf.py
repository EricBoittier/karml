"""Append-RTF helpers must strip a second CHARMM version line."""

from __future__ import annotations

from pathlib import Path


def test_rtf_path_for_append_strips_version_line(tmp_path: Path) -> None:
    from karml.interfaces.pycharmmInterface.nbonds_config import _rtf_path_for_append

    src = tmp_path / "extra.rtf"
    src.write_text(
        "* title\n*\n36  1\nRESI CH3CL 0.00\nEND\n",
        encoding="utf-8",
    )
    out = Path(_rtf_path_for_append(src))
    text = out.read_text(encoding="utf-8")
    assert "RESI CH3CL" in text
    assert "36" not in text.split("RESI")[0] or "36  1" not in text
    assert "36  1" not in text
    assert text.strip().startswith("*")


def test_rtf_path_for_append_noop_without_version(tmp_path: Path) -> None:
    from karml.interfaces.pycharmmInterface.nbonds_config import _rtf_path_for_append

    src = tmp_path / "extra.rtf"
    body = "* title\n*\nRESI FOO 0.00\nEND\n"
    src.write_text(body, encoding="utf-8")
    out = Path(_rtf_path_for_append(src))
    assert out.resolve() == src.resolve()
    assert "RESI FOO" in out.read_text(encoding="utf-8")


