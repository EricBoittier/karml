"""TTY gate and layout for the CLI drawings."""

from __future__ import annotations

import io

from karml.cli.ascii_art import banner_text, print_banner
from karml.cli.__main__ import _KARMLTopLevelParser

_NAMES = ("wordmark", "physnet-train", "md-system", "make-box", "doctor")
_BOX_EDGES = ("┌", "│", "└", "╭", "╰")


class _TTY(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _clear_banner_env(monkeypatch) -> None:
    for name in (
        "KARML_ASCII",
        "KARML_NO_ASCII",
        "KARML_QUIET",
        "KARML_NO_RICH",
        "KARML_RICH",
        "NO_COLOR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_drawings_fit_and_name_their_command():
    for name in _NAMES:
        lines = banner_text(name).splitlines()
        assert lines
        assert all(len(line) <= 78 for line in lines)
    wordmark = banner_text("wordmark").splitlines()
    assert wordmark[-1] == "machine learning for molecules"
    assert all("KARML" not in line for line in wordmark[:-1])
    for name in ("physnet-train", "md-system", "make-box", "doctor"):
        assert banner_text(name).splitlines()[-1] == name


def test_boxed_drawings_keep_equal_rows():
    for name in ("md-system", "make-box", "doctor"):
        rows = [line for line in banner_text(name).splitlines() if line.startswith(_BOX_EDGES)]
        assert len(rows) >= 3
        assert len({len(line) for line in rows}) == 1


def test_print_banner_skips_non_tty(monkeypatch):
    _clear_banner_env(monkeypatch)
    buf = io.StringIO()
    print_banner("wordmark", stream=buf)
    assert buf.getvalue() == ""


def test_print_banner_honors_ascii_and_quiet_env(monkeypatch):
    _clear_banner_env(monkeypatch)
    shown = io.StringIO()
    monkeypatch.setenv("KARML_ASCII", "1")
    print_banner("md-system", stream=shown)
    assert "md-system" in shown.getvalue()
    assert "┌" in shown.getvalue()

    hidden = io.StringIO()
    monkeypatch.setenv("KARML_NO_ASCII", "1")
    print_banner("md-system", stream=hidden)
    assert hidden.getvalue() == ""

    monkeypatch.delenv("KARML_NO_ASCII")
    monkeypatch.delenv("KARML_ASCII")
    quiet_env = _TTY(True)
    monkeypatch.setenv("KARML_QUIET", "1")
    print_banner("doctor", stream=quiet_env)
    assert quiet_env.getvalue() == ""

    monkeypatch.delenv("KARML_QUIET")
    quiet_flag = _TTY(True)
    print_banner("physnet-train", stream=quiet_flag, quiet=True)
    assert quiet_flag.getvalue() == ""


def test_print_banner_no_rich_and_no_color_are_plain(monkeypatch):
    _clear_banner_env(monkeypatch)
    monkeypatch.setenv("KARML_NO_RICH", "1")
    plain = _TTY(True)
    print_banner("make-box", stream=plain)
    text = plain.getvalue()
    assert "make-box" in text
    assert "\x1b[" not in text

    monkeypatch.delenv("KARML_NO_RICH")
    monkeypatch.setenv("NO_COLOR", "1")
    no_color = _TTY(True)
    print_banner("make-box", stream=no_color)
    assert "make-box" in no_color.getvalue()
    assert "\x1b[" not in no_color.getvalue()


def test_top_level_print_help_includes_wordmark_on_tty(monkeypatch):
    _clear_banner_env(monkeypatch)
    monkeypatch.setenv("KARML_NO_RICH", "1")
    tty = _TTY(True)
    _KARMLTopLevelParser(prog="karml").print_help(tty)
    text = tty.getvalue()
    assert "machine learning for molecules" in text
    assert "usage: karml" in text
    assert text.index("machine learning for molecules") < text.index("usage: karml")

    plain = io.StringIO()
    _KARMLTopLevelParser(prog="karml").print_help(plain)
    help_text = plain.getvalue()
    assert "usage: karml" in help_text
    assert "machine learning for molecules" not in help_text
