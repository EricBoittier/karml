"""Compact CLI drawings for ``karml`` and a few themed subcommands.

Printed on a TTY (or when ``KARML_ASCII=1``). ``KARML_NO_ASCII`` and
``KARML_QUIET`` suppress them. ``NO_COLOR`` and ``KARML_NO_RICH`` keep the
drawing and drop ANSI color.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

_TRUTHY = frozenset({"1", "yes", "true"})

# fmt: off
_BANNERS: dict[str, str] = {
    "wordmark": (
        "╻┏  ┏━┓ ╻┏┓ ╻┏┳┓╻ ╻  \n"
        "┣┻┓ ┣━┫ ┣┻┫ ┃┃┃┃┃ ┃  \n"
        "╹ ╹ ╹ ╹ ╹ ╹ ╹╹ ╹╹ ┗━╸\n"
        "machine learning for molecules"
    ),
    "physnet-train": (
        "E │ ●\n"
        "  │  ╰─●╮\n"
        "  │     ╰─●──●\n"
        "  └──────────── step\n"
        "physnet-train"
    ),
    "md-system": (
        "┌─────────────────┐\n"
        "│   H─O─H         │\n"
        "│        · ·      │\n"
        "│     H─O─H    ·  │\n"
        "│ ·         H─C─H │\n"
        "└─────────────────┘\n"
        "md-system"
    ),
    "make-box": (
        "     O         O\n"
        "    / \\       / \\\n"
        "   H   H     H   H\n"
        "┌───────────────────┐\n"
        "│  H─O─H    H─O─H   │\n"
        "│      H─O─H        │\n"
        "└───────────────────┘\n"
        "make-box"
    ),
    "doctor": (
        "╭───────────────────╮\n"
        "│  ●  python        │\n"
        "│  ●  jax           │\n"
        "│  ●  charmm        │\n"
        "│  ●  packmol       │\n"
        "╰───────────────────╯\n"
        "doctor"
    ),
}
# fmt: on


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


def banner_text(name: str) -> str:
    """Return one drawing, including a single trailing newline."""

    try:
        text = _BANNERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_BANNERS))
        raise KeyError(f"unknown banner {name!r} (known: {known})") from exc
    return text if text.endswith("\n") else text + "\n"


def banner_enabled(*, stream: TextIO, quiet: bool = False) -> bool:
    """True when a banner should be written to *stream*."""

    if quiet or _truthy("KARML_QUIET") or _truthy("KARML_NO_ASCII"):
        return False
    if _truthy("KARML_ASCII"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _color_enabled(stream: TextIO) -> bool:
    if (os.environ.get("NO_COLOR") or "").strip():
        return False
    from karml.cli.help_style import _use_color

    return _use_color(stream)


def _write_colored(text: str, stream: TextIO) -> None:
    from rich.console import Console
    from rich.text import Text

    body, _, tag = text.rstrip("\n").rpartition("\n")
    styled = Text()
    if body:
        styled.append(body + "\n", style="bold cyan")
    styled.append(tag + "\n\n", style="dim")
    Console(
        file=stream,
        force_terminal=True,
        color_system="standard",
        soft_wrap=True,
    ).print(styled, end="")


def print_banner(name: str, *, stream: TextIO | None = None, quiet: bool = False) -> None:
    """Print *name* when the stream is a terminal and banners are allowed."""

    target = stream or sys.stdout
    if not banner_enabled(stream=target, quiet=quiet):
        return
    text = banner_text(name)
    if _color_enabled(target):
        _write_colored(text, target)
    else:
        target.write(text if text.endswith("\n\n") else text + "\n")
    target.flush()
