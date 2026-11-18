"""jax_md must import on CI (fresh uv resolve from pyproject.toml).

JAX 0.11.2 deleted ``jax.experimental.hijax.HiPrimitive`` (renamed ``HiPrim``).
flax 0.12.x still subclasses ``HiPrimitive`` when importing ``flax.nnx``, and
``jax_md.energy`` does that at import. Keep ``jax``/``jaxlib`` ``<0.11.2`` in
``pyproject.toml`` until flax follows.
"""

from __future__ import annotations

import os
from importlib.metadata import version

from jax.experimental import hijax
from jax_md import space
import pytest
from packaging.version import Version

# The default pin (jax<0.11.2) is the supported stack. The optional newer stack (docs/newer-jax-stack.md) is
# tested with KARML_NEWER_JAX_STACK=1, which replaces the two pin guards by the flax-matches-hijax check.
_NEWER = os.environ.get("KARML_NEWER_JAX_STACK", "") == "1"


@pytest.mark.skipif(_NEWER, reason="optional newer JAX stack")
def test_jax_below_hiprimitive_deletion() -> None:
    assert Version(version("jax")) < Version("0.11.2")
    assert Version(version("jaxlib")) < Version("0.11.2")


@pytest.mark.skipif(_NEWER, reason="optional newer JAX stack")
def test_hijax_exports_hiprimitive_for_flax() -> None:
    assert hasattr(hijax, "HiPrimitive")


def test_jax_md_space_imports() -> None:
    assert space is not None


def test_flax_matches_hijax_rename() -> None:
    # jax >= 0.11.2 only has HiPrim; flax must then be >= 0.12.10 (default stack: HiPrimitive still exists).
    if not hasattr(hijax, "HiPrimitive"):
        assert Version(version("flax")) >= Version("0.12.10")
