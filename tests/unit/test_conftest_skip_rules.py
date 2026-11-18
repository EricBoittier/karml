"""The conftest turns missing-input failures into explicit skips, and nothing else."""
from __future__ import annotations

from tests.conftest import _CHARMM_UNAVAILABLE_SIGNATURES, missing_input_reason


def test_missing_charmm_tree_and_checkpoint_are_skips():
    assert missing_input_reason(FileNotFoundError("[Errno 2] No such file: '/x/setup/charmm/tool/pycharmm/a.py'"))
    assert missing_input_reason(RuntimeError("Checkpoint not found: examples/ckpts_json/DESdimers_params.json"))
    assert missing_input_reason(FileNotFoundError("No checkpoint found. Set KARML_CKPT or pass --checkpoint."))


def test_real_errors_are_not_skipped():
    assert missing_input_reason(ValueError("setup/charmm/ in a message but not an OS error")) is None
    assert missing_input_reason(RuntimeError("energy conservation violated")) is None
    assert missing_input_reason(AssertionError("Checkpoint not found")) is None


def test_bare_pycharmm_import_error_is_a_charmm_signature():
    assert "No module named 'pycharmm'" in _CHARMM_UNAVAILABLE_SIGNATURES
