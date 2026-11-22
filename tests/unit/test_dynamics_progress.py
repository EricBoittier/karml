"""Dynamics progress lines stay visible when CHARMM itself is quiet."""

from __future__ import annotations

import time

from karml.interfaces.pycharmmInterface.mlpot.dynamics_progress import (
    _lock,
    _report_stall,
    _state,
    begin_force_call,
    dynamics_progress,
    end_force_call,
)


def test_quiet_dynamics_prints_start_each_call_and_finish(capsys) -> None:
    kw = {"timestep": 0.0005, "leap": True}
    with dynamics_progress(kw, nstep=1000, interval_s=0):
        begin_force_call()
        end_force_call()
        begin_force_call()
        end_force_call()
    out = capsys.readouterr().out
    assert "NVE running: 1000 steps × 0.50 fs = 0.50 ps" in out
    assert "CHARMM prints nothing else until this call returns" in out
    assert "force 1/1000 started" in out
    assert "2/1000 steps" in out
    assert "ns/day" in out
    assert "ETA" in out
    assert "NVE finished: 2/1000 steps, 0.00/0.50 ps" in out
    assert _state["armed"] is False


def test_running_call_reports_a_stall_without_a_new_step(capsys) -> None:
    kw = {"timestep": 0.0005, "start": True}
    with dynamics_progress(kw, nstep=200, interval_s=15):
        begin_force_call()
        with _lock:
            _state["last_print"] = time.monotonic() - 20
            _state["in_call"] = True
        _report_stall()
        end_force_call()
    out = capsys.readouterr().out
    assert "HEAT running:" in out
    assert "HEAT force 1/200 started" in out
    assert "HEAT still in force 1/200" in out
    assert "ns/day n/a, ETA n/a" in out
    assert "working; the next count prints when this call returns" in out
