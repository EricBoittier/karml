"""Visible progress for a CHARMM dynamics call that prints nothing else.

``--quiet`` sets CHARMM ``nprint`` to the full step count, so the terminal
sits on the last setup line for the whole trajectory. This module prints one
start line, a line every few seconds of force calls, and a line when a force
call is still inside the first compile.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

_DEFAULT_INTERVAL_S = 15.0

_lock = threading.Lock()
_thread_started = False
_state: dict[str, Any] = {
    "armed": False,
    "label": "dynamics",
    "nstep": 0,
    "timestep_ps": 0.0,
    "interval": _DEFAULT_INTERVAL_S,
    "n": 0,
    "in_call": False,
    "t0": 0.0,
    "last_print": 0.0,
}


def _env_interval() -> float:
    raw = (os.environ.get("KARML_DYNAMICS_PROGRESS_S") or "").strip()
    if not raw:
        return _DEFAULT_INTERVAL_S
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_INTERVAL_S


def _rank0() -> bool:
    for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID"):
        raw = os.environ.get(key)
        if raw is not None and raw.strip() not in ("", "0"):
            return False
    return True


def _label(kw: dict[str, Any]) -> str:
    if kw.get("cpt"):
        return "CPT"
    if kw.get("start"):
        return "HEAT"
    return "NVE"


def _timestep_ps(kw: dict[str, Any]) -> float:
    for key in ("timestep", "timestep_ps"):
        raw = kw.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _fmt_seconds(seconds: float) -> str:
    whole = max(0, int(seconds))
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def _elapsed(now: float) -> str:
    return _fmt_seconds(now - float(_state["t0"]))


def _perf(now: float, n_done: int) -> str:
    """ns/day and ETA from completed force calls. ``n_done`` must already be finished."""
    nstep = max(1, int(_state["nstep"]))
    dt_ps = float(_state["timestep_ps"])
    elapsed = max(now - float(_state["t0"]), 1e-9)
    if n_done <= 0 or dt_ps <= 0:
        return "ns/day n/a, ETA n/a"
    rate = n_done / elapsed
    ns_per_day = rate * dt_ps * 86.4
    eta = max(nstep - n_done, 0) / rate
    return f"{ns_per_day:.3g} ns/day, ETA {_fmt_seconds(eta)}"


def _emit(text: str) -> None:
    if not _rank0():
        return
    try:
        print(text, flush=True)
    except Exception:
        pass


def _step_line(n: int, now: float, *, starting: bool) -> str:
    nstep = max(1, int(_state["nstep"]))
    label = str(_state["label"])
    dt = float(_state["timestep_ps"])
    done_ps = n * dt
    total_ps = nstep * dt
    if starting and n == 1:
        return (
            f"{label} force 1/{nstep} started"
            + (f" ({total_ps:.2f} ps)" if total_ps > 0 else "")
            + " — this first call compiles and will not print again until it returns"
        )
    elapsed = max(now - float(_state["t0"]), 1e-9)
    rate = n / elapsed
    perf = _perf(now, n)
    if total_ps > 0:
        return (
            f"{label} {n}/{nstep} steps, {done_ps:.2f}/{total_ps:.2f} ps, "
            f"{rate:.0f} steps/s, {perf}, elapsed {_elapsed(now)}"
        )
    return (
        f"{label} {n}/{nstep} steps, {rate:.0f} steps/s, {perf}, "
        f"elapsed {_elapsed(now)}"
    )


def _stall_line(now: float) -> str:
    n = max(1, int(_state["n"]))
    nstep = max(1, int(_state["nstep"]))
    label = str(_state["label"])
    # The in-flight call has not finished, so the rate uses the previous count.
    perf = _perf(now, n - 1)
    return (
        f"{label} still in force {n}/{nstep}, {_elapsed(now)}, {perf} "
        "(working; the next count prints when this call returns)"
    )


def _ensure_thread() -> None:
    global _thread_started
    if _thread_started:
        return
    _thread_started = True

    def _watch() -> None:
        while True:
            with _lock:
                interval = float(_state["interval"])
            time.sleep(interval if interval > 0 else _DEFAULT_INTERVAL_S)
            _report_stall()

    threading.Thread(target=_watch, name="karml-dynamics-progress", daemon=True).start()


def _report_stall() -> None:
    with _lock:
        if not _state["armed"] or not _state["in_call"]:
            return
        interval = float(_state["interval"])
        if interval <= 0:
            return
        now = time.monotonic()
        if now - float(_state["last_print"]) < interval:
            return
        text = _stall_line(now)
        _state["last_print"] = now
    _emit(text)


def begin_force_call() -> None:
    """Count one CHARMM energy callback. No-op until a dynamics call is armed."""
    try:
        with _lock:
            if not _state["armed"]:
                return
            _state["n"] = int(_state["n"]) + 1
            _state["in_call"] = True
            now = time.monotonic()
            interval = float(_state["interval"])
            n = int(_state["n"])
            due = n == 1 or interval <= 0 or now - float(_state["last_print"]) >= interval
            text = _step_line(n, now, starting=(n == 1)) if due else None
            if due:
                _state["last_print"] = now
        if text:
            _emit(text)
    except Exception:
        return


def end_force_call() -> None:
    try:
        with _lock:
            _state["in_call"] = False
    except Exception:
        return


@contextmanager
def dynamics_progress(
    kw: dict[str, Any],
    *,
    nstep: int,
    interval_s: float | None = None,
) -> Iterator[None]:
    """Print a start line, let force calls report, then print when the call returns."""
    interval = _env_interval() if interval_s is None else float(interval_s)
    nstep = max(1, int(nstep))
    dt = _timestep_ps(kw)
    label = _label(kw)
    total_ps = nstep * dt
    dt_fs = dt * 1000.0
    with _lock:
        _state.update(
            armed=True,
            label=label,
            nstep=nstep,
            timestep_ps=dt,
            interval=interval,
            n=0,
            in_call=False,
            t0=time.monotonic(),
            last_print=time.monotonic(),
        )
    if interval > 0:
        _ensure_thread()
    if total_ps > 0 and dt_fs > 0:
        span = f"{nstep} steps × {dt_fs:.2f} fs = {total_ps:.2f} ps"
    else:
        span = f"{nstep} steps"
    every = f"every {interval:.0f} s" if interval > 0 else "on every force call"
    _emit(
        f"{label} running: {span}. "
        f"CHARMM prints nothing else until this call returns. "
        f"Progress {every}."
    )
    try:
        yield
    finally:
        with _lock:
            n = int(_state["n"])
            now = time.monotonic()
            elapsed = _elapsed(now)
            done_ps = n * float(_state["timestep_ps"])
            total_ps = nstep * float(_state["timestep_ps"])
            _state["armed"] = False
            _state["in_call"] = False
        if n <= 0:
            _emit(f"{label} finished with no force calls, elapsed {elapsed}")
        else:
            perf = _perf(now, n)
            if total_ps > 0:
                _emit(
                    f"{label} finished: {n}/{nstep} steps, "
                    f"{done_ps:.2f}/{total_ps:.2f} ps, {perf}, elapsed {elapsed}"
                )
            else:
                _emit(
                    f"{label} finished: {n}/{nstep} steps, {perf}, elapsed {elapsed}"
                )
