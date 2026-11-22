"""Triton LJ + Coulomb tail for the dynamic MM pair list.

Blocked Triton was the fastest correct pair force on the 308-DCM cell
(about 0.4 ms versus 5 ms for JAX ``value_and_grad`` on 633k pairs). The
launch goes through ``buffer_callback`` so it uses the step's device buffers.
``KARML_MM_TRITON=1`` selects it.

The scale is the complementary COM handoff. Pair forces are multiplied by
that scale, and ``d(scale)/d(r_com)`` is spread over both monomers.
Orthorhombic cells. Distances use a hard 0.6 Å floor instead of the JAX
softplus.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax.experimental.buffer_callback import buffer_callback
from triton.language.extra import libdevice

COULOMB_K = 332.063711


def triton_mm_enabled() -> bool:
    return (os.environ.get("KARML_MM_TRITON") or "").strip().lower() in ("1", "true", "yes", "on")


@triton.jit
def _mic(delta, length):
    return delta - length * libdevice.rint(delta / length)


@triton.jit
def _sharp(dist, x0, x1, gamma):
    den = x1 - x0
    s0 = tl.minimum(tl.maximum((dist - x0) / den, 0.0), 1.0)
    inside = (dist > x0) & (dist < x1)
    ds0 = tl.where(inside, 1.0 / den, 0.0)
    sp = tl.where(s0 > 0.0, libdevice.pow(s0, gamma), 0.0)
    dsp = gamma * tl.where(s0 > 0.0, libdevice.pow(s0, gamma - 1.0), 0.0) * ds0
    h = sp * sp * sp * (10.0 + sp * (-15.0 + 6.0 * sp))
    dh = (30.0 * sp * sp * (1.0 - sp) * (1.0 - sp)) * dsp
    return h, dh


@lru_cache(maxsize=8)
def _kernel(natom: int):
    @triton.jit
    def mm(
        pos, pairs, mask, q, rmin, eps, monomer, offsets, lam, box,
        force, energy, n_pairs,
        on: tl.constexpr, width: tl.constexpr, ml_width: tl.constexpr,
        kcoul: tl.constexpr, nat: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = row < n_pairs
        m = tl.load(mask + row, mask=valid, other=0.0)
        i = tl.load(pairs + row * 2, mask=valid, other=0).to(tl.int32)
        j = tl.load(pairs + row * 2 + 1, mask=valid, other=0).to(tl.int32)
        valid = valid & (m > 0.0) & (i < j)
        bx = tl.load(box + 0)
        by = tl.load(box + 1)
        bz = tl.load(box + 2)
        dx = _mic(tl.load(pos + i * 3, mask=valid, other=0.0) - tl.load(pos + j * 3, mask=valid, other=0.0), bx)
        dy = _mic(tl.load(pos + i * 3 + 1, mask=valid, other=0.0) - tl.load(pos + j * 3 + 1, mask=valid, other=0.0), by)
        dz = _mic(tl.load(pos + i * 3 + 2, mask=valid, other=0.0) - tl.load(pos + j * 3 + 2, mask=valid, other=0.0), bz)
        r = tl.maximum(tl.sqrt(dx * dx + dy * dy + dz * dz), 0.6)
        r2s = r * r
        sig = tl.load(rmin + i, mask=valid, other=0.0) + tl.load(rmin + j, mask=valid, other=0.0)
        ep = tl.sqrt(tl.load(eps + i, mask=valid, other=0.0) * tl.load(eps + j, mask=valid, other=0.0))
        qq = tl.load(q + i, mask=valid, other=0.0) * tl.load(q + j, mask=valid, other=0.0)
        mi = tl.load(monomer + i, mask=valid, other=0).to(tl.int32)
        mj = tl.load(monomer + j, mask=valid, other=0).to(tl.int32)
        valid = valid & (mi != mj)
        pair_lam = tl.load(lam + mi, mask=valid, other=0.0) * tl.load(lam + mj, mask=valid, other=0.0)
        ep = ep * pair_lam
        qq = qq * pair_lam
        sr = sig / r
        sr2 = sr * sr
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        e_pair = ep * (sr12 - 2.0 * sr6) + kcoul * qq / r
        de_over_r = 12.0 * ep * (sr6 - sr12) / r2s - (kcoul * qq) / (r2s * r)

        si = tl.load(offsets + mi, mask=valid, other=0).to(tl.int32)
        sj = tl.load(offsets + mj, mask=valid, other=0).to(tl.int32)
        cx = tl.zeros([BLOCK], tl.float32)
        cy = tl.zeros([BLOCK], tl.float32)
        cz = tl.zeros([BLOCK], tl.float32)
        gx = tl.zeros([BLOCK], tl.float32)
        gy = tl.zeros([BLOCK], tl.float32)
        gz = tl.zeros([BLOCK], tl.float32)
        for t in range(nat):
            ai = si + t
            aj = sj + t
            cx += tl.load(pos + ai * 3, mask=valid, other=0.0)
            cy += tl.load(pos + ai * 3 + 1, mask=valid, other=0.0)
            cz += tl.load(pos + ai * 3 + 2, mask=valid, other=0.0)
            gx += tl.load(pos + aj * 3, mask=valid, other=0.0)
            gy += tl.load(pos + aj * 3 + 1, mask=valid, other=0.0)
            gz += tl.load(pos + aj * 3 + 2, mask=valid, other=0.0)
        inv_n = 1.0 / nat
        cx *= inv_n
        cy *= inv_n
        cz *= inv_n
        gx *= inv_n
        gy *= inv_n
        gz *= inv_n
        rx = _mic(cx - gx, bx)
        ry = _mic(cy - gy, by)
        rz = _mic(cz - gz, bz)
        rc = tl.maximum(tl.sqrt(rx * rx + ry * ry + rz * rz), 1e-8)
        h1, dh1 = _sharp(rc, on - ml_width, on, 1.0)
        h2, dh2 = _sharp(rc, on, on + width, 3.0)
        scale = h1 * (1.0 - h2)
        dscale = dh1 * (1.0 - h2) - h1 * dh2
        e_sw = tl.where(valid, e_pair * scale, 0.0)
        tl.atomic_add(energy, tl.sum(e_sw, axis=0))

        coef = tl.where(valid, scale * de_over_r, 0.0)
        tl.atomic_add(force + i * 3, coef * dx, mask=valid)
        tl.atomic_add(force + i * 3 + 1, coef * dy, mask=valid)
        tl.atomic_add(force + i * 3 + 2, coef * dz, mask=valid)
        tl.atomic_add(force + j * 3, -coef * dx, mask=valid)
        tl.atomic_add(force + j * 3 + 1, -coef * dy, mask=valid)
        tl.atomic_add(force + j * 3 + 2, -coef * dz, mask=valid)

        spread = tl.where(valid, e_pair * dscale / rc, 0.0) * inv_n
        for t in range(nat):
            ai = si + t
            aj = sj + t
            tl.atomic_add(force + ai * 3, spread * rx, mask=valid)
            tl.atomic_add(force + ai * 3 + 1, spread * ry, mask=valid)
            tl.atomic_add(force + ai * 3 + 2, spread * rz, mask=valid)
            tl.atomic_add(force + aj * 3, -spread * rx, mask=valid)
            tl.atomic_add(force + aj * 3 + 1, -spread * ry, mask=valid)
            tl.atomic_add(force + aj * 3 + 2, -spread * rz, mask=valid)

    return mm


def _as_cupy(buf):
    import cupy as cp

    iface = buf.__cuda_array_interface__
    ptr = int(iface["data"][0])
    shape = tuple(int(s) for s in iface["shape"])
    if not shape:
        shape = (1,)
    dtype = np.dtype(iface["typestr"])
    nbytes = int(np.prod(shape)) * dtype.itemsize
    mem = cp.cuda.UnownedMemory(ptr, max(nbytes, 1), buf)
    return cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0))


def _wait_stream(stream) -> None:
    if stream is None:
        return
    try:
        handle = int(stream)
    except (TypeError, ValueError):
        return
    if handle == 0:
        return
    import cupy as cp

    cp.cuda.ExternalStream(handle).synchronize()


class _Gpu:
    def __init__(self, array):
        self._array = array
        self.dtype = getattr(array, "dtype", None)

    def data_ptr(self):
        return int(self._array.data.ptr)


def make_triton_mm_call(
    *,
    natom: int,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
):
    """JAX callable: pair list in, switched energy and forces out."""
    kernel = _kernel(int(natom))
    on = float(mm_switch_on)
    width = float(mm_switch_width)
    ml_width = float(ml_switch_width)
    nat = int(natom)
    calls: dict[int, object] = {}

    def _launch(ctx, out, pos, pairs, mask, box, q, rmin, eps, monomer, offsets, lam):
        import cupy as cp

        # Triton has no torch here, so it launches on the legacy default stream.
        # Wait for the XLA stream that produced the inputs, then wait for the
        # default stream before returning so the next JAX kernel sees the forces.
        _wait_stream(getattr(ctx, "stream", None))
        energy_buf, force_buf = out
        energy = _as_cupy(energy_buf)
        force = _as_cupy(force_buf)
        energy.fill(0)
        force.fill(0)
        pos_c, pairs_c, mask_c, box_c, q_c, rmin_c, eps_c, mon_c, off_c, lam_c = (
            _as_cupy(buf) for buf in (pos, pairs, mask, box, q, rmin, eps, monomer, offsets, lam)
        )
        n_pairs = int(pairs_c.shape[0])
        block = 128
        kernel[((n_pairs + block - 1) // block,)](
            _Gpu(pos_c), _Gpu(pairs_c.reshape(-1)), _Gpu(mask_c.reshape(-1)),
            _Gpu(q_c.reshape(-1)), _Gpu(rmin_c.reshape(-1)), _Gpu(eps_c.reshape(-1)),
            _Gpu(mon_c.reshape(-1)), _Gpu(off_c.reshape(-1)), _Gpu(lam_c.reshape(-1)),
            _Gpu(box_c.reshape(-1)),
            _Gpu(force.reshape(-1)), _Gpu(energy.reshape(-1)), n_pairs,
            on=on, width=width, ml_width=ml_width, kcoul=COULOMB_K, nat=nat, BLOCK=block,
        )
        cp.cuda.Stream.null.synchronize()

    def run(positions, pair_idx, pair_mask, box_lengths, q, rmin, eps, monomer_id, offsets, lam):
        n = int(positions.shape[0])
        fn = calls.get(n)
        if fn is None:
            fn = buffer_callback(
                _launch,
                (
                    jax.ShapeDtypeStruct((), jnp.float32),
                    jax.ShapeDtypeStruct((n, 3), jnp.float32),
                ),
                has_side_effect=True,
            )
            calls[n] = fn
        return fn(
            positions.astype(jnp.float32),
            pair_idx.astype(jnp.int32),
            pair_mask.astype(jnp.float32),
            box_lengths.astype(jnp.float32),
            q.astype(jnp.float32),
            rmin.astype(jnp.float32),
            eps.astype(jnp.float32),
            monomer_id.astype(jnp.int32),
            offsets.astype(jnp.int32),
            lam.astype(jnp.float32),
        )

    return run
