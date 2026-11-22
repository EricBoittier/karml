"""Analytic NpT virial for the shared jax-md driver.

jax-md differentiates the energy through ``perturbation = 1 + eps``. Doing that
with reverse mode pulls every term, including PhysNet, into the barostat
module. The chain rule splits the same derivative:

    dE/dp = (dE/dreal) · dreal/dp + (dE/dbox)|_real · dbox/dp

``dE/dreal`` is one force evaluation. ``dE/dbox`` is taken only on the terms
that actually read the box. Both are outlined (``inline=False``), so the
integrator compiles a call rather than the energy body.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax_md import space


def strain_box(box, perturbation):
    """Box jax-md means by ``perturbation``.

    Scalar ``p`` scales the box (``box * p``). A matrix ``P`` maps
    displacements as ``d -> P @ d``, i.e. ``box' = P @ box``.
    """
    box = jnp.asarray(box)
    if perturbation is None:
        return box
    p = jnp.asarray(perturbation)
    if p.ndim == 0:
        return box * p
    if box.ndim == 0:
        box_m = jnp.eye(p.shape[0], dtype=box.dtype) * box
    elif box.ndim == 1:
        box_m = jnp.diag(box)
    else:
        box_m = box
    return p @ box_m


def analytic_npt_energy(
    energy_of_real: Callable[..., Any],
    box_energy_of_real: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Fractional NpT energy with an analytic perturbation cotangent.

    ``energy_of_real(real, box, dyn) -> scalar`` is the full energy.
    ``box_energy_of_real`` is the part that depends on the box at fixed real
    positions (defaults to the full energy). ``dyn`` is the neighbor-list pytree
    the driver passes through as keywords.
    """
    if box_energy_of_real is None:
        box_energy_of_real = energy_of_real

    @jax.jit(inline=False)
    def _force_and_real(frac, box, perturbation, dyn):
        box_eff = strain_box(box, perturbation)
        real = space.transform(box_eff, frac)

        def e(R):
            return energy_of_real(R, box_eff, dyn)

        force = -jax.grad(e)(real)
        return force, real, box_eff

    @jax.jit(inline=False)
    def _box_grad(real, box_eff, dyn):
        real_fixed = jax.lax.stop_gradient(real)

        def e(b):
            return box_energy_of_real(real_fixed, b, dyn)

        return jax.grad(e)(box_eff)

    @jax.jit(inline=False)
    def _energy(frac, box, perturbation, dyn):
        box_eff = strain_box(box, perturbation)
        real = space.transform(box_eff, frac)
        return energy_of_real(real, box_eff, dyn)

    @jax.custom_vjp
    def _npt(frac, box, perturbation, dyn):
        return _energy(frac, box, perturbation, dyn)

    def _fwd(frac, box, perturbation, dyn):
        return _energy(frac, box, perturbation, dyn), (frac, box, perturbation, dyn)

    def _bwd(res, g):
        frac, box, perturbation, dyn = res
        force, real, box_eff = _force_and_real(frac, box, perturbation, dyn)
        # Position cotangent is dE/dreal. space.transform's JVP passes the
        # fractional tangent through unscaled, which is the force jax-md applies.
        grad_frac = jnp.asarray(-force * g)

        if perturbation is None:
            return grad_frac, None, None, None

        p = jnp.asarray(perturbation)
        _, vjp_real = jax.vjp(
            lambda pp: space.transform(strain_box(box, pp), frac), p
        )
        (d_pos,) = vjp_real(-force)
        g_box = _box_grad(real, box_eff, dyn)
        _, vjp_box = jax.vjp(lambda pp: strain_box(box, pp), p)
        (d_box,) = vjp_box(g_box)
        grad_pert = jnp.asarray((d_pos + d_box) * g)
        return grad_frac, None, grad_pert, None

    _npt.defvjp(_fwd, _bwd)

    def energy_fn(frac, box=None, perturbation=1.0, **dyn):
        return _npt(frac, box, perturbation, dyn)

    return energy_fn
