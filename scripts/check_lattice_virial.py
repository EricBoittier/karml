"""Regression: analytic ML lattice virial vs plain autodiff dE/dbox (and E, F identical) on a real hybrid ACO/DCM box.
Run via the md-system hook (same calculator as the MD): python check_lattice_virial.py <out.json> -- <md-system args>
f64 on CPU or GPU: JAX_ENABLE_X64=1 KARML_ML_DTYPE=float64.  Reports max |dE_old-dE_new| / max|dE_old| for box as 3-vector and 3x3 matrix."""
import json, os, sys, time
import numpy as np
OUT = sys.argv[1]; assert sys.argv[2] == "--"; MD_ARGS = sys.argv[3:]
import jax, jax.numpy as jnp
import karml.cli.run.md_pbc_suite.jaxmd as J
from karml.interfaces.pycharmmInterface.mm_energy_forces import refresh_mm_pairs
_orig = J.set_up_nhc_sim_routine
DT = jnp.float64 if os.environ.get("KARML_ML_DTYPE") == "float64" else jnp.float32

def hook(atoms, args, spherical_cutoff_calculator, get_update_fn, CUTOFF_PARAMS, n_monomers, monomer_offsets, **kw):
    Z = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
    R0 = np.asarray(atoms.get_positions())
    L = float(args.cell)
    rng = np.random.default_rng(1)
    res = {"n_atoms": len(atoms), "dtype": str(DT), "L": L}
    update_fn = get_update_fn(R0, CUTOFF_PARAMS)
    for tag in ("frame0", "frame_shifted"):
        R = R0 if tag == "frame0" else R0 + 0.3 * rng.standard_normal(R0.shape)   # jitter, some molecules split across faces
        if tag != "frame0":
            R = R % L                                                             # atom-wise wrap: splits molecules (exercises the whole-making shift)
        Rj = jnp.asarray(R, dtype=DT)
        for form in ("vec", "mat"):
            box = jnp.array([L, L, L], dtype=DT) if form == "vec" else jnp.eye(3, dtype=DT) * L
            pi, pm = refresh_mm_pairs(update_fn, Rj, np.array([L, L, L]), positions_are_cartesian=True)
            def calc(b, flag):
                return spherical_cutoff_calculator(atomic_numbers=Z, positions=Rj, n_monomers=n_monomers, cutoff_params=CUTOFF_PARAMS,
                    doML=True, doMM=True, doML_dimer=True, debug=False, mm_pair_idx=pi, mm_pair_mask=pm, box=b, ml_lattice_virial=flag)
            @jax.jit
            def old(b):
                def ea(bb):
                    o = calc(bb, False); return jnp.reshape(o.energy, (-1,))[0], o.forces
                (e, f), g = jax.value_and_grad(ea, has_aux=True)(b); return e, f, g
            @jax.jit
            def new(b):
                def ea(bb):
                    o = calc(bb, True); return jnp.reshape(o.energy, (-1,))[0], (o.forces, o.lattice_G)
                (e, (f, G)), g = jax.value_and_grad(ea, has_aux=True)(b)
                Gc = G if b.ndim == 2 else jnp.diagonal(G)
                return e, f, g + Gc, g, G
            t0 = time.time(); eo, fo, go = jax.block_until_ready(old(box)); t1 = time.time()
            en, fn, gn, g_rest, G = jax.block_until_ready(new(box)); t2 = time.time()
            go, gn = np.asarray(go), np.asarray(gn)
            key = f"{tag}_{form}"
            res[key] = {"E_old": float(eo), "E_new": float(en), "dE_abs": float(abs(eo - en)),
                        "F_maxdiff": float(np.max(np.abs(np.asarray(fo) - np.asarray(fn)))), "F_max": float(np.max(np.abs(np.asarray(fo)))),
                        "dEdbox_old": go.reshape(-1).tolist(), "dEdbox_new": gn.reshape(-1).tolist(),
                        "dEdbox_rel_err": float(np.max(np.abs(go - gn)) / max(np.max(np.abs(go)), 1e-30)),
                        "ad_rest_abs_max": float(np.max(np.abs(np.asarray(g_rest)))), "G_abs_max": float(np.max(np.abs(np.asarray(G)))),
                        "compile+run_s_old": t1 - t0, "compile+run_s_new": t2 - t1}
            print(key, json.dumps({k: v for k, v in res[key].items() if "dEdbox_o" not in k and "dEdbox_n" not in k}), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
    os._exit(0)

J.set_up_nhc_sim_routine = hook
from karml.cli.__main__ import cli
sys.argv = ["karml"] + MD_ARGS
sys.exit(cli())
