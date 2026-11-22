"""i-PI custom PES: the karml hybrid ML/MM calculator (spherical_cutoff_calculator) for a periodic liquid.

Run (client of an i-PI server, unix socket; i-PI branch `next` provides the custom-PES loader and batched requests):
  i-pi-py_driver -m custom -P karml/interfaces/ipi_client.py -u -a <socket> \
      -o "composition=DCM:439,psf=model.psf,crd=start.crd,ckpt=model.json,box=36.3,log=client_log.csv"
Parameters (key=value via -o): composition (e.g. DCM:439), psf, crd, ckpt, box (start cubic side, A), outdir (setup scratch),
  lam_sidecar (optional LJ-scale json), mm_switch_on/mm_switch_width/ml_switch_width (A; defaults 6.0/5.0/1.5), log (per-call csv).
Units: i-PI atomic units in/out.  Returns energy [Ha], forces [Ha/bohr] and the VIRIAL [Ha] as a 3x3 matrix = -dE/d(strain)
(i-PI convention, see ipi/pes/_ase.py: vir = -sigma V = -dE/deps).  For the cubic cell and the isotropic barostat only the trace is
used: the virial returned is  -(dE/ds)/3 * I,  where s is the affine ATOMIC strain (positions and cell scaled together), computed by a
forward-mode jvp of the production calculator (exact, no finite difference).  The calculator is built by running the production
`md-system` front end with a hook on set_up_nhc_sim_routine, so Hamiltonian, switches, LJ scales and neighbour-list code are the
ones of the JAX-MD runs.  Molecules are wrapped (centre into [0,L)) before each call; the strain virial is invariant to that.
"""
__DRIVER_NAME__ = "karml_hybrid"
__DRIVER_CLASS__ = "KarmlHybridDriver"

import json, os, sys, time
import numpy as np
try:
    from ipi.pes.dummy import Dummy_driver
    from ipi.utils.units import unit_to_user, unit_to_internal
except ImportError:  # i-PI (branch next) is an optional extra: `pip install karml[ipi]` + i-PI next
    Dummy_driver = object
    unit_to_user = unit_to_internal = None

def wrap_molecules(pos, L, nmol):
    """Shift each (whole) molecule by a lattice vector so that its geometric centre lies in [0, L). Returns wrapped (N,3) positions.
    i-PI never wraps, so molecules stay whole; the strain virial is invariant under this shift."""
    pos = np.asarray(pos, np.float64)
    R = pos.reshape(nmol, -1, 3)
    shift = L * np.floor(R.mean(1) / L)
    return (R - shift[:, None, :]).reshape(-1, 3)


def verlet_list_valid(pos, ref, L, L_ref, skin, r_list):
    """True while a Verlet list built at (ref, L_ref) with radius r_list (= cutoff + skin) is still exact for (pos, L):
    2 max|R - R_ref L/L_ref| + r_list |L/L_ref - 1| <= skin  (pos and ref are UNWRAPPED, i-PI positions)."""
    if ref is None or skin <= 0.0:
        return False
    sc = L / L_ref
    dr = float(np.sqrt(((np.asarray(pos) - np.asarray(ref) * sc) ** 2).sum(1).max()))
    return (2.0 * dr + r_list * abs(sc - 1.0)) <= skin


class _Captured(Exception):
    pass


_CAP = {}


def _capture_calculator(argv):
    import karml.cli.run.md_pbc_suite.jaxmd as J
    def hook(atoms, args, spherical_cutoff_calculator, get_update_fn, CUTOFF_PARAMS, n_monomers, monomer_offsets, **kw):
        _CAP.update(atoms=atoms, args=args, scc=spherical_cutoff_calculator, get_update_fn=get_update_fn,
                    cutoff=CUTOFF_PARAMS, n_monomers=int(n_monomers), offsets=np.asarray(monomer_offsets))
        raise _Captured()
    J.set_up_nhc_sim_routine = hook
    from karml.cli.__main__ import main as cli
    sys.argv = ["karml"] + argv
    try:
        cli()
    except BaseException as e:   # the cli front end may wrap/re-raise the hook's exception
        if not _CAP:
            raise RuntimeError(f"md-system exited before the calculator was built: {e!r}")
    if not _CAP:
        raise RuntimeError("md-system finished without building the calculator")


class KarmlHybridDriver(Dummy_driver):
    def __init__(self, composition, psf, crd, ckpt, box, outdir="/tmp/ipi_karml_setup", lam_sidecar="", mm_switch_width="5.0",
                 mm_switch_on="6.0", ml_switch_width="1.5", virial="jvp", log="", *args, **kwargs):
        if Dummy_driver is object:
            raise ImportError("i-PI is required: pip install 'karml[ipi]' and install i-PI (branch next), see docs/ipi.md")
        self.composition, self.psf, self.crd, self.ckpt = composition, psf, crd, ckpt
        self.box0 = float(box); self.outdir = outdir; self.lam = lam_sidecar
        self.mmw, self.mmon, self.mlw = mm_switch_width, mm_switch_on, ml_switch_width
        self.mmon_f = float(mm_switch_on); self.virial_mode = virial; self.logpath = log; self._logf = None
        self.ncall = 0; self.t_force = 0.0; self.nover = 0
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        os.makedirs(self.outdir, exist_ok=True)
        argv = ["md-system", "--seed", "1", "--backend", "jaxmd", "--ml-potential-mode", "physnet", "--setup", "pbc_npt",
                "--pbc-ensemble", "npt", "--ps", "0.001", "--dt-fs", "0.5", "--temperature", "300", "--pressure", "1",
                "--checkpoint", self.ckpt, "--from-psf", self.psf, "--from-crd", self.crd, "--skip-cluster-build",
                "--box-size", str(self.box0), "--composition", self.composition,
                "--mm-switch-on", self.mmon, "--mm-switch-width", self.mmw, "--ml-switch-width", self.mlw,
                "--no-calculator-pre-minimize", "--no-charmm-pre-minimize", "--jaxmd-minimize-steps", "0",
                "--jaxmd-pbc-minimize-steps", "0", "--no-mc-density-equalize", "--density-prep-mode", "off",
                "--charmm-image-mlpot-min-distance", "0", "--output-dir", self.outdir]
        if self.lam:
            argv += ["--mm-lj-scales-file", self.lam]
        _capture_calculator(argv)
        import jax, jax.numpy as jnp
        from karml.interfaces.pycharmmInterface.mm_energy_forces import refresh_mm_pairs
        self.jax, self.jnp, self._refresh = jax, jnp, refresh_mm_pairs
        a = _CAP["atoms"]; scc, cut, nm = _CAP["scc"], _CAP["cutoff"], _CAP["n_monomers"]
        self.nat = len(a); self.nmol = nm; self.apm = self.nat // nm
        assert self.apm * nm == self.nat
        self.masses = a.get_masses()
        self.Z = jnp.asarray(a.get_atomic_numbers(), dtype=jnp.int32)
        self.update_fn = _CAP["get_update_fn"](np.asarray(a.get_positions()), cut)
        Z = self.Z

        def energy(X, L, pidx, pmask):
            o = scc(atomic_numbers=Z, positions=X, n_monomers=nm, cutoff_params=cut, doML=True, doMM=True, doML_dimer=True,
                    mm_pair_idx=pidx, mm_pair_mask=pmask, box=jnp.full((3,), L))
            return jnp.asarray(o.energy, dtype=jnp.float64).reshape(-1)[0], o

        @jax.jit
        def evaluate(X, L, pidx, pmask):
            e, o = energy(X, L, pidx, pmask)
            F = jnp.asarray(o.forces, dtype=jnp.float64)
            # affine atomic strain: X -> s X, L -> s L  (dE/ds at s=1) by forward-mode AD
            _, dEds = jax.jvp(lambda s: energy(X * s, L * s, pidx, pmask)[0], (1.0,), (1.0,))
            return e, F, dEds

        @jax.jit
        def energy_only(X, L, pidx, pmask):
            return energy(X, L, pidx, pmask)[0]
        self._evaluate, self._energy_only = evaluate, energy_only
        self._cap_pairs = None; self._ref = None; self._Lref = None; self.nrebuild = 0
        a_ = _CAP['args']; self.skin = float(getattr(a_, 'jax_md_skin_distance', 0.25)); self.r_list = float(self.mmon_f + float(self.mmw) ) + self.skin

    def _prep(self, L_A, pos_A):
        """whole-molecule wrap of the (unwrapped, whole) molecules into the box; returns wrapped positions and the MM pair list.
        The list is a Verlet list (built at cutoff + skin, as in the JAX-MD runs: skin 0.25 A) and is reused while
        2*max|R - R_ref L/L_ref| + r_list*|L/L_ref - 1| <= skin (displacements of the UNWRAPPED positions, i-PI never wraps)."""
        pos_A = np.asarray(pos_A, np.float64)
        Rw = wrap_molecules(pos_A, L_A, self.nmol)
        box = np.array([L_A] * 3)
        reuse = verlet_list_valid(pos_A, self._ref, L_A, self._Lref, self.skin, self.r_list)
        if not reuse:
            self._pidx, self._pmask = self._refresh(self.update_fn, self.jnp.asarray(Rw), box, positions_are_cartesian=True)
            cap = int(self._pmask.shape[0]); nval = int(self.jnp.sum(self._pmask))
            if nval >= cap:     # a full mask means the list may have overflowed (silent wrong forces): refuse
                self.nover += 1
                raise RuntimeError(f"MM pair list full ({nval}/{cap}): possible overflow, rebuild with more capacity")
            self._cap_pairs = (nval, cap); self._ref = pos_A.copy(); self._Lref = float(L_A); self.nrebuild += 1
        return Rw, self._pidx, self._pmask

    def energy_forces_virial_A(self, L_A, pos_A):
        Rw, pidx, pmask = self._prep(L_A, pos_A)
        e, F, dEds = self._evaluate(self.jnp.asarray(Rw), float(L_A), pidx, pmask)
        e, F, dEds = float(e), np.asarray(F, np.float64), float(dEds)
        return e, F, dEds

    def compute_structure(self, cell, pos):
        t0 = time.perf_counter()
        cell_A = unit_to_user("length", "angstrom", np.asarray(cell).T)
        d = np.diag(cell_A)
        if np.abs(cell_A - np.diag(d)).max() > 1e-6 or np.ptp(d) > 1e-6:
            raise ValueError("karml_hybrid PES needs a cubic cell (isotropic barostat)")
        pos_A = unit_to_user("length", "angstrom", np.asarray(pos))
        e, F, dEds = self.energy_forces_virial_A(float(d[0]), pos_A)
        vir_eV = -(dEds / 3.0) * np.eye(3)          # -dE/d(strain) with strain isotropic: each diagonal gets 1/3
        self.ncall += 1; dt = time.perf_counter() - t0; self.t_force += dt
        if self.ncall % 200 == 0 and self.verbose:
            print(f"karml_hybrid calls={self.ncall} mean {1e3*self.t_force/self.ncall:.1f} ms/call pairs={self._cap_pairs}", flush=True)
        if self.logpath:
            if self._logf is None:
                self._logf = open(self.logpath, "a", buffering=1)
                if self._logf.tell() == 0:
                    self._logf.write("# call L_A E_eV dEds_eV P_cfg_atm t_s npairs\n")
            V = float(d[0]) ** 3
            self._logf.write(f"{self.ncall} {d[0]:.8f} {e:.8f} {dEds:.8f} {-dEds/(3*V)*1602176.634/1.01325:.4f} {dt:.4f} {self._cap_pairs[0]}\n")
        extras = json.dumps({"dEds_eV": dEds, "t_s": dt, "pairs": self._cap_pairs})
        return (np.asarray(unit_to_internal("energy", "electronvolt", e), np.float64),
                np.asarray(unit_to_internal("force", "ev/ang", F), np.float64),
                np.asarray(unit_to_internal("energy", "electronvolt", vir_eV.T), np.float64), extras)
