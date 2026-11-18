# Independent barostat: i-PI client for the karml hybrid ML/MM liquid

`karml/interfaces/ipi_client.py` is a custom PES for [i-PI](https://github.com/i-pi/i-pi) (development branch `next`): the same hybrid
ML/MM calculator and neighbour-list code as `karml md-system --backend jaxmd`, evaluated for positions and cell sent by an i-PI
server, so that an NpT (or NVE, PIMD/PIGLET) run uses i-PI's integrators and barostats instead of the JAX-MD or CHARMM ones.

## Install
```bash
pip install 'karml[ipi]'                                            # array-api-compat, greenlet (needed by i-PI next)
pip install git+https://github.com/i-pi/i-pi.git@next              # server + i-pi-py_driver
```

## Run (one job: server on CPU, client on the GPU, unix socket)
```bash
i-pi input.xml &                                                   # <ffsocket mode='unix' pbc='false'> with <address>karml-dcm</address>
i-pi-py_driver -m custom -P karml/interfaces/ipi_client.py -u -a karml-dcm \
   -o "composition=DCM:439,psf=model.psf,crd=start.crd,ckpt=model.json,box=36.3,log=client_log.csv"
```
Use `karml md-system`-compatible inputs (PSF/CRD/checkpoint/box) for the system; the client builds the calculator through the
`md-system` front end with the production switches (6.0 / 5.0 / 1.5 A), so the Hamiltonian is identical to the JAX-MD runs.

## Conventions (checked against `ipi/pes/_ase.py` and by finite differences)
* i-PI atomic units in and out; the client returns energy, forces and the **virial = -dE/d(strain)** (a 3x3 matrix; `vir = -sigma V` of the ASE
  calculator). For the cubic cell and the isotropic barostat the matrix is `-(dE/ds)/3 * I`, with `s` the affine **atomic** strain
  (positions and cell scaled together), from a forward-mode `jvp` of the calculator (exact; no finite difference). Cubic cells only.
* Molecules are wrapped whole into the box before each call (i-PI never wraps); the strain virial is invariant to this.
* The MM pair list is a Verlet list (cutoff + skin 0.25 A) reused while `2 max|R - R_ref L/L_ref| + r_list |L/L_ref-1| <= skin`; a full
  pair mask (possible overflow) raises instead of returning wrong forces.
* The i-PI `isotropic` barostat is the Bussi-Zykova-Parrinello one (use `flexible` for MTTK); i-PI next has `svr`, `langevin`, `pile_l/g`, `gle`/`nm_gle` thermostats, no Nose-Hoover chain.

## Validation recipe
Compare `E` and `dE/ds` with `karml md-system` stored frames (bit-level agreement expected in float64), FD of `E(s R, s L)`, and the i-PI
`pressure_md` against `(2 KE - dE/ds) / (3 V)` recomputed from `client_log.csv`.
