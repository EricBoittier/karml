# karml 0.1.0

First tagged, citable release of karml: machine-learned potentials (PhysNet, DCMNet and others) in molecular-mechanics MD,
with two engines (CHARMM + MLpot and JAX-MD) sharing one model code. See `CHANGELOG.md` for the full history.

## Highlights of this release
- Opt-in analytic ML lattice strain virial for JAX-MD NpT (`KARML_NPT_VIRIAL=analytic KARML_ML_LATTICE_VIRIAL=1`): forces and strain virial in one
  PhysNet pass, validated against the float64 reference; 1.3-1.6x faster NpT on an H200, 3.7x on a 1/64-FP64 GPU. Defaults are unchanged (`fd64` virial).
- Persistent JAX compilation cache by default for `md-system --backend jaxmd` and the trainer (`KARML_NO_JAX_COMPILATION_CACHE=1` disables).
- Opt-in `KARML_ML_CHUNK_REMAT=1` for lower memory in the strain-virial reverse pass; `md-system` logs which virial path and opt-ins are active.
- Clean install fixed: the CHARMM tarball is extracted without macOS `._*` files and with the 15 stale-file fixes (`make install-native`).
- GPU cluster runbook (`docs/gpu-cluster-runbook.md`) and an optional, experimental newer JAX / flax / jax-md stack (`docs/newer-jax-stack.md`).
- Repository hygiene (no personal paths), CI repaired where possible (`docs/RELEASE_BLOCKERS.md` lists what is left).

## Known limitations
- The analytic lattice virial in the CHARMM CPT path is experimental and not validated (`KARML_CHARMM_LATTICE_VIRIAL=1`).
- The learned-charges work, Ewald-tail variants and the call-1 virial check are not part of this release.
- The newer JAX stack is optional; the default pin stays `jax<0.11.2`.

## Licences
karml is MIT. The CHARMM sources and pyCHARMM bindings under `setup/` keep their own terms (`THIRD_PARTY_NOTICES.md`); pyCHARMM
(GPL-3.0-or-later) is not part of the wheel/sdist.

## Citation
See `CITATION.cff`; the DOI is added after Zenodo archives the release.
