# Third-party software and data shipped in this repository

karml itself is MIT-licensed (see `LICENSE`). The repository also contains third-party material under its own terms.
This file records what is there; entries marked TODO need the maintainer's confirmation before a release.

| component | location | licence | note |
|---|---|---|---|
| CHARMM (c52a1 source tree, with MLpot/ML-MM C API patches) | `setup/charmm.tar.xz` (extracted to `setup/charmm/`), fixes in `setup/charmm_tarball_fixes.patch` | TODO: state the CHARMM licence under which redistribution is permitted and add its text or URL here | the tarball carries no top-level CHARMM licence file; the Fortran sources have no licence header |
| pyCHARMM (Python bindings, `setup/charmm/tool/pycharmm`) | inside the CHARMM tarball; also packaged as `pycharmm*` by `pyproject.toml` | GNU GPL v3 or later (file headers and `tool/pycharmm/LICENSE`, copyright J. Buckner) | GPL applies to that code; check the consequence for a combined MIT/GPL wheel before publishing to PyPI |
| Packmol | `karml/generate/packmol/` | MIT (`karml/generate/packmol/LICENSE`) | |
| CHARMM force-field files (CGenFF, topology/parameter files) | `karml/data/charmm/`, `setup/charmm/toppar` | CGenFF/CHARMM force-field terms of the MacKerell lab; TODO confirm | |
| CHARMM source overlays (nonbonded and image routines, incl. DCM/GRaPe modules) | `setup/nbonds/`, `setup/image/` | part of CHARMM: same TODO as the first row | |
