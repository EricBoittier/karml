# Third-party software and data shipped in this repository

karml itself is MIT-licensed (see `LICENSE`). The repository also contains third-party material under its own terms.
This file records what is there; entries marked TODO need the maintainer's confirmation before a release.

| component | location | licence | note |
|---|---|---|---|
| CHARMM (c52a1 source tree, with MLpot/ML-MM C API patches) | `setup/charmm.tar.xz` (extracted to `setup/charmm/`), fixes in `setup/charmm_tarball_fixes.patch`, overlays in `setup/nbonds/`, `setup/image/` | CHARMM licence agreement, see the CHARMM project (<https://www.charmm.org/>) | the tarball carries no top-level CHARMM licence file and the Fortran sources have no licence header; see the statement below |
| pyCHARMM (Python bindings, `setup/charmm/tool/pycharmm`) | inside the CHARMM tarball | **GPL-3.0-or-later** (GNU General Public License v3 or later; file headers and `tool/pycharmm/LICENSE`, copyright J. Buckner) | source: `setup/charmm/tool/pycharmm/` after extracting `setup/charmm.tar.xz`; licence text: `setup/charmm/tool/pycharmm/LICENSE` and <https://www.gnu.org/licenses/gpl-3.0.html>. Kept in the repository tree under its own licence; **not packaged** into the karml wheel/sdist (`pyproject.toml` packages `karml*` only since #6). karml's own code is MIT; the GPL applies to pyCHARMM and to any distribution that bundles it |
| Packmol | `karml/generate/packmol/` | MIT (`karml/generate/packmol/LICENSE`) | |
| CHARMM force-field files (CGenFF, topology/parameter files) | `karml/data/charmm/`, `setup/charmm/toppar` | CGenFF/CHARMM force-field terms of the MacKerell lab; [maintainer to confirm] | |
| CHARMM source overlays (nonbonded and image routines, incl. DCM/GRaPe modules) | `setup/nbonds/`, `setup/image/` | part of CHARMM, see the CHARMM statement below | |

## CHARMM redistribution statement

[maintainer to confirm] Whether and on what terms the CHARMM sources in `setup/` may be redistributed. The exact licence text or URL, the permitted-redistribution wording and the date of the agreement
have not been verified from this repository; replace this line by the confirmed statement before the release is tagged.

## Wheel and sdist

The published wheel and sdist contain the `karml` package only (MIT). `setup/charmm.tar.xz`, `setup/charmm_tarball_fixes.patch`
and the pyCHARMM sources are part of the git repository (and its GitHub/Zenodo source archive) but not of the PyPI artefacts.
