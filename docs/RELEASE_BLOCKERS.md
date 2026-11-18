# Release blockers and known gaps (18 Nov 2026)

Status of the first release (`v0.1.0` proposed; nothing is tagged). Items marked **blocker** must be resolved or accepted before tagging.

## Needs a maintainer decision or data
| item | status |
|---|---|
| **blocker** CITATION.cff placeholders (title, co-authors, ORCID, version/date) and Zenodo toggle | PR #6 (open) |
| **blocker** CHARMM licence statement in `THIRD_PARTY_NOTICES.md` (sources carry no licence file) | PR #6 (open) |
| **blocker** pyCHARMM is GPL-3.0-or-later and is packaged (`pycharmm*`) into the MIT wheel | decide: exclude from wheel / relicense / source-only release |
| **blocker** >= 20 ps NpT density comparison, default vs newer JAX stack (sciCORE job 33294888) | running; newer stack stays optional until read |
| **blocker** CHARMM-path analytic lattice virial: not validated (frame comparison float64 1e-8 / float32 0.1 atm, timed 100-step CPT run on acetone and DCM). Gated behind `KARML_CHARMM_LATTICE_VIRIAL=1` (experimental) | needs a CHARMM build and GPU |
| learned-charges work and call-1 virial check (uncommitted in a side checkout) not part of this release | decide whether the tag waits |
| fresh-clone `make install-native` end to end with `prepare_charmm_source.sh` | CI "PyCHARMM smoke" now exercises it; confirm once green |

## CI
| job | before | after this series |
|---|---|---|
| Lint | red (24 ruff errors) | green (#1) |
| Unit tests | aborted at collection; 11 failures | collection fixed (#1); 7 size-ratchet caps re-recorded (debt register: the functions grew in the 5 Oct port), 2 `--backend ase` tests moved to `jaxmd`, 2 source-inspection tests updated to the current runner code |
| Functionality tests | `test_mm_molpair` used a 15 A box that the L/2 guard now rejects | box enlarged to 20 A |
| PyCHARMM smoke | CHARMM cmake failed on AppleDouble `._*` files in the tarball | `scripts/ci/setup_charmm_lib.sh` now runs `scripts/prepare_charmm_source.sh` (deletes `._*`, applies the fixes) even after the workflow's own `tar`; `.github/workflows/ci.yml` itself was NOT edited (the push token has no `workflow` scope): optionally replace its `tar -xf` step by `bash scripts/prepare_charmm_source.sh` |
| Generated content | needed removed `docs/cli/` and figure assets | CLI, crystal-table and figure generators dropped from the refresh job (their targets are not in this snapshot); `docs/package-architecture.md` regenerated |
| MkDocs HTML / PDF | skipped behind the failing job | RED, not fixed: `mkdocs build --strict` reports 118 warnings, all broken links/images to material deliberately left out of the snapshot (`cli/*`, `examples.md`, `images/*`, `plot-style-gallery-assets/*`). HANDOFF.md says to delete such pages rather than patch them; that is a docs clean-up PR (or a non-strict build, which needs a workflow edit that this token cannot push) |

## History and public materials (cannot be fixed without rewriting history)
* `docs/manuscripts/condensed-phase-hybrid-mlmm/` (draft manuscript, 72 files, about 6.5 MB: LaTeX, 28 PNG, 23 PDF, notes) is public on `main` and in history (commits `87a7c51`, `e64c0b9`). It was left in place.
* Personal home/cluster paths, user names and the earlier `.cursor/mcp.json` stay in history (13 commits total); the tip is cleaned by the hygiene PR (#5).
* Commit metadata carries the maintainer's e-mail (`eric.boittier@icloud.com`, 9 commits; the other 5 use the GitHub no-reply address).
* Options: (a) leave as is; (b) move the draft manuscript to a private repository and delete it from the tip (history keeps it); (c) rewrite history with `git filter-repo` and force-push, which breaks forks, clones and any link to `87a7c51` (the commit cited in the manuscript bibliography), so it needs an explicit decision.
