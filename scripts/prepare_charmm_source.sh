#!/usr/bin/env bash
# Extract setup/charmm.tar.xz into setup/charmm and make it buildable.
#
#  1. Extracts without macOS AppleDouble ("._*") files (the shipped tarball contains thousands of them; they
#     break CHARMM's cmake: "add_custom_target cannot create target _docu_target").
#  2. Applies setup/charmm_tarball_fixes.patch (OUTPUT_NAME charmm, restart-lattice fix in dynio.F90, pyCHARMM
#     cmake/python fixes). Idempotent: skipped when already applied.
#
# Usage: scripts/prepare_charmm_source.sh [--force]   (--force: remove an existing setup/charmm first)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAR="${CHARMM_TAR:-$ROOT/setup/charmm.tar.xz}"
PATCH="${CHARMM_PATCH:-$ROOT/setup/charmm_tarball_fixes.patch}"
DEST="${CHARMM_DEST:-$ROOT/setup/charmm}"

if [[ "${1:-}" == "--force" ]]; then
  rm -rf "$DEST"
fi
FRESH=0
if [[ ! -f "$DEST/CMakeLists.txt" ]]; then
  FRESH=1
  [[ -f "$TAR" ]] || { echo "prepare_charmm_source: $TAR not found" >&2; exit 1; }
  echo "prepare_charmm_source: extracting $TAR"
  # (GNU tar warns about every LIBARCHIVE.xattr.* header of a macOS-made archive; filter that noise.)
  tar -xf "$TAR" -C "$(dirname "$DEST")" --exclude='._*' 2> >(grep -v 'unknown extended header' >&2)
fi
find "$DEST" -name '._*' -delete 2>/dev/null || true

if patch -p1 -R --dry-run -s -d "$DEST" < "$PATCH" >/dev/null 2>&1; then
  echo "prepare_charmm_source: tarball fixes already applied"
elif patch -p1 --dry-run -s -d "$DEST" < "$PATCH" >/dev/null 2>&1; then
  patch -p1 -s -d "$DEST" < "$PATCH"
  echo "prepare_charmm_source: applied $(basename "$PATCH")"
elif [[ "$FRESH" == 1 ]]; then
  echo "prepare_charmm_source: $(basename "$PATCH") does not apply to the freshly extracted tree" >&2
  exit 1
else
  # An existing tree that is neither pristine nor patched (local edits, other CHARMM version): leave it alone.
  echo "prepare_charmm_source: WARNING $DEST was modified locally; not applying $(basename "$PATCH") (use --force to re-extract)" >&2
fi
