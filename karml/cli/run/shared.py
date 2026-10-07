"""Shared geometry helpers for the MD runners."""
from __future__ import annotations

from typing import Optional

import numpy as np


def _wrap_frame_by_monomer(
    positions: np.ndarray,
    cell: np.ndarray,
    monomer_offsets: np.ndarray,
    masses: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap positions so each monomer stays intact (no atoms in another cell)."""
    from karml.interfaces.pycharmmInterface.cell_list import _wrap_groups_np

    cell_matrix = np.asarray(cell, dtype=np.float64)
    if cell_matrix.ndim == 1 and cell_matrix.size >= 3:
        cell_matrix = np.diag(cell_matrix.flat[:3])
    elif cell_matrix.ndim == 2 and cell_matrix.shape[0] >= 3 and cell_matrix.shape[1] >= 3:
        cell_matrix = cell_matrix[:3, :3].copy()
    return _wrap_groups_np(
        np.asarray(positions, dtype=np.float64),
        cell_matrix,
        monomer_offsets,
        masses=masses,
    )
