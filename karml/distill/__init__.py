"""Teacher–student distillation helpers (PET → PhysNet NPZ).

Public API lives in ``geometry_pool``, ``teacher_label``, and ``npz_export``.
Molecule recipes (acetone COM windows, carbonyl stretch) live in
``karml.distill.examples``. Torch is not imported here.
"""

from karml.distill.geometry_pool import (
    POOL_PRESETS,
    Geometry,
    PoolConfig,
    build_geometry_pool,
)
from karml.distill.npz_export import write_distill_npz
from karml.distill.teacher_label import (
    ENERGY_MODE_INTERACTION,
    ENERGY_MODE_TOTAL,
    LabeledSample,
    label_geometries,
)

__all__ = [
    "ENERGY_MODE_INTERACTION",
    "ENERGY_MODE_TOTAL",
    "Geometry",
    "LabeledSample",
    "POOL_PRESETS",
    "PoolConfig",
    "build_geometry_pool",
    "label_geometries",
    "write_distill_npz",
]
