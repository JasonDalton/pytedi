"""Pond volume from surface area.

pytedi relates dam capacity and surface area with a power law,
``Volume (ML) = a * SurfaceArea (m2) ** b``. The model's defaults, a = 0.0001042 and
b = 1.3213, are equation 1 of Fowler et al. (2015) for hillside farm dams in Victoria,
Australia. They are used here so a dam list built from US data matches the model's own
relation when ``vary_surfarea`` is switched on.

The result is an estimate: the relation was fitted to Australian farm dams, not US farm
ponds. Pass ``a`` and ``b`` for a local relation when one is available.

Fowler, K., Morden, R., Lowe, L., Nathan, R., 2015. Advances in assessing the impact of
hillside farm dams on streamflow. Australian Journal of Water Resources 19.
https://doi.org/10.1080/13241583.2015.1116182
"""

from __future__ import annotations

import numpy as np


def default_coefficients() -> tuple[float, float]:
    """The model's own (a, b), read from a fresh ``tedimodel`` so they never drift apart."""
    from ..main import tedimodel

    model = tedimodel()
    return model.vol_surfarea_a, model.vol_surfarea_b


def volume_ml_from_area(area_m2, a: float | None = None, b: float | None = None):
    """Capacity in ML from surface area in m2: ``a * area ** b`` (Fowler et al. 2015 by default)."""
    default_a, default_b = default_coefficients()
    a = default_a if a is None else a
    b = default_b if b is None else b
    area = np.asarray(area_m2, dtype=float)
    if np.any(area < 0):
        raise ValueError("Surface areas cannot be negative.")
    volume = a * area ** b
    return float(volume) if volume.ndim == 0 else volume
