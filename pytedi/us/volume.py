"""Pond volume from surface area.

pytedi relates dam capacity and surface area with a power law,
``Volume (ML) = a * SurfaceArea (m2) ** b``. Two relations are built in (:data:`RELATIONS`):

``"fowler2015"`` (the default)
    a = 0.0001042, b = 1.3213: equation 1 of Fowler et al. (2015) for hillside farm dams in
    Victoria, Australia, and the model's own default, so a dam list built from US data
    matches the model when ``vary_surfarea`` is switched on. Against Virginia's National
    Inventory of Dams records for private farm-type dams of 2 to 20 acres, it gives a median
    of 0.96 to 1.16 times the listed normal storage, depending on the size cut-off.

``"va_nid_farm"``
    a = 0.0122006, b = 0.845694: fitted here (:func:`fit_nid_relation`) to 476 private
    Virginia NID dams with a farm-type primary purpose (recreation, irrigation, fish and
    wildlife, fire protection/stock, other) and 0.26 to 20 acres of surface area
    (R2 = 0.55). Use with care below about 1 ha: the NID only lists dams that are tall and
    store at least 15 to 50 acre-feet, so its small ponds are unusually deep. For a 500 m2
    pond this relation implies a mean depth of about 4.7 m (Fowler: 0.8 m).

Any other ``(a, b)`` pair can be passed instead.

Fowler, K., Morden, R., Lowe, L., Nathan, R., 2015. Advances in assessing the impact of
hillside farm dams on streamflow. Australian Journal of Water Resources 19.
https://doi.org/10.1080/13241583.2015.1116182
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ACRE_FT_TO_ML = 1.2334818375
ACRE_TO_M2 = 4046.8564224
FARM_PURPOSES = ("Recreation", "Irrigation", "Fish and Wildlife Pond",
                 "Fire Protection, Stock, Or Small Fish Pond", "Other")
RELATIONS = {
    "fowler2015": (0.0001042, 1.3213),
    "va_nid_farm": (0.012200642448556794, 0.845694263169227),
}


def default_coefficients() -> tuple[float, float]:
    """The model's own (a, b), read from a fresh ``tedimodel`` so they never drift apart."""
    from ..main import tedimodel

    model = tedimodel()
    return model.vol_surfarea_a, model.vol_surfarea_b


def relation_coefficients(relation) -> tuple[float, float]:
    """(a, b) for a relation name in :data:`RELATIONS` or an explicit ``(a, b)`` pair."""
    if isinstance(relation, str):
        if relation not in RELATIONS:
            raise ValueError(f"Unknown volume relation {relation!r}; use one of {list(RELATIONS)} or (a, b).")
        return RELATIONS[relation]
    a, b = relation
    return float(a), float(b)


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


def fit_area_volume(area_m2, volume_ml) -> dict:
    """Least-squares fit of log(volume) on log(area): returns a, b, r2 and n."""
    x = np.log(np.asarray(area_m2, dtype=float))
    y = np.log(np.asarray(volume_ml, dtype=float))
    b, log_a = np.polyfit(x, y, 1)
    residuals = y - (log_a + b * x)
    r2 = 1.0 - residuals.var() / y.var() if y.var() > 0 else 1.0
    return {"a": float(np.exp(log_a)), "b": float(b), "r2": float(r2), "n": int(x.size)}


def fit_nid_relation(nid_gpkg: str | os.PathLike, state: str = "Virginia", max_acres: float = 20.0,
                     purposes=FARM_PURPOSES, owner: str = "Private") -> dict:
    """Fit ``V = a * A ** b`` to NID dams' surface area and normal storage.

    Parameters
    ----------
    nid_gpkg : path
        The national NID GeoPackage (:func:`~pytedi.us.fetch.nid_inventory_path`).
    state : str
        NID state name, e.g. "Virginia".
    max_acres : float
        Largest surface area to include.
    purposes : sequence of str
        Primary purposes to include (farm-type by default).
    owner : str
        Owner type the dam's owner list must contain.

    Returns
    -------
    dict with a, b (V in ML, A in m2), r2 and n.
    """
    import pyogrio

    columns = ["state", "surfaceArea", "normalStorage", "primaryPurposeId", "ownerTypeIds"]
    nid = pyogrio.read_dataframe(nid_gpkg, columns=columns, read_geometry=False)
    area = pd.to_numeric(nid["surfaceArea"], errors="coerce")
    storage = pd.to_numeric(nid["normalStorage"], errors="coerce")
    keep = ((nid["state"] == state) & (area > 0) & (area <= max_acres) & (storage > 0)
            & nid["primaryPurposeId"].isin(purposes) & nid["ownerTypeIds"].fillna("").str.contains(owner))
    if keep.sum() < 2:
        raise ValueError(f"Too few NID dams to fit for {state} (found {int(keep.sum())}).")
    return fit_area_volume(area[keep] * ACRE_TO_M2, storage[keep] * ACRE_FT_TO_ML)
