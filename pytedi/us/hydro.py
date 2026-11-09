"""Flow routing on a DEM and the pond network pytedi needs.

:func:`flow_grids` conditions the DEM (fill pits, fill depressions, resolve flats) and
computes D8 flow directions and flow accumulation with ``pysheds``.

:func:`link_ponds` then, for each pond:

- takes as its outlet the pond cell with the largest flow accumulation (every cell the
  pond touches counts, so ponds smaller than a cell still get one), which snaps the outlet
  to where the drainage leaves the pond;
- sets ``catchuptotal_km2`` to the area draining to that outlet, including the pond and
  any ponds upstream (never less than the pond's own area);
- follows the flow path from the outlet to the first other pond it enters, which becomes
  ``ds_damid``; a path that leaves the DEM or reaches missing data ends at the catchment
  outlet, ``"-99"``;
- sets ``catchuplocal_km2`` to the total area less the total areas of the ponds directly
  upstream, the same rule pytedi uses for STEDI dam lists (``damslist.damlist_from_list``).

Why pysheds: it is pip-installable, pure Python with numba (already a pytedi dependency),
and works on in-memory arrays. WhiteboxTools needs a separate binary download at run time.
Flow accumulation is computed here (:func:`d8_accumulation`) rather than with pysheds,
whose accumulation calls ``numpy.in1d``, removed in NumPy 2.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features

OUTLET = "-99"
# pysheds' default D8 codes, in its order: N, NE, E, SE, S, SW, W, NW.
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
STEPS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}
LINK_COLUMNS = ["dam_id", "ds_damid", "catchuptotal_km2", "catchuplocal_km2", "outlet_x", "outlet_y", "outlet_found"]


@dataclass
class FlowGrids:
    """D8 flow directions and accumulation on the DEM's grid.

    Attributes
    ----------
    fdir : 2-D int array of pysheds D8 codes (0 or negative where undefined)
    acc : 2-D float array, number of cells draining through each cell, itself included
    valid : 2-D bool array, True where the DEM has data
    transform : affine.Affine of the grid
    crs : the grid's CRS
    cell_area_m2 : float
    """

    fdir: np.ndarray
    acc: np.ndarray
    valid: np.ndarray
    transform: object
    crs: object
    cell_area_m2: float


def flow_grids(dem) -> FlowGrids:
    """Condition a DEM and compute D8 flow directions and accumulation.

    Parameters
    ----------
    dem : xarray.DataArray
        Elevation with a projected CRS in meters (rioxarray), NaN where there is no data;
        :func:`~pytedi.us.fetch.fetch_3dep` returns one.
    """
    import pyproj
    from pysheds.grid import Grid
    from pysheds.sview import Raster, ViewFinder

    crs = dem.rio.crs
    if crs is None or not pyproj.CRS.from_user_input(crs).is_projected:
        raise ValueError("The DEM needs a projected CRS in meters, such as EPSG:5070.")
    transform = dem.rio.transform()
    z = np.asarray(dem.values, dtype="float64")
    valid = np.isfinite(z)
    nodata = -9999.0
    z = np.where(valid, z, nodata)

    view = ViewFinder(affine=transform, shape=z.shape, crs=pyproj.Proj(pyproj.CRS.from_user_input(crs)),
                      nodata=nodata)
    raster = Raster(z, viewfinder=view)
    grid = Grid.from_raster(raster)
    conditioned = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(raster)))
    fdir = np.asarray(grid.flowdir(conditioned, dirmap=DIRMAP)).astype("int64")
    fdir[~valid] = 0
    acc = d8_accumulation(fdir, valid)
    cell_area = abs(transform.a * transform.e)
    return FlowGrids(fdir, acc, valid, transform, crs, cell_area)


def downstream_index(fdir: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Flat index of the cell each cell drains to, or -1 (edge, missing data, no direction)."""
    rows, cols = fdir.shape
    r, c = np.indices(fdir.shape)
    dr = np.zeros(fdir.shape, dtype="int64")
    dc = np.zeros(fdir.shape, dtype="int64")
    for code, (sr, sc) in STEPS.items():
        hit = fdir == code
        dr[hit], dc[hit] = sr, sc
    has_dir = np.isin(fdir, DIRMAP) & valid
    nr, nc = r + dr, c + dc
    inside = has_dir & (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
    target = np.full(fdir.shape, -1, dtype="int64")
    target[inside] = (nr * cols + nc)[inside]
    ok = target >= 0
    target[ok] = np.where(valid.ravel()[target[ok]], target[ok], -1)
    return target.ravel()


def d8_accumulation(fdir: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Number of valid cells draining through each cell, itself included (0 where invalid)."""
    ds = downstream_index(fdir, valid)
    acc = valid.ravel().astype("float64")
    indegree = np.bincount(ds[ds >= 0], minlength=ds.size)
    frontier = np.flatnonzero(valid.ravel() & (indegree == 0))
    while frontier.size:  # one pass per step along the longest flow path
        frontier = frontier[ds[frontier] >= 0]
        targets = ds[frontier]
        np.add.at(acc, targets, acc[frontier])
        np.subtract.at(indegree, targets, 1)
        frontier = np.unique(targets[indegree[targets] == 0])
    return acc.reshape(fdir.shape)


def _label_ponds(ponds: gpd.GeoDataFrame, grids: FlowGrids) -> np.ndarray:
    """Raster of 1-based pond numbers; where ponds share a cell the smaller pond wins."""
    order = ponds.geometry.area.sort_values(ascending=False).index  # small ponds burned last
    shapes = [(ponds.geometry.loc[i], ponds.index.get_loc(i) + 1) for i in order]
    return features.rasterize(shapes, out_shape=grids.fdir.shape, transform=grids.transform,
                              fill=0, all_touched=True, dtype="int32")


def link_ponds(ponds: gpd.GeoDataFrame, grids: FlowGrids) -> pd.DataFrame:
    """Outlet, catchment areas and next pond downstream for each pond.

    Parameters
    ----------
    ponds : GeoDataFrame
        Pond polygons with a unique ``dam_id`` column, in the grid's CRS or any CRS.
    grids : FlowGrids
        From :func:`flow_grids`.

    Returns
    -------
    DataFrame
        dam_id, ds_damid ("-99" for the catchment outlet), catchuptotal_km2,
        catchuplocal_km2, outlet_x, outlet_y (grid CRS) and outlet_found (False when the
        pond lies over missing DEM data; its catchment is then its own area).
    """
    if ponds.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LINK_COLUMNS})
    ponds = ponds.to_crs(grids.crs).reset_index(drop=True)
    ids = ponds["dam_id"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("dam_id values must be unique.")
    labels = _label_ponds(ponds, grids)
    rows, cols = grids.fdir.shape
    km2_per_cell = grids.cell_area_m2 / 1e6
    own_km2 = ponds.geometry.area.to_numpy() / 1e6

    outlets, totals, found = [], [], []
    for n in range(1, len(ponds) + 1):
        rr, cc = np.nonzero((labels == n) & grids.valid)
        if rr.size == 0:
            outlets.append(None)
            totals.append(own_km2[n - 1])
            found.append(False)
            continue
        k = int(np.argmax(grids.acc[rr, cc]))
        outlets.append((rr[k], cc[k]))
        totals.append(max(grids.acc[rr[k], cc[k]] * km2_per_cell, own_km2[n - 1]))
        found.append(True)

    downstream = []
    for n, outlet in enumerate(outlets, start=1):
        target = OUTLET
        if outlet is not None:
            r, c = outlet
            for _ in range(rows * cols):  # a flow path visits each cell at most once
                step = STEPS.get(int(grids.fdir[r, c]))
                if step is None:
                    break
                r, c = r + step[0], c + step[1]
                if not (0 <= r < rows and 0 <= c < cols) or not grids.valid[r, c]:
                    break
                if labels[r, c] not in (0, n):
                    target = ids[labels[r, c] - 1]
                    break
        downstream.append(target)

    links = pd.DataFrame({"dam_id": ids, "ds_damid": downstream, "catchuptotal_km2": totals})
    upstream_total = links.groupby("ds_damid")["catchuptotal_km2"].sum()
    links["catchuplocal_km2"] = (links["catchuptotal_km2"] - links["dam_id"].map(upstream_total).fillna(0.0)).clip(lower=0.0)
    xs, ys = [], []
    for outlet in outlets:
        if outlet is None:
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            x, y = grids.transform @ (outlet[1] + 0.5, outlet[0] + 0.5)
            xs.append(x)
            ys.append(y)
    links["outlet_x"], links["outlet_y"], links["outlet_found"] = xs, ys, found
    return links[LINK_COLUMNS]
