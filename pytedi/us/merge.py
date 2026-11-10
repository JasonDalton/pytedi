"""Combine mapped ponds, detected ponds and official dams into one candidate table.

Rules
-----
- Every NHD pond or reservoir is a candidate. It keeps the NHD outline and area (mapped
  at 1:24,000 or better).
- Detections are then added layer by layer, NAIP (1 m) first, then Sentinel-2 (10 m).
  Detections mostly inside an NHD river polygon are rivers, not ponds, and are dropped. A
  detection that overlaps an existing candidate confirms it (the detection's own polygon
  is dropped); the rest become new candidates with their own outline. So a pond only the
  imagery found takes NAIP's outline when NAIP saw it.
- ``in_nhd``, ``in_naip`` and ``in_sentinel2`` record which sources found each pond, and
  ``source`` joins them with "+", e.g. "nhd", "nhd+naip+sentinel2", "naip+sentinel2".
- Each NID dam is matched to the nearest candidate within ``nid_match_m`` (each dam to one
  pond, each pond to one dam, nearest pairs first); the match fills ``nid_id`` and
  ``nid_name`` without changing ``source``. A dam with no pond in reach becomes its own
  candidate, ``source="nid"``, drawn as a circle of the NID surface area around the dam
  (or ``nid_default_area_m2`` when the NID gives none).
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd

from .aoi import EQUAL_AREA

LAYERS = ("nhd", "naip", "sentinel2")
FLAGS = [f"in_{layer}" for layer in LAYERS]
CANDIDATE_COLUMNS = ["source", "surfacearea_m2", "name", "nhd_id", *FLAGS, "nid_id", "nid_name", "nid_storage_ML",
                     "area_from_default"]


def _col(gdf, name):
    return gdf[name] if name in gdf else pd.Series(None, index=gdf.index, dtype="object")


def _empty_candidates() -> gpd.GeoDataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in CANDIDATE_COLUMNS})
    frame["surfacearea_m2"] = frame["surfacearea_m2"].astype("float64")
    for flag in FLAGS:
        frame[flag] = frame[flag].astype(bool)
    return gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries([], crs=EQUAL_AREA), crs=EQUAL_AREA)


def _new_rows(geometry, **values) -> gpd.GeoDataFrame:
    geometry = gpd.GeoSeries(list(geometry), crs=EQUAL_AREA)
    columns = {"name": None, "nhd_id": None, **{f: False for f in FLAGS}, **values}
    frame = pd.DataFrame({k: (v.to_numpy() if isinstance(v, pd.Series) else v) for k, v in columns.items()},
                         index=pd.RangeIndex(len(geometry)))
    return gpd.GeoDataFrame(frame, geometry=geometry.to_numpy(), crs=EQUAL_AREA)


def merge_candidates(nhd_waterbodies: gpd.GeoDataFrame, s2_ponds: gpd.GeoDataFrame | None,
                     nid_dams: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame | None = None,
                     nid_match_m: float = 150.0, river_overlap: float = 0.5, nid_default_area_m2: float = 500.0,
                     naip_ponds: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """One row per candidate pond or reservoir (see the module notes for the rules).

    Parameters
    ----------
    nhd_waterbodies : GeoDataFrame
        From ``fetch_nhd(...).waterbodies``.
    s2_ponds : GeoDataFrame or None
        From :func:`~pytedi.us.ponds.detect_ponds_sentinel2`.
    nid_dams : GeoDataFrame
        From :func:`~pytedi.us.fetch.fetch_official_dams`.
    rivers : GeoDataFrame, optional
        River polygons (``fetch_nhd(...).rivers``) used to drop river detections.
    nid_match_m : float
        Largest distance in meters between an NID dam and the pond it is matched to.
    river_overlap : float
        Share of a detection's area inside a river polygon above which it is dropped.
    nid_default_area_m2 : float
        Footprint for an unmatched NID dam whose surface area is unknown.
    naip_ponds : GeoDataFrame, optional
        From :func:`~pytedi.us.naip.detect_ponds_naip`.

    Returns
    -------
    GeoDataFrame (EPSG:5070)
        geometry, source, surfacearea_m2, name, nhd_id, in_nhd, in_naip, in_sentinel2,
        nid_id, nid_name, nid_storage_ML, area_from_default.
    """
    nhd = nhd_waterbodies.to_crs(EQUAL_AREA).reset_index(drop=True)
    nid = nid_dams.to_crs(EQUAL_AREA).reset_index(drop=True)
    river_union = None
    if rivers is not None and not rivers.empty:
        river_union = rivers.to_crs(EQUAL_AREA).geometry.union_all()

    candidates = _new_rows(nhd.geometry, name=_col(nhd, "gnis_name"),
                           nhd_id=_col(nhd, "permanent_identifier").astype("object"), in_nhd=True)
    for layer, detections in (("naip", naip_ponds), ("sentinel2", s2_ponds)):
        if detections is None or detections.empty:
            continue
        found = detections.to_crs(EQUAL_AREA).reset_index(drop=True)
        if river_union is not None:
            inside = found.geometry.intersection(river_union).area / found.geometry.area
            found = found[~(inside > river_overlap)]
        if not candidates.empty and not found.empty:
            pairs = gpd.sjoin(found[["geometry"]], candidates[["geometry"]], predicate="intersects")
            candidates.loc[pairs["index_right"].unique(), f"in_{layer}"] = True
            found = found.drop(index=pairs.index.unique())
        if not found.empty:
            added = _new_rows(found.geometry.to_numpy(), **{f"in_{layer}": True})
            candidates = gpd.GeoDataFrame(pd.concat([candidates, added], ignore_index=True), crs=EQUAL_AREA)

    candidates = candidates.reset_index(drop=True)
    candidates["source"] = ["+".join(layer for layer in LAYERS if row[f"in_{layer}"])
                            for _, row in candidates.iterrows()]
    candidates["surfacearea_m2"] = candidates.geometry.area
    candidates["area_from_default"] = False
    for column in ("nid_id", "nid_name", "nid_storage_ML"):
        candidates[column] = None

    unmatched = list(range(len(nid)))
    if not nid.empty and not candidates.empty:
        near = gpd.sjoin_nearest(nid[["geometry"]], candidates[["geometry"]], max_distance=nid_match_m,
                                 distance_col="distance").sort_values("distance")
        used_ponds = set()
        for dam, row in near.iterrows():
            pond = row["index_right"]
            if dam in unmatched and pond not in used_ponds:
                candidates.loc[pond, ["nid_id", "nid_name", "nid_storage_ML"]] = [
                    nid.loc[dam, "nid_id"], _col(nid, "name")[dam], _col(nid, "storage_ML")[dam]]
                used_ponds.add(pond)
                unmatched.remove(dam)

    if unmatched:
        lone = nid.loc[unmatched]
        area = pd.to_numeric(_col(lone, "surfacearea_m2"), errors="coerce")
        default = area.isna() | (area <= 0)
        area = area.where(~default, nid_default_area_m2)
        radius = np.sqrt(area / math.pi)
        circles = [point.buffer(r, quad_segs=32) for point, r in zip(lone.geometry, radius)]
        lone_rows = _new_rows(circles, source="nid", name=_col(lone, "name").to_numpy(),
                              nid_id=lone["nid_id"].to_numpy(), nid_name=_col(lone, "name").to_numpy(),
                              nid_storage_ML=_col(lone, "storage_ML").to_numpy(), area_from_default=default.to_numpy())
        lone_rows["surfacearea_m2"] = lone_rows.geometry.area
        candidates = gpd.GeoDataFrame(pd.concat([candidates, lone_rows], ignore_index=True), crs=EQUAL_AREA)

    if candidates.empty:
        return _empty_candidates()
    candidates["nid_storage_ML"] = pd.to_numeric(candidates["nid_storage_ML"], errors="coerce").astype("float64")
    candidates["area_from_default"] = candidates["area_from_default"].astype(bool)
    for flag in FLAGS:
        candidates[flag] = candidates[flag].astype(bool)
    return candidates[["geometry", *CANDIDATE_COLUMNS]].reset_index(drop=True)
