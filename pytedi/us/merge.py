"""Combine mapped ponds, detected ponds and official dams into one candidate table.

Rules
-----
- Every NHD pond or reservoir is a candidate, ``source="nhd"``.
- A Sentinel-2 polygon that overlaps an NHD pond confirms it: the NHD row becomes
  ``source="both"`` and keeps the NHD outline and area (mapped at 1:24,000 or better,
  which is cleaner than a 10 m pixel outline). The Sentinel-2 polygon is dropped.
- Sentinel-2 polygons mostly inside an NHD river polygon are rivers, not ponds, and are
  dropped. The rest are new candidates, ``source="sentinel2"``.
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

CANDIDATE_COLUMNS = ["source", "surfacearea_m2", "name", "nhd_id", "nid_id", "nid_name", "nid_storage_ML",
                     "area_from_default"]


def _col(gdf, name):
    return gdf[name] if name in gdf else pd.Series(None, index=gdf.index, dtype="object")


def _empty_candidates() -> gpd.GeoDataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in CANDIDATE_COLUMNS})
    frame["surfacearea_m2"] = frame["surfacearea_m2"].astype("float64")
    return gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries([], crs=EQUAL_AREA), crs=EQUAL_AREA)


def merge_candidates(nhd_waterbodies: gpd.GeoDataFrame, s2_ponds: gpd.GeoDataFrame, nid_dams: gpd.GeoDataFrame,
                     rivers: gpd.GeoDataFrame | None = None, nid_match_m: float = 150.0,
                     river_overlap: float = 0.5, nid_default_area_m2: float = 500.0) -> gpd.GeoDataFrame:
    """One row per candidate pond or reservoir (see the module notes for the rules).

    Parameters
    ----------
    nhd_waterbodies : GeoDataFrame
        From ``fetch_nhd(...).waterbodies``.
    s2_ponds : GeoDataFrame
        From :func:`~pytedi.us.ponds.detect_ponds_sentinel2`.
    nid_dams : GeoDataFrame
        From :func:`~pytedi.us.fetch.fetch_official_dams`.
    rivers : GeoDataFrame, optional
        River polygons (``fetch_nhd(...).rivers``) used to drop river detections.
    nid_match_m : float
        Largest distance in meters between an NID dam and the pond it is matched to.
    river_overlap : float
        Share of a Sentinel-2 polygon's area inside a river polygon above which it is dropped.
    nid_default_area_m2 : float
        Footprint for an unmatched NID dam whose surface area is unknown.

    Returns
    -------
    GeoDataFrame (EPSG:5070)
        geometry, source, surfacearea_m2, name, nhd_id, nid_id, nid_name, nid_storage_ML,
        area_from_default.
    """
    nhd = nhd_waterbodies.to_crs(EQUAL_AREA).reset_index(drop=True)
    s2 = s2_ponds.to_crs(EQUAL_AREA).reset_index(drop=True)
    nid = nid_dams.to_crs(EQUAL_AREA).reset_index(drop=True)

    if rivers is not None and not rivers.empty and not s2.empty:
        river_union = rivers.to_crs(EQUAL_AREA).geometry.union_all()
        inside = s2.geometry.intersection(river_union).area / s2.geometry.area
        s2 = s2[~(inside > river_overlap)].reset_index(drop=True)

    nhd_rows = gpd.GeoDataFrame({
        "source": "nhd",
        "name": _col(nhd, "gnis_name"),
        "nhd_id": _col(nhd, "permanent_identifier").astype("object"),
    }, geometry=nhd.geometry, crs=EQUAL_AREA)
    if not nhd_rows.empty and not s2.empty:
        pairs = gpd.sjoin(s2[["geometry"]], nhd_rows[["geometry"]], predicate="intersects")
        nhd_rows.loc[pairs["index_right"].unique(), "source"] = "both"
        s2 = s2.drop(index=pairs.index.unique())
    s2_rows = gpd.GeoDataFrame({"source": "sentinel2"}, geometry=s2.geometry, crs=EQUAL_AREA, index=s2.index)

    candidates = pd.concat([nhd_rows, s2_rows], ignore_index=True)
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=EQUAL_AREA)
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
        lone_rows = gpd.GeoDataFrame({
            "source": "nid", "name": _col(lone, "name").to_numpy(), "nhd_id": None,
            "nid_id": lone["nid_id"].to_numpy(), "nid_name": _col(lone, "name").to_numpy(),
            "nid_storage_ML": _col(lone, "storage_ML").to_numpy(),
            "area_from_default": default.to_numpy(),
        }, geometry=circles, crs=EQUAL_AREA)
        lone_rows["surfacearea_m2"] = lone_rows.geometry.area
        candidates = gpd.GeoDataFrame(pd.concat([candidates, lone_rows], ignore_index=True), crs=EQUAL_AREA)

    if candidates.empty:
        return _empty_candidates()
    return candidates[["geometry", *CANDIDATE_COLUMNS]].reset_index(drop=True)
