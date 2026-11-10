"""Build a pytedi dam list for a US area of interest.

:func:`build_damlist` runs the whole chain:

1. 3DEP DEM for the AOI (:func:`~pytedi.us.fetch.fetch_3dep`);
2. NHD ponds, flowlines and rivers (:func:`~pytedi.us.fetch.fetch_nhd`);
3. NID dams (:func:`~pytedi.us.fetch.fetch_official_dams`);
4. ponds detected in NAIP 1 m imagery (:func:`~pytedi.us.naip.detect_ponds_naip`) and
   Sentinel-2 (:func:`~pytedi.us.ponds.detect_ponds_sentinel2`), each optional;
5. one candidate table (:func:`~pytedi.us.merge.merge_candidates`);
6. outlets, catchment areas and the downstream network (:func:`~pytedi.us.hydro.link_ponds`);
7. volumes from surface areas with an area-volume relation (:mod:`pytedi.us.volume`);
8. the dam list in the schema of ``tedimodel().damlist``.

Phase 1 has no demands, bypasses or winterfill: every dam gets the engine's own neutral
values (group "none", demand factor 0, no bypass or winterfill, operating in all years).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

from ..main import tedimodel
from .aoi import AOI, EQUAL_AREA, as_aoi
from .fetch import NHDData, fetch_3dep, fetch_nhd, fetch_official_dams
from .hydro import flow_grids, link_ponds
from .merge import merge_candidates
from .naip import detect_ponds_naip
from .ponds import DEFAULT_DATE_RANGE, detect_ponds_sentinel2
from .volume import relation_coefficients, volume_ml_from_area

NEUTRAL = {  # the values the engine gives its own catchment-outlet row (engine.damlist_preprocessing)
    "year_start": 0, "year_end": 9999, "group_name": "none", "demfactor": 0.0,
    "bypass_MLperT": 0.0, "bypass_start_mmdd": 101, "bypass_end_mmdd": 1231,
    "wfill_MLperT": 0.0, "wfill_start_mmdd": 101, "wfill_end_mmdd": 1231, "output": 0,
}


def to_pytedi_damlist(candidates: pd.DataFrame) -> pd.DataFrame:
    """The columns and dtypes of ``tedimodel().damlist`` from linked candidates.

    ``candidates`` needs dam_id, ds_damid, surfacearea_m2, catchuptotal_km2,
    catchuplocal_km2 and vol_ML.
    """
    template = tedimodel().damlist
    out = pd.DataFrame(index=range(len(candidates)))
    for column in template.columns:
        if column in NEUTRAL:
            out[column] = NEUTRAL[column]
        else:
            out[column] = candidates[column].to_numpy()
    return out.astype(dict(template.dtypes))


@dataclass
class DamListResult:
    """Everything :func:`build_damlist` produced.

    Attributes
    ----------
    damlist : DataFrame
        Ready for ``tedimodel().damlist``.
    candidates : GeoDataFrame (EPSG:5070)
        One row per pond with geometry, source, links, areas, volume and NID match.
    aoi : AOI
    cat_area_km2 : float
        Area of the AOI, for ``tedimodel().cat_area_km2``.
    dem, nhd, nid_dams, s2_ponds, naip_ponds :
        The inputs, kept for plots and checks.
    settings : dict
        The parameters used.
    """

    damlist: pd.DataFrame
    candidates: gpd.GeoDataFrame
    aoi: AOI
    cat_area_km2: float
    dem: object
    nhd: NHDData
    nid_dams: gpd.GeoDataFrame
    s2_ponds: gpd.GeoDataFrame | None
    settings: dict = field(default_factory=dict)
    naip_ponds: gpd.GeoDataFrame | None = None

    def to_model(self, title: str | None = None) -> tedimodel:
        """A ``tedimodel`` with this dam list and catchment area; add ``ts`` before running."""
        model = tedimodel()
        model.title = title or f"{self.aoi.name} (pytedi.us)"
        model.cat_area_km2 = self.cat_area_km2
        model.damlist = self.damlist.copy()
        model.vol_surfarea_a, model.vol_surfarea_b = self.settings["vol_a"], self.settings["vol_b"]
        return model

    def summary(self) -> dict:
        """Counts and areas for a quick report."""
        c = self.candidates
        area = c["surfacearea_m2"] if not c.empty else pd.Series(dtype=float)
        return {
            "aoi": self.aoi.name,
            "aoi_km2": round(self.aoi.area_km2, 2),
            "cat_area_km2": round(self.cat_area_km2, 2),
            "ponds": len(c),
            "by_source": c["source"].value_counts().sort_index().to_dict() if not c.empty else {},
            "surfacearea_m2": area.describe().round(1).to_dict() if not c.empty else {},
            "total_surfacearea_ha": round(area.sum() / 1e4, 2),
            "total_vol_ML": round(float(c["vol_ML"].sum()), 1) if not c.empty else 0.0,
            "draining_to_outlet": int((c["ds_damid"] == "-99").sum()) if not c.empty else 0,
            "nid_dams_in_aoi": len(self.nid_dams),
            "nid_dams_matched": int(c["nid_id"].notna().sum() - (c["source"] == "nid").sum()) if not c.empty else 0,
            "outlets_not_found": int((~c["outlet_found"].astype(bool)).sum()) if not c.empty else 0,
        }

    def write_geopackage(self, path: str | os.PathLike) -> None:
        """Layers: candidates (polygons), outlets (points), nid_dams, flowlines, aoi."""
        path = os.fspath(path)
        if os.path.exists(path):
            os.remove(path)
        c = self.candidates.copy()
        c.to_file(path, layer="candidates", driver="GPKG")
        outlets = gpd.GeoDataFrame(c.drop(columns="geometry"),
                                   geometry=gpd.points_from_xy(c["outlet_x"], c["outlet_y"]), crs=c.crs)
        outlets.to_file(path, layer="outlets", driver="GPKG")
        for name, layer in (("nid_dams", self.nid_dams), ("flowlines", self.nhd.flowlines)):
            if layer.empty:
                layer = gpd.GeoDataFrame({"empty": pd.Series(dtype="int32")}, geometry=[], crs=EQUAL_AREA)
            layer.to_file(path, layer=name, driver="GPKG")
        self.aoi.to_crs(EQUAL_AREA).to_gdf().to_file(path, layer="aoi", driver="GPKG")


def build_damlist(aoi, *, use_naip: bool = True, naip_year: int | None = None, naip_threshold: float = 0.5,
                  naip_min_area_m2: float = 100.0,
                  use_sentinel2: bool = True, date_range=DEFAULT_DATE_RANGE, ndwi_threshold: float = 0.0,
                  min_area_m2: float = 500.0, max_cloud: float = 20, dem_resolution: int = 10,
                  nhd_dataset: str = "nhd_hr", nid_match_m: float = 150.0, volume_relation="fowler2015",
                  vol_a: float | None = None,
                  vol_b: float | None = None, out_gpkg: str | os.PathLike | None = None,
                  cache: str | os.PathLike | None = None, cat_area_km2: float | None = None) -> DamListResult:
    """Fetch the data for an AOI and build a pytedi dam list (see the module notes).

    Parameters
    ----------
    aoi : anything :func:`~pytedi.us.aoi.as_aoi` accepts, e.g. a HUC12 id
    use_naip : bool
        Add ponds detected in NAIP 1 m imagery (the first run for an AOI takes minutes to
        download it; it is cached).
    naip_year, naip_threshold, naip_min_area_m2 :
        Passed to :func:`~pytedi.us.naip.detect_ponds_naip` (latest year by default).
    use_sentinel2 : bool
        Add ponds detected in Sentinel-2 imagery.
    date_range, ndwi_threshold, min_area_m2, max_cloud :
        Passed to :func:`~pytedi.us.ponds.detect_ponds_sentinel2`.
    dem_resolution : int
        3DEP cell size in meters.
    nhd_dataset : {"nhd_hr", "3dhp"}
    nid_match_m : float
        Largest distance between an NID dam and the pond it is matched to.
    volume_relation : str or (a, b)
        "fowler2015" (the model's default), "va_nid_farm" (fitted to Virginia NID farm-type
        dams; see :mod:`pytedi.us.volume` for its caveats) or an explicit ``(a, b)``.
    vol_a, vol_b : float, optional
        Override ``a`` and/or ``b`` of the chosen relation.
    out_gpkg : path, optional
        Also write the candidates and inputs to this GeoPackage.
    cat_area_km2 : float, optional
        Area draining to the AOI's outlet, for pytedi's ``cat_area_km2``. Defaults to the AOI's
        area; give more when water enters from upstream (e.g. :func:`~pytedi.us.aoi.huc_drainage_area`).
        Ponds are mapped only inside the AOI, so the extra area counts as having none.
    cache : path, optional
        Download cache (see :func:`~pytedi.us.fetch.cache_dir`).
    """
    aoi = as_aoi(aoi)
    if cat_area_km2 is not None and cat_area_km2 < aoi.area_km2 * 0.99:
        raise ValueError(f"cat_area_km2 ({cat_area_km2}) is smaller than the AOI ({aoi.area_km2:.2f} km2).")
    relation_a, relation_b = relation_coefficients(volume_relation)
    vol_a = relation_a if vol_a is None else vol_a
    vol_b = relation_b if vol_b is None else vol_b

    dem = fetch_3dep(aoi, resolution=dem_resolution, cache=cache)
    nhd = fetch_nhd(aoi, dataset=nhd_dataset, cache=cache)
    nid = fetch_official_dams(aoi, cache=cache)
    naip = None
    if use_naip:
        naip = detect_ponds_naip(aoi, year=naip_year, threshold=naip_threshold, min_area_m2=naip_min_area_m2,
                                 cache=cache)
    s2 = None
    if use_sentinel2:
        s2 = detect_ponds_sentinel2(aoi, date_range=tuple(date_range), threshold=ndwi_threshold,
                                    min_area_m2=min_area_m2, max_cloud=max_cloud)
    candidates = merge_candidates(nhd.waterbodies, s2 if s2 is not None else gpd.GeoDataFrame(geometry=[], crs=EQUAL_AREA),
                                  nid, rivers=nhd.rivers, nid_match_m=nid_match_m, naip_ponds=naip)
    candidates.insert(0, "dam_id", pd.Series([str(i) for i in range(1, len(candidates) + 1)],
                                             index=candidates.index, dtype="object"))

    links = link_ponds(candidates[["dam_id", "geometry"]], flow_grids(dem))
    candidates = candidates.merge(links, on="dam_id", how="left")
    candidates["vol_ML"] = volume_ml_from_area(candidates["surfacearea_m2"].to_numpy(dtype=float), vol_a, vol_b)
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=EQUAL_AREA)

    settings = dict(use_naip=use_naip, naip_year=None if naip is None else naip.attrs.get("year", naip_year),
                    naip_threshold=naip_threshold, naip_min_area_m2=naip_min_area_m2, use_sentinel2=use_sentinel2, date_range=tuple(date_range), ndwi_threshold=ndwi_threshold,
                    min_area_m2=min_area_m2, max_cloud=max_cloud, dem_resolution=dem_resolution,
                    nhd_dataset=nhd_dataset, nid_match_m=nid_match_m, volume_relation=volume_relation,
                    vol_a=vol_a, vol_b=vol_b)
    settings["cat_area_km2"] = float(cat_area_km2 or aoi.area_km2)
    result = DamListResult(to_pytedi_damlist(candidates), candidates, aoi, settings["cat_area_km2"], dem, nhd, nid, s2,
                           settings,
                           naip)
    if out_gpkg is not None:
        result.write_geopackage(out_gpkg)
    return result
