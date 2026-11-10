"""Streamflow for an ungauged AOI, transferred from a USGS reference gauge.

1. :func:`find_gauges` lists USGS NWIS stream sites with daily mean discharge (parameter
   00060, statistic 00003) within ``radius_km`` of the AOI centroid whose drainage area is
   ``area_range`` (0.2 to 5) times the AOI's, with their GAGES-II class (reference or not),
   aggregated ecoregion and HCDN-2009 flag (Falcone 2011), and the number of complete calendar
   years their record covers in the study period.
2. :func:`select_gauge` applies a fixed rule, in order: at least ``min_years`` complete
   years in the period; GAGES-II reference class; same aggregated ecoregion as the AOI (that
   of the GAGES-II gauge nearest the AOI); then the nearest. It returns the reason each
   candidate was or wasn't chosen.
3. :func:`fetch_gauge_flow` downloads the daily record in ML/day. HyRiver returns m3/s,
   converted from the USGS cubic feet per second; 1 m3/s = 86.4 ML/day, so 1 cfs =
   2.446575 ML/day.
4. :func:`transfer_flow` scales by drainage area: Q_aoi = Q_gauge * (A_aoi / A_gauge) ** e,
   with e = 1 by default (equal runoff per unit area).

Falcone, J.A., 2011. GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow. U.S.
Geological Survey. https://doi.org/10.5066/P96CPHOT
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd

from .aoi import EQUAL_AREA, WGS84, as_aoi

SQMI_TO_KM2 = 2.589988110336
CMS_TO_ML_PER_DAY = 86.4
CFS_TO_ML_PER_DAY = 0.028316846592 * CMS_TO_ML_PER_DAY  # = 2.446575...
GAGESII_POINTS_URL = ("https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a"
                      "?f=__disk__53%2Fba%2F17%2F53ba17cfb64d2d50d3876864edf519eb7295945a")


def cms_to_ml_per_day(q):
    return q * CMS_TO_ML_PER_DAY


def transfer_flow(flow, gauge_area_km2: float, target_area_km2: float, exponent: float = 1.0):
    """Drainage-area ratio transfer: flow * (target / gauge) ** exponent."""
    return flow * (target_area_km2 / gauge_area_km2) ** exponent


# ---------------------------------------------------------------------------- services

def _nwis_sites(bbox) -> pd.DataFrame:
    from pygeohydro import NWIS

    query = {"bBox": ",".join(f"{v:.4f}" for v in bbox), "parameterCd": "00060", "hasDataTypeCd": "dv",
             "siteType": "ST", "outputDataTypeCd": "dv"}
    return NWIS().get_info(query, expanded=True)


def _gagesii(cache=None) -> gpd.GeoDataFrame:
    import shutil
    import urllib.request

    from .fetch import cache_dir

    path = cache_dir(cache) / "gagesii_9322_point_shapefile.zip"
    if not path.exists():
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(GAGESII_POINTS_URL, timeout=120) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    return gpd.read_file(f"zip://{path}")


def _nwis_streamflow(site_no: str, dates) -> pd.DataFrame:
    from pygeohydro import NWIS

    return NWIS().get_streamflow(site_no, dates)


# ---------------------------------------------------------------------------- search and choice

def _complete_years(begin, end, period) -> int:
    start = max(pd.Timestamp(begin), pd.Timestamp(period[0]))
    stop = min(pd.Timestamp(end), pd.Timestamp(period[1]))
    first = start.year if (start.month, start.day) == (1, 1) else start.year + 1
    last = stop.year if (stop.month, stop.day) == (12, 31) else stop.year - 1
    return max(0, last - first + 1)


def find_gauges(aoi, period, radius_km: float = 150.0, area_range=(0.2, 5.0), target_area_km2: float | None = None,
                cache: str | os.PathLike | None = None) -> pd.DataFrame:
    """Candidate gauges for transferring daily flow to the AOI (see the module notes).

    Returns a DataFrame sorted by distance with site_no, station_nm, drainage_km2, area_ratio,
    dist_km, begin_date, end_date, years_in_period, gagesii_class, ecoregion, hcdn_2009;
    ``attrs`` holds the target area and ecoregion.
    """
    aoi = as_aoi(aoi)
    target_area = target_area_km2 or aoi.area_km2
    centroid = aoi.to_crs(EQUAL_AREA).geometry.centroid
    center = gpd.GeoSeries([centroid], crs=EQUAL_AREA)
    lon, lat = center.to_crs(WGS84).iloc[0].coords[0]
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    sites = _nwis_sites((lon - dlon, lat - dlat, lon + dlon, lat + dlat))
    sites = sites[(sites["parm_cd"] == "00060") & (sites["stat_cd"] == "00003")].drop_duplicates("site_no")

    pts = gpd.GeoSeries(gpd.points_from_xy(sites["dec_long_va"], sites["dec_lat_va"]), crs=WGS84,
                        index=sites.index).to_crs(EQUAL_AREA)
    out = pd.DataFrame({
        "site_no": sites["site_no"].astype(str),
        "station_nm": sites["station_nm"],
        "drainage_km2": pd.to_numeric(sites["drain_area_va"], errors="coerce") * SQMI_TO_KM2,
        "dist_km": pts.distance(centroid) / 1000.0,
        "begin_date": sites["begin_date"].astype(str),
        "end_date": sites["end_date"].astype(str),
        "lat": sites["dec_lat_va"], "lon": sites["dec_long_va"],
    })
    out["area_ratio"] = out["drainage_km2"] / target_area
    out = out[(out["dist_km"] <= radius_km) & out["area_ratio"].between(*area_range)].copy()
    out["years_in_period"] = [_complete_years(b, e, period) for b, e in zip(out["begin_date"], out["end_date"])]

    g2 = _gagesii(cache)
    g2 = g2.assign(STAID=g2["STAID"].astype(str))
    meta = g2.set_index("STAID")
    out["gagesii_class"] = out["site_no"].map(meta["CLASS"])
    out["ecoregion"] = out["site_no"].map(meta["AGGECOREGI"])
    out["hcdn_2009"] = out["site_no"].map(meta["HCDN_2009"]) if "HCDN_2009" in meta else None
    g2_5070 = g2.to_crs(EQUAL_AREA)
    nearest = g2_5070.iloc[int(np.argmin(g2_5070.distance(centroid).to_numpy()))]

    out = out.sort_values("dist_km").reset_index(drop=True)
    out.attrs.update(target_area_km2=target_area, target_ecoregion=nearest["AGGECOREGI"],
                     target_ecoregion_from=f"nearest GAGES-II gauge {nearest['STAID']}", period=tuple(period),
                     radius_km=radius_km, area_range=tuple(area_range))
    return out


def select_gauge(candidates: pd.DataFrame, min_years: int = 30, require_reference: bool = True,
                 same_ecoregion: bool = True):
    """Apply the selection rule (see the module notes). Returns (chosen row, {site_no: reason})."""
    target_eco = candidates.attrs.get("target_ecoregion")
    reasons, qualified = {}, []
    for _, row in candidates.sort_values("dist_km").iterrows():
        if row["years_in_period"] < min_years:
            reasons[row["site_no"]] = f"fewer than {min_years} complete years in the period ({row['years_in_period']})"
        elif require_reference and row["gagesii_class"] != "Ref":
            reasons[row["site_no"]] = "not GAGES-II reference"
        elif same_ecoregion and row["ecoregion"] != target_eco:
            reasons[row["site_no"]] = f"ecoregion {row['ecoregion']}, not {target_eco}"
        else:
            qualified.append(row)
    if not qualified:
        raise ValueError(f"No gauge meets the rule; reasons: {reasons}")
    chosen = qualified[0]
    reasons[chosen["site_no"]] = "chosen"
    for row in qualified[1:]:
        reasons[row["site_no"]] = "farther than the chosen gauge"
    return chosen, reasons


def fetch_gauge_flow(site_no: str, start, end, max_fill_days: int = 7) -> pd.Series:
    """Daily mean flow at a USGS gauge in ML/day; gaps of up to ``max_fill_days`` interpolated and listed."""
    dates = (str(start)[:10], str(end)[:10])
    raw = _nwis_streamflow(site_no, dates)
    cms = raw.iloc[:, 0]
    index = pd.DatetimeIndex(cms.index)
    if index.tz is not None:
        # USGS daily values are local days. HyRiver stamps each at local midnight expressed in UTC
        # (04:00-10:00 UTC across US time zones), which is still the same calendar date in UTC.
        index = index.tz_convert("UTC").tz_localize(None)
    cms.index = index.normalize()
    days = pd.date_range(dates[0], dates[1], freq="D")
    cms = cms.reindex(days)
    missing = cms.isna()
    runs = missing.ne(missing.shift()).cumsum()[missing]
    if not runs.empty and runs.value_counts().max() > max_fill_days:
        raise ValueError(f"Gauge {site_no} is missing more than {max_fill_days} consecutive days in {dates}.")
    filled = list(days[missing].strftime("%Y-%m-%d"))
    q = cms_to_ml_per_day(cms.interpolate(method="time", limit_direction="both"))
    q.name = "flow_ML"
    q.attrs.update(site_no=site_no, units="ML/day", source="USGS NWIS daily values (00060, mean) via HyRiver",
                   filled_dates=filled)
    return q
