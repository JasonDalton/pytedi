"""Fetch US terrain, hydrography and dam data for an area of interest.

- :func:`fetch_3dep`: a DEM from the USGS 3D Elevation Program (via HyRiver ``py3dep``).
- :func:`fetch_nhd`: flowlines, ponds and lakes, and river polygons from the USGS National
  Hydrography Dataset, high resolution (via HyRiver ``pynhd``), or from the newer 3D
  Hydrography Program (3DHP).
- :func:`fetch_official_dams`: the USACE National Inventory of Dams (NID).

Vector results are returned in EPSG:5070 (NAD83 / Conus Albers, meters), so areas and
distances can be measured directly.

Downloads are cached in :func:`cache_dir`.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)

from .aoi import AOI, EQUAL_AREA, WGS84, as_aoi

NID_GPKG_URL = "https://nid.sec.usace.army.mil/api/nation/gpkg"
NHD_DATASETS = ("nhd_hr", "3dhp")
NHD_POND_FTYPES = (390, 436)  # NHD FType 390 LakePond, 436 Reservoir (466 SwampMarsh is left out)
HP3D_POND_TYPES = ("Lake", "Reservoir", "Pond", "LakePond")

ACRE_FT_TO_ML = 1.2334818375
ACRE_TO_M2 = 4046.8564224
FT_TO_M = 0.3048
SQMI_TO_KM2 = 2.589988110336


def cache_dir(path: str | os.PathLike | None = None) -> Path:
    """Folder for downloads: ``path``, else $PYTEDI_US_CACHE, else ./.pytedi_cache. Created if needed."""
    folder = Path(path or os.environ.get("PYTEDI_US_CACHE") or Path.cwd() / ".pytedi_cache")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _use_cache(cache: Path) -> None:
    """Point HyRiver's HTTP cache into our cache folder unless the user has set it."""
    os.environ.setdefault("HYRIVER_CACHE_NAME", str(cache / "hyriver_http_cache.sqlite"))


# ---------------------------------------------------------------------------- 3DEP

def _get_dem(geometry, resolution, crs):
    import py3dep

    return py3dep.get_dem(geometry, resolution, crs=crs)


def fetch_3dep(aoi, resolution: int = 10, crs: str = EQUAL_AREA, cache: str | os.PathLike | None = None):
    """Digital elevation model for the AOI from USGS 3DEP.

    Parameters
    ----------
    aoi : anything :func:`~pytedi.us.aoi.as_aoi` accepts
    resolution : int
        Cell size in meters. 3DEP's seamless 1/3 arc-second (about 10 m) layer is the
        default; 1 m and 3 m exist where lidar has been flown.
    crs : str
        CRS of the returned grid. The default, EPSG:5070, has square cells in meters.

    Returns
    -------
    xarray.DataArray
        Elevation in meters (float32), named "elevation", with its CRS set (rioxarray).
        Cells outside the AOI polygon are NaN.
    """
    aoi = as_aoi(aoi).to_crs(WGS84)
    _use_cache(cache_dir(cache))
    dem = _get_dem(aoi.geometry, resolution, crs=WGS84)
    if dem.rio.crs is None or dem.rio.crs.to_epsg() != _epsg(crs):
        dem = dem.rio.reproject(crs, resolution=resolution)
    dem = dem.astype("float32")
    dem.name = "elevation"
    dem.attrs.update(units="m", source=f"USGS 3DEP {resolution} m")
    return dem


def _epsg(crs: str) -> int | None:
    from pyproj import CRS

    return CRS.from_user_input(crs).to_epsg()


# ---------------------------------------------------------------------------- NHD

@dataclass
class NHDData:
    """Hydrography for an AOI, in EPSG:5070.

    Attributes
    ----------
    flowlines : GeoDataFrame
        Stream and river centerlines, clipped to the AOI.
    waterbodies : GeoDataFrame
        Ponds, lakes and reservoirs that touch the AOI (whole polygons, not clipped).
    rivers : GeoDataFrame
        Polygons of wide rivers (NHD area features), used to tell rivers from ponds.
    catchments : GeoDataFrame or None
        NHDPlus HR catchments when requested and the service answered.
    dataset : str
        "nhd_hr" or "3dhp".
    """

    flowlines: gpd.GeoDataFrame
    waterbodies: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    catchments: gpd.GeoDataFrame | None
    dataset: str


def _nhd_layer(dataset: str, layer: str, aoi: AOI) -> gpd.GeoDataFrame:
    """Raw features of one layer for the AOI's bounding box.

    The services time out on detailed polygon queries, so this asks for the bounding box
    and the caller clips to the polygon.
    """
    import pynhd

    if dataset == "nhd_hr":
        service = pynhd.NHD({"flowline": "flowline_hr", "waterbody": "waterbody_hr", "area": "area_hr"}[layer])
        return service.bygeom(aoi.bounds, geo_crs=WGS84)
    if layer == "area":  # 3DHP has no separate river-area layer
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)
    return pynhd.HP3D(layer).bygeom(aoi.bounds, geo_crs=WGS84)


def _nhdplus_hr_catchments(aoi: AOI) -> gpd.GeoDataFrame:
    import pynhd

    return pynhd.NHDPlusHR("catchment").bygeom(aoi.bounds, geo_crs=WGS84)


def _tidy(gdf: gpd.GeoDataFrame, aoi_5070: AOI) -> gpd.GeoDataFrame:
    """Lower-case column names, EPSG:5070, and only features that touch the AOI."""
    gdf = gdf.rename(columns={c: c.lower() for c in gdf.columns if c != "geometry"})
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    gdf = gdf.to_crs(EQUAL_AREA)
    return gdf[gdf.intersects(aoi_5070.geometry)].reset_index(drop=True)


def fetch_nhd(aoi, dataset: str = "nhd_hr", include_catchments: bool = False,
              cache: str | os.PathLike | None = None) -> NHDData:
    """Flowlines, ponds and river polygons for the AOI.

    Parameters
    ----------
    aoi : anything :func:`~pytedi.us.aoi.as_aoi` accepts
    dataset : {"nhd_hr", "3dhp"}
        "nhd_hr" is the National Hydrography Dataset at 1:24,000 or better (the default).
        "3dhp" is the 3D Hydrography Program, its elevation-derived successor, which does
        not cover every area yet.
    include_catchments : bool
        Also ask for NHDPlus HR catchments. The service is often slow or down; on failure
        this warns and leaves ``catchments`` as None.

    Notes
    -----
    Waterbodies are limited to ponds, lakes and reservoirs (NHD FType 390 and 436; 3DHP
    "Lake" and "Reservoir"); swamps and marshes are left out.
    """
    if dataset not in NHD_DATASETS:
        raise ValueError(f"Unknown NHD dataset {dataset!r}; use one of {NHD_DATASETS}.")
    aoi = as_aoi(aoi).to_crs(WGS84)
    aoi_5070 = aoi.to_crs(EQUAL_AREA)
    _use_cache(cache_dir(cache))

    flowlines = _tidy(_nhd_layer(dataset, "flowline", aoi), aoi_5070)
    flowlines = flowlines.set_geometry(flowlines.geometry.intersection(aoi_5070.geometry))
    flowlines = flowlines[~flowlines.geometry.is_empty].reset_index(drop=True)

    waterbodies = _tidy(_nhd_layer(dataset, "waterbody", aoi), aoi_5070)
    if dataset == "nhd_hr" and "ftype" in waterbodies:
        waterbodies = waterbodies[waterbodies["ftype"].astype(int).isin(NHD_POND_FTYPES)]
    elif dataset == "3dhp" and "featuretypelabel" in waterbodies:
        waterbodies = waterbodies[waterbodies["featuretypelabel"].isin(HP3D_POND_TYPES)]
    waterbodies = waterbodies.reset_index(drop=True)

    rivers = _tidy(_nhd_layer(dataset, "area", aoi), aoi_5070)

    catchments = None
    if include_catchments:
        try:
            catchments = _tidy(_nhdplus_hr_catchments(aoi), aoi_5070)
        except Exception as exc:  # the service fails in many ways; none should stop the run
            warnings.warn(f"NHDPlus HR catchments are unavailable ({type(exc).__name__}); continuing without them.",
                          UserWarning, stacklevel=2)
    return NHDData(flowlines, waterbodies, rivers, catchments, dataset)


# ---------------------------------------------------------------------------- NID

def nid_inventory_path(cache: str | os.PathLike | None = None, refresh: bool = False) -> Path:
    """The national NID GeoPackage (about 80 MB), downloaded once into the cache."""
    path = cache_dir(cache) / "nid_nation.gpkg"
    if refresh or not path.exists():
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(NID_GPKG_URL, timeout=300) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    return path


NID_COLUMNS = {  # NID field -> (our column, factor to metric or None for text)
    "nidId": ("nid_id", None),
    "name": ("name", None),
    "nidStorage": ("storage_ML", ACRE_FT_TO_ML),
    "normalStorage": ("normal_storage_ML", ACRE_FT_TO_ML),
    "surfaceArea": ("surfacearea_m2", ACRE_TO_M2),
    "nidHeight": ("height_m", FT_TO_M),
    "drainageArea": ("drainage_km2", SQMI_TO_KM2),
    "yearCompleted": ("year_completed", None),
    "ownerTypeIds": ("owner_type", None),
    "primaryPurposeId": ("purpose", None),
    "publicHazardId": ("hazard", None),
}


def fetch_official_dams(aoi, cache: str | os.PathLike | None = None) -> gpd.GeoDataFrame:
    """Dams in the National Inventory of Dams (USACE) inside the AOI, in metric units.

    Reads the national NID GeoPackage directly. (HyRiver's ``pygeohydro.NID`` failed
    against the inventory's current column names when this was written.)

    Returns
    -------
    GeoDataFrame (EPSG:5070, points)
        nid_id, name, storage_ML, normal_storage_ML, surfacearea_m2, height_m,
        drainage_km2, year_completed, owner_type, purpose, hazard.
    """
    aoi = as_aoi(aoi)
    path = nid_inventory_path(cache)
    nid_crs = gpd.read_file(path, rows=1).crs or "EPSG:4269"
    aoi_nid = aoi.to_crs(nid_crs)
    raw = gpd.read_file(path, bbox=aoi_nid.bounds)
    raw = raw[raw.within(aoi_nid.geometry)]

    out = pd.DataFrame(index=raw.index)
    for field, (column, factor) in NID_COLUMNS.items():
        values = raw[field] if field in raw else pd.Series(None, index=raw.index, dtype="object")
        if factor is not None:
            out[column] = pd.to_numeric(values, errors="coerce") * factor
        elif column == "year_completed":
            out[column] = pd.to_numeric(values, errors="coerce").astype("Int64")
        else:
            out[column] = values.astype("string")
    dams = gpd.GeoDataFrame(out, geometry=raw.geometry, crs=raw.crs).to_crs(EQUAL_AREA)
    return dams.reset_index(drop=True)
