"""Detect small water bodies (farm ponds) in Sentinel-2 imagery.

Method
------
1. Find Sentinel-2 Level-2A scenes over the AOI in Element 84's Earth Search STAC catalog
   (public, no credentials), with scene cloud cover under ``max_cloud`` percent.
2. Convert digital numbers to surface reflectance with each scene's own scale and offset
   (scenes from processing baseline 04.00 on carry a -0.1 offset).
3. Mask clouds, cloud shadows, cirrus, saturated and missing pixels with the scene
   classification layer (SCL), then take the per-pixel median over the remaining dates.
   The median removes passing clouds, shadows and short-lived flooding.
4. Compute NDWI (McFeeters 1996) = (green - NIR) / (green + NIR) at 10 m and call a pixel
   water where NDWI > ``threshold`` (0 by default, McFeeters' value).
5. Polygonize the water mask (8-connected) and keep polygons of ``min_area_m2`` or more
   (500 m2, 0.05 ha, five 10 m pixels, by default).

Why NDWI rather than MNDWI: MNDWI uses the 20 m SWIR band, and a 0.05 ha pond is about one
20 m pixel. NDWI uses two 10 m bands, which resolves ponds down to a few pixels. Its known
weakness, built-up surfaces scoring as water, matters little in a rural AOI and is
checked against the map.

McFeeters, S.K., 1996. The use of the Normalized Difference Water Index (NDWI) in the
delineation of open water features. International Journal of Remote Sensing 17,
1425-1432. https://doi.org/10.1080/01431169608948714
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
from rasterio import features
from shapely.geometry import shape

from .aoi import AOI, EQUAL_AREA, WGS84, as_aoi

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
DEFAULT_DATE_RANGE = ("2024-06-01", "2024-09-30")
# Scene classification (SCL) values to drop: 0 no data, 1 saturated or defective,
# 3 cloud shadow, 8 cloud (medium probability), 9 cloud (high probability), 10 thin cirrus.
SCL_INVALID = (0, 1, 3, 8, 9, 10)


def ndwi(green, nir):
    """McFeeters NDWI, (green - nir) / (green + nir); NaN where undefined."""
    green = np.asarray(green, dtype="float64")
    nir = np.asarray(nir, dtype="float64")
    total = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (green - nir) / total
    out[~np.isfinite(out)] = np.nan
    return out


def to_reflectance(dn, scale: float, offset: float):
    """Surface reflectance from digital numbers; DN 0 (no data) becomes NaN."""
    dn = np.asarray(dn)
    out = dn.astype("float32") * np.float32(scale) + np.float32(offset)
    out[dn == 0] = np.nan
    return out


def masked_median(values, scl):
    """Per-pixel median over time (axis 0), ignoring dates whose SCL marks the pixel invalid."""
    values = np.where(np.isin(scl, SCL_INVALID), np.nan, values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
        return np.nanmedian(values, axis=0)


def water_polygons(mask, transform, crs, min_area_m2: float = 500.0) -> gpd.GeoDataFrame:
    """Polygons of the True cells of ``mask`` (8-connected), at least ``min_area_m2`` each."""
    mask = np.asarray(mask, dtype=bool)
    geoms = [shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask, transform=transform,
                                                  connectivity=8) if v == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["surfacearea_m2"] = gdf.area
    return gdf[gdf["surfacearea_m2"] >= min_area_m2].reset_index(drop=True)


def _load_scenes(aoi: AOI, date_range, max_cloud: float):
    """Load green, NIR and SCL for the AOI from Earth Search, one layer per solar day.

    Returns ``(dataset, calibration)`` where calibration maps each date to
    ``{"green": (scale, offset), "nir": (scale, offset)}``, or ``(None, {})`` with no scenes.
    """
    import odc.stac
    from pystac_client import Client

    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    aoi = aoi.to_crs(WGS84)
    items = list(Client.open(EARTH_SEARCH).search(
        collections=[COLLECTION], bbox=aoi.bounds, datetime="/".join(date_range),
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).items())
    if not items:
        return None, {}
    calibration = defaultdict(dict)
    for item in items:
        day = item.datetime.date()
        for band in ("green", "nir"):
            info = (item.assets[band].extra_fields.get("raster:bands") or [{}])[0]
            pair = (float(info.get("scale", 0.0001)), float(info.get("offset", 0.0)))
            if calibration[day].setdefault(band, pair) != pair:
                raise ValueError(f"Scenes from {day} disagree on the {band} scale/offset; load them separately.")
    ds = odc.stac.load(items, bands=["green", "nir", "scl"], bbox=aoi.bounds, crs=EQUAL_AREA, resolution=10,
                       groupby="solar_day", resampling="nearest")
    return ds, dict(calibration)


def composite(ds, calibration) -> tuple[np.ndarray, np.ndarray]:
    """Cloud-masked median green and NIR reflectance from a scene stack."""
    green, nir = [], []
    for t in ds["time"].values:
        day = pd.Timestamp(t).date()
        cal = calibration[day]
        green.append(to_reflectance(ds["green"].sel(time=t).values, *cal["green"]))
        nir.append(to_reflectance(ds["nir"].sel(time=t).values, *cal["nir"]))
    scl = ds["scl"].values
    return masked_median(np.stack(green), scl), masked_median(np.stack(nir), scl)


def detect_ponds_sentinel2(aoi, date_range=DEFAULT_DATE_RANGE, threshold: float = 0.0,
                           min_area_m2: float = 500.0, max_cloud: float = 20) -> gpd.GeoDataFrame:
    """Open water polygons in the AOI from a Sentinel-2 median composite (see the module notes).

    Parameters
    ----------
    aoi : anything :func:`~pytedi.us.aoi.as_aoi` accepts
    date_range : (start, end)
        ISO dates. The default, June to September 2024, is leaf-on with a high sun, when
        vegetation is bright in NIR and shadows are short.
    threshold : float
        NDWI above this is water.
    min_area_m2 : float
        Smallest pond kept (default 500 m2 = 0.05 ha).
    max_cloud : float
        Largest scene cloud cover, in percent, to use.

    Returns
    -------
    GeoDataFrame (EPSG:5070)
        One row per pond: geometry, surfacearea_m2, source ("sentinel2"). ``attrs`` records
        the number of scenes, the date range and the threshold.
    """
    aoi = as_aoi(aoi)
    ds, calibration = _load_scenes(aoi, tuple(date_range), max_cloud)
    attrs = dict(date_range=tuple(date_range), threshold=threshold, min_area_m2=min_area_m2,
                 scenes=0 if ds is None else int(ds.sizes["time"]))
    if ds is None:
        warnings.warn(f"No Sentinel-2 scenes under {max_cloud}% cloud for {date_range}.", UserWarning, stacklevel=2)
        found = gpd.GeoDataFrame({"surfacearea_m2": [], "source": []}, geometry=[], crs=EQUAL_AREA)
    else:
        green, nir = composite(ds, calibration)
        water = ndwi(green, nir) > threshold  # NaN compares False
        found = water_polygons(water, ds.rio.transform(), ds.rio.crs, min_area_m2).to_crs(EQUAL_AREA)
        found = found[found.intersects(aoi.to_crs(EQUAL_AREA).geometry)].reset_index(drop=True)
        found["source"] = "sentinel2"
    found.attrs.update(attrs)
    return found
