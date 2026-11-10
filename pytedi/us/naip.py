"""Detect farm ponds in NAIP aerial imagery (about 1 m).

The USDA National Agriculture Imagery Program flies each state every two or three years in
the growing season, with red, green, blue and near-infrared bands at 0.6 to 1 m. This reads
it from Microsoft's Planetary Computer (public; URLs are signed anonymously by the
``planetary-computer`` package), which is enough to see ponds of 100 m2 that Sentinel-2's
10 m pixels miss.

Method
------
1. Pick the latest NAIP year over the AOI (or the year asked for) and mosaic its tiles onto
   a grid in EPSG:5070 at ``resolution`` meters (1 m by default, read from the tiles' first
   overview and averaged). The mosaic is cached as a GeoTIFF, since reading it takes minutes.
2. NDWI = (green - NIR) / (green + NIR) on the 8-bit values. Open water is almost black in
   NAIP's NIR band (a median of 1 in the Hurt, Virginia ponds), so water scores near 1 and
   vegetation below 0. Water is NDWI > ``threshold`` (0.5).
3. Polygonize and keep polygons of ``min_area_m2`` (100 m2) or more.
4. Drop shadows. Deep shadows of tree lines and buildings are as dark in NIR as water, but
   they are thin strips. A polygon is kept only if it is at least ``min_width_m`` wide (the
   diameter of the largest circle that fits inside) and either at least
   ``elongation_free_width_m`` wide or no more than ``max_elongation`` times longer than
   wide. At Hurt, this removed every shadow among 51 hits outside the mapped ponds and kept
   the ponds; 2 of the 56 NHD ponds there would fail the elongation test alone, which is
   why wide water is exempt.

It finds any open water, including sewage-treatment tanks and pools; check the map.
"""

from __future__ import annotations

import hashlib
import os
import warnings

import geopandas as gpd
import numpy as np
from shapely import maximum_inscribed_circle

from .aoi import AOI, EQUAL_AREA, WGS84, as_aoi
from .ponds import ndwi, water_polygons

PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ("red", "green", "blue", "nir")


def naip_water_mask(img, threshold: float = 0.5) -> np.ndarray:
    """True where NDWI of a (4, rows, cols) red-green-blue-NIR array exceeds the threshold."""
    img = np.asarray(img)
    valid = (img > 0).any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        index = ndwi(img[1].astype("float32"), img[3].astype("float32"))
    return valid & (np.nan_to_num(index, nan=-1.0) > threshold)


def _elongation(polygon) -> float:
    xs, ys = polygon.minimum_rotated_rectangle.exterior.coords.xy
    a = np.hypot(xs[1] - xs[0], ys[1] - ys[0])
    b = np.hypot(xs[2] - xs[1], ys[2] - ys[1])
    return max(a, b) / max(min(a, b), 1e-9)


def shape_filter(polygons: gpd.GeoDataFrame, min_width_m: float = 8.0, max_elongation: float = 3.0,
                 elongation_free_width_m: float = 12.0) -> gpd.GeoDataFrame:
    """Keep pond-shaped polygons: wide enough, and compact unless wide (see the module notes)."""
    if polygons.empty:
        return polygons
    width = np.array([2 * maximum_inscribed_circle(g, 0.25).length for g in polygons.geometry])
    elongation = np.array([_elongation(g) for g in polygons.geometry])
    keep = (width >= min_width_m) & ((width >= elongation_free_width_m) | (elongation <= max_elongation))
    return polygons[keep]


def pick_naip_items(items, year: int | None):
    """(year, items) for the requested year, or the latest year available."""
    years = sorted({int(i.properties["naip:year"]) for i in items})
    if not years:
        raise ValueError("No NAIP imagery covers this area.")
    chosen = years[-1] if year is None else int(year)
    if chosen not in years:
        raise ValueError(f"No NAIP imagery from {chosen} here; available years: {years}.")
    return chosen, [i for i in items if int(i.properties["naip:year"]) == chosen]


def _target_grid(aoi: AOI, resolution: float):
    from affine import Affine

    west, south, east, north = aoi.to_crs(EQUAL_AREA).bounds
    west, north = np.floor(west), np.ceil(north)
    width = int(np.ceil((east - west) / resolution))
    height = int(np.ceil((north - south) / resolution))
    return Affine(resolution, 0.0, west, 0.0, -resolution, north), width, height


def _load_naip(aoi: AOI, year: int | None, resolution: float, cache):
    """Mosaic of the NAIP tiles over the AOI: (uint8 array (4, rows, cols), transform, crs, year)."""
    import planetary_computer
    import rasterio
    import rasterio.warp
    import rasterio.windows
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    from .fetch import cache_dir

    aoi = aoi.to_crs(WGS84)
    catalog = Client.open(PLANETARY_COMPUTER, modifier=planetary_computer.sign_inplace)
    year, items = pick_naip_items(list(catalog.search(collections=["naip"], bbox=aoi.bounds).items()), year)
    transform, width, height = _target_grid(aoi, resolution)
    key = hashlib.sha1(f"{aoi.geometry.wkb_hex}{year}{resolution}".encode()).hexdigest()[:16]
    path = cache_dir(cache) / f"naip_{year}_{resolution:g}m_{key}.tif"
    if path.exists():
        with rasterio.open(path) as src:
            return src.read(), src.transform, src.crs.to_string(), year

    mosaic = np.zeros((4, height, width), dtype="uint8")
    full = rasterio.windows.Window(0, 0, width, height)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY=3, GDAL_CACHEMAX=512):
        for item in items:
            # overview_level=0 is the first overview (about 1.2 m), a quarter of the full-resolution data.
            open_kwargs = {"overview_level": 0} if resolution >= 1.0 else {}
            with rasterio.open(item.assets["image"].href, **open_kwargs) as src, \
                    WarpedVRT(src, crs=EQUAL_AREA, transform=transform, width=width, height=height,
                              resampling=Resampling.average, nodata=0) as vrt:
                bounds = rasterio.warp.transform_bounds(src.crs, EQUAL_AREA, *src.bounds)
                window = rasterio.windows.from_bounds(*bounds, transform).round_offsets().round_lengths()
                try:
                    window = window.intersection(full)
                except rasterio.errors.WindowError:
                    continue
                data = vrt.read(indexes=[1, 2, 3, 4], window=window)
                r0, c0 = int(window.row_off), int(window.col_off)
                view = mosaic[:, r0:r0 + data.shape[1], c0:c0 + data.shape[2]]
                np.copyto(view, data, where=(data > 0) & (view == 0))
    profile = dict(driver="GTiff", width=width, height=height, count=4, dtype="uint8", crs=EQUAL_AREA,
                   transform=transform, nodata=0, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mosaic)
    return mosaic, transform, EQUAL_AREA, year


def detect_ponds_naip(aoi, year: int | None = None, threshold: float = 0.5, min_area_m2: float = 100.0,
                      min_width_m: float = 8.0, max_elongation: float = 3.0, elongation_free_width_m: float = 12.0,
                      resolution: float = 1.0, cache: str | os.PathLike | None = None) -> gpd.GeoDataFrame:
    """Open water polygons in the AOI from NAIP imagery (see the module notes).

    Returns
    -------
    GeoDataFrame (EPSG:5070)
        geometry, surfacearea_m2, source ("naip"). ``attrs`` records the year and settings.
    """
    aoi = as_aoi(aoi)
    img, transform, crs, year = _load_naip(aoi, year, resolution, cache)
    found = water_polygons(naip_water_mask(img, threshold), transform, crs, min_area_m2).to_crs(EQUAL_AREA)
    found = found[found.intersects(aoi.to_crs(EQUAL_AREA).geometry)]
    found = shape_filter(found, min_width_m, max_elongation, elongation_free_width_m).reset_index(drop=True)
    found["source"] = "naip"
    found.attrs.update(year=year, threshold=threshold, min_area_m2=min_area_m2, min_width_m=min_width_m,
                       max_elongation=max_elongation, resolution=resolution)
    return found
