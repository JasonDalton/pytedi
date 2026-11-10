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
4. Drop shadows by shape. Deep shadows of tree lines and buildings are as dark in NIR as
   water, but they are often thin strips. A polygon is kept only if it is at least
   ``min_width_m`` wide (the diameter of the largest circle that fits inside) and either at
   least ``elongation_free_width_m`` wide or no more than ``max_elongation`` times longer
   than wide. (2 of the 56 NHD ponds at Hurt would fail the elongation test alone, which is
   why wide water is exempt.)
5. Drop shadows by persistence. Shadows move with the flight date and time; ponds stay. A
   polygon smaller than ``confirm_below_m2`` (500 m2) is kept only if at least
   ``confirm_fraction`` (20%) of it, with a 3 m margin, is also water in the previous NAIP
   year (``confirm_year``). Larger ponds are kept without the check.

Checked by eye on the 45 ponds that only NAIP 2023 found at Hurt, Virginia: steps 1-4 kept
about 21 real ponds and 20 shadows or treatment tanks (4 unclear); step 5 then kept 18 of
those 22 real ponds and 2 of the 18 shadows (the tanks are real water and stay). The rule
was tuned on those same ponds, so expect somewhat worse elsewhere.

It finds any open water, including sewage-treatment tanks and pools; check the map.
"""

from __future__ import annotations

import hashlib
import os

import geopandas as gpd
import numpy as np
from shapely import maximum_inscribed_circle

from .aoi import AOI, EQUAL_AREA, WGS84, as_aoi
from .ponds import water_polygons

PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ("red", "green", "blue", "nir")


def naip_water_mask(img, threshold: float = 0.5) -> np.ndarray:
    """True where NDWI of a (4, rows, cols) red-green-blue-NIR array exceeds the threshold.

    Computed in float32 without division (NDWI > t  <=>  g - n > t * (g + n) for g + n > 0)
    to keep memory down on 1 m mosaics.
    """
    img = np.asarray(img)
    green = img[1].astype("float32")
    nir = img[3].astype("float32")
    total = green + nir
    water = (green - nir) > np.float32(threshold) * total
    water &= total > 0
    water &= (img > 0).any(axis=0)
    return water


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


def _naip_years(aoi: AOI) -> list[int]:
    """NAIP years available over the AOI."""
    import planetary_computer
    from pystac_client import Client

    catalog = Client.open(PLANETARY_COMPUTER, modifier=planetary_computer.sign_inplace)
    items = catalog.search(collections=["naip"], bbox=aoi.to_crs(WGS84).bounds).items()
    return sorted({int(i.properties["naip:year"]) for i in items})


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
                      confirm_year="previous", confirm_below_m2: float = 500.0, confirm_fraction: float = 0.2,
                      resolution: float = 1.0, cache: str | os.PathLike | None = None) -> gpd.GeoDataFrame:
    """Open water polygons in the AOI from NAIP imagery (see the module notes).

    Parameters
    ----------
    year : int, optional
        NAIP year; the latest over the AOI by default.
    confirm_year : "previous", int or None
        Year whose imagery must also show water for ponds under ``confirm_below_m2``;
        "previous" is the NAIP year before ``year``; None skips the check.

    Returns
    -------
    GeoDataFrame (EPSG:5070)
        geometry, surfacearea_m2, source ("naip"). ``attrs`` records the years and settings.
    """
    from rasterio import features

    aoi = as_aoi(aoi)
    years = _naip_years(aoi)
    if not years:
        raise ValueError("No NAIP imagery covers this area.")
    year = years[-1] if year is None else int(year)
    if confirm_year == "previous":
        earlier = [y for y in years if y < year]
        confirm_year = earlier[-1] if earlier else None

    img, transform, crs, year = _load_naip(aoi, year, resolution, cache)
    water = naip_water_mask(img, threshold)
    del img
    found = water_polygons(water, transform, crs, min_area_m2).to_crs(EQUAL_AREA)
    del water
    found = found[found.intersects(aoi.to_crs(EQUAL_AREA).geometry)]
    found = shape_filter(found, min_width_m, max_elongation, elongation_free_width_m).reset_index(drop=True)

    dropped = 0
    small = found.index[found["surfacearea_m2"] < confirm_below_m2]
    if confirm_year is not None and len(small):
        before, before_transform, before_crs, _ = _load_naip(aoi, int(confirm_year), resolution, cache)
        water_before = naip_water_mask(before, threshold)
        del before
        zones = found.loc[small].to_crs(before_crs).buffer(3.0)
        labels = features.rasterize([(g, n + 1) for n, g in enumerate(zones)], out_shape=water_before.shape,
                                    transform=before_transform, fill=0, dtype="int32")
        counts = np.bincount(labels.ravel(), minlength=len(zones) + 1)[1:]
        wet = np.bincount(labels.ravel(), weights=water_before.ravel(), minlength=len(zones) + 1)[1:]
        share = np.divide(wet, counts, out=np.zeros_like(wet), where=counts > 0)
        unconfirmed = small[share < confirm_fraction]
        dropped = len(unconfirmed)
        found = found.drop(index=unconfirmed).reset_index(drop=True)

    found["source"] = "naip"
    found.attrs.update(year=year, confirm_year=confirm_year, unconfirmed_dropped=dropped, threshold=threshold,
                       min_area_m2=min_area_m2, min_width_m=min_width_m, max_elongation=max_elongation,
                       confirm_below_m2=confirm_below_m2, confirm_fraction=confirm_fraction, resolution=resolution)
    return found
