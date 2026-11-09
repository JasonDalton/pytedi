"""Sentinel-2 pond detection on synthetic scenes."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

from pytedi.us import ponds
from shapely.geometry import box

from pytedi.us.aoi import AOI

T = Affine(10.0, 0.0, 1_000_000.0, 0.0, -10.0, 1_600_000.0)  # 10 m cells in EPSG:5070
GRID_AOI = AOI(box(1_000_000, 1_599_700, 1_000_300, 1_600_000), "EPSG:5070", "synthetic grid")


def test_ndwi_is_mcfeeters_and_safe_with_zeros():
    green = np.array([0.10, 0.05, 0.0, np.nan])
    nir = np.array([0.02, 0.30, 0.0, 0.1])
    out = ponds.ndwi(green, nir)
    assert out[0] == pytest.approx((0.10 - 0.02) / 0.12)
    assert out[1] < 0
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_reflectance_applies_each_scenes_scale_and_offset_and_nodata():
    dn = np.array([0, 1000, 2000], dtype="uint16")
    out = ponds.to_reflectance(dn, scale=0.0001, offset=-0.1)
    assert np.isnan(out[0])
    assert out[1:] == pytest.approx([0.0, 0.1], abs=1e-6)
    assert ponds.to_reflectance(dn, scale=0.0001, offset=0.0)[1] == pytest.approx(0.1)


def test_masked_median_skips_clouds_and_shadows():
    # Three dates, one pixel: a cloud (SCL 9) and a shadow (SCL 3) must not count.
    values = np.array([[[0.9]], [[0.2]], [[0.05]]])
    scl = np.array([[[9]], [[4]], [[3]]])
    assert ponds.masked_median(values, scl)[0, 0] == pytest.approx(0.2)
    all_cloud = ponds.masked_median(values, np.full_like(scl, 9))
    assert np.isnan(all_cloud[0, 0])


def test_water_polygons_drop_specks_and_measure_area():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True      # 16 cells = 1600 m2, kept
    mask[10:12, 10:12] = True  # 4 cells = 400 m2, below 500 m2
    mask[15, 15] = True        # one cell
    gdf = ponds.water_polygons(mask, T, "EPSG:5070", min_area_m2=500)
    assert len(gdf) == 1
    assert gdf["surfacearea_m2"].iloc[0] == pytest.approx(1600.0)
    assert gdf.crs.to_epsg() == 5070
    minx, miny, maxx, maxy = gdf.total_bounds
    assert (minx, maxy) == (1_000_020.0, 1_599_980.0)


def test_water_polygons_join_diagonal_cells():
    mask = np.zeros((10, 10), dtype=bool)
    for i in range(6):
        mask[i, i] = True
    gdf = ponds.water_polygons(mask, T, "EPSG:5070", min_area_m2=500)
    assert len(gdf) == 1 and gdf["surfacearea_m2"].iloc[0] == pytest.approx(600.0)


def synthetic_scenes():
    """Two dates over a 30 x 30 grid: a 6 x 6 pond, vegetation elsewhere, one cloudy date."""
    shape = (2, 30, 30)
    green = np.full(shape, 1000 + 600, dtype="uint16")   # reflectance 0.06 after the -0.1 offset
    nir = np.full(shape, 1000 + 3000, dtype="uint16")    # 0.30: vegetation
    nir[:, 10:16, 10:16] = 1000 + 100                     # 0.01: water
    scl = np.full(shape, 4, dtype="uint8")
    scl[1] = 9  # second date all cloud: it must not change the answer
    nir[1] = 1000 + 50
    times = pd.to_datetime(["2024-07-01", "2024-07-11"])
    coords = {"time": times, "y": 1_600_000.0 - 5 - 10 * np.arange(30), "x": 1_000_000.0 + 5 + 10 * np.arange(30)}
    ds = xr.Dataset({"green": (("time", "y", "x"), green), "nir": (("time", "y", "x"), nir),
                     "scl": (("time", "y", "x"), scl)}, coords=coords)
    ds = ds.rio.write_crs("EPSG:5070")
    calibration = {t.date(): {"green": (0.0001, -0.1), "nir": (0.0001, -0.1)} for t in times}
    return ds, calibration


def test_detect_ponds_end_to_end_on_synthetic_scenes(monkeypatch):
    seen = {}

    def fake_load(aoi, date_range, max_cloud):
        seen.update(date_range=date_range, max_cloud=max_cloud)
        return synthetic_scenes()

    monkeypatch.setattr(ponds, "_load_scenes", fake_load)
    found = ponds.detect_ponds_sentinel2(GRID_AOI, ("2024-06-01", "2024-09-30"))
    assert seen == {"date_range": ("2024-06-01", "2024-09-30"), "max_cloud": 20}
    assert len(found) == 1
    assert found["surfacearea_m2"].iloc[0] == pytest.approx(3600.0)
    assert found["source"].iloc[0] == "sentinel2"
    assert found.attrs["scenes"] == 2 and found.attrs["threshold"] == 0.0


def test_detect_ponds_threshold_is_tunable(monkeypatch):
    monkeypatch.setattr(ponds, "_load_scenes", lambda *a: synthetic_scenes())
    # Water NDWI here is (0.06 - 0.01) / 0.07 = 0.71; a stricter threshold above that finds nothing.
    found = ponds.detect_ponds_sentinel2(GRID_AOI, threshold=0.8)
    assert found.empty and "surfacearea_m2" in found.columns


def test_no_scenes_is_an_empty_result(monkeypatch):
    monkeypatch.setattr(ponds, "_load_scenes", lambda *a: (None, {}))
    with pytest.warns(UserWarning, match="No Sentinel-2 scenes"):
        found = ponds.detect_ponds_sentinel2(GRID_AOI)
    assert found.empty and found.crs.to_epsg() == 5070


def test_ponds_outside_the_aoi_polygon_are_dropped(monkeypatch):
    monkeypatch.setattr(ponds, "_load_scenes", lambda *a: synthetic_scenes())
    corner = AOI(box(1_000_000, 1_599_900, 1_000_050, 1_600_000), "EPSG:5070")  # far from the pond
    assert ponds.detect_ponds_sentinel2(corner).empty
