"""Fetchers, with every network call replaced by a fake."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, Polygon, box

from pytedi.us import fetch
from pytedi.us.aoi import as_aoi

# A triangle, so features inside its bounding box can still fall outside it.
AOI = as_aoi(Polygon([(-79.40, 37.00), (-79.30, 37.00), (-79.40, 37.10)]), name="test")
INSIDE = (-79.38, 37.02)
OUTSIDE_BUT_IN_BBOX = (-79.31, 37.09)


def square(lon, lat, half=0.0005):
    return box(lon - half, lat - half, lon + half, lat + half)


def test_fetch_3dep_asks_for_the_aoi_and_resolution(monkeypatch):
    seen = {}

    def fake_get_dem(geometry, resolution, crs):
        seen.update(geometry=geometry, resolution=resolution, crs=crs)
        dem = xr.DataArray(np.ones((3, 4), dtype="float64"), dims=("y", "x"),
                           coords={"y": [3.0, 2.0, 1.0], "x": [0.0, 1.0, 2.0, 3.0]})
        return dem.rio.write_crs("EPSG:5070")

    monkeypatch.setattr(fetch, "_get_dem", fake_get_dem)
    dem = fetch.fetch_3dep(AOI)
    assert seen["resolution"] == 10 and seen["crs"] == "EPSG:4326"
    assert seen["geometry"].equals(AOI.geometry)
    assert dem.name == "elevation" and dem.dtype == np.float32
    assert dem.rio.crs.to_epsg() == 5070


def fake_layers(dataset, layer, aoi):
    """What the services return for a bbox query: some features lie outside the AOI polygon."""
    if layer == "waterbody":
        return gpd.GeoDataFrame(
            {"PERMANENT_IDENTIFIER": ["a", "b", "c", "d"], "FTYPE": [390, 436, 466, 390],
             "GNIS_NAME": [None, "Mill Pond", None, None]},
            geometry=[square(*INSIDE), square(-79.37, 37.03), square(-79.36, 37.02), square(*OUTSIDE_BUT_IN_BBOX)],
            crs="EPSG:4326")
    if layer == "flowline":
        return gpd.GeoDataFrame({"ftype": [460, 460]},
                                geometry=[LineString([(-79.39, 37.01), (-79.35, 37.02)]),
                                          LineString([(-79.31, 37.09), (-79.30, 37.099)])], crs="EPSG:4326")
    if layer == "area":
        return gpd.GeoDataFrame({"FTYPE": [460]}, geometry=[square(-79.39, 37.05, 0.003)], crs="EPSG:4326")
    raise AssertionError(layer)


def test_fetch_nhd_clips_to_the_aoi_and_keeps_ponds_and_reservoirs(monkeypatch):
    monkeypatch.setattr(fetch, "_nhd_layer", fake_layers)
    nhd = fetch.fetch_nhd(AOI)
    assert nhd.dataset == "nhd_hr" and nhd.catchments is None
    wb = nhd.waterbodies
    assert wb.crs.to_epsg() == 5070
    assert list(wb["permanent_identifier"]) == ["a", "b"]  # swamp "c" dropped, "d" outside the polygon
    assert list(wb["gnis_name"].fillna("")) == ["", "Mill Pond"]
    assert len(nhd.flowlines) == 1 and len(nhd.rivers) == 1
    assert nhd.flowlines.crs.to_epsg() == 5070


def test_fetch_nhd_tolerates_missing_catchments(monkeypatch):
    def failing(*args):
        raise RuntimeError("service down")

    monkeypatch.setattr(fetch, "_nhd_layer", fake_layers)
    monkeypatch.setattr(fetch, "_nhdplus_hr_catchments", failing)
    with pytest.warns(UserWarning, match="catchments"):
        nhd = fetch.fetch_nhd(AOI, include_catchments=True)
    assert nhd.catchments is None


def test_fetch_nhd_rejects_unknown_datasets():
    with pytest.raises(ValueError, match="dataset"):
        fetch.fetch_nhd(AOI, dataset="nhd_mr")


def write_fake_nid(path):
    gpd.GeoDataFrame(
        {"name": ["Inside Dam", "Outside Dam"], "federalId": ["VA00001", "VA00002"], "nidId": ["VA00001", "VA00002"],
         "nidStorage": ["100", "50"], "normalStorage": ["80", None], "surfaceArea": ["10", "2"],
         "nidHeight": ["30", "12"], "yearCompleted": ["1965", None], "drainageArea": ["1.5", "0.2"],
         "ownerTypeIds": ["Private", "Private"], "primaryPurposeId": ["Recreation", "Fire Protection"],
         "publicHazardId": ["Low", "Low"]},
        geometry=[Point(*INSIDE), Point(*OUTSIDE_BUT_IN_BBOX)], crs="EPSG:4269",
    ).to_file(path, driver="GPKG")


def test_fetch_official_dams_reads_the_inventory_in_metric_units(monkeypatch, tmp_path):
    gpkg = tmp_path / "nid.gpkg"
    write_fake_nid(gpkg)
    monkeypatch.setattr(fetch, "nid_inventory_path", lambda cache_dir=None, refresh=False: gpkg)
    dams = fetch.fetch_official_dams(AOI)
    assert list(dams["nid_id"]) == ["VA00001"]
    row = dams.iloc[0]
    assert row["name"] == "Inside Dam" and row["owner_type"] == "Private" and row["purpose"] == "Recreation"
    assert row["storage_ML"] == pytest.approx(100 * 1.2334818375)
    assert row["normal_storage_ML"] == pytest.approx(80 * 1.2334818375)
    assert row["surfacearea_m2"] == pytest.approx(10 * 4046.8564224)
    assert row["height_m"] == pytest.approx(30 * 0.3048)
    assert row["drainage_km2"] == pytest.approx(1.5 * 2.589988110336)
    assert row["year_completed"] == 1965
    assert dams.crs.to_epsg() == 5070


def test_fetch_official_dams_with_none_in_the_aoi(monkeypatch, tmp_path):
    gpkg = tmp_path / "nid.gpkg"
    write_fake_nid(gpkg)
    monkeypatch.setattr(fetch, "nid_inventory_path", lambda cache_dir=None, refresh=False: gpkg)
    dams = fetch.fetch_official_dams(as_aoi((-100.0, 40.0, -99.9, 40.1)))
    assert dams.empty and "nid_id" in dams.columns and dams.crs.to_epsg() == 5070


def test_cache_dir_honors_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTEDI_US_CACHE", str(tmp_path / "c"))
    assert fetch.cache_dir() == tmp_path / "c" and (tmp_path / "c").is_dir()
    assert fetch.cache_dir(tmp_path / "d") == tmp_path / "d"


def test_fetch_nhd_carries_on_without_river_polygons(monkeypatch):
    def flaky(dataset, layer, aoi):
        if layer == "area":
            raise RuntimeError("504 Gateway Timeout")
        return fake_layers(dataset, layer, aoi)

    monkeypatch.setattr(fetch, "_nhd_layer", flaky)
    with pytest.warns(UserWarning, match="river polygons"):
        nhd = fetch.fetch_nhd(AOI)
    assert nhd.rivers.empty and nhd.rivers.crs.to_epsg() == 5070 and len(nhd.waterbodies) == 2


def test_fetch_nhd_retries_then_explains(monkeypatch):
    attempts = []

    def once_flaky(dataset, layer, aoi):
        attempts.append(layer)
        if attempts.count(layer) == 1:
            raise RuntimeError("504 Gateway Timeout")
        return fake_layers(dataset, layer, aoi)

    monkeypatch.setattr(fetch, "_nhd_layer", once_flaky)
    monkeypatch.setattr(fetch, "RETRY_WAIT_S", 0)
    assert len(fetch.fetch_nhd(AOI).waterbodies) == 2

    monkeypatch.setattr(fetch, "_nhd_layer", lambda *a: (_ for _ in ()).throw(RuntimeError("504")))
    with pytest.raises(RuntimeError, match="3dhp"):
        fetch.fetch_nhd(AOI)
