"""Merging NHD ponds, Sentinel-2 detections and NID dams into one candidate table."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from pytedi.us.merge import merge_candidates

CRS = "EPSG:5070"


def sq(x, y, size=50.0):
    return box(x, y, x + size, y + size)


@pytest.fixture
def nhd():
    return gpd.GeoDataFrame({"permanent_identifier": ["n1", "n2"], "gnis_name": ["Mill Pond", None]},
                            geometry=[sq(0, 0), sq(1000, 0)], crs=CRS)


@pytest.fixture
def s2():
    return gpd.GeoDataFrame({"surfacearea_m2": [1600.0, 900.0, 2500.0], "source": "sentinel2"},
                            geometry=[sq(10, 10, 40), sq(500, 500, 30), sq(3000, 0, 50)], crs=CRS)


@pytest.fixture
def nid():
    return gpd.GeoDataFrame(
        {"nid_id": ["VA1", "VA2"], "name": ["Mill Dam", "Lone Dam"], "storage_ML": [60.0, 30.0],
         "surfacearea_m2": [2500.0, 4046.86]},
        geometry=[Point(25, -20), Point(5000, 5000)], crs=CRS)


def test_sources_and_dedupe(nhd, s2, nid):
    c = merge_candidates(nhd, s2, nid)
    assert sorted(c["source"]) == ["both", "nhd", "nid", "sentinel2", "sentinel2"]
    both = c[c["source"] == "both"].iloc[0]
    assert both["nhd_id"] == "n1" and both["name"] == "Mill Pond"
    assert both.geometry.equals(sq(0, 0))  # NHD outline kept
    assert both["surfacearea_m2"] == pytest.approx(2500.0)


def test_nid_dams_attach_to_the_nearest_pond_within_reach(nhd, s2, nid):
    c = merge_candidates(nhd, s2, nid, nid_match_m=150)
    matched = c[c["nid_id"] == "VA1"]
    assert len(matched) == 1 and matched["nhd_id"].iloc[0] == "n1" and matched["nid_name"].iloc[0] == "Mill Dam"
    lone = c[c["source"] == "nid"].iloc[0]
    assert lone["nid_id"] == "VA2"
    assert lone["surfacearea_m2"] == pytest.approx(4046.86, rel=1e-3)  # a circle of the NID area
    assert lone.geometry.centroid.distance(Point(5000, 5000)) < 1


def test_out_of_reach_nid_dam_is_not_matched(nhd, s2, nid):
    c = merge_candidates(nhd, s2, nid, nid_match_m=10)
    assert (c["source"] == "nid").sum() == 2


def test_sentinel2_polygons_on_rivers_are_dropped(nhd, s2, nid):
    rivers = gpd.GeoDataFrame(geometry=[box(2900, -100, 3100, 100)], crs=CRS)
    c = merge_candidates(nhd, s2, nid, rivers=rivers)
    assert (c["source"] == "sentinel2").sum() == 1


def test_nid_dam_without_area_gets_a_small_default_footprint(nhd, s2, nid):
    nid.loc[1, "surfacearea_m2"] = np.nan
    c = merge_candidates(nhd, s2, nid, nid_default_area_m2=500.0)
    lone = c[c["source"] == "nid"].iloc[0]
    assert lone["surfacearea_m2"] == pytest.approx(500.0, rel=1e-3) and lone["area_from_default"]


def test_everything_empty_gives_an_empty_typed_table():
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS)
    c = merge_candidates(empty, empty, empty)
    assert c.empty and c.crs.to_epsg() == 5070
    assert {"source", "surfacearea_m2", "nhd_id", "nid_id"} <= set(c.columns)


def test_one_sentinel2_polygon_touching_two_nhd_ponds_marks_both():
    nhd = gpd.GeoDataFrame({"permanent_identifier": ["a", "b"]}, geometry=[sq(0, 0), sq(60, 0)], crs=CRS)
    s2 = gpd.GeoDataFrame({"surfacearea_m2": [3000.0]}, geometry=[box(30, 10, 80, 40)], crs=CRS)
    c = merge_candidates(nhd, s2, gpd.GeoDataFrame(geometry=[], crs=CRS))
    assert list(c["source"]) == ["both", "both"]


def test_numeric_columns_stay_numeric(nhd, s2, nid):
    c = merge_candidates(nhd, s2, nid)
    assert c["nid_storage_ML"].dtype == "float64" and c["surfacearea_m2"].dtype == "float64"
    assert c.loc[c["nid_id"] == "VA1", "nid_storage_ML"].iloc[0] == 60.0
    assert c["area_from_default"].dtype == bool
