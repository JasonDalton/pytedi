"""build_damlist end to end on synthetic data, and the pytedi schema of its output."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import LineString, Point, box

from pytedi import tedimodel
from pytedi.engine import damlist_preprocessing
from pytedi.us import damlist as dl
from pytedi.us.aoi import AOI
from pytedi.us.fetch import NHDData
from test_us_hydro import COLS, ROWS, X0, Y0, cells, valley_dem

CRS = "EPSG:5070"
GRID = AOI(box(X0, Y0 - ROWS * 10, X0 + COLS * 10, Y0), CRS, "synthetic valley")


def empty(geom_type=None):
    return gpd.GeoDataFrame(geometry=[], crs=CRS)


@pytest.fixture
def fake_sources(monkeypatch):
    """A valley with an NHD pond upstream, a Sentinel-2 pond downstream and one NID dam."""
    nhd = NHDData(
        flowlines=gpd.GeoDataFrame(geometry=[LineString([(X0 + 105, Y0), (X0 + 105, Y0 - 400)])], crs=CRS),
        waterbodies=gpd.GeoDataFrame({"permanent_identifier": ["nhd-1"], "gnis_name": ["Upper Pond"]},
                                     geometry=[cells(5, 7, 9, 11)], crs=CRS),
        rivers=empty(), catchments=None, dataset="nhd_hr")
    s2 = gpd.GeoDataFrame({"surfacearea_m2": [900.0], "source": "sentinel2"}, geometry=[cells(25, 27, 9, 11)], crs=CRS)
    nid = gpd.GeoDataFrame({"nid_id": ["VA9"], "name": ["Lower Dam"], "storage_ML": [5.0], "surfacearea_m2": [900.0]},
                           geometry=[Point(X0 + 105, Y0 - 285)], crs=CRS)
    # NAIP sees the downstream pond and a small pond on the hillslope that nothing else found.
    naip = gpd.GeoDataFrame({"surfacearea_m2": [900.0, 200.0], "source": "naip"},
                            geometry=[cells(25, 27, 9, 11), cells(15, 16, 2, 3)], crs=CRS)
    calls = {}
    monkeypatch.setattr(dl, "fetch_3dep", lambda aoi, resolution, cache=None: valley_dem())
    monkeypatch.setattr(dl, "fetch_nhd", lambda aoi, dataset, cache=None: nhd)
    monkeypatch.setattr(dl, "fetch_official_dams", lambda aoi, cache=None: nid)

    def fake_detect(aoi, date_range, threshold, min_area_m2, max_cloud):
        calls.update(date_range=date_range, threshold=threshold, min_area_m2=min_area_m2)
        return s2

    monkeypatch.setattr(dl, "detect_ponds_sentinel2", fake_detect)

    def fake_naip(aoi, year, threshold, min_area_m2, cache=None):
        calls.update(naip_year=year, naip_threshold=threshold)
        return naip

    monkeypatch.setattr(dl, "detect_ponds_naip", fake_naip)
    return calls


def test_build_damlist_links_the_ponds(fake_sources):
    r = dl.build_damlist(GRID, date_range=("2024-06-01", "2024-09-30"), ndwi_threshold=0.1)
    assert fake_sources["threshold"] == 0.1 and fake_sources["min_area_m2"] == 500.0
    c = r.candidates.set_index("source")
    assert set(c.index) == {"nhd", "naip+sentinel2", "naip"}
    up, down, side = c.loc["nhd"], c.loc["naip+sentinel2"], c.loc["naip"]
    assert up["ds_damid"] == down["dam_id"] and side["ds_damid"] == down["dam_id"] and down["ds_damid"] == "-99"
    assert down["nid_id"] == "VA9"
    assert r.cat_area_km2 == pytest.approx(GRID.area_km2)
    assert len(r.damlist) == 3
    assert fake_sources["naip_year"] is None and fake_sources["naip_threshold"] == 0.5


def test_damlist_has_the_tedimodel_schema_and_passes_engine_checks(fake_sources):
    r = dl.build_damlist(GRID)
    reference = tedimodel().damlist
    assert list(r.damlist.columns) == list(reference.columns)
    assert dict(r.damlist.dtypes) == dict(reference.dtypes)
    d = r.damlist.set_index("dam_id")
    assert (d["year_start"] == 0).all() and (d["year_end"] == 9999).all()
    assert (d["group_name"] == "none").all() and (d["demfactor"] == 0.0).all()
    assert np.isfinite(d[["surfacearea_m2", "vol_ML", "catchuptotal_km2", "catchuplocal_km2"]].to_numpy()).all()
    demands = pd.DataFrame({"stock": [0.0]}, index=pd.date_range("2024-01-01", periods=1))
    damlist_preprocessing(r.damlist, demands, r.cat_area_km2, 0.8)  # raises on any schema problem


def test_volume_follows_the_area_relation(fake_sources):
    r = dl.build_damlist(GRID, vol_a=0.0002, vol_b=1.3)
    d = r.damlist
    assert d["vol_ML"].to_numpy() == pytest.approx(0.0002 * d["surfacearea_m2"].to_numpy() ** 1.3)


def test_damlist_runs_in_the_model(fake_sources):
    r = dl.build_damlist(GRID)
    model = r.to_model(title="synthetic valley")
    days = pd.date_range("2024-01-01", periods=60)
    model.ts = pd.DataFrame({"rain": 2.0, "evap": 3.0, "flowup": 5.0, "flowdown": np.nan, "stock": 0.0}, index=days)
    model.run_model()
    assert len(model.catch_results) == 60
    assert model.catch_results["flow_down"].iloc[-1] <= 5.0


def test_without_sentinel2_and_writing_a_geopackage(fake_sources, tmp_path):
    out = tmp_path / "ponds.gpkg"
    r = dl.build_damlist(GRID, use_sentinel2=False, use_naip=False, out_gpkg=out)
    assert "date_range" not in fake_sources and "naip_year" not in fake_sources
    assert set(r.candidates["source"]) == {"nhd", "nid"}
    layers = {name for name, _ in pyogrio.list_layers(out)}
    assert {"candidates", "outlets", "nid_dams", "flowlines", "aoi"} <= layers
    written = gpd.read_file(out, layer="candidates")
    assert {"dam_id", "ds_damid", "source", "vol_ML", "catchuptotal_km2"} <= set(written.columns)


def test_no_ponds_gives_an_empty_valid_damlist(monkeypatch):
    nothing = NHDData(empty(), empty(), empty(), None, "nhd_hr")
    monkeypatch.setattr(dl, "fetch_3dep", lambda aoi, resolution, cache=None: valley_dem())
    monkeypatch.setattr(dl, "fetch_nhd", lambda aoi, dataset, cache=None: nothing)
    monkeypatch.setattr(dl, "fetch_official_dams", lambda aoi, cache=None: empty())
    monkeypatch.setattr(dl, "detect_ponds_sentinel2", lambda *a, **k: empty())
    monkeypatch.setattr(dl, "detect_ponds_naip", lambda *a, **k: empty())
    r = dl.build_damlist(GRID)
    assert r.damlist.empty and list(r.damlist.columns) == list(tedimodel().damlist.columns)
    assert r.candidates.empty


def test_summary_counts(fake_sources):
    s = dl.build_damlist(GRID).summary()
    assert s["ponds"] == 3 and s["by_source"] == {"naip": 1, "naip+sentinel2": 1, "nhd": 1}
    assert s["nid_dams_in_aoi"] == 1 and s["nid_dams_matched"] == 1


def test_volume_relation_presets(fake_sources):
    from pytedi.us.volume import RELATIONS

    r = dl.build_damlist(GRID, volume_relation="va_nid_farm")
    a, b = RELATIONS["va_nid_farm"]
    assert r.damlist["vol_ML"].to_numpy() == pytest.approx(a * r.damlist["surfacearea_m2"].to_numpy() ** b)
    assert r.settings["volume_relation"] == "va_nid_farm"
    model = r.to_model()
    assert (model.vol_surfarea_a, model.vol_surfarea_b) == (a, b)
    with pytest.raises(ValueError, match="relation"):
        dl.build_damlist(GRID, volume_relation="unknown")


def test_catchment_area_can_include_area_upstream_of_the_aoi(fake_sources):
    r = dl.build_damlist(GRID, cat_area_km2=5.0)
    assert r.cat_area_km2 == 5.0 and r.settings["cat_area_km2"] == 5.0
    assert r.to_model().cat_area_km2 == 5.0
    with pytest.raises(ValueError, match="smaller"):
        dl.build_damlist(GRID, cat_area_km2=GRID.area_km2 / 2)


def test_summary_reports_aoi_and_catchment_areas(fake_sources):
    s = dl.build_damlist(GRID, cat_area_km2=5.0).summary()
    assert s["aoi_km2"] == pytest.approx(round(GRID.area_km2, 2)) and s["cat_area_km2"] == 5.0
