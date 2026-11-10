"""Live checks against the real US services. Deselected by default; run with:

    pytest -m network tests/us

They need internet access and download about 80 MB once (the NID GeoPackage).
"""
import pytest

from pytedi.us import (as_aoi, detect_ponds_naip, detect_ponds_sentinel2, fetch_3dep, fetch_nhd, fetch_official_dams, huc12_at)

pytestmark = pytest.mark.network
HURT_HUC12 = "030101011304"
SMALL = (-79.33, 37.02, -79.31, 37.04)  # a 2 km square inside that HUC12


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("pytedi_us_cache")


def test_huc12_lookup_for_hurt_va():
    assert huc12_at(37.09, -79.30) == HURT_HUC12
    aoi = as_aoi(HURT_HUC12)
    assert 70 < aoi.area_km2 < 85


def test_3dep_dem(cache):
    dem = fetch_3dep(SMALL, cache=cache)
    assert dem.rio.crs.to_epsg() == 5070 and dem.sizes["x"] > 100
    assert 100 < float(dem.min()) < float(dem.max()) < 500


def test_nhd_high_resolution(cache):
    nhd = fetch_nhd(as_aoi(HURT_HUC12), cache=cache)
    assert len(nhd.waterbodies) > 20 and len(nhd.flowlines) > 50


def test_national_inventory_of_dams(cache):
    dams = fetch_official_dams(as_aoi(HURT_HUC12), cache=cache)
    assert "VA143021" in set(dams["nid_id"])  # Brushy Mountain Dam


def test_sentinel2_finds_water():
    ponds = detect_ponds_sentinel2(SMALL)
    assert ponds.attrs["scenes"] > 0


def test_naip_finds_water(cache):
    ponds = detect_ponds_naip(SMALL, cache=cache)
    assert ponds.attrs["year"] >= 2023 and "surfacearea_m2" in ponds.columns


def test_gridmet_climate(cache):
    from pytedi.us import fetch_climate

    c = fetch_climate(as_aoi(HURT_HUC12), "2020-12-28", "2021-01-03", cache=cache)
    assert len(c) == 7 and c.attrs["filled_dates"] == ["2020-12-31"] and (c["evap"] >= 0).all()


def test_reference_gauges_and_flow(cache):
    from pytedi.us import fetch_gauge_flow, find_gauges, select_gauge

    candidates = find_gauges(as_aoi(HURT_HUC12), period=("1980-01-01", "2025-12-31"), cache=cache)
    chosen, reasons = select_gauge(candidates)
    assert chosen["site_no"] == "02056900" and reasons["02065500"] == "farther than the chosen gauge"
    q = fetch_gauge_flow("02056900", "2020-01-01", "2020-01-31")
    assert len(q) == 31 and (q > 0).all()


def test_drainage_area_of_a_huc12_that_receives_a_river():
    from pytedi.us import huc_drainage_area, upstream_hucs

    assert upstream_hucs("020700030504") == ["020700030501", "020700030502", "020700030503"]
    assert huc_drainage_area("020700030504") == pytest.approx(389.29, abs=0.01)
