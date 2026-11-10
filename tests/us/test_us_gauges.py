"""Gauge search, selection and transfer, with NWIS and GAGES-II mocked."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from pytedi.us import gauges
from pytedi.us.aoi import as_aoi

AOI = as_aoi((-79.40, 37.00, -79.30, 37.10))  # centroid (-79.35, 37.05); about 87 km2


def test_units_and_area_transfer():
    assert gauges.CFS_TO_ML_PER_DAY == pytest.approx(2.446575, rel=1e-6)
    assert gauges.cms_to_ml_per_day(1.0) == pytest.approx(86.4)
    q = pd.Series([10.0, 0.0, 30.0])
    assert gauges.transfer_flow(q, gauge_area_km2=200.0, target_area_km2=50.0).tolist() == [2.5, 0.0, 7.5]
    assert gauges.transfer_flow(q, 200.0, 50.0, exponent=0.5).tolist() == pytest.approx([5.0, 0.0, 15.0])


def nwis_sites():
    rows = [  # site, name, lat, lon, drainage sq mi, begin, end
        ("01", "Near But Short", 37.10, -79.30, 30.0, "2005-10-01", "2014-06-30"),
        ("02", "Mountain Ref", 37.20, -79.50, 40.0, "1950-10-01", "2026-11-01"),
        ("03", "Piedmont Ref", 37.40, -79.00, 90.0, "1946-10-01", "2026-11-01"),
        ("04", "Piedmont NonRef", 37.15, -79.40, 35.0, "1970-10-01", "2026-11-01"),
        ("05", "Too Big", 37.12, -79.33, 2000.0, "1930-10-01", "2026-11-01"),
        ("06", "Piedmont Ref Far", 37.80, -78.60, 60.0, "1950-10-01", "2026-11-01"),
    ]
    df = pd.DataFrame(rows, columns=["site_no", "station_nm", "dec_lat_va", "dec_long_va", "drain_area_va",
                                     "begin_date", "end_date"])
    df["parm_cd"], df["stat_cd"] = "00060", "00003"
    return df


def gagesii():
    rows = [("01", "Ref", "SEPlains"), ("02", "Ref", "EastHghlnds"), ("03", "Ref", "SEPlains"),
            ("04", "Non-ref", "SEPlains"), ("06", "Ref", "SEPlains"), ("99", "Non-ref", "SEPlains")]
    pts = [Point(-79.30, 37.10), Point(-79.50, 37.20), Point(-79.00, 37.40), Point(-79.40, 37.15),
           Point(-78.60, 37.80), Point(-79.35, 37.06)]  # "99" sits next to the AOI: it sets the ecoregion
    return gpd.GeoDataFrame({"STAID": [r[0] for r in rows], "CLASS": [r[1] for r in rows],
                             "AGGECOREGI": [r[2] for r in rows], "HCDN_2009": ["yes", None, "yes", None, None, None]},
                            geometry=pts, crs="EPSG:4326")


@pytest.fixture
def services(monkeypatch):
    monkeypatch.setattr(gauges, "_nwis_sites", lambda bbox: nwis_sites())
    monkeypatch.setattr(gauges, "_gagesii", lambda cache=None: gagesii())


def test_find_gauges_lists_candidates_in_range(services):
    c = gauges.find_gauges(AOI, period=("1980-01-01", "2025-12-31"))
    assert "05" not in set(c["site_no"])  # 5180 km2 is more than 5x the AOI
    row = c.set_index("site_no").loc["03"]
    assert row["drainage_km2"] == pytest.approx(90 * 2.589988110336)
    assert row["gagesii_class"] == "Ref" and row["ecoregion"] == "SEPlains" and row["hcdn_2009"] == "yes"
    assert row["years_in_period"] == 46  # complete calendar years 1980-2025 inside both its record and the period
    assert list(c["dist_km"]) == sorted(c["dist_km"])
    assert c.attrs["target_ecoregion"] == "SEPlains" and c.attrs["target_area_km2"] == pytest.approx(AOI.area_km2)


def test_select_gauge_applies_the_rule_in_order(services):
    c = gauges.find_gauges(AOI, period=("1980-01-01", "2025-12-31"))
    chosen, reasons = gauges.select_gauge(c)
    # 01 is nearest but has 8 years; 04 is not reference; 02 is in another ecoregion; 03 beats 06 on distance.
    assert chosen["site_no"] == "03"
    assert reasons["01"].startswith("fewer than 30") and reasons["04"] == "not GAGES-II reference"
    assert reasons["02"].startswith("ecoregion") and reasons["06"] == "farther than the chosen gauge"
    assert reasons["03"] == "chosen"


def test_select_gauge_says_when_nothing_qualifies(services):
    c = gauges.find_gauges(AOI, period=("1980-01-01", "2025-12-31"))
    with pytest.raises(ValueError, match="No gauge"):
        gauges.select_gauge(c, min_years=200)


def test_fetch_gauge_flow_converts_and_fills_short_gaps(monkeypatch):
    days = pd.date_range("2000-01-01", periods=6)
    cms = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], index=days.tz_localize("UTC") + pd.Timedelta(hours=5))
    monkeypatch.setattr(gauges, "_nwis_streamflow", lambda site, dates: cms.to_frame(f"USGS-{site}"))
    q = gauges.fetch_gauge_flow("03", "2000-01-01", "2000-01-06")
    assert q.index.equals(days) and q.name == "flow_ML"
    assert q.tolist() == pytest.approx([86.4, 172.8, 259.2, 345.6, 432.0, 518.4])
    assert q.attrs["filled_dates"] == ["2000-01-03"] and q.attrs["site_no"] == "03"


def test_fetch_gauge_flow_refuses_long_gaps(monkeypatch):
    days = pd.date_range("2000-01-01", periods=20)
    cms = pd.Series(1.0, index=days); cms.iloc[3:15] = np.nan
    monkeypatch.setattr(gauges, "_nwis_streamflow", lambda site, dates: cms.to_frame("USGS-03"))
    with pytest.raises(ValueError, match="consecutive"):
        gauges.fetch_gauge_flow("03", "2000-01-01", "2000-01-20")


def test_daylight_saving_stamps_keep_their_date(monkeypatch):
    # Local midnight in July (EDT) is 04:00 UTC; in January (EST) 05:00 UTC.
    stamps = pd.DatetimeIndex(["2000-07-01 04:00", "2000-07-02 04:00"], tz="UTC")
    monkeypatch.setattr(gauges, "_nwis_streamflow",
                        lambda site, dates: pd.Series([1.0, 2.0], index=stamps).to_frame("USGS-03"))
    q = gauges.fetch_gauge_flow("03", "2000-07-01", "2000-07-02")
    assert q.tolist() == pytest.approx([86.4, 172.8])
