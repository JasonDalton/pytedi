"""Hypothetical pond, before/after runs and the comparison, offline on the synthetic valley."""
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from pytedi.us import damlist as dl
from pytedi.us import scenario
from pytedi.us import timeseries as tsm
from test_us_damlist import GRID, fake_sources  # noqa: F401  (fixture)
from test_us_hydro import CELL, X0, Y0
from test_us_timeseries import DAYS, inputs  # noqa: F401  (fixture)

TO_LONLAT = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
SITE_LON, SITE_LAT = TO_LONLAT.transform(X0 + 10 * CELL + 5, Y0 - 15 * CELL - 5)  # channel, between the ponds


def test_seven_day_minima_use_complete_climatic_years():
    days = pd.date_range("2000-04-01", "2003-03-31")
    q = pd.Series(10.0, index=days)
    q["2000-09-01":"2000-09-07"] = 1.0
    q["2001-08-01":"2001-08-03"] = 0.0     # 3 zero days inside a 7-day window: mean 4/7 * 10
    q["2002-07-01":"2002-07-10"] = 2.0
    minima = scenario.seven_day_minima(q)
    assert list(minima.index) == [2000, 2001, 2002]  # climatic years starting 1 April
    assert minima.tolist() == pytest.approx([1.0, 40.0 / 7.0, 2.0])
    assert scenario.seven_day_minima(q["2000-05-01":]).index.tolist() == [2001, 2002]  # 2000 incomplete


def test_q7_10_weibull():
    minima = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0])  # n = 9: rank 1 has p = 0.1
    assert scenario.q7_10(minima) == pytest.approx(1.0)
    assert np.isnan(scenario.q7_10(pd.Series([1.0, 2.0])))  # too short to say


def test_add_hypothetical_pond_relinks_and_flags_output(fake_sources):
    before = dl.build_damlist(GRID)
    after = scenario.add_hypothetical_pond(before, lat=SITE_LAT, lon=SITE_LON, surface_area_m2=400.0)
    assert len(after.damlist) == len(before.damlist) + 1 and len(before.damlist) == 3  # input untouched
    c = after.candidates.set_index("dam_id")
    new = c.loc["new"]
    assert new["source"] == "hypothetical" and new["surfacearea_m2"] == pytest.approx(400.0, rel=0.01)
    upstream = c[c["source"] == "nhd"].iloc[0]
    downstream = c[c["source"] == "naip+sentinel2"].iloc[0]
    assert upstream["ds_damid"] == "new" and new["ds_damid"] == downstream.name
    d = after.damlist.set_index("dam_id")
    assert d.loc["new", "output"] == 1 and (d.drop(index="new")["output"] == 0).all()
    a, b = after.settings["vol_a"], after.settings["vol_b"]
    assert d.loc["new", "vol_ML"] == pytest.approx(a * d.loc["new", "surfacearea_m2"] ** b)
    assert after.settings["hypothetical"]["dam_id"] == "new"


def test_volume_can_be_given_instead_of_area(fake_sources):
    after = scenario.add_hypothetical_pond(dl.build_damlist(GRID), lat=SITE_LAT, lon=SITE_LON, volume_ml=2.0)
    d = after.damlist.set_index("dam_id")
    a, b = after.settings["vol_a"], after.settings["vol_b"]
    assert d.loc["new", "vol_ML"] == pytest.approx(2.0)
    assert d.loc["new", "surfacearea_m2"] == pytest.approx((2.0 / a) ** (1 / b), rel=0.01)
    with pytest.raises(ValueError, match="one of"):
        scenario.add_hypothetical_pond(dl.build_damlist(GRID), lat=SITE_LAT, lon=SITE_LON)


def test_before_after_comparison(fake_sources, inputs, tmp_path):
    base = dl.build_damlist(GRID)
    series = tsm.build_timeseries(base, start="2000-01-01", end="2002-12-31", min_years=1)
    natural = tsm.natural_flow(base, series)
    after = scenario.add_hypothetical_pond(base, lat=SITE_LAT, lon=SITE_LON, surface_area_m2=400.0,
                                           group_name="stock", demfactor=0.5)
    runs = scenario.run_before_after(after, series, natural, dam_id="new")
    assert runs.before.damlist.set_index("dam_id").loc["new", "vol_ML"] == 0.0
    assert "stock" in runs.after.ts.columns and runs.after.ts["stock"].iloc[0] == pytest.approx(1 / 365.25)
    cmp = scenario.compare_scenarios(runs.before, runs.after, dam_id="new", label="TEST")
    d = cmp.daily
    assert list(d.columns) == ["site_before", "site_after", "outlet_before", "outlet_after"]
    assert d.index.equals(DAYS)
    assert d["site_after"].sum() < d["site_before"].sum()            # evaporation and demand take water
    assert d["outlet_after"].sum() < d["outlet_before"].sum()
    s = cmp.summary
    assert s["label"] == "TEST" and s["period"] == ("2000-01-01", "2002-12-31")
    assert s["site"]["mean_annual_captured_ML"] == pytest.approx(
        (d["site_before"] - d["site_after"]).sum() / (len(d) / 365.25))
    assert set(cmp.fdc.columns) == {"site_before", "site_after", "outlet_before", "outlet_after"}
    assert (cmp.fdc["site_before"].diff().dropna() <= 1e-12).all()   # exceedance curve falls
    assert list(cmp.monthly.index) == list(range(1, 13))
    out = cmp.to_csv(tmp_path / "cmp")
    assert all(p.exists() for p in out)
    png = tmp_path / "cmp.png"
    scenario.plot_comparison(cmp, png)
    assert png.stat().st_size > 10_000
