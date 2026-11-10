"""build_timeseries and the natural-flow back-calculation, offline."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pytedi.us import damlist as dl
from pytedi.us import timeseries as tsm
from test_us_damlist import GRID, fake_sources  # noqa: F401  (fixture)

DAYS = pd.date_range("2000-01-01", "2002-12-31")


@pytest.fixture
def inputs(monkeypatch):
    rng = np.random.default_rng(1)
    rain = pd.Series(rng.gamma(0.4, 8.0, len(DAYS)), index=DAYS)
    climate = pd.DataFrame({"rain": rain, "evap": 3.0}, index=DAYS)
    climate.attrs.update(source="fake gridMET", evap_factor=1.05, filled_dates=[])
    gauge_q = pd.Series(50.0 + 400.0 * (rain > 15), index=DAYS, name="flow_ML")
    gauge_q.attrs.update(site_no="99999999", filled_dates=[])
    candidates = pd.DataFrame({"site_no": ["99999999"], "station_nm": ["Fake Creek"], "drainage_km2": [10.0],
                               "dist_km": [5.0], "begin_date": ["1950-10-01"], "end_date": ["2026-11-01"],
                               "years_in_period": [3], "gagesii_class": ["Ref"], "ecoregion": ["SEPlains"],
                               "hcdn_2009": ["yes"], "area_ratio": [1.0]})
    candidates.attrs.update(target_ecoregion="SEPlains")
    calls = {}
    monkeypatch.setattr(tsm, "fetch_climate", lambda aoi, start, end, evap_factor, cache=None:
                        calls.setdefault("climate", (start, end, evap_factor)) and climate.loc[start:end])
    monkeypatch.setattr(tsm, "find_gauges", lambda aoi, period, target_area_km2=None, cache=None: candidates)
    monkeypatch.setattr(tsm, "fetch_gauge_flow", lambda site, start, end: gauge_q.loc[start:end])
    return calls


def test_build_timeseries_columns_units_and_provenance(fake_sources, inputs):
    build = dl.build_damlist(GRID)
    s = tsm.build_timeseries(build, start="2000-01-01", end="2002-12-31", min_years=1)
    ts = s.ts
    assert list(ts.columns) == ["rain", "evap", "flowup", "flowdown", "none"]
    assert ts.index.equals(DAYS) and ts["flowup"].isna().all() and (ts["none"] == 0.0).all()
    ratio = build.cat_area_km2 / 10.0
    assert ts["flowdown"].iloc[0] == pytest.approx(50.0 * ratio)
    assert s.flow_is == "observed" and s.gauge["site_no"] == "99999999"
    assert s.provenance["transfer"] == {"method": "drainage-area ratio", "exponent": 1.0,
                                        "gauge_area_km2": 10.0, "target_area_km2": pytest.approx(build.cat_area_km2)}
    assert s.reasons == {"99999999": "chosen"}
    assert inputs["climate"] == ("2000-01-01", "2002-12-31", 1.05)


def test_flow_can_be_treated_as_natural(fake_sources, inputs):
    s = tsm.build_timeseries(dl.build_damlist(GRID), start="2000-01-01", end="2002-12-31", min_years=1, flow_is="natural")
    assert s.ts["flowdown"].isna().all() and s.ts["flowup"].notna().all()
    with pytest.raises(ValueError, match="flow_is"):
        tsm.build_timeseries(dl.build_damlist(GRID), start="2000-01-01", end="2002-12-31", min_years=1, flow_is="x")


def test_default_period_is_the_last_complete_years(fake_sources, inputs):
    s = tsm.build_timeseries(dl.build_damlist(GRID), min_years=1, today=date(2003, 3, 15),
                             earliest="2000-01-01")
    assert (s.start, s.end) == ("2000-01-01", "2002-12-31")


def test_natural_flow_round_trips_through_the_model(fake_sources, inputs):
    build = dl.build_damlist(GRID)
    s = tsm.build_timeseries(build, start="2000-01-01", end="2002-12-31", min_years=1)
    natural = tsm.natural_flow(build, s)
    assert natural.index.equals(DAYS) and natural.notna().all()
    # Forward run with the natural flow and the same ponds gives the observed flow back.
    model = tsm.model_with_flow(build, s, natural, title="check")
    model.run_model()
    # Within the reverse solver's absolute tolerance, 0.02 ML (engine.py, residual_atol).
    assert model.catch_results["flow_down"].to_numpy() == pytest.approx(s.ts["flowdown"].to_numpy(), rel=1e-6, abs=0.02)
