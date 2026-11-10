"""Watershed climate from gridMET, with the service mocked."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pytedi.us import climate
from pytedi.us.aoi import as_aoi

AOI = as_aoi((-79.4, 37.0, -79.3, 37.1))


def fake_gridmet(days, missing=()):
    """Two by two cells: rain 1..4 mm in the cells, ET0 2 mm everywhere; some days absent."""
    days = days.difference(pd.DatetimeIndex(missing))
    pr = np.broadcast_to(np.array([[1.0, 2.0], [3.0, np.nan]]), (len(days), 2, 2)).copy()  # one cell outside the AOI
    pet = np.full((len(days), 2, 2), 2.0)
    pet[:, 1, 1] = np.nan
    return xr.Dataset({"pr": (("time", "lat", "lon"), pr), "pet": (("time", "lat", "lon"), pet)},
                      coords={"time": days, "lat": [37.1, 37.0], "lon": [-79.4, -79.3]})


def test_watershed_mean_and_open_water_evaporation(monkeypatch):
    days = pd.date_range("2000-01-01", "2000-01-10")
    seen = {}

    def fake(geometry, dates, variables):
        seen.update(dates=dates, variables=variables)
        return fake_gridmet(days)

    monkeypatch.setattr(climate, "_get_gridmet", fake)
    c = climate.fetch_climate(AOI, "2000-01-01", "2000-01-10")
    assert seen["variables"] == ["pr", "pet"] and seen["dates"] == ("2000-01-01", "2000-01-10")
    assert list(c.columns) == ["rain", "evap"] and c.index.equals(days)
    assert c["rain"].to_numpy() == pytest.approx(2.0)          # mean of the 3 cells with data
    assert c["evap"].to_numpy() == pytest.approx(1.05 * 2.0)    # FAO-56 open water Kc
    assert c.attrs["evap_factor"] == 1.05 and "gridMET" in c.attrs["source"]
    assert c.attrs["filled_dates"] == []


def test_evap_factor_is_a_parameter(monkeypatch):
    monkeypatch.setattr(climate, "_get_gridmet", lambda *a: fake_gridmet(pd.date_range("2000-01-01", periods=3)))
    c = climate.fetch_climate(AOI, "2000-01-01", "2000-01-03", evap_factor=0.8)
    assert c["evap"].to_numpy() == pytest.approx(1.6)


def test_missing_days_are_filled_from_neighbors_and_listed(monkeypatch):
    days = pd.date_range("2000-12-29", "2001-01-02")
    monkeypatch.setattr(climate, "_get_gridmet", lambda *a: fake_gridmet(days, missing=["2000-12-31"]))
    c = climate.fetch_climate(AOI, "2000-12-29", "2001-01-02")
    assert c.index.equals(days) and not c.isna().any().any()
    assert c.attrs["filled_dates"] == ["2000-12-31"]


def test_long_gaps_are_refused(monkeypatch):
    days = pd.date_range("2000-01-01", "2000-01-20")
    gap = pd.date_range("2000-01-05", "2000-01-12")
    monkeypatch.setattr(climate, "_get_gridmet", lambda *a: fake_gridmet(days, missing=gap))
    with pytest.raises(ValueError, match="consecutive"):
        climate.fetch_climate(AOI, "2000-01-01", "2000-01-20")
