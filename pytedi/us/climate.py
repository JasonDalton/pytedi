"""Daily watershed climate for pytedi from gridMET.

gridMET (Abatzoglou 2013) is a daily, 4 km (1/24 degree) gridded surface-meteorology dataset
for the contiguous US from 1980 (the earliest date HyRiver's ``pygridmet`` serves). It is
used here because it provides both inputs pytedi needs directly:

- ``pr``: daily accumulated precipitation (mm), used as ``rain``;
- ``pet``: daily ASCE Penman-Monteith reference evapotranspiration for short grass, ET0
  (mm), turned into open-water evaporation as ``evap = evap_factor * ET0``.

Daymet has no evapotranspiration variable, so evaporation would have to be modeled from
temperature; gridMET's ET0 already uses its own humidity, wind and radiation.

The default ``evap_factor`` is 1.05, the FAO-56 crop coefficient for "Open Water, < 2 m depth
or in subhumid climates or tropics" (Allen et al. 1998, Table 12). Farm ponds are shallow.

The watershed value for each day is the mean over the grid cells HyRiver returns for the
AOI (cells outside it are masked).

The service returns no value for 31 December of leap years. Such missing days (up to
``max_fill_days`` in a row) are filled with a straight line between the neighboring days and
listed in ``attrs["filled_dates"]``; longer gaps raise an error.

Abatzoglou, J.T., 2013. Development of gridded surface meteorological data for ecological
applications and modelling. International Journal of Climatology 33, 121-131.
https://doi.org/10.1002/joc.3413

Allen, R.G., Pereira, L.S., Raes, D., Smith, M., 1998. Crop evapotranspiration: guidelines for
computing crop water requirements. FAO Irrigation and Drainage Paper 56, Rome. Table 12.
"""

from __future__ import annotations

import os

import pandas as pd

from .aoi import WGS84, as_aoi

OPEN_WATER_KC = 1.05


def _get_gridmet(geometry, dates, variables):
    import pygridmet

    return pygridmet.get_bygeom(geometry, dates, crs=WGS84, variables=variables)


def fetch_climate(aoi, start, end, evap_factor: float = OPEN_WATER_KC, max_fill_days: int = 3,
                  cache: str | os.PathLike | None = None) -> pd.DataFrame:
    """Daily watershed-mean rainfall and open-water evaporation in mm (columns rain, evap).

    Parameters
    ----------
    aoi : anything :func:`~pytedi.us.aoi.as_aoi` accepts
    start, end : str
        ISO dates, inclusive.
    evap_factor : float
        Open-water evaporation as a multiple of gridMET grass reference ET.
    max_fill_days : int
        Longest run of missing days to fill by interpolation.

    Returns
    -------
    DataFrame indexed by day, with ``attrs`` describing the source, method and filled dates.
    """
    from .fetch import _use_cache, cache_dir

    aoi = as_aoi(aoi).to_crs(WGS84)
    _use_cache(cache_dir(cache))
    dates = (str(start)[:10], str(end)[:10])
    ds = _get_gridmet(aoi.geometry, dates, ["pr", "pet"])
    spatial = [d for d in ds["pr"].dims if d != "time"]
    daily = ds[["pr", "pet"]].mean(dim=spatial, skipna=True).to_dataframe()[["pr", "pet"]]
    daily.index = pd.DatetimeIndex(daily.index).normalize()

    days = pd.date_range(dates[0], dates[1], freq="D")
    daily = daily.reindex(days)
    missing = daily["pr"].isna() | daily["pet"].isna()
    runs = missing.ne(missing.shift()).cumsum()[missing]
    if not runs.empty and runs.value_counts().max() > max_fill_days:
        raise ValueError(f"gridMET is missing more than {max_fill_days} consecutive days; not filling.")
    filled = list(days[missing].strftime("%Y-%m-%d"))
    daily = daily.interpolate(method="time", limit_direction="both")

    out = pd.DataFrame({"rain": daily["pr"], "evap": evap_factor * daily["pet"]}, index=days)
    out.index.name = "Date"
    out.attrs.update(
        source="gridMET (Abatzoglou 2013) via HyRiver pygridmet: pr and pet (ASCE grass reference ET0)",
        method="mean of the grid cells covering the AOI",
        evap="evap_factor x ET0",
        evap_factor=evap_factor,
        evap_factor_source="FAO-56 Table 12, open water < 2 m depth (Allen et al. 1998)" if evap_factor == OPEN_WATER_KC
        else "user supplied",
        filled_dates=filled,
        units="mm/day",
    )
    return out
