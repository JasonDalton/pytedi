"""Daily inputs for a pytedi model of a US AOI: rainfall, evaporation and streamflow.

:func:`build_timeseries` puts together

- ``rain`` and ``evap`` (mm/day) from gridMET (:func:`~pytedi.us.climate.fetch_climate`);
- the AOI outlet's flow (ML/day), transferred by drainage-area ratio from the reference gauge
  chosen by :func:`~pytedi.us.gauges.select_gauge`;
- a demand column ``none`` of zeros (the group every dam in a phase-1 dam list belongs to).

Natural or observed flow
------------------------
pytedi takes either the flow upstream of all dams ("natural", column ``flowup``, run with
``calc_nat_or_obs = "obs"``) or the flow below them ("observed", ``flowdown``, run with
``"nat"``). A GAGES-II reference gauge is minimally disturbed but its basin still holds farm
ponds, so by default (``flow_is="observed"``) the transferred record is treated as the AOI's
present-day flow with its existing ponds. :func:`natural_flow` then back-solves the flow
without them, and scenario runs route that natural flow through the existing ponds (plus any
new one) downstream. Treating the gauge as natural instead would apply the AOI's ponds on
top of a record that already reflects its own basin's ponds.

All runs start with the dams full (``startstore_propfull = 1.0``) so the first months don't
count a one-off filling of every pond as an impact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .climate import OPEN_WATER_KC, fetch_climate
from .gauges import fetch_gauge_flow, find_gauges, select_gauge, transfer_flow

GRIDMET_START = "1980-01-01"
START_FULL = 1.0
FLOW_IS = ("observed", "natural")


@dataclass
class InputSeries:
    """What :func:`build_timeseries` produced.

    Attributes
    ----------
    ts : DataFrame
        Columns rain, evap (mm/day), flowup, flowdown (ML/day; one of them NaN), none (0).
    start, end : str
        The period (inclusive).
    flow_is : str
        "observed" (flowdown) or "natural" (flowup).
    gauge : dict
        The chosen gauge's row from :func:`~pytedi.us.gauges.find_gauges`.
    candidates : DataFrame
        Every candidate gauge, with ``reasons`` explaining the choice.
    provenance : dict
        Sources, methods, parameters and filled dates, for reports.
    """

    ts: pd.DataFrame
    start: str
    end: str
    flow_is: str
    gauge: dict
    candidates: pd.DataFrame
    reasons: dict
    provenance: dict = field(default_factory=dict)

    @property
    def flow(self) -> pd.Series:
        return self.ts["flowdown" if self.flow_is == "observed" else "flowup"]


def default_period(gauge_end: str, today: date, earliest: str = GRIDMET_START) -> tuple[str, str]:
    """From ``earliest`` to the end of the last calendar year both the gauge and gridMET cover."""
    last = min(pd.Timestamp(gauge_end), pd.Timestamp(today) - pd.Timedelta(days=1))
    last_year = last.year if (last.month, last.day) == (12, 31) else last.year - 1
    return str(pd.Timestamp(earliest).date()), f"{last_year}-12-31"


def build_timeseries(build, start: str | None = None, end: str | None = None, gauge: str | None = None,
                     flow_is: str = "observed", evap_factor: float = OPEN_WATER_KC, transfer_exponent: float = 1.0,
                     min_years: int = 30, today: date | None = None, earliest: str = GRIDMET_START,
                     cache: str | os.PathLike | None = None) -> InputSeries:
    """Daily rain, evaporation and transferred flow for ``build`` (a :class:`DamListResult`).

    Parameters
    ----------
    start, end : str, optional
        Period. By default from 1980-01-01 (the start of gridMET) to the end of the last
        complete calendar year in the gauge's record (``today`` caps it; pass it in tests).
    gauge : str, optional
        A USGS site number to use instead of the selection rule (still listed with candidates).
    flow_is : {"observed", "natural"}
        How to treat the transferred flow (see the module notes).
    evap_factor : float
        Open-water evaporation / gridMET grass reference ET.
    transfer_exponent : float
        Exponent of the drainage-area ratio.
    min_years : int
        Complete years in the period a gauge needs (selection rule).
    """
    if flow_is not in FLOW_IS:
        raise ValueError(f"flow_is must be one of {FLOW_IS}, not {flow_is!r}.")
    aoi = build.aoi
    target_area = build.cat_area_km2
    today = today or date.today()
    search_period = (start or earliest, end or str(today))
    candidates = find_gauges(aoi, period=search_period, target_area_km2=target_area, cache=cache)
    if gauge is None:
        chosen, reasons = select_gauge(candidates, min_years=min_years)
    else:
        match = candidates[candidates["site_no"] == str(gauge)]
        if match.empty:
            raise ValueError(f"Gauge {gauge} is not among the candidates.")
        chosen, reasons = match.iloc[0], {str(gauge): "chosen by the user"}
    if start is None or end is None:
        default_start, default_end = default_period(chosen["end_date"], today, earliest)
        start, end = start or default_start, end or default_end

    climate = fetch_climate(aoi, start, end, evap_factor, cache=cache)
    gauge_flow = fetch_gauge_flow(chosen["site_no"], start, end)
    flow = transfer_flow(gauge_flow, chosen["drainage_km2"], target_area, transfer_exponent)
    days = pd.date_range(start, end, freq="D")
    nan = pd.Series(float("nan"), index=days)
    ts = pd.DataFrame({
        "rain": climate["rain"].reindex(days),
        "evap": climate["evap"].reindex(days),
        "flowup": flow.reindex(days) if flow_is == "natural" else nan,
        "flowdown": flow.reindex(days) if flow_is == "observed" else nan,
        "none": 0.0,
    }, index=days)
    ts.index.name = "Date"

    provenance = {
        "climate": dict(climate.attrs),
        "gauge": {k: chosen[k] for k in ("site_no", "station_nm", "drainage_km2", "dist_km", "begin_date", "end_date",
                                         "gagesii_class", "ecoregion", "hcdn_2009") if k in chosen},
        "gauge_flow": dict(gauge_flow.attrs),
        "transfer": {"method": "drainage-area ratio", "exponent": transfer_exponent,
                     "gauge_area_km2": float(chosen["drainage_km2"]), "target_area_km2": float(target_area)},
        "selection_rule": {"min_years": min_years, "gagesii_reference": True, "same_ecoregion": True,
                           "then": "nearest", "target_ecoregion": candidates.attrs.get("target_ecoregion")},
        "flow_is": flow_is,
        "start_storage": "full (startstore_propfull = 1.0)",
    }
    return InputSeries(ts, str(start)[:10], str(end)[:10], flow_is, dict(chosen), candidates, reasons, provenance)


def model_with_flow(build, inputs: InputSeries, natural: pd.Series, title: str | None = None):
    """A ``tedimodel`` of ``build`` that routes ``natural`` flow downstream ("obs" mode)."""
    model = build.to_model(title=title)
    ts = inputs.ts.copy()
    ts["flowup"] = natural.reindex(ts.index)
    ts["flowdown"] = float("nan")
    model.ts = ts
    model.calc_nat_or_obs = "obs"
    model.startstore_propfull = START_FULL
    return model


def natural_flow(build, inputs: InputSeries) -> pd.Series:
    """Flow without the AOI's existing ponds (ML/day).

    With observed input this back-solves it with pytedi's "nat" mode and ``build``'s ponds;
    with natural input it returns the input.
    """
    if inputs.flow_is == "natural":
        return inputs.ts["flowup"].copy()
    model = build.to_model(title=f"{build.aoi.name}: natural flow")
    model.ts = inputs.ts.copy()
    model.calc_nat_or_obs = "nat"
    model.startstore_propfull = START_FULL
    model.run_model()
    natural = model.catch_results["flow_up"].astype("float64").reindex(inputs.ts.index)
    natural.name = "flow_natural_ML"
    return natural
