"""Before/after streamflow for a hypothetical new pond.

Workflow::

    after = add_hypothetical_pond(build, lat, lon, surface_area_m2=...)  # new dam list
    series = build_timeseries(build)                                     # rain, evap, flow
    natural = natural_flow(build, series)                                # flow without existing ponds
    runs = run_before_after(after, series, natural, dam_id="new")
    cmp = compare_scenarios(runs.before, runs.after, dam_id="new")
    plot_comparison(cmp, "comparison.png"); cmp.to_csv("comparison")

Where the flow is reported
--------------------------
pytedi reports flow only at the catchment outlet and leaving each dam flagged for output.
To get the flow in the stream just below the new dam in both cases, the "before" model holds
the same network with the new pond as a pass-through node (no volume, no surface area), so
its outflow equals its inflow. "After" uses the real pond. Both route the same natural flow
downstream with the same existing ponds (``calc_nat_or_obs = "obs"``), start with ponds
full, and differ only in the new pond.

Low flows
---------
- 7-day minimum: the lowest 7-day moving average in each complete climatic year
  (1 April to 31 March, the USGS low-flow convention), labeled by the year it starts in.
- 7Q10: estimated empirically as the 0.1 non-exceedance quantile of the annual 7-day minima
  with Weibull plotting positions (p = rank / (n + 1)), interpolated; it needs at least 9
  years so it is not extrapolated. USGS publishes 7Q10 from a log-Pearson Type III fit, which
  can differ, especially where minima are zero.
- Zero-flow days: days below ``ZERO_FLOW_ML`` (0.001 ML/day, 1 m3/day).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .aoi import EQUAL_AREA, WGS84
from .damlist import to_pytedi_damlist
from .hydro import LINK_COLUMNS, flow_grids, link_ponds
from .timeseries import InputSeries, model_with_flow

ZERO_FLOW_ML = 0.001
EXCEEDANCE = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
SERIES = ("site_before", "site_after", "outlet_before", "outlet_after")


# ---------------------------------------------------------------------------- low flows

def seven_day_minima(flow: pd.Series) -> pd.Series:
    """Lowest 7-day mean flow in each complete climatic year (April-March), by starting year."""
    flow = flow.asfreq("D")
    rolling = flow.rolling(7).mean()
    climatic = (flow.index - pd.DateOffset(months=3)).year  # 1 April -> same year; Jan-Mar -> previous
    out = {}
    for year in np.unique(climatic):
        days = flow.index[climatic == year]
        full = pd.date_range(f"{year}-04-01", f"{year + 1}-03-31")
        if len(days) == len(full) and flow.loc[days].notna().all():
            out[int(year)] = float(rolling.loc[days].min())
    return pd.Series(out, name="seven_day_min", dtype="float64")


def q7_10(minima: pd.Series) -> float:
    """Empirical 7Q10: Weibull 0.1 non-exceedance quantile of annual 7-day minima (n >= 9)."""
    values = np.sort(minima.dropna().to_numpy())
    n = len(values)
    if n < 9:
        return float("nan")
    p = np.arange(1, n + 1) / (n + 1)
    return float(np.interp(0.1, p, values))


# ---------------------------------------------------------------------------- the new pond

def add_hypothetical_pond(build, lat: float, lon: float, surface_area_m2: float | None = None,
                          volume_ml: float | None = None, dam_id: str = "new", name: str = "Hypothetical pond",
                          group_name: str = "none", demfactor: float = 0.0, snap_m: float = 20.0):
    """A copy of ``build`` with a new pond whose dam is at (lat, lon).

    Give ``surface_area_m2`` or ``volume_ml``; the other follows from the build's area-volume
    relation. The dam is snapped to the highest-accumulation DEM cell within ``snap_m``; the
    pond's footprint is drawn as a circle of its area around the dam (only used to tell which
    upstream flow paths enter it). Every pond is relinked, so ponds upstream now drain into
    the new one. ``group_name`` and ``demfactor`` set its demand (mean annual demand as a
    share of capacity; see :func:`run_before_after` for the pattern). The new pond is the
    only one flagged for output.
    """
    if (surface_area_m2 is None) == (volume_ml is None):
        raise ValueError("Give one of surface_area_m2 or volume_ml.")
    a, b = build.settings["vol_a"], build.settings["vol_b"]
    if surface_area_m2 is None:
        surface_area_m2 = (volume_ml / a) ** (1.0 / b)
    else:
        volume_ml = a * surface_area_m2 ** b

    point = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=WGS84).to_crs(EQUAL_AREA).iloc[0]
    footprint = point.buffer(math.sqrt(surface_area_m2 / math.pi), quad_segs=32)
    old = build.candidates.drop(columns=[c for c in LINK_COLUMNS if c != "dam_id"] + ["vol_ML"], errors="ignore")
    if dam_id in set(old["dam_id"]):
        raise ValueError(f"dam_id {dam_id!r} is already used.")
    new_row = gpd.GeoDataFrame({"dam_id": [dam_id], "source": ["hypothetical"], "name": [name]},
                               geometry=[footprint], crs=EQUAL_AREA)
    candidates = gpd.GeoDataFrame(pd.concat([old, new_row], ignore_index=True), crs=EQUAL_AREA)
    for flag in ("in_nhd", "in_naip", "in_sentinel2", "area_from_default"):
        if flag in candidates:
            candidates[flag] = candidates[flag].fillna(False).astype(bool)
    links = link_ponds(candidates[["dam_id", "geometry"]], flow_grids(build.dem),
                       fixed_outlets={dam_id: (point.x, point.y)}, snap_m=snap_m)
    candidates = gpd.GeoDataFrame(candidates.merge(links, on="dam_id", how="left"), crs=EQUAL_AREA)
    candidates["surfacearea_m2"] = candidates["surfacearea_m2"].astype("float64")
    candidates.loc[candidates["dam_id"] == dam_id, "surfacearea_m2"] = surface_area_m2
    candidates["vol_ML"] = pd.concat([build.candidates.set_index("dam_id")["vol_ML"],
                                      pd.Series({dam_id: volume_ml})]).reindex(candidates["dam_id"]).to_numpy()

    damlist = to_pytedi_damlist(candidates)
    damlist["output"] = (damlist["dam_id"] == dam_id).astype("int32")
    is_new = damlist["dam_id"] == dam_id
    damlist.loc[is_new, "group_name"] = group_name
    damlist.loc[is_new, "demfactor"] = float(demfactor)
    settings = dict(build.settings)
    settings["hypothetical"] = dict(dam_id=dam_id, name=name, lat=lat, lon=lon, surface_area_m2=surface_area_m2,
                                    volume_ml=volume_ml, group_name=group_name, demfactor=demfactor, snap_m=snap_m)
    return replace(build, damlist=damlist, candidates=candidates, settings=settings)


def pass_through(build, dam_id: str):
    """A copy of ``build`` where ``dam_id`` holds nothing: no volume, no surface area, no demand."""
    damlist = build.damlist.copy()
    row = damlist["dam_id"] == dam_id
    if not row.any():
        raise ValueError(f"No dam {dam_id!r} in the dam list.")
    damlist.loc[row, ["vol_ML", "surfacearea_m2", "demfactor"]] = 0.0
    return replace(build, damlist=damlist)


# ---------------------------------------------------------------------------- runs

@dataclass
class ScenarioRuns:
    before: object
    after: object


def run_before_after(after_build, inputs: InputSeries, natural: pd.Series, dam_id: str = "new") -> ScenarioRuns:
    """Run the before (pass-through) and after models with the same natural flow.

    A demand group other than "none" gets a constant daily pattern of 1/365.25 in ``ts``, so
    the pond's ``demfactor`` is its mean annual demand as a share of capacity.
    """
    models = []
    for label, build in (("before", pass_through(after_build, dam_id)), ("after", after_build)):
        model = model_with_flow(build, inputs, natural, title=f"{after_build.aoi.name}: {label} new pond")
        for group in set(build.damlist["group_name"].dropna()) - {"none"} - set(model.ts.columns):
            model.ts[group] = 1.0 / 365.25
        model.run_model()
        models.append(model)
    return ScenarioRuns(*models)


# ---------------------------------------------------------------------------- comparison

@dataclass
class Comparison:
    """Daily flows, monthly means, flow duration, low flows and a summary (ML/day unless noted)."""

    daily: pd.DataFrame
    monthly: pd.DataFrame
    fdc: pd.DataFrame
    low_flow: pd.DataFrame
    summary: dict

    def to_csv(self, prefix: str | os.PathLike) -> list[Path]:
        prefix = Path(prefix)
        paths = []
        for name in ("daily", "monthly", "fdc", "low_flow"):
            path = prefix.with_name(f"{prefix.name}_{name}.csv")
            getattr(self, name).to_csv(path)
            paths.append(path)
        path = prefix.with_name(f"{prefix.name}_summary.csv")
        rows = [(loc, k, v) for loc in ("site", "outlet") for k, v in self.summary[loc].items()]
        pd.DataFrame(rows, columns=["location", "statistic", "value"]).to_csv(path, index=False)
        paths.append(path)
        return paths


def _dam_outflow(model, dam_id: str) -> pd.Series:
    ind = model.indiv_results
    rows = ind[ind["dam_id"].astype(str) == str(dam_id)]
    if rows.empty:
        raise ValueError(f"No individual results for dam {dam_id!r}; is it flagged for output?")
    return rows.set_index(pd.DatetimeIndex(rows["date"]))["outflow"].astype("float64")


def _location_summary(before: pd.Series, after: pd.Series, years: float) -> dict:
    out = {}
    for tag, q in (("before", before), ("after", after)):
        minima = seven_day_minima(q)
        out[f"mean_flow_{tag}_ML_per_day"] = float(q.mean())
        out[f"median_flow_{tag}_ML_per_day"] = float(q.median())
        out[f"zero_flow_days_{tag}"] = int((q < ZERO_FLOW_ML).sum())
        out[f"zero_flow_percent_{tag}"] = float(100.0 * (q < ZERO_FLOW_ML).mean())
        out[f"mean_7day_min_{tag}_ML_per_day"] = float(minima.mean()) if len(minima) else float("nan")
        out[f"q7_10_{tag}_ML_per_day"] = q7_10(minima)
    out["mean_flow_change_percent"] = 100.0 * (out["mean_flow_after_ML_per_day"] / out["mean_flow_before_ML_per_day"] - 1.0)
    out["mean_annual_captured_ML"] = float((before - after).sum() / years)
    return out


def compare_scenarios(before_model, after_model, dam_id: str = "new", label: str = "") -> Comparison:
    """Flow just below the new dam and at the catchment outlet, before and after (see module notes)."""
    daily = pd.DataFrame({
        "site_before": _dam_outflow(before_model, dam_id),
        "site_after": _dam_outflow(after_model, dam_id),
        "outlet_before": before_model.catch_results["flow_down"].astype("float64"),
        "outlet_after": after_model.catch_results["flow_down"].astype("float64"),
    })
    daily.index = pd.DatetimeIndex(daily.index)
    daily.index.name = "Date"
    years = len(daily) / 365.25
    monthly = daily.groupby(daily.index.month).mean()
    monthly.index.name = "month"
    fdc = pd.DataFrame({c: [daily[c].quantile(1 - p / 100.0) for p in EXCEEDANCE] for c in SERIES},
                       index=pd.Index(EXCEEDANCE, name="exceedance_percent"))
    low_flow = pd.DataFrame({c: seven_day_minima(daily[c]) for c in SERIES})
    low_flow.index.name = "climatic_year_starting_april"
    summary = {
        "label": label,
        "period": (str(daily.index[0].date()), str(daily.index[-1].date())),
        "days": len(daily),
        "site": _location_summary(daily["site_before"], daily["site_after"], years),
        "outlet": _location_summary(daily["outlet_before"], daily["outlet_after"], years),
        "units": "ML/day unless noted; 1 ML = 1,000 m3",
        "zero_flow_threshold_ML_per_day": ZERO_FLOW_ML,
        "q7_10_method": "empirical Weibull 0.1 quantile of April-March annual 7-day minima (not log-Pearson III)",
    }
    return Comparison(daily, monthly, fdc, low_flow, summary)


def plot_comparison(cmp: Comparison, path: str | os.PathLike | None = None, title: str | None = None):
    """Flow duration curves and monthly mean flows, before and after, below the dam and at the outlet."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"before": "#3987e5", "after": "#d95926"}
    for col, loc, name in ((0, "site", "Just below the new dam"), (1, "outlet", "Watershed outlet")):
        ax = axes[0, col]
        for tag in ("before", "after"):
            q = np.sort(cmp.daily[f"{loc}_{tag}"].to_numpy())[::-1]
            exceed = 100.0 * np.arange(1, len(q) + 1) / (len(q) + 1)
            ax.plot(exceed, np.maximum(q, ZERO_FLOW_ML), color=colors[tag], lw=1.4, label=tag.capitalize())
        ax.set_yscale("log")
        ax.set_xlabel("Percent of days flow is equaled or exceeded")
        ax.set_ylabel("Daily flow (ML/day, log scale)")
        ax.set_title(f"{name}: flow duration curve")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax = axes[1, col]
        months = np.arange(1, 13)
        width = 0.4
        ax.bar(months - width / 2, cmp.monthly[f"{loc}_before"], width, color=colors["before"], label="Before")
        ax.bar(months + width / 2, cmp.monthly[f"{loc}_after"], width, color=colors["after"], label="After")
        ax.set_xticks(months, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("Mean daily flow (ML/day)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name}: mean flow by month")
        ax.legend()
    s = cmp.summary
    fig.suptitle(title or f"{s['label']}  {s['period'][0]} to {s['period'][1]}", fontsize=12)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
