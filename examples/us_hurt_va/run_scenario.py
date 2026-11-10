"""Before/after streamflow for a PLACEHOLDER new farm pond near Hurt, Virginia.

The site and size here are placeholders that exercise the method; they are not a proposed
project. The placeholder dam sits on an intermittent stream (NHD FCode 46003): the midpoint
of the longest intermittent NHD flowline in the HUC12 whose midpoint is at least 200 m from
every existing pond. The placeholder pond is 0.25 ha (2,500 m2, close to the median pond at
Hurt) with no water use.

Steps: build the dam list (cached after the first run of run_hurt.py), daily rain and
evaporation (gridMET) and flow (a reference gauge, drainage-area transfer) from 1980, the
natural flow without the existing ponds, then before and after runs. Writes CSVs, a JSON
summary and a figure to ``output/scenario_placeholder/``.

    python examples/us_hurt_va/run_scenario.py
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from pytedi.us import (add_hypothetical_pond, build_damlist, build_timeseries, compare_scenarios, natural_flow,
                       plot_comparison, run_before_after)

HUC12 = "030101011304"
HERE = Path(__file__).resolve().parent
LABEL = "PLACEHOLDER pond (not a proposed project)"


def placeholder_site(build, min_distance_m=200.0):
    """Midpoint of the longest intermittent (FCode 46003) flowline far enough from existing ponds."""
    lines = build.nhd.flowlines
    fcode = lines["fcode"].astype(float)
    lines = lines[fcode == 46003].copy()
    lines["length_m"] = lines.length
    ponds = build.candidates.geometry.union_all()
    for _, line in lines.sort_values("length_m", ascending=False).iterrows():
        mid = line.geometry.interpolate(0.5, normalized=True)
        if mid.distance(ponds) >= min_distance_m:
            lonlat = gpd.GeoSeries([mid], crs=lines.crs).to_crs(4326).iloc[0]
            return lonlat.y, lonlat.x, line
    raise ValueError("No intermittent flowline is far enough from the existing ponds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "output" / "scenario_placeholder")
    parser.add_argument("--cache", type=Path, default=HERE / "output" / "cache")
    parser.add_argument("--area-m2", type=float, default=2500.0, help="placeholder pond surface area")
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    build = build_damlist(HUC12, cache=args.cache)
    lat, lon, line = placeholder_site(build)
    print(f"Placeholder dam: {lat:.6f}, {lon:.6f} on NHD flowline {line.get('permanent_identifier')} "
          f"(FCode {int(float(line['fcode']))}, {line.geometry.length:.0f} m)")

    series = build_timeseries(build, cache=args.cache)
    print(f"Period {series.start} to {series.end}; gauge {series.gauge['site_no']} {series.gauge['station_nm']}")
    cols = ["site_no", "station_nm", "drainage_km2", "area_ratio", "dist_km", "begin_date", "end_date",
            "years_in_period", "gagesii_class", "ecoregion", "hcdn_2009"]
    table = series.candidates[cols].copy()
    table["decision"] = table["site_no"].map(series.reasons)
    table.to_csv(args.out / "gauge_candidates.csv", index=False)

    natural = natural_flow(build, series)
    after = add_hypothetical_pond(build, lat=lat, lon=lon, surface_area_m2=args.area_m2, name=LABEL)
    runs = run_before_after(after, series, natural, dam_id="new")
    cmp = compare_scenarios(runs.before, runs.after, dam_id="new", label=LABEL)
    cmp.to_csv(args.out / "comparison")
    figure = args.figure or args.out / "comparison.png"
    plot_comparison(cmp, figure, title=f"Hurt, VA (HUC12 {HUC12}): {LABEL}\n"
                                       f"{args.area_m2:,.0f} m2 pond on an intermittent stream, "
                                       f"{cmp.summary['period'][0]} to {cmp.summary['period'][1]}")

    ts = series.ts
    years = len(ts) / 365.25
    new = after.candidates.set_index("dam_id").loc["new"]
    baseline = {
        "period": [series.start, series.end],
        "mean_annual_rain_mm": round(float(ts["rain"].sum() / years), 1),
        "mean_annual_evap_mm": round(float(ts["evap"].sum() / years), 1),
        "outlet_area_km2": round(build.cat_area_km2, 2),
        "mean_annual_outlet_flow_ML": round(float(series.flow.sum() / years), 1),
        "mean_annual_outlet_flow_mm": round(float(series.flow.sum() / years / build.cat_area_km2), 1),
        "mean_annual_natural_flow_ML": round(float(natural.sum() / years), 1),
        "existing_ponds": int(len(build.damlist)),
        "existing_pond_volume_ML": round(float(build.damlist["vol_ML"].sum()), 1),
    }
    placeholder = {
        "label": LABEL, "lat": lat, "lon": lon,
        "surface_area_m2": float(new["surfacearea_m2"]), "volume_ML": round(float(new["vol_ML"]), 2),
        "catchuptotal_km2": round(float(new["catchuptotal_km2"]), 4),
        "catchuplocal_km2": round(float(new["catchuplocal_km2"]), 4),
        "drains_to": new["ds_damid"],
        "upstream_ponds": sorted(after.candidates.loc[after.candidates["ds_damid"] == "new", "dam_id"].tolist()),
    }
    out = {"baseline": baseline, "placeholder_pond": placeholder, "comparison": cmp.summary,
           "provenance": series.provenance, "gauge_decisions": series.reasons}
    (args.out / "summary.json").write_text(json.dumps(out, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    print(json.dumps({"baseline": baseline, "placeholder_pond": placeholder,
                      "site": cmp.summary["site"], "outlet": cmp.summary["outlet"]}, indent=2, default=str))
    print(table.round(2).to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
