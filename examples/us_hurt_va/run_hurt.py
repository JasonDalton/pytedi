"""Build a pytedi dam list for the HUC12 watershed around Hurt, Virginia.

Hurt is in Pittsylvania County, Virginia, on the Staunton (Roanoke) River. This script:

1. looks up the HUC12 that contains Hurt in the USGS Watershed Boundary Dataset;
2. fetches 3DEP terrain, NHD hydrography, National Inventory of Dams records and
   Sentinel-2 imagery for it, and detects ponds;
3. links the ponds on the DEM and writes the dam list, a GeoPackage and a map to
   ``output/`` (git-ignored), then prints a summary.

Run from the repository root after ``pip install -e ".[us]"``::

    python examples/us_hurt_va/run_hurt.py
"""

import argparse
import json
from pathlib import Path

from pytedi.us import build_damlist, huc12_at, plot_candidates

HURT_LAT, HURT_LON = 37.09, -79.30
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "output", help="output folder (default: %(default)s)")
    parser.add_argument("--start", default="2024-06-01", help="Sentinel-2 start date")
    parser.add_argument("--end", default="2024-09-30", help="Sentinel-2 end date")
    parser.add_argument("--threshold", type=float, default=0.0, help="NDWI water threshold")
    parser.add_argument("--min-area", type=float, default=500.0, help="smallest detected pond, m2")
    parser.add_argument("--volume-relation", default="fowler2015", choices=["fowler2015", "va_nid_farm"],
                        help="area-volume relation (default: %(default)s)")
    parser.add_argument("--no-sentinel2", action="store_true", help="use NHD and NID only")
    parser.add_argument("--map", type=Path, default=None, help="map PNG path (default: OUT/hurt_va_map.png)")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    huc = huc12_at(HURT_LAT, HURT_LON)
    print(f"HUC12 containing Hurt, VA ({HURT_LAT}, {HURT_LON}): {huc}")

    result = build_damlist(
        huc,
        use_sentinel2=not args.no_sentinel2,
        date_range=(args.start, args.end),
        ndwi_threshold=args.threshold,
        min_area_m2=args.min_area,
        volume_relation=args.volume_relation,
        out_gpkg=out / "hurt_va_candidates.gpkg",
        cache=out / "cache",
    )
    result.damlist.to_csv(out / "hurt_va_damlist.csv", index=False)
    map_path = args.map or out / "hurt_va_map.png"
    plot_candidates(result, map_path, title=f"Hurt, VA: HUC12 {huc}, candidate farm ponds")

    summary = result.summary()
    if result.s2_ponds is not None:
        summary["sentinel2_scenes"] = result.s2_ponds.attrs.get("scenes")
        summary["sentinel2_polygons_before_merge"] = len(result.s2_ponds)
    summary["nid_dams"] = result.nid_dams[["nid_id", "name", "storage_ML", "surfacearea_m2", "year_completed"]] \
        .astype(object).where(result.nid_dams.notna(), None).to_dict("records")
    summary["settings"] = {k: v for k, v in result.settings.items()}
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out / 'hurt_va_damlist.csv'}, {out / 'hurt_va_candidates.gpkg'} and {map_path}")


if __name__ == "__main__":
    main()
