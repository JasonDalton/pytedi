"""US data helpers for pytedi.

Build a pytedi dam list for an area of interest in the United States from public data:

- 3DEP terrain (USGS 3D Elevation Program) for flow routing and catchment areas,
- NHD hydrography (USGS National Hydrography Dataset) for mapped ponds and streams,
- the National Inventory of Dams (USACE) for official dams, and
- Sentinel-2 imagery for small farm ponds that the maps miss.

This subpackage is optional. Install its dependencies with::

    pip install "pytedi[us]"

The pytedi core never imports it.
"""

try:
    from .aoi import AOI, as_aoi, huc12_at, huc_geometry
    from .climate import fetch_climate
    from .gauges import fetch_gauge_flow, find_gauges, select_gauge, transfer_flow
    from .scenario import (add_hypothetical_pond, compare_scenarios, plot_comparison, q7_10, run_before_after,
                           seven_day_minima)
    from .timeseries import InputSeries, build_timeseries, model_with_flow, natural_flow
    from .damlist import DamListResult, build_damlist, to_pytedi_damlist
    from .fetch import NHDData, cache_dir, fetch_3dep, fetch_nhd, fetch_official_dams, nid_inventory_path
    from .hydro import FlowGrids, flow_grids, link_ponds
    from .merge import merge_candidates
    from .plot import plot_candidates
    from .naip import detect_ponds_naip
    from .ponds import detect_ponds_sentinel2, ndwi
    from .volume import volume_ml_from_area
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'pytedi.us needs optional dependencies. Install them with: pip install "pytedi[us]"'
    ) from exc

__all__ = [
    "fetch_climate",
    "find_gauges",
    "select_gauge",
    "fetch_gauge_flow",
    "transfer_flow",
    "build_timeseries",
    "InputSeries",
    "natural_flow",
    "model_with_flow",
    "add_hypothetical_pond",
    "run_before_after",
    "compare_scenarios",
    "plot_comparison",
    "seven_day_minima",
    "q7_10",
    "build_damlist",
    "DamListResult",
    "to_pytedi_damlist",
    "plot_candidates",
    "AOI",
    "as_aoi",
    "huc12_at",
    "huc_geometry",
    "volume_ml_from_area",
    "NHDData",
    "cache_dir",
    "fetch_3dep",
    "fetch_nhd",
    "fetch_official_dams",
    "nid_inventory_path",
    "detect_ponds_sentinel2",
    "detect_ponds_naip",
    "ndwi",
    "FlowGrids",
    "flow_grids",
    "link_ponds",
    "merge_candidates",
]
