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
    from .damlist import DamListResult, build_damlist, to_pytedi_damlist
    from .fetch import NHDData, cache_dir, fetch_3dep, fetch_nhd, fetch_official_dams, nid_inventory_path
    from .hydro import FlowGrids, flow_grids, link_ponds
    from .merge import merge_candidates
    from .plot import plot_candidates
    from .ponds import detect_ponds_sentinel2, ndwi
    from .volume import volume_ml_from_area
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'pytedi.us needs optional dependencies. Install them with: pip install "pytedi[us]"'
    ) from exc

__all__ = [
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
    "ndwi",
    "FlowGrids",
    "flow_grids",
    "link_ponds",
    "merge_candidates",
]
