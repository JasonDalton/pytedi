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
    from .volume import volume_ml_from_area
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'pytedi.us needs optional dependencies. Install them with: pip install "pytedi[us]"'
    ) from exc

__all__ = [
    "AOI",
    "as_aoi",
    "huc12_at",
    "huc_geometry",
    "volume_ml_from_area",
]
