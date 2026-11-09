"""Areas of interest: a bounding box, a polygon (shapely or GeoJSON) or a HUC id."""

from __future__ import annotations

import json
from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry

WGS84 = "EPSG:4326"
EQUAL_AREA = "EPSG:5070"  # NAD83 / Conus Albers, the CRS 3DEP and NHDPlus use for area work
HUC_LEVELS = {8: "huc8", 10: "huc10", 12: "huc12"}


@dataclass(frozen=True)
class AOI:
    """An area of interest: one polygon and its CRS.

    Attributes
    ----------
    geometry : shapely Polygon or MultiPolygon
    crs : str
        Any CRS string pyproj understands. Defaults to WGS84 longitude/latitude.
    name : str
        A label for reports and plots, e.g. "HUC12 030101011304".
    """

    geometry: BaseGeometry
    crs: str = WGS84
    name: str = "area of interest"

    def __post_init__(self):
        if self.geometry is None or self.geometry.is_empty:
            raise ValueError("The area of interest is empty.")
        if self.geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"The area of interest must be a polygon, not a {self.geometry.geom_type}.")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) in the AOI's CRS."""
        return tuple(self.geometry.bounds)

    def to_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"name": [self.name]}, geometry=[self.geometry], crs=self.crs)

    def to_crs(self, crs: str) -> "AOI":
        geom = self.to_gdf().to_crs(crs).geometry.iloc[0]
        return AOI(geom, crs, self.name)

    @property
    def area_km2(self) -> float:
        return float(self.to_gdf().to_crs(EQUAL_AREA).area.iloc[0]) / 1e6


def as_aoi(obj, name: str | None = None) -> AOI:
    """Turn a bbox, polygon, GeoJSON or HUC id into an :class:`AOI` (WGS84 unless an AOI is given).

    Parameters
    ----------
    obj :
        - ``AOI``: returned unchanged;
        - ``(west, south, east, north)`` in longitude/latitude;
        - a shapely Polygon or MultiPolygon in longitude/latitude;
        - a GeoJSON geometry or Feature, as a dict or a string;
        - a HUC8, HUC10 or HUC12 id as a string of digits (looked up in the USGS
          Watershed Boundary Dataset; needs internet access).
    name : str, optional
        Label for the AOI.
    """
    if isinstance(obj, AOI):
        return obj
    if isinstance(obj, str):
        text = obj.strip()
        if text.startswith("{"):
            return as_aoi(json.loads(text), name)
        if text.isdigit() and len(text) in HUC_LEVELS:
            level = HUC_LEVELS[len(text)].upper()
            return AOI(huc_geometry(text), WGS84, name or f"{level} {text}")
        raise ValueError(f"Not a HUC8/10/12 id or GeoJSON: {obj!r}")
    if isinstance(obj, dict):
        geom = obj.get("geometry", obj) if obj.get("type") == "Feature" else obj
        return AOI(shape(geom), WGS84, name or "area of interest")
    if isinstance(obj, BaseGeometry):
        return AOI(obj, WGS84, name or "area of interest")
    if isinstance(obj, (tuple, list)) and len(obj) == 4:
        west, south, east, north = (float(v) for v in obj)
        if not (west < east and south < north):
            raise ValueError("A bounding box must be (west, south, east, north) with west < east and south < north.")
        return AOI(box(west, south, east, north), WGS84, name or "bounding box")
    raise TypeError(f"Cannot make an area of interest from {type(obj).__name__}.")


def huc_geometry(huc_id: str) -> BaseGeometry:
    """Boundary of a HUC8, HUC10 or HUC12 from the USGS Watershed Boundary Dataset (WGS84)."""
    from pygeohydro import WBD

    level = HUC_LEVELS.get(len(huc_id))
    if level is None or not huc_id.isdigit():
        raise ValueError(f"Not a HUC8/10/12 id: {huc_id!r}")
    hucs = WBD(level).byids(level, [huc_id]).to_crs(WGS84)
    if hucs.empty:
        raise ValueError(f"{level.upper()} {huc_id} was not found in the Watershed Boundary Dataset.")
    return hucs.geometry.union_all()


def huc12_at(lat: float, lon: float) -> str:
    """Id of the HUC12 that contains a point (USGS Watershed Boundary Dataset)."""
    from pygeohydro import WBD

    hucs = WBD("huc12").bygeom(Point(lon, lat), geo_crs=WGS84)
    if hucs.empty:
        raise ValueError(f"No HUC12 contains {lat}, {lon}.")
    return str(hucs["huc12"].iloc[0])
