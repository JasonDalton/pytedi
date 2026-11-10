"""A static map of a :class:`~pytedi.us.damlist.DamListResult` for checking pond detection."""

from __future__ import annotations

import os
import warnings

import numpy as np

_DEFAULT = object()  # marks "use the default basemap" (USGS US Topo)


def _default_basemap():
    import xyzservices.providers as xyz

    return xyz.USGS.USTopo


def _attribution(provider) -> str:
    credit = provider.get("attribution") or provider.name
    if provider.name.startswith("USGS."):
        return f"Basemap: USGS The National Map ({provider.name.split('.', 1)[1]}). {credit}"
    return f"Basemap: {credit}"


def _add_basemap(ax, source, crs, zoom):
    """Draw web tiles under the current axis limits (contextily); raises if they can't be fetched."""
    import contextily

    contextily.add_basemap(ax, source=source, crs=crs, zoom=zoom, attribution=False, zorder=0)

CATEGORIES = (  # (legend label, fill color, test on the source label)
    ("NHD, not seen in imagery", "#3987e5", lambda s: s == "nhd"),
    ("NHD, confirmed by imagery", "#199e70", lambda s: s.startswith("nhd+")),
    ("New: NAIP (1 m)", "#d95926", lambda s: s.startswith("naip")),
    ("New: Sentinel-2 only", "#c9a227", lambda s: s == "sentinel2"),
    ("NID only (footprint from NID area)", "#8b6fd6", lambda s: s == "nid"),
)


def plot_candidates(result, path: str | os.PathLike | None = None, title: str | None = None, dpi: int = 200,
                    figsize=(9, 12), basemap=_DEFAULT, basemap_zoom="auto"):
    """Map of the candidate ponds: a reference basemap (or hillshaded DEM), NHD flowlines,
    ponds colored by what found them, pond-to-pond links and NID dams.

    Parameters
    ----------
    basemap : xyzservices.TileProvider or None
        Web tiles drawn underneath so readers can find roads, towns and creeks. Defaults to
        USGS The National Map US Topo (``xyzservices.providers.USGS.USTopo``; no API key).
        None draws a hillshade of the DEM instead. If the tiles can't be fetched (offline),
        this warns and falls back to the hillshade.
    basemap_zoom : int or "auto"
        Tile zoom level (14 suits a HUC12).

    Returns the matplotlib Figure; also saves it as a PNG when ``path`` is given.
    """
    if basemap is _DEFAULT:
        basemap = _default_basemap()
    if basemap is not None and not hasattr(basemap, "build_url"):
        raise TypeError("basemap must be an xyzservices TileProvider (e.g. xyzservices.providers.USGS.USTopo) or None.")
    import matplotlib

    if path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dem = result.dem
    z = np.asarray(dem.values, dtype=float)
    x0, x1 = float(dem.x.min()), float(dem.x.max())
    y0, y1 = float(dem.y.min()), float(dem.y.max())
    cell = abs(float(dem.x[1] - dem.x[0])) if dem.sizes["x"] > 1 else 1.0
    extent = (x0 - cell / 2, x1 + cell / 2, y0 - cell / 2, y1 + cell / 2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    drew_basemap = False
    if basemap is not None:
        try:
            _add_basemap(ax, basemap, result.candidates.crs.to_string(), basemap_zoom)
            drew_basemap = True
        except Exception as exc:  # offline, tile server down, bad zoom: the map is still useful without it
            warnings.warn(f"The {basemap.name} basemap could not be drawn ({type(exc).__name__}: {exc}); "
                          "showing a hillshade instead.", UserWarning, stacklevel=2)
    if not drew_basemap:
        filled = np.where(np.isfinite(z), z, np.nanmin(z) if np.isfinite(z).any() else 0.0)
        shade = LightSource(azdeg=315, altdeg=45).hillshade(filled, vert_exag=3, dx=cell, dy=cell)
        shade = np.where(np.isfinite(z), shade, np.nan)
        ax.imshow(shade, cmap="gray", extent=extent, origin="upper", alpha=0.85, interpolation="nearest")

    aoi = result.aoi.to_crs(result.candidates.crs.to_string()).to_gdf()
    aoi.boundary.plot(ax=ax, color="black", linewidth=1.2)
    if not result.nhd.flowlines.empty:
        result.nhd.flowlines.plot(ax=ax, color="#5aa9e6", linewidth=0.7)

    c = result.candidates
    handles = [Line2D([], [], color="#5aa9e6", lw=1, label="NHD flowline"),
               Line2D([], [], color="black", lw=1.2, label="AOI boundary")]
    for label, color, test in CATEGORIES:
        part = c[c["source"].map(test)] if not c.empty else c
        if part.empty:
            continue
        part.plot(ax=ax, facecolor=color, edgecolor="#222222", linewidth=0.25, alpha=0.95)
        # Most farm ponds are a few pixels at watershed scale: mark each one so none disappear.
        part.geometry.representative_point().plot(ax=ax, color=color, markersize=14, edgecolor="#222222",
                                                  linewidth=0.4, zorder=4)
        handles.append(Patch(facecolor=color, edgecolor="black", label=f"{label} ({len(part)})"))

    if not c.empty:
        by_id = c.set_index("dam_id")
        for _, row in c.iterrows():
            if row["ds_damid"] != "-99" and row["ds_damid"] in by_id.index:
                target = by_id.loc[row["ds_damid"]]
                ax.annotate("", xy=(target["outlet_x"], target["outlet_y"]), xytext=(row["outlet_x"], row["outlet_y"]),
                            arrowprops=dict(arrowstyle="->", color="#444444", lw=0.6, alpha=0.8))
        handles.append(Line2D([], [], color="#444444", lw=1, label="flows to next pond"))

    if not result.nid_dams.empty:
        result.nid_dams.plot(ax=ax, marker="^", color="#d4588b", edgecolor="black", markersize=60, zorder=5)
        handles.append(Line2D([], [], marker="^", color="w", markerfacecolor="#d4588b", markeredgecolor="black",
                              markersize=9, label=f"NID dam ({len(result.nid_dams)})"))

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Easting, EPSG:5070 (m)")
    ax.set_ylabel("Northing, EPSG:5070 (m)")
    ax.ticklabel_format(style="plain")
    ax.set_title(title or result.aoi.name)
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    if drew_basemap:
        ax.text(0.995, 0.995, _attribution(basemap), transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi)
    return fig
