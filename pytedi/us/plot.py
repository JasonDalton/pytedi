"""A static map of a :class:`~pytedi.us.damlist.DamListResult` for checking pond detection."""

from __future__ import annotations

import os

import numpy as np

SOURCE_STYLE = {  # source -> (legend label, fill color)
    "nhd": ("NHD only", "#3987e5"),
    "both": ("NHD and Sentinel-2", "#199e70"),
    "sentinel2": ("Sentinel-2 only", "#d95926"),
    "nid": ("NID only (footprint from NID area)", "#8b6fd6"),
}


def plot_candidates(result, path: str | os.PathLike | None = None, title: str | None = None, dpi: int = 200,
                    figsize=(9, 12)):
    """Hillshaded DEM, NHD flowlines, ponds colored by source, outlets, links and NID dams.

    Returns the matplotlib Figure; also saves it as a PNG when ``path`` is given.
    """
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
    for source, (label, color) in SOURCE_STYLE.items():
        part = c[c["source"] == source]
        if part.empty:
            continue
        part.plot(ax=ax, facecolor=color, edgecolor=color, linewidth=1.2, alpha=0.9)
        handles.append(Patch(facecolor=color, edgecolor=color, label=f"{label} ({len(part)})"))

    if not c.empty:
        by_id = c.set_index("dam_id")
        for _, row in c.iterrows():
            if row["ds_damid"] != "-99" and row["ds_damid"] in by_id.index:
                target = by_id.loc[row["ds_damid"]]
                ax.annotate("", xy=(target["outlet_x"], target["outlet_y"]), xytext=(row["outlet_x"], row["outlet_y"]),
                            arrowprops=dict(arrowstyle="->", color="#c9a227", lw=0.6, alpha=0.8))
        handles.append(Line2D([], [], color="#c9a227", lw=1, label="flows to next pond"))

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
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi)
    return fig
