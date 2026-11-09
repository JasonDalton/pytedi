"""Flow routing and pond linking on a synthetic valley.

The DEM is a V-shaped valley 40 rows by 21 columns of 10 m cells, draining south down
column 10 and off the bottom edge. Hillslope cells flow diagonally toward the channel, so a
cell at (row r, column c) joins the channel at row r + |c - 10|. That gives an independent
count of each channel cell's upstream area (away from the bottom edge).
"""
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from pytedi.us import hydro

ROWS, COLS, CELL = 40, 21, 10.0
X0, Y0 = 1_000_000.0, 1_600_000.0


def valley_dem(nodata_rows=0):
    r, c = np.mgrid[0:ROWS, 0:COLS]
    z = 200.0 + (ROWS - r) * 1.0 + np.abs(c - 10) * 2.0
    z = z.astype("float32")
    if nodata_rows:
        z[:nodata_rows, :] = np.nan
    dem = xr.DataArray(z, dims=("y", "x"),
                       coords={"y": Y0 - CELL / 2 - CELL * np.arange(ROWS), "x": X0 + CELL / 2 + CELL * np.arange(COLS)})
    return dem.rio.write_crs("EPSG:5070")


def cells(r0, r1, c0, c1):
    """Polygon covering rows r0..r1 and columns c0..c1 (inclusive)."""
    return box(X0 + c0 * CELL, Y0 - (r1 + 1) * CELL, X0 + (c1 + 1) * CELL, Y0 - r0 * CELL)


def analytic_area_km2(row):
    """Upstream area of channel cell (row, 10): cells with r + |c - 10| <= row."""
    r, c = np.mgrid[0:ROWS, 0:COLS]
    return np.count_nonzero(r + np.abs(c - 10) <= row) * CELL * CELL / 1e6


@pytest.fixture
def ponds():
    return gpd.GeoDataFrame(
        {"dam_id": ["A", "B", "C", "D"]},
        geometry=[cells(5, 7, 9, 11),     # on the channel, upstream
                  cells(25, 27, 9, 11),   # on the channel, downstream
                  cells(15, 16, 2, 3),    # hillslope; reaches the channel at row ~23, above B
                  cells(33, 34, 2, 3)],   # hillslope; leaves the grid before reaching the channel
        crs="EPSG:5070")


def test_flow_grids_route_the_valley_down_its_channel():
    g = hydro.flow_grids(valley_dem())
    assert g.cell_area_m2 == pytest.approx(100.0)
    acc = g.acc
    # The channel accumulates. D8 cannot leave the grid, so the bottom row flows sideways
    # into the channel and the channel's last cell drains every cell.
    assert acc[39, 10] == ROWS * COLS
    assert acc[20, 10] == pytest.approx(analytic_area_km2(20) * 1e6 / 100.0, rel=0.02)
    assert acc[39, 10] > acc[20, 10] > acc[5, 10]


def test_link_ponds_finds_the_next_pond_downstream(ponds):
    links = hydro.link_ponds(ponds, hydro.flow_grids(valley_dem())).set_index("dam_id")
    assert links.loc["A", "ds_damid"] == "B"
    assert links.loc["C", "ds_damid"] == "B"
    assert links.loc["B", "ds_damid"] == "-99"
    assert links.loc["D", "ds_damid"] == "-99"
    assert links["outlet_found"].all()


def test_total_and_local_areas(ponds):
    links = hydro.link_ponds(ponds, hydro.flow_grids(valley_dem())).set_index("dam_id")
    tot, loc = links["catchuptotal_km2"], links["catchuplocal_km2"]
    assert tot["A"] == pytest.approx(analytic_area_km2(7), rel=0.10)
    assert tot["B"] == pytest.approx(analytic_area_km2(27), rel=0.10)
    # Ponds with nothing upstream: local equals total.
    for d in ("A", "C", "D"):
        assert loc[d] == pytest.approx(tot[d])
    # B's local area excludes what A and C capture, and stays positive.
    assert loc["B"] == pytest.approx(tot["B"] - tot["A"] - tot["C"])
    assert 0 < loc["B"] < tot["B"]
    # A hillslope pond's catchment is small but at least its own footprint.
    assert ponds.set_index("dam_id").area["C"] / 1e6 <= tot["C"] < 0.01


def test_outlet_points_lie_in_their_pond(ponds):
    links = hydro.link_ponds(ponds, hydro.flow_grids(valley_dem())).set_index("dam_id")
    geoms = ponds.set_index("dam_id").geometry
    for d in "ABCD":
        x, y = links.loc[d, ["outlet_x", "outlet_y"]]
        assert geoms[d].buffer(CELL).contains(gpd.points_from_xy([x], [y])[0])


def test_pond_over_missing_dem_keeps_its_own_area():
    dem = valley_dem(nodata_rows=10)
    ponds = gpd.GeoDataFrame({"dam_id": ["A", "B"]}, geometry=[cells(2, 4, 9, 11), cells(25, 27, 9, 11)], crs="EPSG:5070")
    links = hydro.link_ponds(ponds, hydro.flow_grids(dem)).set_index("dam_id")
    assert not links.loc["A", "outlet_found"] and links.loc["B", "outlet_found"]
    assert links.loc["A", "ds_damid"] == "-99"
    assert links.loc["A", "catchuptotal_km2"] == pytest.approx(0.0009)  # its own 9 cells
    assert np.isfinite(links["catchuptotal_km2"]).all()


def test_tiny_pond_smaller_than_a_cell_still_gets_an_outlet():
    ponds = gpd.GeoDataFrame({"dam_id": ["T"]}, geometry=[box(X0 + 102, Y0 - 98, X0 + 106, Y0 - 94)], crs="EPSG:5070")
    links = hydro.link_ponds(ponds, hydro.flow_grids(valley_dem()))
    assert links["outlet_found"].iloc[0]


def test_no_ponds_gives_an_empty_table():
    empty = gpd.GeoDataFrame({"dam_id": []}, geometry=[], crs="EPSG:5070")
    links = hydro.link_ponds(empty, hydro.flow_grids(valley_dem()))
    assert links.empty and {"dam_id", "ds_damid", "catchuptotal_km2", "catchuplocal_km2"} <= set(links.columns)


def test_d8_accumulation_by_hand():
    # Everything drains to the bottom-middle cell; the top-right cell is missing data.
    E, SE, S, SW, W = 1, 2, 4, 8, 16
    fdir = np.array([[SE, S, 0],
                     [SE, S, SW],
                     [E, 0, W]])
    valid = np.array([[True, True, False], [True, True, True], [True, True, True]])
    acc = hydro.d8_accumulation(fdir, valid)
    assert acc.tolist() == [[1, 1, 0],
                            [1, 3, 1],
                            [1, 8, 1]]
