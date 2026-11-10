"""NAIP (1 m) pond detection on synthetic imagery."""
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from pytedi.us import naip as ponds
from pytedi.us.aoi import AOI

T = Affine(1.0, 0.0, 1_000_000.0, 0.0, -1.0, 1_600_000.0)  # 1 m cells, EPSG:5070
GRID = AOI(box(1_000_000, 1_599_800, 1_000_200, 1_600_000), "EPSG:5070", "synthetic 200 m")


def synthetic_naip():
    """Pasture (bright NIR) with a compact pond, a shadow strip and a long, wide pond."""
    img = np.zeros((4, 200, 200), dtype="uint8")
    img[0], img[1], img[2], img[3] = 90, 110, 70, 200      # red, green, blue, NIR: vegetation
    def water(rows, cols):
        img[0][rows, cols], img[1][rows, cols], img[2][rows, cols], img[3][rows, cols] = 40, 70, 60, 1
    water(slice(20, 40), slice(20, 40))      # 20 x 20 m pond: kept
    water(slice(60, 64), slice(20, 120))     # 4 m x 100 m dark strip, like a tree shadow: dropped
    water(slice(100, 116), slice(20, 90))    # 16 m x 70 m pond: elongated but wide, kept
    water(slice(150, 160), slice(150, 185))  # 10 m x 35 m: narrow and elongated, dropped
    water(slice(180, 186), slice(20, 26))    # 6 x 6 m: under the 100 m2 minimum
    img[:, 195:, :] = 0                      # missing data along the bottom
    return img


def test_naip_water_mask_uses_ndwi_and_ignores_missing_data():
    img = synthetic_naip()
    mask = ponds.naip_water_mask(img, threshold=0.5)
    assert mask[30, 30] and not mask[5, 5] and not mask[197, 5]


def test_shape_filter_keeps_ponds_and_drops_shadow_strips():
    import geopandas as gpd

    shapes = gpd.GeoDataFrame(geometry=[box(0, 0, 20, 20), box(0, 0, 100, 4), box(0, 0, 70, 16), box(0, 0, 35, 10)],
                              crs="EPSG:5070")
    kept = ponds.shape_filter(shapes, min_width_m=8, max_elongation=3, elongation_free_width_m=12)
    assert list(kept.index) == [0, 2]


def fake_years_and_images(monkeypatch, images):
    """images: {year: array}, all in one state ("va"). Records the plans loaded."""
    loaded = []
    monkeypatch.setattr(ponds, "_naip_years", lambda aoi: {"va": sorted(images)})

    def fake_load(aoi, plan, resolution, cache):
        loaded.append((plan, resolution))
        return images[plan["va"]], T, "EPSG:5070", plan

    monkeypatch.setattr(ponds, "_load_naip", fake_load)
    return loaded


def test_detect_ponds_naip_end_to_end(monkeypatch):
    img = synthetic_naip()
    loaded = fake_years_and_images(monkeypatch, {2021: img, 2023: img})
    found = ponds.detect_ponds_naip(GRID)
    assert loaded == [({"va": 2023}, 1.0), ({"va": 2021}, 1.0)]  # the latest year, checked against the one before
    assert sorted(found["surfacearea_m2"].round()) == [400.0, 1120.0]
    assert (found["source"] == "naip").all() and found.crs.to_epsg() == 5070
    assert found.attrs["year"] == {"va": 2023} and found.attrs["confirm_year"] == {"va": 2021}
    assert found.attrs["threshold"] == 0.5


def test_small_ponds_must_also_be_water_in_the_previous_year(monkeypatch):
    now, before = synthetic_naip(), synthetic_naip()
    # In the earlier year the 20 x 20 m pond (400 m2, under 500 m2) is dry; the big one is still there.
    before[3][20:40, 20:40] = 200
    fake_years_and_images(monkeypatch, {2021: before, 2023: now})
    found = ponds.detect_ponds_naip(GRID)
    assert sorted(found["surfacearea_m2"].round()) == [1120.0]
    assert found.attrs["unconfirmed_dropped"] == 1
    # Without a check year, or with a lower size limit, it stays.
    assert len(ponds.detect_ponds_naip(GRID, confirm_year=None)) == 2
    assert len(ponds.detect_ponds_naip(GRID, confirm_below_m2=300)) == 2


def test_only_one_naip_year_skips_the_check(monkeypatch):
    fake_years_and_images(monkeypatch, {2023: synthetic_naip()})
    found = ponds.detect_ponds_naip(GRID)
    assert len(found) == 2 and found.attrs["confirm_year"] is None


def test_latest_year_per_state_and_previous_year():
    # NAIP is flown state by state: an AOI on a state line mixes years.
    years = {"wv": [2018, 2020, 2022], "va": [2018, 2021, 2023]}
    assert ponds.naip_plan(years) == {"wv": 2022, "va": 2023}
    assert ponds.previous_plan(years, {"wv": 2022, "va": 2023}) == {"wv": 2020, "va": 2021}
    assert ponds.naip_plan(years, 2018) == {"wv": 2018, "va": 2018}
    with pytest.raises(ValueError, match="2022"):
        ponds.naip_plan(years, 2022)  # Virginia has no 2022
    assert ponds.previous_plan(years, {"wv": 2018, "va": 2018}) is None  # nothing earlier to check against


def test_pick_items_by_state_and_year():
    class Item:
        def __init__(self, state, year):
            self.properties = {"naip:state": state, "naip:year": year}

    items = [Item("wv", 2022), Item("va", 2023), Item("va", 2021), Item("wv", 2020), Item("wv", 2022)]
    assert ponds.pick_naip_items(items, {"wv": 2022, "va": 2023}) == [items[0], items[1], items[4]]
    with pytest.raises(ValueError, match="No NAIP"):
        ponds.pick_naip_items(items, {"md": 2023})
