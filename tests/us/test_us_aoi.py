import json

import pytest
from shapely.geometry import Polygon, box, mapping

from pytedi.us import aoi as aoi_mod
from pytedi.us.aoi import AOI, as_aoi


def test_bbox_becomes_a_polygon_in_wgs84():
    a = as_aoi((-79.4, 37.0, -79.3, 37.1))
    assert a.crs == "EPSG:4326"
    assert a.bounds == pytest.approx((-79.4, 37.0, -79.3, 37.1))
    assert a.geometry.equals(box(-79.4, 37.0, -79.3, 37.1))


def test_bbox_must_be_ordered():
    with pytest.raises(ValueError, match="west < east"):
        as_aoi((-79.3, 37.0, -79.4, 37.1))


def test_polygon_geojson_dict_and_string_are_accepted():
    poly = Polygon([(-79.4, 37.0), (-79.3, 37.0), (-79.3, 37.1), (-79.4, 37.0)])
    assert as_aoi(poly).geometry.equals(poly)
    assert as_aoi(mapping(poly)).geometry.equals(poly)
    assert as_aoi(json.dumps(mapping(poly))).geometry.equals(poly)
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(poly)}
    assert as_aoi(feature).geometry.equals(poly)


def test_aoi_passes_through_and_reprojects():
    a = as_aoi((-79.4, 37.0, -79.3, 37.1))
    assert as_aoi(a) is a
    projected = a.to_crs("EPSG:5070")
    assert projected.crs == "EPSG:5070"
    # About 8.9 km by 11.1 km at this latitude.
    assert 90 < a.area_km2 < 110
    assert projected.area_km2 == pytest.approx(a.area_km2, rel=1e-6)


def test_huc_ids_are_looked_up(monkeypatch):
    calls = []

    def fake_huc_geometry(huc_id):
        calls.append(huc_id)
        return box(-79.4, 37.0, -79.3, 37.1)

    monkeypatch.setattr(aoi_mod, "huc_geometry", fake_huc_geometry)
    a = as_aoi("030101011304")
    assert calls == ["030101011304"] and a.name == "HUC12 030101011304"
    assert as_aoi("03010101").name == "HUC8 03010101"


@pytest.mark.parametrize("bad", ["0301010113", "12345", "abc", 42, None])
def test_rejects_other_inputs(bad, monkeypatch):
    monkeypatch.setattr(aoi_mod, "huc_geometry", lambda h: box(0, 0, 1, 1))
    if bad == "0301010113":  # a HUC10 is valid
        assert as_aoi(bad).name == "HUC10 0301010113"
        return
    with pytest.raises((ValueError, TypeError)):
        as_aoi(bad)


def test_empty_geometry_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        AOI(Polygon())


def test_core_import_does_not_load_the_us_stack():
    import subprocess
    import sys

    code = "import sys, pytedi; print(any(m in sys.modules for m in ('pytedi.us', 'geopandas', 'rasterio')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
