import numpy as np
import pytest

from pytedi import tedimodel
from pytedi.us.volume import volume_ml_from_area


def test_defaults_match_the_model_relation():
    m = tedimodel()
    area = np.array([500.0, 2_000.0, 10_000.0])
    expected = m.vol_surfarea_a * area ** m.vol_surfarea_b
    assert volume_ml_from_area(area) == pytest.approx(expected)


def test_known_value_and_custom_coefficients():
    # Fowler et al. (2015): V = 0.0001042 * A^1.3213; a 1 ha pond holds about 20 ML.
    assert volume_ml_from_area(10_000.0) == pytest.approx(0.0001042 * 10_000.0 ** 1.3213)
    assert volume_ml_from_area(10_000.0, a=0.000145, b=1.314) == pytest.approx(0.000145 * 10_000.0 ** 1.314)


def test_zero_and_negative_areas():
    assert volume_ml_from_area(0.0) == 0.0
    with pytest.raises(ValueError):
        volume_ml_from_area(-1.0)


def test_presets_and_custom_relations():
    from pytedi.us.volume import RELATIONS, relation_coefficients

    assert relation_coefficients("fowler2015") == (0.0001042, 1.3213)
    a, b = relation_coefficients("va_nid_farm")
    assert a == pytest.approx(0.01220064) and b == pytest.approx(0.8456943)
    assert relation_coefficients((0.001, 1.2)) == (0.001, 1.2)
    assert set(RELATIONS) >= {"fowler2015", "va_nid_farm"}
    with pytest.raises(ValueError, match="relation"):
        relation_coefficients("nope")


def test_fit_area_volume_recovers_a_power_law():
    from pytedi.us.volume import fit_area_volume

    area = np.array([1_000.0, 5_000.0, 20_000.0, 60_000.0])
    fit = fit_area_volume(area, 0.002 * area ** 1.1)
    assert fit["a"] == pytest.approx(0.002) and fit["b"] == pytest.approx(1.1) and fit["n"] == 4
    assert fit["r2"] == pytest.approx(1.0)


def test_fit_nid_relation_filters_state_purpose_owner_and_size(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    from pytedi.us.volume import ACRE_FT_TO_ML, ACRE_TO_M2, fit_nid_relation

    acres = np.array([2.0, 5.0, 10.0, 15.0])
    rows = {
        "state": ["Virginia"] * 4 + ["Ohio", "Virginia", "Virginia", "Virginia"],
        "surfaceArea": [*acres, 5.0, 5.0, 5.0, 40.0],
        "normalStorage": [*(3.0 * acres ** 1.2), 999.0, 999.0, 999.0, 999.0],
        "primaryPurposeId": ["Recreation"] * 5 + ["Flood Risk Reduction", "Recreation", "Recreation"],
        "ownerTypeIds": ["Private"] * 6 + ["State", "Private"],
    }
    path = tmp_path / "nid.gpkg"
    gpd.GeoDataFrame(rows, geometry=[Point(0, 0)] * 8, crs="EPSG:4269").to_file(path, driver="GPKG")
    fit = fit_nid_relation(path, state="Virginia", max_acres=20)
    assert fit["n"] == 4  # Ohio, flood control, state-owned and the 40-acre dam are left out
    assert fit["b"] == pytest.approx(1.2)
    expected_a = 3.0 * ACRE_FT_TO_ML / ACRE_TO_M2 ** 1.2
    assert fit["a"] == pytest.approx(expected_a)
