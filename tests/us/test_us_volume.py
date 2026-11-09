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
