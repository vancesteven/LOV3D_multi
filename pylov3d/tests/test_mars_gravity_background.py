from __future__ import annotations

import math

import pytest

from pylov3d.mars_gmm3 import GMM3Coefficient
from pylov3d.mars_gravity_background import (
    degree_coefficient_norm,
    degree_coefficient_rms,
    signal_fraction_of_degree_norm,
    signal_over_degree_rms,
)


def _rows():
    return [
        GMM3Coefficient(2, 0, 3.0, 0.0, 0.1, 0.0),
        GMM3Coefficient(2, 1, 4.0, 0.0, 0.1, 0.1),
        GMM3Coefficient(2, 2, 0.0, 0.0, 0.1, 0.1),
    ]


def test_degree_norm():
    assert degree_coefficient_norm(_rows(), 2) == pytest.approx(5.0)


def test_degree_rms_uses_all_real_modes():
    assert degree_coefficient_rms(_rows(), 2) == pytest.approx(5.0 / math.sqrt(5.0))


def test_signal_fraction_of_norm():
    assert signal_fraction_of_degree_norm(1.0, _rows(), 2) == pytest.approx(0.2)


def test_signal_over_rms():
    assert signal_over_degree_rms(1.0, _rows(), 2) == pytest.approx(math.sqrt(5.0) / 5.0)


def test_missing_degree_raises():
    with pytest.raises(ValueError):
        degree_coefficient_norm(_rows(), 3)


def test_negative_degree_rms_rejected():
    with pytest.raises(ValueError):
        degree_coefficient_rms(_rows(), -1)
