import math

import pytest

from pylov3d.mars_em_sensitivity import (
    conductivity_contrast,
    layer_conductance_siemens,
    period_for_skin_depth_s,
    skin_depth_km,
)


def test_conductance_is_sigma_times_thickness():
    assert layer_conductance_siemens(0.01, 10.0) == pytest.approx(100.0)


def test_skin_depth_period_round_trip():
    sigma = 0.1
    depth = 15.0
    period = period_for_skin_depth_s(sigma, depth)
    assert skin_depth_km(sigma, period) == pytest.approx(depth)


def test_skin_depth_scales_as_inverse_sqrt_conductivity():
    d1 = skin_depth_km(0.01, 100.0)
    d2 = skin_depth_km(0.04, 100.0)
    assert d2 == pytest.approx(d1 / 2.0)


def test_period_scales_linearly_with_conductivity_for_fixed_depth():
    t1 = period_for_skin_depth_s(0.01, 20.0)
    t2 = period_for_skin_depth_s(0.1, 20.0)
    assert t2 == pytest.approx(10.0 * t1)


def test_conductivity_contrast():
    assert conductivity_contrast(1.0, 1e-4) == pytest.approx(1e4)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        skin_depth_km(0.0, 1.0)
