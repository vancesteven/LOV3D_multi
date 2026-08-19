import math

import pytest

from pylov3d.mars_gravity_coefficients import (
    MARS_RADIUS_M,
    gravity_from_thickness_coefficient,
    radial_gravity_from_coefficient,
    thin_sheet_potential_coefficient,
)


def test_uncompensated_coefficient_scales_linearly_with_thickness_and_density():
    q1 = thin_sheet_potential_coefficient(10, 1_000.0, 420.0)
    q2 = thin_sheet_potential_coefficient(10, 2_000.0, 420.0)
    q3 = thin_sheet_potential_coefficient(10, 1_000.0, 840.0)
    assert q2 == pytest.approx(2.0 * q1)
    assert q3 == pytest.approx(2.0 * q1)


def test_full_compensation_at_zero_depth_cancels_exactly():
    q = thin_sheet_potential_coefficient(
        10,
        1_000.0,
        420.0,
        compensation_fraction=1.0,
        compensation_depth_m=0.0,
    )
    assert q == pytest.approx(0.0, abs=1e-30)


def test_deep_compensation_is_less_effective_at_high_degree():
    q10 = abs(
        thin_sheet_potential_coefficient(
            10,
            1_000.0,
            420.0,
            compensation_fraction=1.0,
            compensation_depth_m=50e3,
        )
    )
    q80 = abs(
        thin_sheet_potential_coefficient(
            80,
            1_000.0,
            420.0,
            compensation_fraction=1.0,
            compensation_depth_m=50e3,
        )
    )
    q10_un = abs(thin_sheet_potential_coefficient(10, 1_000.0, 420.0))
    q80_un = abs(thin_sheet_potential_coefficient(80, 1_000.0, 420.0))
    assert q10 / q10_un < q80 / q80_un


def test_radial_gravity_has_exact_degree_altitude_attenuation():
    degree = 20
    q = 1e-6
    g0 = radial_gravity_from_coefficient(degree, q, 0.0)
    h = 300e3
    gh = radial_gravity_from_coefficient(degree, q, h)
    expected = (MARS_RADIUS_M / (MARS_RADIUS_M + h)) ** (degree + 2)
    assert gh / g0 == pytest.approx(expected, rel=1e-14)


def test_wrapper_matches_two_step_evaluation():
    degree = 12
    q = thin_sheet_potential_coefficient(
        degree,
        2_500.0,
        420.0,
        compensation_fraction=0.4,
        compensation_depth_m=40e3,
    )
    g1 = radial_gravity_from_coefficient(degree, q, 250e3)
    g2 = gravity_from_thickness_coefficient(
        degree,
        2_500.0,
        420.0,
        250e3,
        compensation_fraction=0.4,
        compensation_depth_m=40e3,
    )
    assert g2 == pytest.approx(g1, rel=1e-15)


def test_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        thin_sheet_potential_coefficient(0, 1.0, 1.0)
    with pytest.raises(ValueError):
        thin_sheet_potential_coefficient(2, 1.0, 1.0, compensation_fraction=1.1)
    with pytest.raises(ValueError):
        radial_gravity_from_coefficient(2, 1e-6, -1.0)
