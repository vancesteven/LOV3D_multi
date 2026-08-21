import math

import numpy as np
import pytest

from pylov3d.mars_gravity_coefficients import (
    MARS_MASS_KG,
    MARS_RADIUS_M,
    finite_shell_potential_coefficient,
    gravity_from_thickness_coefficient,
    layered_density_potential_coefficient,
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


def test_finite_shell_formula_matches_direct_analytic_expression():
    degree = 8
    rho_lm = -350.0
    ri = MARS_RADIUS_M - 20e3
    ro = MARS_RADIUS_M - 5e3
    got = finite_shell_potential_coefficient(degree, rho_lm, ri, ro)
    moment = (ro ** (degree + 3) - ri ** (degree + 3)) / (degree + 3)
    expected = (
        4.0
        * math.pi
        * rho_lm
        * moment
        / ((2 * degree + 1) * MARS_MASS_KG * MARS_RADIUS_M**degree)
    )
    assert got == pytest.approx(expected, rel=2e-15)


def test_finite_shell_is_stable_at_gmm3_degree_120():
    q = finite_shell_potential_coefficient(
        120,
        -250.0,
        MARS_RADIUS_M - 20e3,
        MARS_RADIUS_M,
    )
    assert math.isfinite(q)
    assert q < 0.0
    # A shallower shell contributes more strongly at high degree than the same
    # density coefficient buried deeper.
    q_deep = finite_shell_potential_coefficient(
        120,
        -250.0,
        MARS_RADIUS_M - 40e3,
        MARS_RADIUS_M - 20e3,
    )
    assert abs(q) > abs(q_deep)


def test_thin_sheet_is_small_thickness_limit_of_exact_shell():
    degree = 20
    rho_lm = 420.0
    thickness = 10.0  # deliberately thin compared with Mars radius
    exact = finite_shell_potential_coefficient(
        degree,
        rho_lm,
        MARS_RADIUS_M - thickness,
        MARS_RADIUS_M,
    )
    sheet = thin_sheet_potential_coefficient(degree, thickness, rho_lm)
    # First neglected term scales as thickness/R.
    assert exact == pytest.approx(sheet, rel=5e-5)


def test_layered_profile_is_sum_of_exact_shell_moments():
    r = np.array([MARS_RADIUS_M - 30e3, MARS_RADIUS_M - 20e3, MARS_RADIUS_M])
    rho = np.array([-400.0, 150.0])
    got = layered_density_potential_coefficient(11, r, rho)
    expected = finite_shell_potential_coefficient(11, rho[0], r[0], r[1]) + finite_shell_potential_coefficient(
        11, rho[1], r[1], r[2]
    )
    assert got == pytest.approx(expected, rel=1e-15)


def test_equal_and_opposite_density_shells_can_cancel_by_radial_moment():
    degree = 5
    r = np.array([MARS_RADIUS_M - 100e3, MARS_RADIUS_M - 50e3, MARS_RADIUS_M])
    # Choose second-shell coefficient to cancel the first shell's exact radial moment.
    m1 = (r[1] ** (degree + 3) - r[0] ** (degree + 3)) / (degree + 3)
    m2 = (r[2] ** (degree + 3) - r[1] ** (degree + 3)) / (degree + 3)
    rho = np.array([1.0, -m1 / m2])
    q = layered_density_potential_coefficient(degree, r, rho)
    assert q == pytest.approx(0.0, abs=1e-20)


def test_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        thin_sheet_potential_coefficient(0, 1.0, 1.0)
    with pytest.raises(ValueError):
        thin_sheet_potential_coefficient(2, 1.0, 1.0, compensation_fraction=1.1)
    with pytest.raises(ValueError):
        radial_gravity_from_coefficient(2, 1e-6, -1.0)
    with pytest.raises(ValueError):
        finite_shell_potential_coefficient(2, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError):
        layered_density_potential_coefficient(2, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
