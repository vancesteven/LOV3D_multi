import math

import numpy as np
import pytest

from pylov3d.mars_alteration_gravity import (
    density_contrast_from_alteration,
    grid_to_orthonormal_density_coefficients,
    layered_density_gravity_coefficients,
    orthonormal_gravity_arrays_to_gmm3,
)
from pylov3d.mars_gravity_coefficients import (
    MARS_RADIUS_M,
    finite_shell_potential_coefficient,
)
from pylov3d.mars_gravity_normalization import orthonormal_to_gmm3_normalized
from pylov3d.matlab_sph import stokes_to_grid


def test_density_contrast_tracks_hydrated_and_reactive_fractions():
    got = density_contrast_from_alteration(
        hydrated_fraction=np.array([0.0, 0.5, 1.0]),
        reactive_fraction=0.4,
        rho_dry_kg_m3=3200.0,
        rho_hydrated_kg_m3=2600.0,
    )
    np.testing.assert_allclose(got, [0.0, -120.0, -240.0])


def test_uniform_density_grid_has_no_nonzero_degree_coefficients():
    lmax = 6
    grid = np.full((2 * lmax, 4 * lmax), -100.0)
    c, s = grid_to_orthonormal_density_coefficients(grid, lmax)
    assert c[0, 0] == pytest.approx(-100.0 * math.sqrt(4.0 * math.pi), rel=1e-14)
    np.testing.assert_allclose(c[1:, :], 0.0, rtol=0, atol=5e-12)
    np.testing.assert_allclose(s, 0.0, rtol=0, atol=5e-12)


def test_phase_corrected_transform_recovers_nonaxisymmetric_stokes_pair():
    lmax = 8
    c0 = np.zeros((lmax + 1, lmax + 1))
    s0 = np.zeros_like(c0)
    c0[3, 2] = 40.0 / math.sqrt(4.0 * math.pi)
    s0[3, 2] = -25.0 / math.sqrt(4.0 * math.pi)
    _, _, grid = stokes_to_grid(c0, s0, lmax)

    c, s = grid_to_orthonormal_density_coefficients(grid, lmax)
    assert c[3, 2] == pytest.approx(40.0, rel=0, abs=2e-10)
    assert s[3, 2] == pytest.approx(-25.0, rel=0, abs=2e-10)


def test_axisymmetric_degree2_density_map_matches_exact_finite_shell_gravity():
    lmax = 6
    desired_unit_norm_rho_lm = -250.0
    c = np.zeros((lmax + 1, lmax + 1))
    s = np.zeros_like(c)
    # stokes_to_grid uses the 4pi-normalized basis, so divide by sqrt(4pi)
    # to synthesize a desired unit-norm density coefficient.
    c[2, 0] = desired_unit_norm_rho_lm / math.sqrt(4.0 * math.pi)
    _, _, grid = stokes_to_grid(c, s, lmax)

    inner = MARS_RADIUS_M - 20e3
    outer = MARS_RADIUS_M
    q_cos, q_sin = layered_density_gravity_coefficients(
        grid[None, :, :],
        np.array([inner, outer]),
        lmax,
    )
    expected = finite_shell_potential_coefficient(
        2,
        desired_unit_norm_rho_lm,
        inner,
        outer,
    )
    assert q_cos[2, 0] == pytest.approx(expected, rel=2e-12)
    np.testing.assert_allclose(q_sin, 0.0, rtol=0, atol=1e-18)


def test_gmm3_array_bridge_matches_scalar_normalization_for_cosine_and_sine():
    q_cos = np.zeros((6, 6))
    q_sin = np.zeros_like(q_cos)
    q_cos[5, 2] = 2.1e-7
    q_sin[5, 2] = -7.0e-8
    c, s = orthonormal_gravity_arrays_to_gmm3(q_cos, q_sin)
    assert c[5, 2] == pytest.approx(
        orthonormal_to_gmm3_normalized(q_cos[5, 2], 5), rel=1e-15
    )
    assert s[5, 2] == pytest.approx(
        orthonormal_to_gmm3_normalized(q_sin[5, 2], 5), rel=1e-15
    )
    assert c[2, 5] == 0.0
    assert s[2, 5] == 0.0


def test_invalid_fraction_and_grid_inputs_are_rejected():
    with pytest.raises(ValueError, match="hydrated_fraction"):
        density_contrast_from_alteration(1.2, 1.0, 3000.0, 2600.0)
    with pytest.raises(ValueError, match="shape"):
        layered_density_gravity_coefficients(
            np.zeros((1, 3, 4)),
            np.array([MARS_RADIUS_M - 1e3, MARS_RADIUS_M]),
            2,
        )
    with pytest.raises(ValueError, match="square"):
        orthonormal_gravity_arrays_to_gmm3(np.zeros((2, 3)), np.zeros((2, 3)))
