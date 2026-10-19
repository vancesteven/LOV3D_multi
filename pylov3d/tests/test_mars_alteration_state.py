import math

import numpy as np
import pytest

from pylov3d.mars_alteration_state import (
    AlterationEndmembers,
    alteration_bulk_fraction,
    alteration_material_fields,
    elastic_layer_state_from_3d,
    radial_volume_average,
)
from pylov3d.matlab_sph import stokes_to_grid


ENDMEMBERS = AlterationEndmembers(
    rho_dry_kg_m3=3200.0,
    rho_hydrated_kg_m3=2600.0,
    mu_dry_Pa=60e9,
    mu_hydrated_Pa=20e9,
    K_dry_Pa=100e9,
    K_hydrated_Pa=50e9,
)


def test_alteration_fraction_and_endmember_limits():
    x = alteration_bulk_fraction(np.array([0.0, 0.5, 1.0]), 0.4)
    np.testing.assert_allclose(x, [0.0, 0.2, 0.4])

    rho0, mu0, K0 = alteration_material_fields(0.0, 1.0, ENDMEMBERS)
    assert float(rho0) == ENDMEMBERS.rho_dry_kg_m3
    assert float(mu0) == ENDMEMBERS.mu_dry_Pa
    assert float(K0) == ENDMEMBERS.K_dry_Pa

    rho1, mu1, K1 = alteration_material_fields(1.0, 1.0, ENDMEMBERS, mixing_law="voigt")
    assert float(rho1) == ENDMEMBERS.rho_hydrated_kg_m3
    assert float(mu1) == ENDMEMBERS.mu_hydrated_Pa
    assert float(K1) == ENDMEMBERS.K_hydrated_Pa


def test_voigt_hill_reuss_ordering_for_soft_hydrated_phase():
    _, mu_v, K_v = alteration_material_fields(0.5, 1.0, ENDMEMBERS, mixing_law="voigt")
    _, mu_h, K_h = alteration_material_fields(0.5, 1.0, ENDMEMBERS, mixing_law="hill")
    _, mu_r, K_r = alteration_material_fields(0.5, 1.0, ENDMEMBERS, mixing_law="reuss")
    assert float(mu_v) > float(mu_h) > float(mu_r)
    assert float(K_v) > float(K_h) > float(K_r)


def test_radial_volume_average_uses_exact_spherical_weights():
    r = np.array([1.0, 2.0, 3.0])
    field = np.array([10.0, 20.0])
    got = radial_volume_average(field, r)
    w = (r[1:] ** 3 - r[:-1] ** 3) / 3.0
    expected = np.sum(field * w) / np.sum(w)
    assert got == pytest.approx(expected, rel=1e-15)


def test_constant_3d_elastic_field_reduces_to_mean_without_lateral_modes():
    lmax = 6
    nz = 2
    mu0 = 30e9
    K0 = 70e9
    mu = np.full((2 * lmax, 4 * lmax, nz), mu0)
    K = np.full_like(mu, K0)
    r = np.array([3.34e6, 3.365e6, 3.39e6])
    state = elastic_layer_state_from_3d(mu, K, r, lmax)
    assert state.mean_mu_Pa == pytest.approx(mu0, rel=2e-14)
    assert state.mean_K_Pa == pytest.approx(K0, rel=2e-14)
    # Numerical transform leakage should be suppressed by the physical bridge.
    assert state.mu_variable == {}
    assert state.K_variable == {}


def test_known_nonaxisymmetric_modulus_map_recovers_fractional_solver_modes():
    lmax = 8
    mu0 = 40e9
    K0 = 80e9
    c_mu = np.zeros((lmax + 1, lmax + 1))
    s_mu = np.zeros_like(c_mu)
    c_mu[0, 0] = mu0
    c_mu[3, 2] = 0.10 * mu0
    s_mu[3, 2] = -0.04 * mu0
    _, _, mu_grid = stokes_to_grid(c_mu, s_mu, lmax)

    c_K = np.zeros_like(c_mu)
    s_K = np.zeros_like(c_mu)
    c_K[0, 0] = K0
    c_K[2, 1] = -0.06 * K0
    _, _, K_grid = stokes_to_grid(c_K, s_K, lmax)

    mu = mu_grid[:, :, None]
    K = K_grid[:, :, None]
    r = np.array([3.34e6, 3.39e6])
    state = elastic_layer_state_from_3d(mu, K, r, lmax, layer_index=3)

    mu_map = {(n, m): amp for n, m, amp in state.mu_variable[3]}
    root2 = math.sqrt(2.0)
    assert mu_map[(3, 2)] == pytest.approx((0.10 + 0.04j) / root2, abs=2e-12)
    assert mu_map[(3, -2)] == pytest.approx((0.10 - 0.04j) / root2, abs=2e-12)

    K_map = {(n, m): amp for n, m, amp in state.K_variable[3]}
    assert K_map[(2, 1)] == pytest.approx(-0.06 / root2, abs=2e-12)
    assert K_map[(2, -1)] == pytest.approx(+0.06 / root2, abs=2e-12)
