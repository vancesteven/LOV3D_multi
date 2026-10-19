# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from pylov3d.mars_seismic import (
    WRIGHT_2024_MIDCRUST,
    isotropic_velocities,
    moduli_from_velocities,
    seismic_chi2,
    seismic_loglike,
)


def test_isotropic_velocity_modulus_roundtrip():
    K0 = 42.0e9
    mu0 = 21.0e9
    rho0 = 2700.0
    vp, vs = isotropic_velocities(K0, mu0, rho0)
    K1, mu1 = moduli_from_velocities(vp, vs, rho0)
    np.testing.assert_allclose(K1, K0, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(mu1, mu0, rtol=2e-15, atol=0.0)


def test_wright2024_observed_vector_has_zero_chi2():
    c = WRIGHT_2024_MIDCRUST
    assert seismic_chi2(c.vp_m_s, c.vs_m_s, c.rho_kg_m3, c) == 0.0
    assert seismic_loglike(c.vp_m_s, c.vs_m_s, c.rho_kg_m3, c) == 0.0


def test_one_sigma_offset_contributes_unit_chi2():
    c = WRIGHT_2024_MIDCRUST
    sigma_vp = np.sqrt(c.covariance[0, 0])
    chi2 = seismic_chi2(c.vp_m_s + sigma_vp, c.vs_m_s, c.rho_kg_m3, c)
    np.testing.assert_allclose(chi2, 1.0, rtol=1e-14, atol=1e-14)


def test_stiffer_faster_state_is_penalized():
    c = WRIGHT_2024_MIDCRUST
    near = seismic_chi2(4.2e3, 2.55e3, 2600.0, c)
    far = seismic_chi2(5.5e3, 3.5e3, 3000.0, c)
    assert far > near > 0.0
