# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.rheology — normalization and Maxwell rheology."""

import math

import jax.numpy as jnp
import pytest

from pylov3d.constants import G
from pylov3d.rheology import normalize, compute_complex_rheology, get_rheology
from pylov3d.types import make_interior_model, make_forcing


@pytest.fixture
def io_model():
    return make_interior_model(
        R0_km=[965.0, 1591.6, 1791.6, 1821.6],
        rho0=[5150.0, 3244.0, 3244.0, 3244.0],
        mu0=[0.0, 6e10, 7.8e5, 6.5e10],
        Ks0=[0.0, 200e16, 200e16, 200e16],
        eta0=[None, 1e20, 1e11, 1e23],
        Delta_rho0=[5150.0 - 3244.0, 5150.0 - 3244.0, 0.0, 0.0],
    )


@pytest.fixture
def io_Td():
    omega0 = 4.1086e-05
    return 2 * math.pi / omega0


class TestNormalize:

    def test_surface_radius_is_one(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        assert float(m.R[3]) == pytest.approx(1.0)

    def test_surface_density_is_one(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        assert float(m.rho[3]) == pytest.approx(1.0)

    def test_surface_shear_is_one(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        assert float(m.mu[3]) == pytest.approx(1.0)

    def test_core_radius_fraction(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        expected = 965.0 / 1821.6
        assert float(m.R[0]) == pytest.approx(expected)

    def test_core_density_ratio(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        expected = 5150.0 / 3244.0
        assert float(m.rho[0]) == pytest.approx(expected)

    def test_gg_formula(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        R0_surf = 1821.6
        rho0_surf = 3244.0
        mu0_surf = 6.5e10
        expected = G * (R0_surf * 1e3) ** 2 * rho0_surf ** 2 / mu0_surf
        assert m.Gg == pytest.approx(expected, rel=1e-10)

    def test_maxwell_time(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        # Layer 2 (index 1): MaxTime = 2*pi*eta0/(mu0*Td)
        expected = 2 * math.pi * 1e20 / (6e10 * io_Td)
        assert float(m.MaxTime[1]) == pytest.approx(expected, rel=1e-10)

    def test_elastic_layer(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        # Core (index 0) has no shear modulus, eta is NaN → elastic flag
        assert int(m.elastic[0]) == 0  # core has no elastic flag set
        # But a truly elastic layer would have elastic=1
        # Let's check that NaN eta → elastic=1 if we set it
        model2 = make_interior_model(
            R0_km=[100.0, 200.0],
            rho0=[5000.0, 3000.0],
            mu0=[0.0, 1e10],
            eta0=[None, None],  # both elastic
        )
        m2 = normalize(model2, io_Td)
        assert int(m2.elastic[1]) == 1
        assert jnp.isnan(m2.MaxTime[1])

    def test_gravity_at_surface(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        # gs[3] should be Gg * M_total / R_surface^2 = Gg * M_total / 1^2
        # Just check it's positive and reasonable
        assert float(m.gs[3]) > 0

    def test_density_contrast(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        # Layer 2 (index 1): Delta_rho should be rho[0] - rho[1]
        expected = float(m.rho[0]) - float(m.rho[1])
        assert float(m.Delta_rho[1]) == pytest.approx(expected, rel=1e-10)


class TestComplexRheology:

    def test_viscoelastic_muC(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        m = compute_complex_rheology(m)
        # Layer 2 (index 1): muC = mu / (1 - 1j/MaxTime)
        mu_1 = float(m.mu[1])
        mt_1 = float(m.MaxTime[1])
        expected = mu_1 / (1.0 - 1j / mt_1)
        assert complex(m.muC[1]) == pytest.approx(expected, rel=1e-10)

    def test_elastic_muC(self, io_Td):
        model = make_interior_model(
            R0_km=[100.0, 200.0],
            rho0=[5000.0, 3000.0],
            mu0=[0.0, 1e10],
            eta0=[None, None],
        )
        m = normalize(model, io_Td)
        m = compute_complex_rheology(m)
        # Elastic: muC = mu (real, no imaginary part)
        assert complex(m.muC[1]).imag == pytest.approx(0.0)
        assert complex(m.muC[1]).real == pytest.approx(float(m.mu[1]))

    def test_lambda(self, io_model, io_Td):
        m = normalize(io_model, io_Td)
        m = compute_complex_rheology(m)
        # lam = Ks - 2/3 * muC
        for i in range(1, 4):
            Ks_i = float(m.Ks[i])
            muC_i = complex(m.muC[i])
            expected = Ks_i - (2.0 / 3.0) * muC_i
            assert complex(m.lam[i]) == pytest.approx(expected, rel=1e-10)

    def test_core_muC_is_zero(self, io_model, io_Td):
        m = get_rheology(io_model, make_forcing(Td=io_Td, n=2, m=0, F=1.0))
        assert complex(m.muC[0]) == 0j


class TestGetRheology:

    def test_full_pipeline(self, io_model, io_Td):
        forcing = make_forcing(Td=io_Td, n=2, m=0, F=1.0)
        m = get_rheology(io_model, forcing)
        # Check everything is filled
        assert float(m.R[3]) == pytest.approx(1.0)
        assert m.Gg > 0
        assert complex(m.muC[1]) != 0j
        assert complex(m.lam[1]) != 0j

    def test_accepts_forcing_list(self, io_model, io_Td):
        forcings = [
            make_forcing(Td=io_Td, n=2, m=0, F=1.0),
            make_forcing(Td=io_Td, n=2, m=2, F=0.5),
        ]
        m = get_rheology(io_model, forcings)
        assert float(m.R[3]) == pytest.approx(1.0)
