# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.mars — TASK-011 Mars 1D radial reference model.

Validates the fitted Mars interior model (built via
``pylov3d.mars.build_mars_model``) against the published bulk constraints in
``pylov3d.mars.MARS`` (see that module's docstring for full citations),
using the existing pylov3d tidal solver pipeline (``get_love``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylov3d.mars import (
    MARS,
    MARS_MU_SCALE,
    _solve_densities,
    build_mars_model,
    fit_mu_scale,
    mars_moi_factor,
)
from pylov3d.types import make_forcing, make_numerics
from pylov3d.love import get_love

# Densities pinned as regression constants (see TestDensityProfile.
# test_fitted_densities_pinned for why this must be checked independently of
# the mass/MoI ratio tests).
_EXPECTED_RHO_CORE = 6128.07599551278
_EXPECTED_RHO_LM = 4136.503827041973

# h2/l2 pinned at the fitted MARS_MU_SCALE (see TestLoveNumberSanity).
_EXPECTED_H2 = 0.3156322056816353
_EXPECTED_L2 = 0.05159595220204703


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mars_model():
    return build_mars_model()


@pytest.fixture(scope="module")
def mars_forcing():
    return make_forcing(Td=44387.62, n=2, m=0, F=1.0)


@pytest.fixture(scope="module")
def mars_numerics():
    return make_numerics(n_layers=4, method="combination", Nrbase=100)


@pytest.fixture(scope="module")
def mars_love(mars_model, mars_forcing, mars_numerics):
    love, y_rad, model = get_love(mars_model, mars_forcing, mars_numerics)
    return love, y_rad, model


# ---------------------------------------------------------------------------
# 1. Mass
# ---------------------------------------------------------------------------

class TestMass:

    def test_mass_within_tolerance(self, mars_model):
        n = mars_model.n_layers
        radii_km = [float(x) for x in np.asarray(mars_model.R0[:n])]
        rho = [float(x) for x in np.asarray(mars_model.rho0[:n])]

        boundaries_m = [0.0] + [r * 1e3 for r in radii_km]
        M = 0.0
        for i, rho_i in enumerate(rho):
            r_in, r_out = boundaries_m[i], boundaries_m[i + 1]
            M += (4 * math.pi / 3) * rho_i * (r_out**3 - r_in**3)

        target_M = MARS["GM"] / MARS["G"]
        assert M == pytest.approx(target_M, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. Moment of inertia factor (mean I/MR^2, NOT the published polar C/MR^2 —
#    see MARS module docstring for the C/MR^2 -> I/MR^2 derivation via J2)
# ---------------------------------------------------------------------------

class TestMoI:

    def test_moi_factor_within_uncertainty(self):
        factor = mars_moi_factor()
        assert factor == pytest.approx(
            MARS["MoI_factor"], abs=MARS["MoI_factor_sigma"]
        )

    def test_moi_target_is_mean_not_polar(self):
        """The mean-moment fit target must differ from the published polar
        moment by exactly (2/3)*J2 (guards against silently re-fitting to
        the polar value, which is the bug this correction addresses)."""
        derived = MARS["MoI_polar_factor"] - (2.0 / 3.0) * MARS["J2"]
        assert derived == pytest.approx(MARS["MoI_factor"], abs=5e-6)
        assert MARS["MoI_factor"] != MARS["MoI_polar_factor"]


# ---------------------------------------------------------------------------
# 3. k2 through get_love
# ---------------------------------------------------------------------------

class TestK2:

    def test_k2_within_observational_uncertainty(self, mars_love):
        love, _, _ = mars_love
        k2 = complex(love.k[0])
        assert k2.real == pytest.approx(MARS["k2"], abs=MARS["k2_sigma"])

    def test_k2_within_fit_tolerance(self, mars_love):
        """Regression pin: hardcoded MARS_MU_SCALE should reproduce the fit
        target to well within the observational error bar (1e-3 << 0.006)."""
        love, _, _ = mars_love
        k2 = complex(love.k[0])
        assert k2.real == pytest.approx(MARS["k2"], abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Love number sanity (elastic, real; h2 > k2 > 0) + regression pins
# ---------------------------------------------------------------------------

class TestLoveNumberSanity:

    def test_k2_is_real(self, mars_love):
        love, _, _ = mars_love
        k2 = complex(love.k[0])
        assert abs(k2.imag) < 1e-10

    def test_h2_positive_and_greater_than_k2(self, mars_love):
        love, _, _ = mars_love
        h2 = complex(love.h[0]).real
        k2 = complex(love.k[0]).real
        assert h2 > 0
        assert h2 > k2

    def test_h2_regression_pin(self, mars_love):
        love, _, _ = mars_love
        h2 = complex(love.h[0]).real
        assert h2 == pytest.approx(_EXPECTED_H2, abs=1e-4)

    def test_l2_regression_pin(self, mars_love):
        love, _, _ = mars_love
        l2 = complex(love.l[0]).real
        assert l2 == pytest.approx(_EXPECTED_L2, abs=1e-4)


# ---------------------------------------------------------------------------
# 5. Density profile
# ---------------------------------------------------------------------------

class TestDensityProfile:

    def test_monotonic_non_increasing_outward(self, mars_model):
        n = mars_model.n_layers
        rho = [float(x) for x in np.asarray(mars_model.rho0[:n])]
        assert all(rho[i] >= rho[i + 1] for i in range(len(rho) - 1))

    def test_fitted_densities_pinned(self):
        """Pin the fitted densities themselves (not just the mass/MoI ratio
        check), to ~1e-6 relative. The mass and MoI *ratio* checks alone
        cannot catch a wrong shell coefficient (e.g. an incorrect 8*pi/15
        prefactor) applied consistently to both the fit and the diagnostic,
        because such an error cancels in the I/(M*R^2) ratio; pinning the
        actual density values catches that class of bug."""
        rho_core, rho_lm = _solve_densities()
        assert rho_core == pytest.approx(_EXPECTED_RHO_CORE, rel=1e-6)
        assert rho_lm == pytest.approx(_EXPECTED_RHO_LM, rel=1e-6)

    def test_core_density_plausible(self):
        rho_core, _ = _solve_densities()
        assert 5700.0 <= rho_core <= 6300.0

    def test_lower_mantle_denser_than_upper_mantle(self):
        _, rho_lm = _solve_densities()
        assert rho_lm > MARS["upper_mantle_density"]

    def test_bad_moi_target_raises_at_call_time(self):
        """_solve_densities is lazily evaluated (not run at import time),
        so a deliberately bad MoI target must raise ValueError when the
        function is actually called, exercising the [5700, 6300] kg/m^3
        core-density guard rail."""
        with pytest.raises(ValueError):
            _solve_densities(moi_factor=0.5)


# ---------------------------------------------------------------------------
# 6. Mantle mu plausibility
# ---------------------------------------------------------------------------

class TestMuPlausibility:

    def test_lower_mantle_mu_plausible(self, mars_model):
        n = mars_model.n_layers
        mu = [float(x) for x in np.asarray(mars_model.mu0[:n])]
        assert 50e9 <= mu[1] <= 200e9

    def test_upper_mantle_mu_plausible(self, mars_model):
        n = mars_model.n_layers
        mu = [float(x) for x in np.asarray(mars_model.mu0[:n])]
        assert 50e9 <= mu[2] <= 200e9


# ---------------------------------------------------------------------------
# 7. Reproducibility of the fit (slow-ok: runs a few get_love calls)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFitReproducibility:

    def test_fit_mu_scale_reproduces_hardcoded_constant(self):
        s = fit_mu_scale()
        assert s == pytest.approx(MARS_MU_SCALE, abs=1e-3)

    def test_fit_mu_scale_exact_at_tight_tolerance(self):
        """MARS_MU_SCALE's provenance comment states it was computed by
        fit_mu_scale(tol=1e-12); verify that invocation exactly reproduces
        the hardcoded constant."""
        s = fit_mu_scale(tol=1e-12)
        # Not exact equality: bisection branch decisions near convergence can
        # flip on ~1e-14 BLAS/environment differences in k2.
        assert s == pytest.approx(MARS_MU_SCALE, abs=1e-15)
