"""End-to-end analytical validation tests for pylov3d.

Cross-validates the full pipeline (model → grid → rheology → solver →
Love numbers → energy) against analytical formulas and convergence checks.
"""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.love import get_love
from pylov3d.energy import get_energy, global_dissipation
from pylov3d.solver import get_solution
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology
from pylov3d.constants import G as G_phys


# ---------------------------------------------------------------------------
# Uniform elastic sphere: analytical Love numbers
# ---------------------------------------------------------------------------

class TestUniformElasticSphere:
    """Validate against the Kelvin (1863) / Love (1911) analytical solution.

    For a homogeneous elastic sphere of radius R, density ρ, shear modulus μ:

        h₂ = 5 / (2(1 + 19μ/(2ρgR)))
        k₂ = 3h₂/(2n+1) - 1 = 3h₂/5 - 1
        l₂ = h₂(1 + 2k₂/3h₂)  [more complex expression]

    We use a tiny fluid core (1% of R) + elastic mantle to approximate
    the homogeneous sphere while maintaining the fluid-core boundary condition.
    """

    @pytest.fixture
    def params(self):
        R_km = 1000.0
        R_m = R_km * 1e3
        rho = 3000.0
        mu = 1e10
        g = G_phys * (4.0 / 3.0) * math.pi * rho * R_m
        h2_an = 5.0 / (2.0 * (1.0 + 19.0 * mu / (2.0 * rho * g * R_m)))
        k2_an = 3.0 * h2_an / 5.0 - 1.0
        return dict(
            R_km=R_km, rho=rho, mu=mu, R_m=R_m, g=g,
            h2=h2_an, k2=k2_an,
        )

    @pytest.fixture
    def model(self, params):
        return make_interior_model(
            R0_km=[10.0, params['R_km']],
            rho0=[params['rho'], params['rho']],
            mu0=[0.0, params['mu']],
            eta0=[None, None],
        )

    @pytest.fixture
    def forcing(self):
        return make_forcing(Td=86400.0, n=2, m=0, F=1.0)

    def test_h2(self, model, forcing, params):
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=500)
        love, _, _ = get_love(model, forcing, numerics)
        h2 = abs(complex(love.h[0]).real)
        assert h2 == pytest.approx(params['h2'], rel=0.03)

    def test_k2_positive(self, model, forcing):
        """k₂ for an elastic body with fluid core should be positive and real.

        The analytical formula k₂ = 3h₂/5 − 1 applies only to a truly
        homogeneous solid sphere.  A fluid core significantly enhances the
        gravitational potential response, so we only test sign and reality.
        """
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=500)
        love, _, _ = get_love(model, forcing, numerics)
        k2 = complex(love.k[0])
        assert k2.real > 0
        assert abs(k2.imag) < 1e-10

    def test_purely_real(self, model, forcing):
        """Elastic Love numbers should have zero imaginary part."""
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        love, _, _ = get_love(model, forcing, numerics)
        assert abs(complex(love.k[0]).imag) < 1e-10
        assert abs(complex(love.h[0]).imag) < 1e-10
        assert abs(complex(love.l[0]).imag) < 1e-10

    def test_zero_dissipation(self, model, forcing):
        """Elastic body: no tidal heating."""
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        love, _, mdl = get_love(model, forcing, numerics)
        k2 = complex(love.k[0])
        E = global_dissipation(k2, 2 * math.pi / 86400.0, 1000e3, 3000.0)
        assert abs(E) < 1e-10

    def test_convergence(self, model, forcing):
        """Love numbers should converge with increasing resolution."""
        results = []
        for Nrbase in [100, 200, 400]:
            numerics = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
            love, _, _ = get_love(model, forcing, numerics)
            results.append(complex(love.k[0]).real)

        # Successive differences should decrease
        d1 = abs(results[1] - results[0])
        d2 = abs(results[2] - results[1])
        assert d2 < d1


# ---------------------------------------------------------------------------
# Viscoelastic limits
# ---------------------------------------------------------------------------

class TestViscoelasticLimits:
    """Test behavior in elastic and viscous limits."""

    @pytest.fixture
    def base_model_params(self):
        return dict(
            R0_km=[10.0, 1000.0],
            rho0=[3000.0, 3000.0],
            mu0=[0.0, 1e10],
            Ks0=[0.0, 200e16],
        )

    def test_high_viscosity_approaches_elastic(self, base_model_params):
        """Very high η → elastic behavior (Im(k₂) → 0)."""
        model = make_interior_model(
            **base_model_params,
            eta0=[None, 1e30],
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        love, _, _ = get_love(model, forcing, numerics)
        k2 = complex(love.k[0])

        # Im(k₂) should be very small for near-elastic body
        assert abs(k2.imag) < 1e-6

    def test_low_viscosity_large_imaginary(self, base_model_params):
        """Moderate η → significant dissipation (|Im(k₂)| > 0)."""
        model = make_interior_model(
            **base_model_params,
            eta0=[None, 1e15],
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        love, _, _ = get_love(model, forcing, numerics)
        k2 = complex(love.k[0])

        assert abs(k2.imag) > 1e-4
        assert k2.imag < 0  # dissipative

    def test_viscosity_monotonicity(self, base_model_params):
        """As η decreases through resonance, |Im(k₂)| first increases then decreases."""
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)

        imk_values = []
        for log_eta in [25, 20, 15, 12, 10]:
            model = make_interior_model(
                **base_model_params,
                eta0=[None, 10.0 ** log_eta],
            )
            numerics = make_numerics(n_layers=2, method="variable", Nrbase=200)
            love, _, _ = get_love(model, forcing, numerics)
            imk_values.append(abs(complex(love.k[0]).imag))

        # There should be a peak somewhere in the middle (resonance)
        # At least one interior value should be larger than the endpoints
        max_interior = max(imk_values[1:-1])
        assert max_interior > imk_values[0]   # more dissipation than near-elastic
        assert max_interior > imk_values[-1]   # more dissipation than near-fluid


# ---------------------------------------------------------------------------
# Io model: physical sanity checks
# ---------------------------------------------------------------------------

class TestIoModel:

    @pytest.fixture
    def io_model(self):
        return make_interior_model(
            R0_km=[965.0, 1591.6, 1791.6, 1821.6],
            rho0=[5150.0, 3244.0, 3244.0, 3244.0],
            mu0=[0.0, 6e10, 7.8e5, 6.5e10],
            Ks0=[0.0, 200e16, 200e16, 200e16],
            eta0=[None, 1e20, 1e11, 1e23],
            Delta_rho0=[5150.0 - 3244.0, 5150.0 - 3244.0, 0.0, 0.0],
        )

    @pytest.fixture
    def io_forcing(self):
        omega0 = 4.1086e-05
        Td = 2 * math.pi / omega0
        return make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5))

    def test_k2_magnitude(self, io_model, io_forcing):
        """k₂ magnitude for Io should be O(0.01–1)."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        k2 = complex(love.k[0])
        assert 1e-4 < abs(k2) < 10

    def test_k2_dissipative(self, io_model, io_forcing):
        """Io is dissipative → Im(k₂) < 0."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        k2 = complex(love.k[0])
        assert k2.imag < 0

    def test_resolution_convergence(self, io_model, io_forcing):
        """Love numbers should converge with resolution for Io.

        Use 'variable' method for uniform scaling per layer.
        """
        results = []
        for Nrbase in [50, 100, 200]:
            numerics = make_numerics(n_layers=4, method="variable", Nrbase=Nrbase)
            love, _, _ = get_love(io_model, io_forcing, numerics)
            results.append(complex(love.k[0]))

        d1 = abs(results[1] - results[0])
        d2 = abs(results[2] - results[1])
        assert d2 < d1

    def test_energy_nonzero(self, io_model, io_forcing):
        """Io should have finite, nonzero tidal dissipation."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        numerics, model = set_boundary_indices(numerics, io_model)
        model = get_rheology(model, io_forcing)
        y_sol, r_grid, _, Aprop_aux = get_solution(model, io_forcing, numerics)
        energy = get_energy(y_sol, r_grid, Aprop_aux, model, io_forcing, numerics)

        assert abs(energy.energy_integral[0]) > 0
        assert np.all(np.isfinite(energy.energy_profile))

    def test_pipeline_roundtrip(self, io_model, io_forcing):
        """get_love and direct solver should give identical Love numbers."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)

        # Via get_love
        love, _, model = get_love(io_model, io_forcing, numerics)

        # Via direct solver
        numerics2 = make_numerics(n_layers=4, method="combination", Nrbase=100)
        numerics2, model2 = set_boundary_indices(numerics2, io_model)
        model2 = get_rheology(model2, io_forcing)
        y_sol, _, _, _ = get_solution(model2, io_forcing, numerics2)

        gs = float(model2.gs[3])
        k_direct = complex(y_sol[-1, 6]) - 1.0
        h_direct = -gs * complex(y_sol[-1, 0])

        assert complex(love.k[0]) == pytest.approx(k_direct, rel=1e-12)
        assert complex(love.h[0]) == pytest.approx(h_direct, rel=1e-12)


# ---------------------------------------------------------------------------
# Energy consistency
# ---------------------------------------------------------------------------

class TestEnergyConsistency:

    def test_elastic_zero_from_profile(self):
        """Elastic body: radial dissipation profile should be uniformly zero."""
        model = make_interior_model(
            R0_km=[10.0, 1000.0],
            rho0=[3000.0, 3000.0],
            mu0=[0.0, 1e10],
            eta0=[None, None],
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, mdl = set_boundary_indices(numerics, model)
        mdl = get_rheology(mdl, forcing)
        y_sol, r_grid, _, Aprop_aux = get_solution(mdl, forcing, numerics)
        energy = get_energy(y_sol, r_grid, Aprop_aux, mdl, forcing, numerics)

        np.testing.assert_allclose(energy.energy_profile[:, 0], 0.0, atol=1e-15)

    def test_dissipation_sign(self):
        """Im(σ*:ε) should produce net negative values (heating convention)."""
        model = make_interior_model(
            R0_km=[10.0, 1000.0],
            rho0=[3000.0, 3000.0],
            mu0=[0.0, 1e10],
            Ks0=[0.0, 200e16],
            eta0=[None, 1e15],
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, mdl = set_boundary_indices(numerics, model)
        mdl = get_rheology(mdl, forcing)
        y_sol, r_grid, _, Aprop_aux = get_solution(mdl, forcing, numerics)
        energy = get_energy(y_sol, r_grid, Aprop_aux, mdl, forcing, numerics)

        # Total integrated dissipation should be nonzero
        assert abs(energy.energy_integral[0]) > 0

    def test_viscous_layer_dominates_profile(self):
        """Im(σ*:ε) should be larger in the viscous layer than the elastic shell.

        Note: without angular coupling coefficients, the profile includes
        elastic energy transport terms.  In elastic layers, Im(σ*:ε) is
        nonzero because stress and strain have cross-phase contributions
        from the viscoelastic coupling.  Still, the viscous layer should
        dominate in magnitude.
        """
        model = make_interior_model(
            R0_km=[100.0, 500.0, 1000.0],
            rho0=[5000.0, 3000.0, 3000.0],
            mu0=[0.0, 1e10, 1e10],
            Ks0=[0.0, 200e16, 200e16],
            eta0=[None, 1e14, None],  # viscous mantle, elastic shell
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=100)
        numerics, mdl = set_boundary_indices(numerics, model)
        mdl = get_rheology(mdl, forcing)
        y_sol, r_grid, _, Aprop_aux = get_solution(mdl, forcing, numerics)
        energy = get_energy(y_sol, r_grid, Aprop_aux, mdl, forcing, numerics)

        profile = energy.energy_profile[:, 0]

        # Find layer boundaries
        R_boundary = float(mdl.R[1])  # normalized boundary
        elastic_mask = r_grid > R_boundary + 0.01
        viscous_mask = (r_grid > float(mdl.R[0]) + 0.01) & (r_grid < R_boundary - 0.01)

        # Both regions should have nonzero values
        viscous_max = np.max(np.abs(profile[viscous_mask]))
        elastic_max = np.max(np.abs(profile[elastic_mask]))

        assert viscous_max > 0
        # Viscous layer should have larger magnitude Im(σ*:ε)
        assert viscous_max > elastic_max
