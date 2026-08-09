# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.solver — radial integration and boundary conditions."""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution, cash_karp_increment
from pylov3d.propagator import build_aprop, compute_gravity
from pylov3d.boundary_conditions import assemble_bc_no_ocean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def io_model():
    """Io 4-layer model (from conftest)."""
    return make_interior_model(
        R0_km=[965.0, 1591.6, 1791.6, 1821.6],
        rho0=[5150.0, 3244.0, 3244.0, 3244.0],
        mu0=[0.0, 6e10, 7.8e5, 6.5e10],
        Ks0=[0.0, 200e16, 200e16, 200e16],
        eta0=[None, 1e20, 1e11, 1e23],
        Delta_rho0=[5150.0 - 3244.0, 5150.0 - 3244.0, 0.0, 0.0],
    )


@pytest.fixture
def io_forcing():
    omega0 = 4.1086e-05
    Td = 2 * math.pi / omega0
    return make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5))


@pytest.fixture
def uniform_elastic():
    """Two-layer (core + mantle) uniform elastic sphere for analytical comparison.

    Core radius is 10 km (1% of R) — small enough that Love numbers
    approximate the homogeneous sphere, large enough that dr/r stays
    reasonable for the fixed-step CK-RK5 integrator.
    """
    R = 1000.0   # km
    rho = 3000.0  # kg/m^3
    mu = 1e10     # Pa
    return make_interior_model(
        R0_km=[10.0, R],  # small fluid core + elastic mantle
        rho0=[rho, rho],
        mu0=[0.0, mu],
        eta0=[None, None],  # elastic
    )


@pytest.fixture
def uniform_forcing():
    return make_forcing(Td=86400.0, n=2, m=0, F=1.0)


# ---------------------------------------------------------------------------
# Cash-Karp increment
# ---------------------------------------------------------------------------

class TestCashKarpIncrement:

    def test_shape(self):
        inc, Ap = cash_karp_increment(
            0.5, 0.001, 2, 1.0+0j, 1.0+0j, 1.0, 1.0, 0.1, 0.3,
        )
        assert inc.shape == (8, 8)
        assert Ap.shape == (8, 8)

    def test_small_step_near_identity(self):
        """Very small step should give inc ≈ dr * Aprop."""
        dr = 1e-8
        inc, Ap = cash_karp_increment(
            0.5, dr, 2, 1.0+0j, 1.0+0j, 1.0, 1.0, 0.1, 0.3,
        )
        expected = dr * Ap
        np.testing.assert_allclose(inc, expected, atol=1e-12)

    def test_increment_is_finite(self):
        inc, _ = cash_karp_increment(
            0.5, 0.01, 2, 1.0+0j, 1.0+0j, 1.0, 1.0, 0.5, 0.3,
        )
        assert np.all(np.isfinite(inc))


# ---------------------------------------------------------------------------
# Boundary conditions assembly
# ---------------------------------------------------------------------------

class TestBCAssembly:

    def test_bc_matrix_shape(self):
        I8 = np.eye(8, dtype=np.complex128)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        B, B2 = assemble_bc_no_ocean(
            I8, I8, 2, 0,
            gc=1.0, Rc=0.5, rho2=1.5, rhoK=1.0, Gg=1.0,
            rho_layer2=1.0, forcing=forcing,
        )
        assert B.shape == (8, 8)
        assert B2.shape == (8,)

    def test_forcing_rhs(self):
        I8 = np.eye(8, dtype=np.complex128)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        _, B2 = assemble_bc_no_ocean(
            I8, I8, 2, 0,
            gc=1.0, Rc=0.5, rho2=1.5, rhoK=1.0, Gg=1.0,
            rho_layer2=1.0, forcing=forcing,
        )
        assert B2[7] == pytest.approx(5.0)  # 2*2+1

    def test_no_forcing_for_different_mode(self):
        I8 = np.eye(8, dtype=np.complex128)
        forcing = make_forcing(Td=1.0, n=2, m=2, F=1.0)
        _, B2 = assemble_bc_no_ocean(
            I8, I8, 2, 0,  # m=0, forcing is m=2
            gc=1.0, Rc=0.5, rho2=1.5, rhoK=1.0, Gg=1.0,
            rho_layer2=1.0, forcing=forcing,
        )
        assert B2[7] == pytest.approx(0.0)

    def test_bc_nonsingular(self):
        """B matrix should be non-singular for reasonable parameters."""
        I8 = np.eye(8, dtype=np.complex128)
        # Use a slightly perturbed Y (as if after some propagation)
        Y_cmb = I8.copy()
        Y_surf = I8 + 0.01 * np.random.RandomState(42).randn(8, 8)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        B, _ = assemble_bc_no_ocean(
            Y_cmb, Y_surf, 2, 0,
            gc=1.0, Rc=0.5, rho2=1.5, rhoK=1.0, Gg=1.0,
            rho_layer2=1.0, forcing=forcing,
        )
        assert abs(np.linalg.det(B)) > 1e-20


# ---------------------------------------------------------------------------
# Full integration
# ---------------------------------------------------------------------------

class TestGetSolution:

    def test_io_solution_shape(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        numerics, model = set_boundary_indices(numerics, io_model)
        model = get_rheology(model, io_forcing)
        y_sol, r_grid, Y, Aprop_aux = get_solution(model, io_forcing, numerics)
        Nr = numerics.Nr
        assert y_sol.shape == (Nr + 1, 8)
        assert r_grid.shape == (Nr + 1,)
        assert Y.shape == (Nr + 1, 8, 8)

    def test_io_solution_finite(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        numerics, model = set_boundary_indices(numerics, io_model)
        model = get_rheology(model, io_forcing)
        y_sol, _, _, _ = get_solution(model, io_forcing, numerics)
        assert np.all(np.isfinite(y_sol))

    def test_surface_stress_free(self, io_model, io_forcing):
        """R, S, T should be approximately zero at the surface."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        numerics, model = set_boundary_indices(numerics, io_model)
        model = get_rheology(model, io_forcing)
        y_sol, _, _, _ = get_solution(model, io_forcing, numerics)
        # R, S, T at surface
        R_surf = abs(y_sol[-1, 3])
        S_surf = abs(y_sol[-1, 4])
        T_surf = abs(y_sol[-1, 5])
        y_max = np.max(np.abs(y_sol[-1, :]))
        assert R_surf / y_max < 1e-6
        assert S_surf / y_max < 1e-6
        assert T_surf / y_max < 1e-6

    def test_radial_grid_endpoints(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        numerics, model = set_boundary_indices(numerics, io_model)
        model = get_rheology(model, io_forcing)
        _, r_grid, _, _ = get_solution(model, io_forcing, numerics)
        # First point is the core radius
        assert r_grid[0] == pytest.approx(float(model.R[0]))
        # Last point is the surface radius
        assert r_grid[-1] == pytest.approx(float(model.R[3]))

    def test_r_monotonically_increasing(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        numerics, model = set_boundary_indices(numerics, io_model)
        model = get_rheology(model, io_forcing)
        _, r_grid, _, _ = get_solution(model, io_forcing, numerics)
        assert np.all(np.diff(r_grid) > 0)


# ---------------------------------------------------------------------------
# Analytical validation: uniform elastic sphere
# ---------------------------------------------------------------------------

class TestUniformSphere:

    def test_h2_analytical(self, uniform_elastic, uniform_forcing):
        """h₂ for fluid-core + elastic-shell model.

        For a tiny fluid core, h₂ ≈ 5/(2*(1 + 19μ/(2ρgR))) from the
        uniform solid sphere formula.  The displacement is insensitive to
        the (tiny) core, so this comparison works within a few percent.
        """
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, model = set_boundary_indices(numerics, uniform_elastic)
        model = get_rheology(model, uniform_forcing)

        y_sol, _, _, _ = get_solution(model, uniform_forcing, numerics)

        gs_surface = float(model.gs[1])
        U_surf = y_sol[-1, 0]
        h2_numerical = -gs_surface * complex(U_surf)

        R_m = 1000.0e3
        rho = 3000.0
        mu = 1e10
        from pylov3d.constants import G as G_phys
        g_surf = G_phys * (4.0 / 3.0) * math.pi * rho * R_m
        h2_analytical = 5.0 / (2.0 * (1.0 + 19.0 * mu / (2.0 * rho * g_surf * R_m)))

        assert abs(h2_numerical.real) == pytest.approx(h2_analytical, rel=0.05)

    def test_bc_satisfaction(self, uniform_elastic, uniform_forcing):
        """All 8 boundary conditions should be satisfied to machine precision."""
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, model = set_boundary_indices(numerics, uniform_elastic)
        model = get_rheology(model, uniform_forcing)

        y_sol, _, _, _ = get_solution(model, uniform_forcing, numerics)

        Gg = model.Gg
        Rc = float(model.R[0])
        gc = Gg * (4.0 / 3.0) * math.pi * float(model.rho[0]) * Rc
        rho2 = float(model.Delta_rho[0]) + float(model.rho[1])
        rhoK = float(model.rho[1])
        n = 2

        # CMB: U - R/(gc*rho2) + Phi/gc = 0
        bc1 = y_sol[0, 0] - y_sol[0, 3] / (gc * rho2) + y_sol[0, 6] / gc
        assert abs(bc1) < 1e-10

        # CMB: S = 0
        assert abs(y_sol[0, 4]) < 1e-10

        # CMB: T = 0
        assert abs(y_sol[0, 5]) < 1e-10

        # Surface: R = 0
        assert abs(y_sol[-1, 3]) < 1e-10

        # Surface: S = 0
        assert abs(y_sol[-1, 4]) < 1e-10

        # Surface: T = 0
        assert abs(y_sol[-1, 5]) < 1e-10

        # Surface: BC8
        bc8 = (4 * math.pi * Gg * rhoK * y_sol[-1, 0]
               + (n + 1) * y_sol[-1, 6] + y_sol[-1, 7])
        assert abs(bc8 - 5.0) < 1e-10

    def test_k2_positive_real(self, uniform_elastic, uniform_forcing):
        """k₂ for an elastic body should be positive and real."""
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, model = set_boundary_indices(numerics, uniform_elastic)
        model = get_rheology(model, uniform_forcing)

        y_sol, _, _, _ = get_solution(model, uniform_forcing, numerics)
        k2 = complex(y_sol[-1, 6]) - 1.0

        assert k2.real > 0
        assert abs(k2.imag) < 1e-10

    def test_matches_scipy(self, uniform_elastic, uniform_forcing):
        """CK-RK5 integration should match scipy.integrate.solve_ivp."""
        from scipy.integrate import solve_ivp

        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, model = set_boundary_indices(numerics, uniform_elastic)
        model = get_rheology(model, uniform_forcing)

        y_sol, _, Y_ck, _ = get_solution(model, uniform_forcing, numerics)

        # Independently integrate with scipy
        Gg = model.Gg
        muC_k = complex(model.muC[1])
        lam_k = complex(model.lam[1])
        rho_k = float(model.rho[1])
        R_inner = float(model.R[0])
        M_inner = (4.0 / 3.0) * math.pi * float(model.rho[0]) * R_inner ** 3

        def ode_func(r, Y_flat):
            Y = Y_flat.reshape(8, 8)
            g, dg = compute_gravity(r, rho_k, M_inner, R_inner, Gg)
            Ap = build_aprop(r, g, dg, 2, muC_k, lam_k, rho_k, Gg)
            return (Ap @ Y).flatten()

        Y0 = np.eye(8, dtype=complex).flatten()
        sol = solve_ivp(ode_func,
                        (float(model.R[0]), float(model.R[1])),
                        Y0, method='RK45', rtol=1e-10, atol=1e-12,
                        t_eval=[float(model.R[1])])
        Y_scipy = sol.y[:, -1].reshape(8, 8)

        # Compare fundamental matrices at the surface
        rel_err = np.linalg.norm(Y_ck[-1] - Y_scipy) / np.linalg.norm(Y_scipy)
        assert rel_err < 1e-4  # CK-RK5 with 300 points vs adaptive RK45
