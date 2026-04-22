"""Tests for pylov3d.energy — strain matrices, stress/strain, dissipation."""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.energy import (
    build_A14_A15,
    compute_stress_strain,
    get_energy,
    global_dissipation,
)
from pylov3d.propagator import build_A1_A2, build_A3
from pylov3d.constants import G as G_phys


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def io_forcing():
    omega0 = 4.1086e-05
    Td = 2 * math.pi / omega0
    return make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5))


@pytest.fixture
def uniform_elastic():
    return make_interior_model(
        R0_km=[10.0, 1000.0],
        rho0=[3000.0, 3000.0],
        mu0=[0.0, 1e10],
        eta0=[None, None],
    )


@pytest.fixture
def uniform_forcing():
    return make_forcing(Td=86400.0, n=2, m=0, F=1.0)


def _run_solver(model, forcing, Nrbase=100, method="combination"):
    n_layers = model.n_layers
    numerics = make_numerics(n_layers=n_layers, method=method, Nrbase=Nrbase)
    numerics, model = set_boundary_indices(numerics, model)
    model = get_rheology(model, forcing)
    y_sol, r_grid, Y, Aprop_aux = get_solution(model, forcing, numerics)
    return y_sol, r_grid, Aprop_aux, model, numerics


# ---------------------------------------------------------------------------
# A14/A15 strain matrices
# ---------------------------------------------------------------------------

class TestBuildA14A15:

    def test_shape_n2(self):
        A14, A15 = build_A14_A15(2)
        assert A14.shape == (6, 3)
        assert A15.shape == (6, 3)

    def test_shape_n0(self):
        A14, A15 = build_A14_A15(0)
        assert A14.shape == (6, 3)
        assert A15.shape == (6, 3)

    def test_n2_finite(self):
        A14, A15 = build_A14_A15(2)
        assert np.all(np.isfinite(A14))
        assert np.all(np.isfinite(A15))

    def test_n2_real(self):
        """Strain matrices are purely geometric (real-valued)."""
        A14, A15 = build_A14_A15(2)
        assert np.allclose(A14.imag, 0.0)
        assert np.allclose(A15.imag, 0.0)

    def test_n0_mostly_zero(self):
        """n=0 should have only two nonzero rows (ε_{0,0,0} and ε_{0,2,2})."""
        A14, A15 = build_A14_A15(0)
        # Only rows 0 and 5 have nonzero entries
        for i in [1, 2, 3, 4]:
            assert np.allclose(A14[i], 0.0)
            assert np.allclose(A15[i], 0.0)
        assert np.any(A14[0] != 0)
        assert np.any(A14[5] != 0)

    def test_n2_eps_nn0(self):
        """ε_{n,n,0} row for n=2: known analytical values."""
        A14, A15 = build_A14_A15(2)
        s5 = math.sqrt(5)
        s2 = math.sqrt(2)
        s3 = math.sqrt(3)
        c0 = 1.0 / s3 / s5
        assert A14[0, 0] == pytest.approx(-c0 * s2)
        assert A14[0, 2] == pytest.approx(c0 * s3)
        assert A15[0, 0] == pytest.approx(c0 * 1 * s2)   # (n-1)*sqrt(n), n=2
        assert A15[0, 2] == pytest.approx(c0 * s3 * 4)    # sqrt(n+1)*(n+2), n=2

    def test_n2_eps_nm2_2(self):
        """ε_{n,n-2,2} row for n=2."""
        A14, A15 = build_A14_A15(2)
        c1 = math.sqrt(1 / 3)
        assert A14[1, 0] == pytest.approx(c1)
        assert A14[1, 1] == pytest.approx(0.0)
        assert A14[1, 2] == pytest.approx(0.0)
        assert A15[1, 0] == pytest.approx(c1 * 2)

    def test_higher_degrees(self):
        """Matrices should be finite for various degrees."""
        for n in [1, 3, 5, 10]:
            A14, A15 = build_A14_A15(n)
            assert np.all(np.isfinite(A14))
            assert np.all(np.isfinite(A15))

    def test_A1_A2_proportional_to_mu(self):
        """A1/A2 scale with μ, while A14/A15 are μ-independent.

        For rows 1-5 (deviatoric), A1[j,k] should scale as 2μ·f(n)
        while A14[j,k] = f'(n) is purely geometric.
        Doubling μ should double A1 for deviatoric rows.
        """
        n = 2
        lam1 = 1.0 + 0j
        mu1 = 1.0 + 0j
        mu2 = 2.0 + 0j
        # lam adjusts with mu: lam = K - 2/3 mu
        lam2 = lam1 + 2.0 / 3.0 * (mu1 - mu2)

        A1_1, _ = build_A1_A2(n, mu1, lam1)
        A1_2, _ = build_A1_A2(n, mu2, lam2)
        A14, _ = build_A14_A15(n)

        # Deviatoric rows (1-5) of A1 should scale with μ
        for j in [1, 2, 3, 4, 5]:
            for k in range(3):
                if abs(A1_1[j, k]) > 1e-14:
                    ratio = A1_2[j, k] / A1_1[j, k]
                    assert ratio == pytest.approx(mu2 / mu1, rel=1e-10)

        # A14/A15 should be the same regardless of μ
        A14_check, _ = build_A14_A15(n)
        np.testing.assert_allclose(A14, A14_check)


# ---------------------------------------------------------------------------
# Stress and strain computation
# ---------------------------------------------------------------------------

class TestComputeStressStrain:

    def test_shapes(self, io_model, io_forcing):
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        u_gsh, stress, strain = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        Nr = numerics.Nr
        assert u_gsh.shape == (Nr + 1, 3)
        assert stress.shape == (Nr + 1, 6)
        assert strain.shape == (Nr + 1, 6)

    def test_finite(self, io_model, io_forcing):
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        u_gsh, stress, strain = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        assert np.all(np.isfinite(u_gsh))
        assert np.all(np.isfinite(stress))
        assert np.all(np.isfinite(strain))

    def test_core_zero(self, io_model, io_forcing):
        """Stress and strain should be zero at the core (fluid layer 0)."""
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        _, stress, strain = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        np.testing.assert_allclose(stress[0], 0.0)
        np.testing.assert_allclose(strain[0], 0.0)

    def test_elastic_stress_real(self, uniform_elastic, uniform_forcing):
        """For an elastic body, stress and strain should be real."""
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            uniform_elastic, uniform_forcing, Nrbase=300, method="variable",
        )
        _, stress, strain = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, uniform_forcing, numerics,
        )
        # Skip first point (core) and check mantle
        assert np.allclose(stress[1:].imag, 0.0, atol=1e-10)
        assert np.allclose(strain[1:].imag, 0.0, atol=1e-10)

    def test_viscoelastic_stress_complex(self, io_model, io_forcing):
        """Io's viscoelastic layers should produce complex stress."""
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        _, stress, _ = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        # Some interior point should have nonzero imaginary part
        assert np.any(np.abs(stress.imag) > 1e-10)

    def test_u_gsh_consistent_with_A3(self, io_model, io_forcing):
        """u_gsh should satisfy [U, V, W] = A3 @ u_gsh."""
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        u_gsh, _, _ = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        A3 = build_A3(2)
        # Check at a few interior points (skip core)
        for k in [5, 20, 50]:
            if k < len(r_grid):
                UVW_from_sol = y_sol[k, :3]
                UVW_from_gsh = A3 @ u_gsh[k]
                np.testing.assert_allclose(UVW_from_sol, UVW_from_gsh, atol=1e-12)


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

class TestGetEnergy:

    def test_returns_energy_spectra(self, io_model, io_forcing):
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        energy = get_energy(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        from pylov3d.types import EnergySpectra
        assert isinstance(energy, EnergySpectra)

    def test_profile_shape(self, io_model, io_forcing):
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        energy = get_energy(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        Nr = numerics.Nr
        assert energy.energy_profile.shape == (Nr + 1, 1)
        assert energy.energy_integral.shape == (1,)

    def test_elastic_zero_dissipation(self, uniform_elastic, uniform_forcing):
        """An elastic body should have zero dissipation."""
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            uniform_elastic, uniform_forcing, Nrbase=300, method="variable",
        )
        energy = get_energy(
            y_sol, r_grid, Aprop_aux, model, uniform_forcing, numerics,
        )
        # Im(σ*:ε) should vanish for elastic (real) stress and strain
        assert abs(energy.energy_integral[0]) < 1e-15

    def test_io_nonzero_dissipation(self, io_model, io_forcing):
        """Io should have nonzero dissipation."""
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=100,
        )
        energy = get_energy(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        assert abs(energy.energy_integral[0]) > 1e-20

    def test_profile_finite(self, io_model, io_forcing):
        y_sol, r_grid, Aprop_aux, model, numerics = _run_solver(
            io_model, io_forcing, Nrbase=50,
        )
        energy = get_energy(
            y_sol, r_grid, Aprop_aux, model, io_forcing, numerics,
        )
        assert np.all(np.isfinite(energy.energy_profile))


# ---------------------------------------------------------------------------
# Global dissipation formula
# ---------------------------------------------------------------------------

class TestGlobalDissipation:

    def test_elastic_zero(self):
        """No dissipation when Im(k) = 0."""
        k = 0.3 + 0j  # purely real
        assert global_dissipation(k, 1e-5, 1e6, 3000.0) == pytest.approx(0.0)

    def test_dissipative_positive(self):
        """Im(k) < 0 should give positive dissipation (heating)."""
        k = 0.3 - 0.01j
        E = global_dissipation(k, 1e-5, 1e6, 3000.0)
        assert E > 0

    def test_scales_with_imk(self):
        """Dissipation should scale linearly with -Im(k)."""
        k1 = 0.3 - 0.01j
        k2 = 0.3 - 0.02j
        E1 = global_dissipation(k1, 1e-5, 1e6, 3000.0)
        E2 = global_dissipation(k2, 1e-5, 1e6, 3000.0)
        assert E2 == pytest.approx(2 * E1, rel=1e-10)

    def test_scales_with_omega(self):
        """Dissipation should scale linearly with ω."""
        k = 0.3 - 0.01j
        E1 = global_dissipation(k, 1e-5, 1e6, 3000.0)
        E2 = global_dissipation(k, 2e-5, 1e6, 3000.0)
        assert E2 == pytest.approx(2 * E1, rel=1e-10)
