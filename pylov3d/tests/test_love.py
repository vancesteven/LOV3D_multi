"""Tests for pylov3d.love — pipeline orchestrator and Love number extraction."""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.love import get_love, extract_love_numbers
from pylov3d.constants import G as G_phys


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def io_model():
    """Io 4-layer model."""
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
    R = 1000.0
    rho = 3000.0
    mu = 1e10
    return make_interior_model(
        R0_km=[10.0, R],
        rho0=[rho, rho],
        mu0=[0.0, mu],
        eta0=[None, None],
    )


@pytest.fixture
def uniform_forcing():
    return make_forcing(Td=86400.0, n=2, m=0, F=1.0)


# ---------------------------------------------------------------------------
# get_love pipeline
# ---------------------------------------------------------------------------

class TestGetLove:

    def test_returns_correct_types(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        love, y_rad, model = get_love(io_model, io_forcing, numerics)

        from pylov3d.types import LoveSpectra, RadialSolution, InteriorModel
        assert isinstance(love, LoveSpectra)
        assert isinstance(y_rad, RadialSolution)
        assert isinstance(model, InteriorModel)

    def test_love_spectra_shape(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        assert love.k.shape == (1,)
        assert love.h.shape == (1,)
        assert love.l.shape == (1,)
        assert love.nf == 2
        assert love.mf == 0

    def test_y_rad_shape(self, io_model, io_forcing):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        _, y_rad, _ = get_love(io_model, io_forcing, numerics)
        assert y_rad.r.ndim == 1
        assert y_rad.y.ndim == 2
        assert y_rad.y.shape[1] == 8
        assert len(y_rad.r) == len(y_rad.y)

    def test_love_matches_direct_solver(self, io_model, io_forcing):
        """get_love should give same result as calling solver directly."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        love, _, model = get_love(io_model, io_forcing, numerics)

        # Direct path
        numerics2 = make_numerics(n_layers=4, method="combination", Nrbase=50)
        numerics2, model2 = set_boundary_indices(numerics2, io_model)
        model2 = get_rheology(model2, io_forcing)
        y_sol, _, _, _ = get_solution(model2, io_forcing, numerics2)

        gs = float(model2.gs[3])
        k_direct = complex(y_sol[-1, 6]) - 1.0
        h_direct = -gs * complex(y_sol[-1, 0])
        l_direct = -gs * complex(y_sol[-1, 1])

        assert complex(love.k[0]) == pytest.approx(k_direct, rel=1e-12)
        assert complex(love.h[0]) == pytest.approx(h_direct, rel=1e-12)
        assert complex(love.l[0]) == pytest.approx(l_direct, rel=1e-12)


# ---------------------------------------------------------------------------
# Love number physics
# ---------------------------------------------------------------------------

class TestLovePhysics:

    def test_io_k2_has_imaginary_part(self, io_model, io_forcing):
        """Io is viscoelastic → k₂ should have nonzero imaginary part."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        k2 = complex(love.k[0])
        assert abs(k2.imag) > 1e-6

    def test_io_k2_dissipative(self, io_model, io_forcing):
        """Im(k₂) should be negative for a dissipative body."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        k2 = complex(love.k[0])
        assert k2.imag < 0

    def test_elastic_k2_real(self, uniform_elastic, uniform_forcing):
        """k₂ for an elastic body should be purely real."""
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        love, _, _ = get_love(uniform_elastic, uniform_forcing, numerics)
        k2 = complex(love.k[0])
        assert abs(k2.imag) < 1e-10
        assert k2.real > 0

    def test_elastic_h2_analytical(self, uniform_elastic, uniform_forcing):
        """h₂ for a near-uniform elastic sphere.

        h₂ ≈ 5/(2(1 + 19μ/(2ρgR))) from the uniform solid sphere formula.
        """
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        love, _, _ = get_love(uniform_elastic, uniform_forcing, numerics)
        h2 = complex(love.h[0])

        R_m = 1000.0e3
        rho = 3000.0
        mu = 1e10
        g_surf = G_phys * (4.0 / 3.0) * math.pi * rho * R_m
        h2_analytical = 5.0 / (2.0 * (1.0 + 19.0 * mu / (2.0 * rho * g_surf * R_m)))

        assert abs(h2.real) == pytest.approx(h2_analytical, rel=0.05)

    def test_h2_nonzero(self, io_model, io_forcing):
        """h₂ should be nonzero for a viscoelastic body."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        h2 = complex(love.h[0])
        assert abs(h2) > 1e-6

    def test_l2_nonzero(self, io_model, io_forcing):
        """l₂ should be nonzero for a viscoelastic body."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
        love, _, _ = get_love(io_model, io_forcing, numerics)
        l2 = complex(love.l[0])
        assert abs(l2) > 1e-6

    def test_forcing_list(self, io_model, io_forcing):
        """get_love should accept a list of forcings."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)
        love, _, _ = get_love(io_model, [io_forcing], numerics)
        assert love.k.shape == (1,)


# ---------------------------------------------------------------------------
# extract_love_numbers directly
# ---------------------------------------------------------------------------

class TestExtractLoveNumbers:

    def test_identity_solution(self):
        """Known surface values should give predictable Love numbers."""
        # Construct a fake y_sol where surface Φ = 1.5, U = 0.1, V = 0.05
        y_sol = np.zeros((10, 8), dtype=np.complex128)
        y_sol[-1, 0] = 0.1   # U
        y_sol[-1, 1] = 0.05  # V
        y_sol[-1, 6] = 1.5   # Φ

        model = make_interior_model(
            R0_km=[500.0, 1000.0],
            rho0=[3000.0, 3000.0],
            mu0=[0.0, 1e10],
        )
        # Set gs for the surface layer
        model = model._replace(
            gs=model.gs.at[1].set(1.0),
            n_layers=2,
        )
        forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)

        love = extract_love_numbers(y_sol, model, forcing)
        assert complex(love.k[0]) == pytest.approx(0.5)       # 1.5 - 1
        assert complex(love.h[0]) == pytest.approx(-0.1)      # -1.0 * 0.1
        assert complex(love.l[0]) == pytest.approx(-0.05)     # -1.0 * 0.05
