"""Tests for the coupled (multi-mode) solver and boundary conditions."""

import math

import numpy as np

from pylov3d.types import (
    make_interior_model,
    make_forcing,
    make_numerics,
    LateralRheology,
)
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.couplings import Couplings, get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.solver import (
    get_solution,
    _get_solution_coupled,
    cash_karp_increment_coupled,
)
from pylov3d.boundary_conditions import (
    assemble_bc_no_ocean,
    assemble_bc_no_ocean_coupled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_io_pipeline(Nrbase=20):
    """Build Io 3-layer model through the full pipeline.

    Returns (model, forcing, numerics) with model already normalized.
    """
    raw_model = make_interior_model(
        R0_km=[800.0, 1600.0, 1821.6],
        rho0=[5150.0, 3300.0, 3000.0],
        mu0=[0.0, 60e9, 65e9],
        eta0=[None, 1e19, None],
    )
    forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=3, method="combination", Nrbase=Nrbase)
    numerics, raw_model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(raw_model, forcing)
    return model, forcing, numerics


# ---------------------------------------------------------------------------
# cash_karp_increment_coupled
# ---------------------------------------------------------------------------

class TestCashKarpIncrementCoupled:

    def test_shape(self):
        model, forcing, numerics = _make_io_pipeline()
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        N = len(C.n_s)

        r = float(model.R[0])
        rho = float(model.rho[1])
        M = (4.0 / 3.0) * math.pi * float(model.rho[0]) * r ** 3

        inc, Ap = cash_karp_increment_coupled(
            r, 0.01, C.n_s,
            complex(model.muC[1]), complex(model.lam[1]),
            rho, model.Gg, M, r,
            C.Coup, np.array([0.0]), np.array([0.0]),
        )
        assert inc.shape == (8 * N, 8 * N)
        assert Ap.shape == (8 * N, 8 * N)

    def test_finite(self):
        model, forcing, numerics = _make_io_pipeline()
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)

        r = float(model.R[0])
        rho = float(model.rho[1])
        M = (4.0 / 3.0) * math.pi * float(model.rho[0]) * r ** 3

        inc, Ap = cash_karp_increment_coupled(
            r, 0.01, C.n_s,
            complex(model.muC[1]), complex(model.lam[1]),
            rho, model.Gg, M, r,
            C.Coup, np.array([0.1 + 0.01j]), np.array([0.0]),
        )
        assert np.all(np.isfinite(inc))
        assert np.all(np.isfinite(Ap))


# ---------------------------------------------------------------------------
# assemble_bc_no_ocean_coupled
# ---------------------------------------------------------------------------

class TestAssembleBCCoupled:

    def test_shape(self):
        model, forcing, _ = _make_io_pipeline()
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        N = len(C.n_s)
        N8 = 8 * N

        Y_cmb = np.eye(N8, dtype=complex)
        Y_surf = np.eye(N8, dtype=complex)

        B, B2 = assemble_bc_no_ocean_coupled(
            Y_cmb, Y_surf, C.n_s, C.m_s,
            1.0, 0.5, 1.0, 1.0, 1.0, 1.0, forcing,
        )
        assert B.shape == (N8, N8)
        assert B2.shape == (N8,)

    def test_single_mode_matches_scalar(self):
        """N=1 coupled BCs should match scalar BCs."""
        forcing = make_forcing(Td=86400, n=2, m=0, F=1.0)
        n_s = np.array([2])
        m_s = np.array([0])

        rng = np.random.default_rng(42)
        Y = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        Y2 = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))

        B_ref, B2_ref = assemble_bc_no_ocean(
            Y, Y2, 2, 0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, forcing,
        )
        B_coup, B2_coup = assemble_bc_no_ocean_coupled(
            Y, Y2, n_s, m_s, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, forcing,
        )

        assert np.allclose(B_coup, B_ref)
        assert np.allclose(B2_coup, B2_ref)

    def test_forcing_only_in_forcing_mode(self):
        """B2 should be nonzero only at the forcing mode's surface BC row."""
        forcing = make_forcing(Td=86400, n=2, m=0, F=1.0)
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        N = len(C.n_s)
        N8 = 8 * N

        Y_cmb = np.eye(N8, dtype=complex)
        Y_surf = np.eye(N8, dtype=complex)

        B, B2 = assemble_bc_no_ocean_coupled(
            Y_cmb, Y_surf, C.n_s, C.m_s,
            1.0, 0.5, 1.0, 1.0, 1.0, 1.0, forcing,
        )

        # Forcing mode (n=2, m=0)
        forcing_k = np.where((C.n_s == 2) & (C.m_s == 0))[0][0]
        surf_base = 4 * N + 4 * forcing_k
        assert B2[surf_base + 3] != 0

        # All other entries should be zero
        for i in range(N8):
            if i != surf_base + 3:
                assert B2[i] == 0

    def test_nonsingular(self):
        """BC matrix built from realistic Y should be non-singular."""
        model, forcing, numerics = _make_io_pipeline(Nrbase=10)
        Nr = numerics.Nr

        # Run 1D solver to get realistic Y_cmb, Y_surf
        y_ref, r_ref, Y_ref, _ = get_solution(model, forcing, numerics)

        n_s = np.array([2])
        m_s = np.array([0])
        rhoC = float(model.rho[0])
        Rc = float(model.R[0])
        gc = model.Gg * (4.0 / 3.0) * math.pi * rhoC * Rc ** 3 / Rc ** 2

        B, B2 = assemble_bc_no_ocean_coupled(
            Y_ref[0], Y_ref[Nr], n_s, m_s,
            gc, Rc,
            float(model.Delta_rho[0]) + float(model.rho[1]),
            float(model.rho[model.n_layers - 1]),
            model.Gg, float(model.rho[1]), forcing,
        )
        assert abs(np.linalg.det(B)) > 1e-30


# ---------------------------------------------------------------------------
# get_solution coupled
# ---------------------------------------------------------------------------

class TestGetSolutionCoupled:

    def test_single_mode_matches_1d(self):
        """N=1 coupled solver should match 1D solver exactly."""
        model, forcing, numerics = _make_io_pipeline()

        # 1D reference
        y_ref, r_ref, Y_ref, Ap_ref = get_solution(model, forcing, numerics)

        # Coupled with N=1 and zero coupling
        n_s = np.array([2])
        m_s = np.array([0])
        Coup = np.zeros((1, 1, 27, 1))
        couplings = Couplings(
            n_s=n_s, m_s=m_s, order=np.array([0]), Coup=Coup,
        )
        lateral = LateralRheology(
            variations=np.zeros((1, 2), dtype=int),
            muC_amp=np.zeros((model.n_layers, 1), dtype=complex),
            K_amp=np.zeros((model.n_layers, 1), dtype=complex),
            uniform=np.ones(model.n_layers, dtype=bool),
        )

        y_coup, r_coup, Y_coup, _ = _get_solution_coupled(
            model, forcing, numerics, couplings, lateral,
        )

        assert np.allclose(r_coup, r_ref)
        assert np.allclose(y_coup, y_ref, atol=1e-10)

    def test_multi_mode_shape(self):
        model, forcing, numerics = _make_io_pipeline()
        Nr = numerics.Nr

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)
        N = len(C.n_s)

        y_sol, r_grid, Y_all, Ap_aux = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )

        assert y_sol.shape == (Nr + 1, 8 * N)
        assert r_grid.shape == (Nr + 1,)
        assert Y_all.shape == (Nr + 1, 8 * N, 8 * N)
        assert Ap_aux.shape == (Nr + 1, 3 * N, 8 * N)

    def test_finite(self):
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)

        y_sol, r_grid, Y_all, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )
        assert np.all(np.isfinite(y_sol))

    def test_forcing_mode_nonzero(self):
        """Forcing mode should have nonzero displacement at the surface."""
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )

        # Find the forcing mode (n=2, m=0)
        k_f = np.where((C.n_s == 2) & (C.m_s == 0))[0][0]
        # U at surface for forcing mode
        U_f = y_sol[-1, 3 * k_f]
        assert abs(U_f) > 0

    def test_coupling_excites_other_modes(self):
        """With coupling, non-forcing modes should have nonzero response."""
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)
        N = len(C.n_s)

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )

        # Non-forcing modes should have some nonzero response
        k_f = np.where((C.n_s == 2) & (C.m_s == 0))[0][0]
        N6 = 6 * N
        has_nonzero = False
        for k in range(N):
            if k != k_f and int(C.n_s[k]) > 0:
                U_k = y_sol[-1, 3 * k]
                Phi_k = y_sol[-1, N6 + 2 * k]
                if abs(U_k) > 1e-15 or abs(Phi_k) > 1e-15:
                    has_nonzero = True
                    break
        assert has_nonzero

    def test_surface_stress_free(self):
        """Surface radial stress R should be ~0 for all modes."""
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)
        N = len(C.n_s)
        N3 = 3 * N

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )

        for k in range(N):
            R_k = y_sol[-1, N3 + 3 * k]
            assert abs(R_k) < 1e-8, f"Mode {k} (n={C.n_s[k]}): R_surf={R_k}"

    def test_backward_compat_dispatch(self):
        """get_solution without couplings should use the 1D path."""
        model, forcing, numerics = _make_io_pipeline()

        y_sol, r_grid, Y, Ap = get_solution(model, forcing, numerics)
        assert y_sol.shape[1] == 8  # 1D path returns (Nr+1, 8)
