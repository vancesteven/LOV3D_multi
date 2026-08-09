# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for coupled (multi-mode) Love number extraction."""


import numpy as np
import pytest

from pylov3d.types import (
    make_interior_model,
    make_forcing,
    make_numerics,
    LoveSpectra,
    LateralRheology,
)
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.couplings import Couplings, get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.solver import get_solution, _get_solution_coupled
from pylov3d.love import extract_love_numbers, get_love


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_io_pipeline(Nrbase=20):
    """Build Io 3-layer model through the full pipeline."""
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
# extract_love_numbers — coupled
# ---------------------------------------------------------------------------

class TestExtractLoveNumbersCoupled:

    def test_single_mode_matches_1d(self):
        """N=1 coupled extraction matches 1D."""
        model, forcing, numerics = _make_io_pipeline()

        # 1D reference
        y_ref, r_ref, _, _ = get_solution(model, forcing, numerics)
        love_ref = extract_love_numbers(y_ref, model, forcing)

        # Coupled N=1
        couplings = Couplings(
            n_s=np.array([2]), m_s=np.array([0]),
            order=np.array([0]), Coup=np.zeros((1, 1, 27, 1)),
        )
        lateral = LateralRheology(
            variations=np.zeros((1, 2), dtype=int),
            muC_amp=np.zeros((model.n_layers, 1), dtype=complex),
            K_amp=np.zeros((model.n_layers, 1), dtype=complex),
            uniform=np.ones(model.n_layers, dtype=bool),
        )
        y_coup, _, _, _ = _get_solution_coupled(
            model, forcing, numerics, couplings, lateral,
        )
        love_coup = extract_love_numbers(y_coup, model, forcing, couplings)

        assert love_coup.k[0] == pytest.approx(love_ref.k[0], abs=1e-10)
        assert love_coup.h[0] == pytest.approx(love_ref.h[0], abs=1e-10)
        assert love_coup.l[0] == pytest.approx(love_ref.l[0], abs=1e-10)

    def test_multi_mode_shapes(self):
        """Coupled Love numbers have correct array shapes."""
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)
        N = len(C.n_s)

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )
        love = extract_love_numbers(y_sol, model, forcing, couplings=C)

        assert isinstance(love, LoveSpectra)
        assert len(love.k) == N
        assert len(love.h) == N
        assert len(love.l) == N
        assert len(love.n) == N
        assert len(love.m) == N
        assert love.nf == 2
        assert love.mf == 0

    def test_forcing_mode_has_subtracted_one(self):
        """k for the forcing mode should have the -1 subtraction."""
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )
        love = extract_love_numbers(y_sol, model, forcing, couplings=C)

        # Compare forcing-mode k with raw Phi
        k_f_idx = np.where((C.n_s == 2) & (C.m_s == 0))[0][0]
        N = len(C.n_s)
        Phi_f = complex(y_sol[-1, 6 * N + 2 * k_f_idx])
        assert love.k[k_f_idx] == pytest.approx(Phi_f - 1.0, abs=1e-14)

        # Non-forcing modes should NOT have the -1
        for k in range(N):
            if k != k_f_idx:
                Phi_k = complex(y_sol[-1, 6 * N + 2 * k])
                assert love.k[k] == pytest.approx(Phi_k, abs=1e-14)

    def test_forcing_mode_dominates(self):
        """Forcing mode should have the largest response."""
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)
        N = len(C.n_s)

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )
        love = extract_love_numbers(y_sol, model, forcing, couplings=C)

        k_f_idx = np.where((C.n_s == 2) & (C.m_s == 0))[0][0]
        h_forcing = abs(love.h[k_f_idx])
        for k in range(N):
            if k != k_f_idx and int(C.n_s[k]) > 0:
                assert abs(love.h[k]) < h_forcing

    def test_finite(self):
        model, forcing, numerics = _make_io_pipeline()

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable={1: [(2, 0, 0.1)]},
        )
        C = get_couplings(lateral.variations, 2, 0)

        y_sol, _, _, _ = get_solution(
            model, forcing, numerics, couplings=C, lateral=lateral,
        )
        love = extract_love_numbers(y_sol, model, forcing, couplings=C)

        assert np.all(np.isfinite(love.k))
        assert np.all(np.isfinite(love.h))
        assert np.all(np.isfinite(love.l))


# ---------------------------------------------------------------------------
# get_love — lateral variation pipeline
# ---------------------------------------------------------------------------

class TestGetLoveLateral:

    def test_no_lateral_matches_1d(self):
        """get_love without lateral variations matches 1D."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)

        love_1d, _, _ = get_love(raw_model, forcing, numerics)
        assert len(love_1d.k) == 1

    def test_lateral_produces_multi_mode(self):
        """get_love with mu_variable produces multi-mode Love numbers."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)

        love, y_rad, model = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        assert len(love.k) > 1
        assert len(love.n) == len(love.k)
        assert 2 in love.n  # forcing mode present
        assert love.nf == 2
        assert love.mf == 0

    def test_radial_solution_shape(self):
        """RadialSolution should contain all modes."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)

        love, y_rad, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        N = len(love.k)
        assert y_rad.y.shape[1] == 8 * N
        assert len(y_rad.n_s) == N
        assert len(y_rad.m_s) == N

    def test_small_perturbation_close_to_1d(self):
        """Very small lateral variation should give Love numbers close to 1D."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)

        love_1d, _, _ = get_love(raw_model, forcing, numerics)
        love_3d, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.001)]},
        )

        # Find forcing mode in 3D result
        k_f_idx = np.where((love_3d.n == 2) & (love_3d.m == 0))[0][0]
        assert love_3d.k[k_f_idx] == pytest.approx(love_1d.k[0], rel=0.05)
        assert love_3d.h[k_f_idx] == pytest.approx(love_1d.h[0], rel=0.05)
