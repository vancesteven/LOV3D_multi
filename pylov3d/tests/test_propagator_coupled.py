# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for coupled (multi-mode) propagator matrices."""


import numpy as np
import pytest

from pylov3d.propagator import (
    build_aprop,
    build_aprop_coupled,
    build_A1_A2,
    build_A1_A2_coupled,
    _a1a2_geometric,
    _coupling_A1_A2,
)
from pylov3d.couplings import get_couplings


# ---------------------------------------------------------------------------
# _a1a2_geometric
# ---------------------------------------------------------------------------

class TestA1A2Geometric:

    def test_shape(self):
        A1g, A2g = _a1a2_geometric(2)
        assert A1g.shape == (6, 3)
        assert A2g.shape == (6, 3)

    def test_n0_only_rows_0_5(self):
        """For n=0, only rows 0 and 5 should be nonzero."""
        A1g, A2g = _a1a2_geometric(0)
        for row in [1, 2, 3, 4]:
            assert np.allclose(A1g[row, :], 0)
            assert np.allclose(A2g[row, :], 0)
        assert np.max(np.abs(A1g[0, :])) > 0 or np.max(np.abs(A1g[5, :])) > 0

    def test_n2_matches_build_A1_A2(self):
        """Geometric × material should reproduce build_A1_A2."""
        n = 2
        muC = 0.5 + 0.1j
        lam = 0.3 + 0.05j

        A1_ref, A2_ref = build_A1_A2(n, muC, lam)
        A1g, A2g = _a1a2_geometric(n)

        # Row 0: scale by (3*lam + 2*muC)
        mat0 = 3 * lam + 2 * muC
        assert np.allclose(mat0 * A1g[0, :], A1_ref[0, :])
        assert np.allclose(mat0 * A2g[0, :], A2_ref[0, :])

        # Rows 1-5: scale by 2*muC
        mat1 = 2 * muC
        for row in range(1, 6):
            assert np.allclose(mat1 * A1g[row, :], A1_ref[row, :], atol=1e-14)
            assert np.allclose(mat1 * A2g[row, :], A2_ref[row, :], atol=1e-14)

    def test_n4_matches(self):
        """Check for degree 4 as well."""
        n = 4
        muC = 1.0 + 0.2j
        lam = 0.5
        A1_ref, A2_ref = build_A1_A2(n, muC, lam)
        A1g, A2g = _a1a2_geometric(n)

        mat0 = 3 * lam + 2 * muC
        assert np.allclose(mat0 * A1g[0, :], A1_ref[0, :])
        mat1 = 2 * muC
        for row in range(1, 6):
            assert np.allclose(mat1 * A1g[row, :], A1_ref[row, :], atol=1e-14)


# ---------------------------------------------------------------------------
# _coupling_A1_A2
# ---------------------------------------------------------------------------

class TestCouplingA1A2:

    def test_zero_coupling(self):
        """Zero coupling coefficients → zero contribution."""
        Cp = np.zeros(26)
        A1c, A2c = _coupling_A1_A2(2, 0.1, 0.2, Cp)
        assert np.allclose(A1c, 0)
        assert np.allclose(A2c, 0)

    def test_shape(self):
        Cp = np.ones(26)
        A1c, A2c = _coupling_A1_A2(2, 0.1, 0.2, Cp)
        assert A1c.shape == (6, 3)
        assert A2c.shape == (6, 3)

    def test_slot0_uses_K(self):
        """Slot 0 should use K_nm, not mu_nm."""
        Cp = np.zeros(26)
        Cp[0] = 1.0
        A1c_K, _ = _coupling_A1_A2(2, 1.0, 0.0, Cp)
        A1c_mu, _ = _coupling_A1_A2(2, 0.0, 1.0, Cp)
        # Only row 0 should be nonzero, and it should use K_nm
        assert np.max(np.abs(A1c_K[0, :])) > 0
        assert np.allclose(A1c_mu, 0)

    def test_finite(self):
        Cp = np.random.randn(26)
        A1c, A2c = _coupling_A1_A2(2, 0.1+0.01j, 0.2+0.02j, Cp)
        assert np.all(np.isfinite(A1c))
        assert np.all(np.isfinite(A2c))


# ---------------------------------------------------------------------------
# build_A1_A2_coupled
# ---------------------------------------------------------------------------

class TestBuildA1A2Coupled:

    def test_single_mode_matches_diagonal(self):
        """N=1 with zero coupling should match single-mode build_A1_A2."""
        n_s = np.array([2])
        muC = 0.5 + 0.1j
        lam = 0.3 + 0.05j
        Coup = np.zeros((1, 1, 27, 1))
        muC_amp = np.zeros(1, dtype=complex)
        K_amp = np.zeros(1, dtype=complex)

        A1, A2 = build_A1_A2_coupled(n_s, muC, lam, Coup, muC_amp, K_amp)
        A1_ref, A2_ref = build_A1_A2(2, muC, lam)

        assert A1.shape == (6, 3)
        assert np.allclose(A1, A1_ref)
        assert np.allclose(A2, A2_ref)

    def test_multi_mode_shape(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0, perturbation_order=2)
        muC_amp = np.array([0.1 + 0.01j])
        K_amp = np.array([0.0])

        A1, A2 = build_A1_A2_coupled(
            C.n_s, 0.5+0.1j, 0.3, C.Coup, muC_amp, K_amp,
        )
        N = len(C.n_s)
        assert A1.shape == (6 * N, 3 * N)
        assert A2.shape == (6 * N, 3 * N)

    def test_diagonal_blocks_present(self):
        """Diagonal blocks should always be nonzero (for n>0 modes)."""
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        A1, _ = build_A1_A2_coupled(
            C.n_s, 0.5+0.1j, 0.3, C.Coup,
            np.array([0.1]), np.array([0.0]),
        )
        for k in range(len(C.n_s)):
            n = C.n_s[k]
            if n > 0:
                block = A1[6*k:6*k+6, 3*k:3*k+3]
                assert np.max(np.abs(block)) > 0

    def test_off_diagonal_nonzero_with_coupling(self):
        """With nonzero coupling, some off-diagonal blocks should be nonzero."""
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        A1, _ = build_A1_A2_coupled(
            C.n_s, 0.5+0.1j, 0.3, C.Coup,
            np.array([0.1 + 0.02j]), np.array([0.0]),
        )
        N = len(C.n_s)
        has_offdiag = False
        for i in range(N):
            for j in range(N):
                if i != j:
                    block = A1[6*i:6*i+6, 3*j:3*j+3]
                    if np.max(np.abs(block)) > 0:
                        has_offdiag = True
        assert has_offdiag

    def test_finite(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        A1, A2 = build_A1_A2_coupled(
            C.n_s, 0.5+0.1j, 0.3, C.Coup,
            np.array([0.1]), np.array([0.0]),
        )
        assert np.all(np.isfinite(A1))
        assert np.all(np.isfinite(A2))


# ---------------------------------------------------------------------------
# build_aprop_coupled
# ---------------------------------------------------------------------------

class TestBuildApropCoupled:

    def test_single_mode_matches_scalar(self):
        """N=1 coupled propagator should match single-mode build_aprop."""
        n = 2
        muC = 0.5 + 0.1j
        lam = 0.3 + 0.05j
        rho = 1.0
        Gg = 0.5
        r = 0.8
        g = 0.4
        dg = 0.1

        Aprop_ref = build_aprop(r, g, dg, n, muC, lam, rho, Gg)

        n_s = np.array([n])
        Coup = np.zeros((1, 1, 27, 1))
        Aprop = build_aprop_coupled(
            r, g, dg, n_s, muC, lam, rho, Gg,
            Coup, np.zeros(1), np.zeros(1),
        )

        assert Aprop.shape == (8, 8)
        assert np.allclose(Aprop, Aprop_ref, atol=1e-12)

    def test_multi_mode_shape(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        N = len(C.n_s)

        Aprop = build_aprop_coupled(
            0.8, 0.4, 0.1, C.n_s, 0.5+0.1j, 0.3, 1.0, 0.5,
            C.Coup, np.array([0.1]), np.array([0.0]),
        )
        assert Aprop.shape == (8 * N, 8 * N)

    def test_finite(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)

        Aprop = build_aprop_coupled(
            0.8, 0.4, 0.1, C.n_s, 0.5+0.1j, 0.3, 1.0, 0.5,
            C.Coup, np.array([0.1+0.01j]), np.array([0.0]),
        )
        assert np.all(np.isfinite(Aprop))

    def test_coupling_modifies_propagator(self):
        """Nonzero coupling should modify the propagator vs zero coupling."""
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        args = (0.8, 0.4, 0.1, C.n_s, 0.5+0.1j, 0.3, 1.0, 0.5, C.Coup)

        Ap0 = build_aprop_coupled(*args, np.zeros(1), np.zeros(1))
        Ap1 = build_aprop_coupled(*args, np.array([0.1+0.02j]), np.zeros(1))

        # Should differ
        assert not np.allclose(Ap0, Ap1)

    def test_degree0_mode_handled(self):
        """Degree-0 mode should have V/W/S/T rows set to trivial."""
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        Aprop = build_aprop_coupled(
            0.8, 0.4, 0.1, C.n_s, 0.5+0.1j, 0.3, 1.0, 0.5,
            C.Coup, np.array([0.0]), np.array([0.0]),
        )

        # Find degree-0 mode index
        k0 = np.where(C.n_s == 0)[0]
        if len(k0) > 0:
            k = k0[0]
            # V row (displacement index 3*k+1) should be trivial: Aprop[row,row]=1
            v_row = 3 * k + 1
            assert Aprop[v_row, v_row] == pytest.approx(1.0, abs=1e-12)
