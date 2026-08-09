# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.propagator — sub-matrix builders and Aprop assembly."""

import math

import numpy as np
import pytest

from pylov3d.propagator import (
    build_A1_A2,
    build_A3,
    build_A4,
    build_A5,
    build_others,
    build_aprop,
    compute_gravity,
    CK_A,
    CK_B,
    CK_C,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def n2_params():
    """Parameters for a normalized n=2 elastic layer."""
    return dict(n=2, muC=1.0+0j, lam=1.0+0j, rho=1.0, Gg=1.0)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:

    def test_A1_A2_shape(self):
        A1, A2 = build_A1_A2(2, 1.0+0j, 1.0+0j)
        assert A1.shape == (6, 3)
        assert A2.shape == (6, 3)

    def test_A3_shape(self):
        assert build_A3(2).shape == (3, 3)

    def test_A4_shape(self):
        assert build_A4(2).shape == (3, 6)

    def test_A5_shape(self):
        assert build_A5(2).shape == (3, 6)

    def test_others_shapes(self):
        A13, A6, A71, A72, A81, A82, A9, A100, A101, A102, A11, A12 = \
            build_others(2, 1.0, 1.0)
        assert A13.shape == (3, 3)
        assert A6.shape == (3, 3)
        assert A71.shape == (3, 3)
        assert A72.shape == (3, 3)
        assert A81.shape == (3, 2)
        assert A82.shape == (3, 2)
        assert A9.shape == (2, 2)
        assert A100.shape == (2, 2)
        assert A101.shape == (2, 2)
        assert A102.shape == (2, 2)
        assert A11.shape == (2, 3)
        assert A12.shape == (2, 3)

    def test_aprop_shape(self):
        Ap = build_aprop(0.5, 0.1, 0.01, 2, 1.0+0j, 1.0+0j, 1.0, 1.0)
        assert Ap.shape == (8, 8)

    def test_n0_shapes(self):
        """n=0 special-case matrices should have the same shapes."""
        A1, A2 = build_A1_A2(0, 1.0+0j, 1.0+0j)
        assert A1.shape == (6, 3)
        assert A2.shape == (6, 3)
        assert build_A3(0).shape == (3, 3)
        assert build_A4(0).shape == (3, 6)
        assert build_A5(0).shape == (3, 6)


# ---------------------------------------------------------------------------
# A3: displacement transform
# ---------------------------------------------------------------------------

class TestA3:

    def test_n2_invertible(self):
        A3 = build_A3(2)
        assert abs(np.linalg.det(A3)) > 1e-10

    def test_n2_U_row(self):
        """A3[0,:] = [√n/√(2n+1), 0, -√(n+1)/√(2n+1)] for n=2."""
        A3 = build_A3(2)
        s5 = math.sqrt(5)
        assert A3[0, 0] == pytest.approx(math.sqrt(2) / s5)
        assert A3[0, 1] == pytest.approx(0.0)
        assert A3[0, 2] == pytest.approx(-math.sqrt(3) / s5)

    def test_n2_V_row(self):
        """A3[1,:] = [1/(√(2n+1)·√n), 0, 1/(√(2n+1)·√(n+1))]."""
        A3 = build_A3(2)
        s5 = math.sqrt(5)
        assert A3[1, 0] == pytest.approx(1.0 / (s5 * math.sqrt(2)))
        assert A3[1, 1] == pytest.approx(0.0)
        assert A3[1, 2] == pytest.approx(1.0 / (s5 * math.sqrt(3)))

    def test_n2_W_row(self):
        """A3[2,:] = [0, i/√(n(n+1)), 0]."""
        A3 = build_A3(2)
        assert A3[2, 0] == pytest.approx(0.0)
        assert A3[2, 1] == pytest.approx(1j / math.sqrt(6))
        assert A3[2, 2] == pytest.approx(0.0)

    def test_n0_special(self):
        """n=0 has a degenerate A3."""
        A3 = build_A3(0)
        assert A3[0, 2] == pytest.approx(-1.0)  # -√1/√1
        assert A3[2, 1] == pytest.approx(1.0)
        assert A3[1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A1, A2: constitutive relation
# ---------------------------------------------------------------------------

class TestA1A2:

    def test_n2_sigma_nn0_entries(self):
        """First row of A1/A2 for n=2: (3λ+2μ)/√3/√5 factor."""
        muC = 1.0 + 0j
        lam = 1.0 + 0j
        A1, A2 = build_A1_A2(2, muC, lam)
        fac = 5.0 / math.sqrt(15)  # (3+2)/√3/√5

        assert A1[0, 0] == pytest.approx(fac * math.sqrt(2) * 1)  # n-1=1
        assert A1[0, 2] == pytest.approx(fac * math.sqrt(3) * 4)  # n+2=4
        assert A2[0, 0] == pytest.approx(-fac * math.sqrt(2))
        assert A2[0, 2] == pytest.approx(fac * math.sqrt(3))

    def test_n2_sigma_nm2_entries(self):
        """Row 1 (sigma_{n,n-2,2}): 2μ√((n-1)/(2n-1)) factor."""
        muC = 1.0 + 0j
        lam = 1.0 + 0j
        A1, A2 = build_A1_A2(2, muC, lam)
        fac2 = 2.0 * math.sqrt(1.0 / 3.0)

        assert A1[1, 0] == pytest.approx(fac2 * 2)
        assert A2[1, 0] == pytest.approx(fac2)
        # All other entries in this row are zero
        assert A1[1, 1] == pytest.approx(0.0)
        assert A1[1, 2] == pytest.approx(0.0)

    def test_complex_modulus(self):
        """Viscoelastic muC should propagate into A1/A2 entries."""
        muC_real = 1.0 + 0j
        muC_visc = 1.0 / (1.0 - 1j / 5.0)
        lam = 1.0 + 0j

        A1r, _ = build_A1_A2(2, muC_real, lam)
        A1v, _ = build_A1_A2(2, muC_visc, lam)

        # Row 1 (sigma_{n,n-2,2}) depends only on μ, not λ
        ratio = A1v[1, 0] / A1r[1, 0]
        assert ratio == pytest.approx(muC_visc / muC_real, rel=1e-10)

    def test_n0_mostly_zero(self):
        """n=0 should have only two non-zero entries in A1."""
        A1, A2 = build_A1_A2(0, 1.0+0j, 1.0+0j)
        # Only [0,2] and [5,2] are non-zero
        assert A1[0, 2] != 0
        assert A1[5, 2] != 0
        # Everything else is zero
        mask = np.zeros((6, 3), dtype=bool)
        mask[0, 2] = True
        mask[5, 2] = True
        assert np.allclose(A1[~mask], 0.0)


# ---------------------------------------------------------------------------
# A4: stress transform
# ---------------------------------------------------------------------------

class TestA4:

    def test_n2_R_diagonal(self):
        """A4[0,0] = -1/√3 for all n>0."""
        A4 = build_A4(2)
        assert A4[0, 0] == pytest.approx(-1.0 / math.sqrt(3))

    def test_n2_T_imaginary(self):
        """T row entries should be purely imaginary."""
        A4 = build_A4(2)
        assert A4[2, 2].real == pytest.approx(0.0)
        assert A4[2, 4].real == pytest.approx(0.0)
        assert abs(A4[2, 2].imag) > 0
        assert abs(A4[2, 4].imag) > 0

    def test_n0_sparse(self):
        """n=0 A4 should have only 2 non-zero entries."""
        A4 = build_A4(0)
        assert A4[0, 0] == pytest.approx(-1.0 / math.sqrt(3))
        nnz = np.count_nonzero(A4)
        assert nnz == 2


# ---------------------------------------------------------------------------
# A5: stress divergence
# ---------------------------------------------------------------------------

class TestA5:

    def test_n2_T_row_imaginary(self):
        """T row of A5 should be purely imaginary."""
        A5 = build_A5(2)
        # Row 2 = T
        assert A5[2, 2].real == pytest.approx(0.0)
        assert A5[2, 4].real == pytest.approx(0.0)

    def test_n2_S_row_sigma_nn0(self):
        """S row, sigma_{n,n,0} entry = 1/√3 (after negation)."""
        A5 = build_A5(2)
        # A5 is negated: the raw S row [1,0] = -1/√3, after negation → +1/√3
        assert A5[1, 0] == pytest.approx(1.0 / math.sqrt(3))

    def test_n0_sparse(self):
        """n=0 A5 should have only 1 non-zero entry (in the R row)."""
        A5 = build_A5(0)
        nnz = np.count_nonzero(A5)
        assert nnz == 1


# ---------------------------------------------------------------------------
# build_others: gravity, potential, identity
# ---------------------------------------------------------------------------

class TestOthers:

    def test_A13_is_identity(self):
        A13, *_ = build_others(2, 1.0, 1.0)
        np.testing.assert_allclose(A13, np.eye(3))

    def test_A6_is_zero(self):
        _, A6, *_ = build_others(2, 1.0, 1.0)
        np.testing.assert_allclose(A6, 0.0)

    def test_A71_gravity(self):
        rho = 1.5
        _, _, A71, A72, *_ = build_others(2, rho, 1.0)
        nn = 2 * 3  # n(n+1)
        assert A71[0, 0] == pytest.approx(-2 * rho)
        assert A71[0, 1] == pytest.approx(rho * nn)
        assert A72[0, 0] == pytest.approx(rho)
        assert A71[1, 0] == pytest.approx(rho)

    def test_A81_potential_coupling(self):
        rho = 1.0
        *_, A81, A82, _, _, _, _, _, _ = build_others(2, rho, 1.0)
        assert A81[0, 1] == pytest.approx(rho)
        assert A82[1, 0] == pytest.approx(rho)

    def test_A100_potential_ode(self):
        *_, A100, A101, A102, _, _ = build_others(2, 1.0, 1.0)
        nn = 6
        assert A100[0, 1] == pytest.approx(1.0)
        assert A101[1, 1] == pytest.approx(-2.0)
        assert A102[1, 0] == pytest.approx(nn)

    def test_A11_poisson(self):
        rho = 1.0
        Gg = 2.0
        *_, A11, A12 = build_others(2, rho, Gg)
        nn = 6
        assert A11[1, 0] == pytest.approx(-2 * 4 * math.pi * Gg * rho)
        assert A11[1, 1] == pytest.approx(4 * math.pi * Gg * rho * nn)
        assert A12[1, 0] == pytest.approx(-4 * math.pi * Gg * rho)

    def test_n0_longman(self):
        """n=0 uses Longman 1963 formulation for potential."""
        *_, A100, _, _, _, _ = build_others(0, 1.0, 1.0)
        assert A100[0, 0] == pytest.approx(1.0)
        assert A100[1, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_gravity
# ---------------------------------------------------------------------------

class TestComputeGravity:

    def test_uniform_sphere(self):
        """g(r) = Gg * (4/3 π ρ r³) / r² = Gg * 4/3 π ρ r for uniform."""
        rho = 1.0
        Gg = 1.0
        r = 0.5
        R_inner = 0.0
        M_inner = 0.0
        g, dg = compute_gravity(r, rho, M_inner, R_inner, Gg)
        expected_g = Gg * (4.0 / 3.0) * math.pi * rho * r
        assert g == pytest.approx(expected_g)

    def test_gravity_at_surface(self):
        """Total mass enclosed → g = Gg * M / R²."""
        rho = 1.0
        Gg = 1.0
        r = 1.0
        g, _ = compute_gravity(r, rho, 0.0, 0.0, Gg)
        M = (4.0 / 3.0) * math.pi * rho
        assert g == pytest.approx(Gg * M)

    def test_hollow_shell(self):
        """Layer from R_inner to r with pre-existing mass."""
        rho_layer = 2.0
        R_inner = 0.5
        M_inner = 1.0
        Gg = 1.0
        r = 0.8
        g, dg = compute_gravity(r, rho_layer, M_inner, R_inner, Gg)
        M_r = M_inner + (4.0 / 3.0) * math.pi * rho_layer * (r**3 - R_inner**3)
        assert g == pytest.approx(Gg * M_r / r**2)


# ---------------------------------------------------------------------------
# Full Aprop assembly
# ---------------------------------------------------------------------------

class TestBuildAprop:

    def test_finite_entries(self):
        """Aprop should have no NaN or Inf for typical parameters."""
        Ap = build_aprop(0.5, 0.1, 0.01, 2, 1.0+0j, 1.0+0j, 1.0, 1.0)
        assert np.all(np.isfinite(Ap))

    def test_spheroidal_toroidal_decoupling(self):
        """For 1D model, toroidal (W,T) decouple from spheroidal (U,V,R,S,Φ,dΦ/dr).

        State vector: [U, V, W, R, S, T, Φ, dΦ/dr] = indices [0,1,2,3,4,5,6,7].
        Toroidal indices: W=2, T=5.
        Spheroidal indices: U=0, V=1, R=3, S=4, Φ=6, dΦ/dr=7.

        Decoupling means:
        - dW/dr depends only on W, T (not U,V,R,S,Φ,dΦ/dr)
        - dT/dr depends only on W, T
        - dU/dr, dV/dr, etc. don't depend on W, T
        """
        Ap = build_aprop(0.5, 0.1, 0.01, 2, 1.0+0j, 1.0+0j, 1.0, 1.0)
        tor = [2, 5]
        sph = [0, 1, 3, 4, 6, 7]

        # Toroidal rows should have zero spheroidal columns
        for i in tor:
            for j in sph:
                assert abs(Ap[i, j]) < 1e-12, f"Ap[{i},{j}] = {Ap[i,j]}"

        # Spheroidal rows should have zero toroidal columns
        for i in sph:
            for j in tor:
                assert abs(Ap[i, j]) < 1e-12, f"Ap[{i},{j}] = {Ap[i,j]}"

    def test_complex_valued(self):
        """With viscoelastic muC, Aprop should have imaginary parts."""
        muC = 1.0 / (1.0 - 1j / 5.0)
        lam = 2.0 - (2.0 / 3.0) * muC
        Ap = build_aprop(0.5, 0.1, 0.01, 2, muC, lam, 1.0, 1.0)
        assert np.any(np.abs(Ap.imag) > 1e-10)

    def test_scales_with_radius(self):
        """Aprop at different radii should differ (r appears in denominators)."""
        kwargs = dict(g=0.1, dg=0.01, n=2, muC=1.0+0j, lam=1.0+0j, rho=1.0, Gg=1.0)
        Ap1 = build_aprop(r=0.3, **kwargs)
        Ap2 = build_aprop(r=0.7, **kwargs)
        assert not np.allclose(Ap1, Ap2)

    def test_n2_matches_matlab_structure(self):
        """Verify Aprop block structure matches MATLAB get_Aprop.

        MATLAB: Aprop = Adotx \\ Ax where Adotx and Ax have blocks:
        Block 1 (rows 0:3): constitutive
        Block 2 (rows 3:6): momentum
        Block 3 (rows 6:8): Poisson
        """
        Ap = build_aprop(0.5, 0.1, 0.01, 2, 1.0+0j, 1.0+0j, 1.0, 1.0)
        # Should be non-trivially filled in the spheroidal block
        sph = [0, 1, 3, 4, 6, 7]
        sph_block = Ap[np.ix_(sph, sph)]
        assert np.count_nonzero(np.abs(sph_block) > 1e-14) > 10


# ---------------------------------------------------------------------------
# Cash-Karp coefficients
# ---------------------------------------------------------------------------

class TestCashKarp:

    def test_6_stages(self):
        assert len(CK_A) == 6
        assert len(CK_B) == 6
        assert len(CK_C) == 6

    def test_weights_sum_to_one(self):
        """5th-order weights should sum to 1."""
        assert sum(CK_C) == pytest.approx(1.0, rel=1e-14)

    def test_butcher_tableau_structure(self):
        """B[i] should have i entries (lower-triangular)."""
        for i, row in enumerate(CK_B):
            assert len(row) == i

    def test_node_values(self):
        """Spot-check a few node coefficients."""
        assert CK_A[0] == pytest.approx(0.0)
        assert CK_A[1] == pytest.approx(0.2)
        assert CK_A[5] == pytest.approx(7 / 8)
