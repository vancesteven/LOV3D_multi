"""Tests for pylov3d.couplings — mode selection and coupling coefficients."""


import numpy as np
import pytest

from pylov3d.couplings import (
    next_coupling,
    get_active_modes,
    coupling_coefficients,
    get_couplings,
    Couplings,
)


# ---------------------------------------------------------------------------
# next_coupling
# ---------------------------------------------------------------------------

class TestNextCoupling:

    def test_n2m0_rheo_n2m0(self):
        """Forcing (2,0) + rheology (2,0) → n_new in {0,2,4}, m_new=0."""
        modes = next_coupling(2, 0, 1, 2, 0, 1)
        n_values = sorted(set(m[0] for m in modes))
        # Even parity: |2-2|=0, 2, 4
        assert 0 in n_values
        assert 2 in n_values
        assert 4 in n_values
        # All m = 0
        for m in modes:
            assert m[1] == 0

    def test_n2m0_rheo_n2m2(self):
        """Forcing (2,0) + rheology (2,2) → m_new=2."""
        modes = next_coupling(2, 0, 1, 2, 2, 1)
        for m in modes:
            assert m[1] == 2

    def test_n2m2_rheo_n2m_neg2(self):
        """Forcing (2,2) + rheology (2,-2) → m_new=0."""
        modes = next_coupling(2, 2, 1, 2, -2, 1)
        for m in modes:
            assert m[1] == 0

    def test_spheroidal_preserved_even(self):
        """Even-parity coupling preserves spheroidal/toroidal type."""
        modes = next_coupling(2, 0, 1, 2, 0, 1)
        even_modes = [m for m in modes if (m[0] - abs(2 - 2)) % 2 == 0]
        for m in even_modes:
            assert m[2] == 1  # spheroidal preserved

    def test_returns_list(self):
        modes = next_coupling(2, 0, 1, 2, 0, 1)
        assert isinstance(modes, list)
        assert len(modes) > 0

    def test_high_m_excluded(self):
        """Modes with |m| >= n+1 should not appear."""
        modes = next_coupling(1, 1, 1, 1, 1, 1)
        for m in modes:
            assert abs(m[1]) < m[0] + 1


# ---------------------------------------------------------------------------
# get_active_modes
# ---------------------------------------------------------------------------

class TestGetActiveModes:

    def test_uniform_body(self):
        """No rheology variations → only forcing mode."""
        # A "uniform" body still needs a variations array; use (0,0)
        variations = np.array([[0, 0]])
        modes = get_active_modes(2, variations, 2, 0)
        assert len(modes) >= 1
        # Forcing mode should be present
        assert any(modes[i, 0] == 2 and modes[i, 1] == 0 for i in range(len(modes)))

    def test_forcing_mode_order_zero(self):
        """Forcing mode should have perturbation order 0."""
        variations = np.array([[2, 0]])
        modes = get_active_modes(2, variations, 2, 0)
        forcing_idx = np.where((modes[:, 0] == 2) & (modes[:, 1] == 0))[0]
        assert len(forcing_idx) == 1
        assert modes[forcing_idx[0], 2] == 0

    def test_degree_2_rheology_excites_degree_0_and_4(self):
        """Degree-2 rheology coupling to degree-2 forcing excites n=0,2,4."""
        variations = np.array([[2, 0]])
        modes = get_active_modes(2, variations, 2, 0)
        n_values = set(modes[:, 0])
        assert 0 in n_values
        assert 2 in n_values
        assert 4 in n_values

    def test_perturbation_order_0_single_mode(self):
        """Order 0 → only forcing mode."""
        variations = np.array([[2, 0]])
        modes = get_active_modes(0, variations, 2, 0)
        assert len(modes) == 1
        assert modes[0, 0] == 2
        assert modes[0, 1] == 0

    def test_higher_order_more_modes(self):
        """Higher perturbation order should produce >= as many modes."""
        variations = np.array([[2, 0]])
        modes1 = get_active_modes(1, variations, 2, 0)
        modes2 = get_active_modes(2, variations, 2, 0)
        assert len(modes2) >= len(modes1)

    def test_output_shape(self):
        variations = np.array([[2, 0]])
        modes = get_active_modes(2, variations, 2, 0)
        assert modes.ndim == 2
        assert modes.shape[1] == 3


# ---------------------------------------------------------------------------
# coupling_coefficients
# ---------------------------------------------------------------------------

class TestCouplingCoefficients:

    def test_output_shape(self):
        """Should return 27-element array."""
        Cp = coupling_coefficients(2, 0, 2, 0, 2, 0)
        assert Cp.shape == (27,)

    def test_self_coupling_nonzero(self):
        """Mode (2,0) coupled with itself via (2,0) rheology should have nonzero coefficients."""
        Cp = coupling_coefficients(2, 0, 2, 0, 2, 0)
        # Sparsity flag should be 1
        assert Cp[26] == 1.0
        # At least some coupling coefficients nonzero
        assert np.max(np.abs(Cp[:26])) > 0

    def test_selection_rule_m(self):
        """When ma != m - mb, all couplings should be zero.

        This is enforced externally in get_couplings, but if called directly
        with invalid quantum numbers, the Wigner symbols should give zero.
        """
        # na=2, ma=1, but m=0, mb=0 → need ma=0, not 1
        Cp = coupling_coefficients(2, 0, 2, 1, 2, 0)
        # Wigner 3j with ma=1, -mc=0, mb=0 → m1+m2+m3=1 ≠ 0 → zero
        assert np.max(np.abs(Cp[:26])) == pytest.approx(0.0, abs=1e-14)
        assert Cp[26] == 0.0

    def test_finite_values(self):
        """All coupling coefficients should be finite."""
        Cp = coupling_coefficients(2, 0, 0, 0, 2, 0)
        assert np.all(np.isfinite(Cp))

    def test_real_values(self):
        """Coupling coefficients should be real (float, not complex)."""
        Cp = coupling_coefficients(2, 0, 2, 0, 2, 0)
        assert Cp.dtype in (np.float64, np.float32)

    def test_n2_n0_nb2(self):
        """Coupling (2,0)-(0,0)-(2,0): should excite degree-0 response."""
        Cp = coupling_coefficients(2, 0, 0, 0, 2, 0)
        # The isotropic coupling (slot 0) should be nonzero
        # since n=2, na=0, nb=2 satisfies triangle
        assert Cp[26] == 1.0

    def test_various_degrees(self):
        """Coupling coefficients should be finite for various degree combinations."""
        for n, na, nb in [(2, 2, 2), (2, 4, 2), (4, 2, 2), (2, 0, 2), (3, 1, 2)]:
            Cp = coupling_coefficients(n, 0, na, 0, nb, 0)
            assert np.all(np.isfinite(Cp))


# ---------------------------------------------------------------------------
# get_couplings
# ---------------------------------------------------------------------------

class TestGetCouplings:

    def test_returns_couplings_type(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0, perturbation_order=2)
        assert isinstance(C, Couplings)

    def test_array_shapes(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0, perturbation_order=2)
        Nsol = len(C.n_s)
        assert C.m_s.shape == (Nsol,)
        assert C.order.shape == (Nsol,)
        assert C.Coup.shape == (Nsol, Nsol, 27, 1)  # 1 rheology mode

    def test_forcing_mode_present(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        idx = np.where((C.n_s == 2) & (C.m_s == 0))[0]
        assert len(idx) == 1

    def test_sparsity_flag(self):
        """Coup[:,:,26,:] should be 0 or 1."""
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        flags = C.Coup[:, :, 26, :]
        assert np.all((flags == 0) | (flags == 1))

    def test_multiple_rheology_modes(self):
        """Multiple rheology variations should produce Nreo>1 in Coup."""
        variations = np.array([[2, 0], [2, 2]])
        C = get_couplings(variations, 2, 0)
        assert C.Coup.shape[3] == 2

    def test_selection_rule_enforced(self):
        """Only entries satisfying selection rules should be nonzero."""
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        Nsol = len(C.n_s)
        for i in range(Nsol):
            for j in range(Nsol):
                n_eq = C.n_s[i]
                na = C.n_s[j]
                ma = C.m_s[j]
                m_eq = C.m_s[i]
                nb, mb = 2, 0
                if not (na >= abs(n_eq - nb) and na <= n_eq + nb and ma == m_eq - mb):
                    # Should be all zeros
                    assert np.max(np.abs(C.Coup[i, j, :, 0])) == 0.0

    def test_order_0_gives_single_mode(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0, perturbation_order=0)
        assert len(C.n_s) == 1
        assert C.n_s[0] == 2
        assert C.m_s[0] == 0

    def test_coupling_tensor_finite(self):
        variations = np.array([[2, 0]])
        C = get_couplings(variations, 2, 0)
        assert np.all(np.isfinite(C.Coup))
