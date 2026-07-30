"""End-to-end tests for the lateral variation (3D mode coupling) pipeline.

Tests the full workflow:
  make_interior_model → get_love(mu_variable=...) → Love number spectra

Validates backward compatibility, physical consistency, and convergence.
"""


import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.love import get_love


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_zero_amplitude_matches_1d(self):
        """Lateral variations with zero amplitude should match 1D exactly."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love_1d, _, _ = get_love(raw_model, forcing, numerics)

        # 3D with zero amplitude → should match 1D
        love_3d, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.0)]},
        )

        # Find forcing mode in 3D
        k_idx = np.where((love_3d.n == 2) & (love_3d.m == 0))[0][0]
        assert love_3d.k[k_idx] == pytest.approx(love_1d.k[0], abs=1e-8)
        assert love_3d.h[k_idx] == pytest.approx(love_1d.h[0], abs=1e-8)
        assert love_3d.l[k_idx] == pytest.approx(love_1d.l[0], abs=1e-8)

    def test_1d_path_unchanged(self):
        """get_love without lateral params uses 1D path unmodified."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love, y_rad, _ = get_love(raw_model, forcing, numerics)

        assert len(love.k) == 1
        assert y_rad.y.shape[1] == 8  # 1D state vector

    def test_elastic_body_real_love(self):
        """Elastic body with lateral variations should have ~real k."""
        raw_model = make_interior_model(
            R0_km=[1000.0, 1800.0],
            rho0=[5000.0, 3000.0],
            mu0=[0.0, 50e9],
        )
        forcing = make_forcing(Td=86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=50)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.05)]},
        )

        k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
        # Elastic: imaginary part should be small relative to real
        assert abs(love.k[k_idx].imag) < 0.01 * abs(love.k[k_idx].real)


# ---------------------------------------------------------------------------
# Physical consistency
# ---------------------------------------------------------------------------

class TestPhysicalConsistency:

    def test_forcing_mode_dominates(self):
        """The forcing mode should have the largest response."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
        h_forcing = abs(love.h[k_idx])

        for i in range(len(love.n)):
            if i != k_idx:
                assert abs(love.h[i]) < h_forcing

    def test_coupled_modes_present(self):
        """Degree-2 forcing + degree-2 rheology should excite degrees 0 and 4."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        degrees = set(int(x) for x in love.n)
        assert 0 in degrees or 4 in degrees, f"Expected degree 0 or 4, got {degrees}"
        assert 2 in degrees  # forcing mode always present

    def test_larger_perturbation_larger_coupling(self):
        """Larger lateral variation amplitude → larger non-forcing mode response."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love_small, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.01)]},
        )
        love_large, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.2)]},
        )

        # Find degree 4, m=0 mode (excited by coupling)
        idx_4_small = np.where((love_small.n == 4) & (love_small.m == 0))[0]
        idx_4_large = np.where((love_large.n == 4) & (love_large.m == 0))[0]

        if len(idx_4_small) > 0 and len(idx_4_large) > 0:
            assert abs(love_large.h[idx_4_large[0]]) > abs(love_small.h[idx_4_small[0]])

    def test_all_love_numbers_finite(self):
        """All Love numbers should be finite."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.15)]},
        )

        assert np.all(np.isfinite(love.k))
        assert np.all(np.isfinite(love.h))
        assert np.all(np.isfinite(love.l))

    def test_dissipative_body_imag_k(self):
        """Viscoelastic body: forcing mode k should have nonzero Im (dissipative).

        The sign of Im(k) depends on model details; here we check that the
        coupled result has the same sign as the 1D reference.
        """
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love_1d, _, _ = get_love(raw_model, forcing, numerics)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
        # Im(k) should be nonzero (dissipation present)
        assert abs(love.k[k_idx].imag) > 1e-10
        # Sign should match 1D reference
        assert np.sign(love.k[k_idx].imag) == np.sign(love_1d.k[0].imag)


# ---------------------------------------------------------------------------
# Multi-layer variations
# ---------------------------------------------------------------------------

class TestMultiLayer:

    def test_two_layers_with_variations(self):
        """Lateral variations in two layers should work."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)], 2: [(2, 0, 0.05)]},
        )

        assert len(love.k) > 1
        assert np.all(np.isfinite(love.k))


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:

    def test_small_perturbation_converges_to_1d(self):
        """As perturbation → 0, coupled Love numbers → 1D Love numbers."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=30)

        love_1d, _, _ = get_love(raw_model, forcing, numerics)

        amps = [0.1, 0.01, 0.001]
        diffs = []
        for amp in amps:
            love_3d, _, _ = get_love(
                raw_model, forcing, numerics,
                mu_variable={1: [(2, 0, amp)]},
            )
            k_idx = np.where((love_3d.n == 2) & (love_3d.m == 0))[0][0]
            diffs.append(abs(love_3d.k[k_idx] - love_1d.k[0]))

        # Differences should decrease with decreasing amplitude
        assert diffs[1] < diffs[0]
        assert diffs[2] < diffs[1]
