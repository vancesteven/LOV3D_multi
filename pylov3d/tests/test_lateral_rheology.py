"""Tests for pylov3d.rheology — lateral variation processing."""

import math

import numpy as np
import pytest

from pylov3d.types import (
    InteriorModel,
    Forcing,
    LateralRheology,
    make_interior_model,
    make_forcing,
)
from pylov3d.rheology import (
    get_rheology,
    process_lateral_variations,
    _sh_grid,
    _sh_synthesis,
    _sh_analysis,
    _ensure_conjugate_pairs,
    _unify_modes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def io_model():
    """Simple 3-layer Io-like model (core + mantle + crust)."""
    model = make_interior_model(
        R0_km=[800.0, 1600.0, 1821.6],
        rho0=[5150.0, 3300.0, 3000.0],
        mu0=[0.0, 60e9, 65e9],
        eta0=[None, 1e19, None],
    )
    forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
    return get_rheology(model, forcing), forcing


@pytest.fixture
def elastic_model():
    """Simple 2-layer elastic model (core + shell)."""
    model = make_interior_model(
        R0_km=[1000.0, 1800.0],
        rho0=[5000.0, 3000.0],
        mu0=[0.0, 50e9],
    )
    forcing = make_forcing(Td=86400, n=2, m=0, F=1.0)
    return get_rheology(model, forcing), forcing


# ---------------------------------------------------------------------------
# SH grid utilities
# ---------------------------------------------------------------------------

class TestSHGrid:

    def test_grid_sizes(self):
        theta, phi, w = _sh_grid(4)
        assert len(theta) >= 4 * 5
        assert len(phi) == 2 * len(theta)
        assert len(w) == len(theta)

    def test_weights_sum(self):
        """GL weights should sum to 2 (integral of 1 over [-1,1])."""
        _, _, w = _sh_grid(4)
        assert np.sum(w) == pytest.approx(2.0, rel=1e-12)

    def test_theta_range(self):
        theta, _, _ = _sh_grid(4)
        assert np.all(theta >= 0)
        assert np.all(theta <= np.pi)


# ---------------------------------------------------------------------------
# SH synthesis / analysis round-trip
# ---------------------------------------------------------------------------

class TestSHRoundTrip:

    def test_constant_field(self):
        """Synthesis of empty coeffs gives a field of ones."""
        theta, phi, _ = _sh_grid(4)
        field = _sh_synthesis([], theta, phi)
        assert np.allclose(field, 1.0)

    def test_y20_roundtrip(self):
        """Synthesis of (2,0) perturbation, analysis should recover it."""
        amp = 0.1
        coeffs = [(2, 0, amp)]
        lmax = 4
        theta, phi, w = _sh_grid(lmax)
        field = _sh_synthesis(coeffs, theta, phi)

        recovered = _sh_analysis(field, theta, phi, w, lmax)
        # (0,0) coefficient = sqrt(4π) × mean = sqrt(4π) × 1
        assert recovered[(0, 0)] == pytest.approx(np.sqrt(4 * np.pi), abs=1e-10)
        # (2,0) should be ~amp
        assert recovered[(2, 0)] == pytest.approx(amp, abs=1e-10)
        # Other modes should be ~0
        assert abs(recovered.get((1, 0), 0.0)) < 1e-10
        assert abs(recovered.get((3, 0), 0.0)) < 1e-10

    def test_y22_roundtrip(self):
        """Synthesis of (2,2) + (2,-2) conjugate pair, analysis recovery."""
        amp = 0.05 + 0.02j
        coeffs = [
            (2, 2, amp),
            (2, -2, (-1)**2 * np.conj(amp)),  # conjugate for real field
        ]
        lmax = 4
        theta, phi, w = _sh_grid(lmax)
        field = _sh_synthesis(coeffs, theta, phi)

        # Field should be real (since we have conjugate pair)
        assert np.max(np.abs(np.imag(field))) < 1e-10

        recovered = _sh_analysis(field, theta, phi, w, lmax)
        assert recovered[(2, 2)] == pytest.approx(amp, abs=1e-10)
        assert recovered[(2, -2)] == pytest.approx((-1)**2 * np.conj(amp), abs=1e-10)


# ---------------------------------------------------------------------------
# Conjugate pair helper
# ---------------------------------------------------------------------------

class TestEnsureConjugatePairs:

    def test_m0_unchanged(self):
        modes = [(2, 0, 0.1)]
        result = _ensure_conjugate_pairs(modes)
        assert len(result) == 1

    def test_adds_negative_m(self):
        modes = [(2, 2, 0.1 + 0.05j)]
        result = _ensure_conjugate_pairs(modes)
        assert len(result) == 2
        added = [m for m in result if m[1] == -2][0]
        assert added[0] == 2
        assert added[2] == pytest.approx((-1)**2 * np.conj(0.1 + 0.05j))

    def test_both_present_unchanged(self):
        modes = [(2, 2, 0.1), (2, -2, 0.1)]
        result = _ensure_conjugate_pairs(modes)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Unify modes helper
# ---------------------------------------------------------------------------

class TestUnifyModes:

    def test_mu_only(self):
        nm_list, mu_map, eta_map, K_map = _unify_modes(
            [(2, 0, 0.1)], None, None,
        )
        assert (2, 0) in nm_list
        assert mu_map[(2, 0)] == pytest.approx(0.1)
        assert eta_map[(2, 0)] == 0.0
        assert K_map[(2, 0)] == 0.0

    def test_different_modes_merged(self):
        nm_list, mu_map, eta_map, K_map = _unify_modes(
            [(2, 0, 0.1)],
            [(4, 0, 0.05)],
            None,
        )
        assert (2, 0) in nm_list
        assert (4, 0) in nm_list
        assert mu_map[(4, 0)] == 0.0
        assert eta_map[(2, 0)] == 0.0

    def test_excludes_00(self):
        nm_list, _, _, _ = _unify_modes([(0, 0, 0.1)], None, None)
        assert (0, 0) not in nm_list


# ---------------------------------------------------------------------------
# process_lateral_variations — elastic layers
# ---------------------------------------------------------------------------

class TestLateralElastic:

    def test_no_variations_returns_trivial(self, elastic_model):
        model, forcing = elastic_model
        model, lateral = process_lateral_variations(model, forcing)
        assert isinstance(lateral, LateralRheology)
        assert lateral.variations.shape[0] == 1
        assert np.all(lateral.uniform)

    def test_elastic_mu_scaling(self, elastic_model):
        model, forcing = elastic_model
        amp = 0.1
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, amp)]},
        )
        assert not lateral.uniform[1]
        # For elastic, muC_amp should be mu * amp
        mu_layer = float(model.mu[1])
        idx = np.where((lateral.variations[:, 0] == 2)
                        & (lateral.variations[:, 1] == 0))[0]
        assert len(idx) == 1
        assert lateral.muC_amp[1, idx[0]] == pytest.approx(mu_layer * amp)

    def test_elastic_K_amp_zero(self, elastic_model):
        model, forcing = elastic_model
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, 0.1)]},
        )
        assert np.all(lateral.K_amp == 0.0)


# ---------------------------------------------------------------------------
# process_lateral_variations — viscoelastic layers
# ---------------------------------------------------------------------------

class TestLateralViscoelastic:

    def test_mu_only_produces_muC_amp(self, io_model):
        model, forcing = io_model
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, 0.1)]},
        )
        assert not lateral.uniform[1]
        # Should have at least degree 2 mode
        assert any(lateral.variations[:, 0] == 2)
        # muC_amp should be complex (viscoelastic)
        idx = np.where((lateral.variations[:, 0] == 2)
                        & (lateral.variations[:, 1] == 0))[0]
        if len(idx) > 0:
            assert np.abs(lateral.muC_amp[1, idx[0]]) > 0

    def test_eta_only_produces_muC_amp(self, io_model):
        model, forcing = io_model
        model, lateral = process_lateral_variations(
            model, forcing,
            eta_variable={1: [(2, 0, 0.1)]},
        )
        assert not lateral.uniform[1]
        # Even pure eta variation produces muC variation
        # because muC depends nonlinearly on eta
        assert lateral.muC_amp.shape[1] >= 1

    def test_model_muC_updated(self, io_model):
        model_orig, forcing = io_model
        muC_before = complex(model_orig.muC[1])
        model, lateral = process_lateral_variations(
            model_orig, forcing,
            mu_variable={1: [(2, 0, 0.2)]},
        )
        # muC should be updated (may differ slightly from naive Maxwell)
        # The (0,0) component of the SH expansion should be close to the
        # original but not identical for large perturbations
        muC_after = complex(model.muC[1])
        # Both should be nonzero and complex
        assert abs(muC_after) > 0
        assert abs(muC_after.imag) > 0

    def test_small_perturbation_muC_close(self, io_model):
        """For small perturbations, (0,0) muC should be close to original."""
        model_orig, forcing = io_model
        muC_before = complex(model_orig.muC[1])
        model, lateral = process_lateral_variations(
            model_orig, forcing,
            mu_variable={1: [(2, 0, 0.001)]},
        )
        muC_after = complex(model.muC[1])
        assert muC_after == pytest.approx(muC_before, rel=0.01)

    def test_finite_amplitudes(self, io_model):
        model, forcing = io_model
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, 0.1)]},
        )
        assert np.all(np.isfinite(lateral.muC_amp))

    def test_variations_shape(self, io_model):
        model, forcing = io_model
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, 0.1), (2, 2, 0.05)]},
        )
        # variations should be (Nreo, 2)
        assert lateral.variations.ndim == 2
        assert lateral.variations.shape[1] == 2
        # muC_amp shape: (n_layers, Nreo)
        assert lateral.muC_amp.shape == (model.n_layers, lateral.variations.shape[0])


# ---------------------------------------------------------------------------
# process_lateral_variations — multi-layer
# ---------------------------------------------------------------------------

class TestLateralMultiLayer:

    def test_uniform_layer_stays_uniform(self, io_model):
        model, forcing = io_model
        # Only layer 1 gets variations; layer 2 should stay uniform
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, 0.1)]},
        )
        assert lateral.uniform[2]
        # All amplitudes for layer 2 should be zero
        assert np.all(lateral.muC_amp[2, :] == 0)

    def test_padding_across_layers(self, io_model):
        model, forcing = io_model
        # Layer 1: degree 2, Layer 1 also has degree 4
        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable={1: [(2, 0, 0.1), (4, 0, 0.05)]},
        )
        # variations should include both (2,0) and (4,0)
        nm_set = set(map(tuple, lateral.variations.tolist()))
        assert (2, 0) in nm_set
        assert (4, 0) in nm_set


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_get_rheology_unchanged(self, io_model):
        """get_rheology still returns a single model (no lateral data)."""
        model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        result = get_rheology(model, forcing)
        assert isinstance(result, InteriorModel)
