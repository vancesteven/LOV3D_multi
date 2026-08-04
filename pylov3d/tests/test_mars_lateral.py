"""Tests for pylov3d.mars_lateral (TASK-016).

Covers: the real->complex SH conversion round-trip (the load-bearing
convention check) plus an independent rotational-covariance regression on
its imaginary/sine branch, zero-amplitude reduction to the 1D Mars k2,
linearity in the lateral amplitude (including the first-order (4,0)
forcing-mode scaling found in review), JAX/NumPy coupled equivalence on the
real lateral case, sanity of the (D2 areoid-corrected) Airy numbers, the
forcing-mode perturbation being small, and an lmax=5 truncation-sensitivity
spot check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.jax_coupled import jax_get_solution_coupled_scan
from pylov3d.love import extract_love_numbers, get_love
from pylov3d.mapping import latlon_grid, sh_to_latlon
from pylov3d.mars import MARS, MARS_FORCING_TD, build_mars_model
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.solver import _get_solution_coupled
from pylov3d.types import make_forcing, make_numerics

from pylov3d.mars_lateral import (
    CRUST_LAYER_INDEX,
    TOPO_PATH,
    _real_sh_to_complex_mu_variable,
    complex_sh_synthesis,
    crustal_thickness_diagnostics,
    crustal_thickness_variation,
    mu_variable_from_topography,
)

pytestmark = pytest.mark.skipif(
    not TOPO_PATH.exists(), reason="MarsTopo719.shape.gz not present",
)


def _relative_error(actual, expected):
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected))) / scale


def _scale_entries(entries, eps):
    return [(n, m, eps * amp) for n, m, amp in entries]


# ---------------------------------------------------------------------------
# 1. Round-trip convention test (load-bearing) + rotational covariance
# ---------------------------------------------------------------------------

class TestCSHRoundTrip:
    """The real (Clm, Slm) -> complex mu_variable conversion must exactly
    reconstruct the real field it was derived from, using the complex-SH
    convention implicit in the (MATLAB-cross-validated) mu_variable
    machinery -- not the unrelated scipy-sph_harm_y convention used only
    by process_lateral_variations's viscoelastic grid path."""

    def test_random_field_roundtrip(self):
        rng = np.random.default_rng(0)
        lmax = 4
        real_coeffs = {}
        for n in range(1, lmax + 1):
            for m in range(n + 1):
                real_coeffs[(n, m)] = float(rng.normal())
                if m >= 1:
                    real_coeffs[(n, -m)] = float(rng.normal())

        entries = _real_sh_to_complex_mu_variable(real_coeffs)
        nlat, nlon = 40, 80
        lat, lon = latlon_grid(nlat=nlat, nlon=nlon)
        real_grid = sh_to_latlon(real_coeffs, nlat=nlat, nlon=nlon).z
        complex_field = complex_sh_synthesis(entries, lat, lon)

        assert np.max(np.abs(complex_field.imag)) < 1e-10
        rel = _relative_error(complex_field.real, real_grid)
        assert rel < 1e-10, f"CSH round-trip relative error {rel:.3e} >= 1e-10"

    def test_actual_mars_dt_field_roundtrip(self):
        """Same check on the real Airy dt(theta, phi) field, not a synthetic one."""
        dt = crustal_thickness_variation(lmax=4)
        entries = _real_sh_to_complex_mu_variable(dt)
        nlat, nlon = 60, 120
        lat, lon = latlon_grid(nlat=nlat, nlon=nlon)
        real_grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon).z
        complex_field = complex_sh_synthesis(entries, lat, lon)
        rel = _relative_error(complex_field.real, real_grid)
        assert rel < 1e-10, f"Mars dt round-trip relative error {rel:.3e} >= 1e-10"

    def test_conjugate_pairs_present(self):
        """Every m != 0 mode from mu_variable_from_topography has both signs."""
        entries = mu_variable_from_topography(lmax=4)[CRUST_LAYER_INDEX]
        nm = {(n, m) for n, m, _a in entries}
        for n, m in nm:
            if m != 0:
                assert (n, -m) in nm, f"missing conjugate partner for ({n},{m})"


class TestRotationCovariance:
    """The only in-repo guard on the imaginary-amplitude (sine/S_nm) branch
    of _real_sh_to_complex_mu_variable, which TestCSHRoundTrip exercises
    only via a purely mathematical round-trip, not through the solver.

    A pure-sine C22=0,S22=amp rheology field is the pure-cosine field
    rotated by +pi/4 in longitude (sin(2*phi) = cos(2*(phi - pi/4))).
    Because the reference body is azimuthally isotropic and the forcing
    (2,0) is itself axisymmetric, the whole coupled response must rotate
    the same way: every response mode picks up |k| unchanged and a phase
    factor exp(-i*m*pi/4) (derived and verified against a probe run to
    machine precision on this exact case -- see docs/MARS_MODEL.md,
    "Lateral variations (TASK-016)"). A sign error anywhere in the i*S_nm
    terms of _real_sh_to_complex_mu_variable would break this.
    """

    def test_cos_vs_sin_c22_equal_magnitude_and_rotated_phase(self):
        amp = 0.02
        cos_entries = _real_sh_to_complex_mu_variable({(2, 2): amp})
        sin_entries = _real_sh_to_complex_mu_variable({(2, -2): amp})

        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=15, perturbation_order=2)

        love_cos, _, _ = get_love(model, forcing, numerics, mu_variable={CRUST_LAYER_INDEX: cos_entries})
        love_sin, _, _ = get_love(model, forcing, numerics, mu_variable={CRUST_LAYER_INDEX: sin_entries})

        compared = 0
        for i in range(len(love_cos.n)):
            n_m, m_m = int(love_cos.n[i]), int(love_cos.m[i])
            j = np.where((love_sin.n == n_m) & (love_sin.m == m_m))[0]
            if len(j) == 0:
                continue
            kc, ks = complex(love_cos.k[i]), complex(love_sin.k[j[0]])
            if abs(kc) < 1e-16 or abs(ks) < 1e-16:
                continue
            assert abs(ks) == pytest.approx(abs(kc), rel=1e-6), (
                f"mode (n={n_m},m={m_m}): |k| not preserved under rotation"
            )
            expected_phase = (-m_m * math.pi / 4.0) % (2 * math.pi)
            phase = math.atan2((ks / kc).imag, (ks / kc).real) % (2 * math.pi)
            diff = min(abs(phase - expected_phase), 2 * math.pi - abs(phase - expected_phase))
            assert diff < 1e-4, (
                f"mode (n={n_m},m={m_m}): phase {math.degrees(phase):.3f} deg vs "
                f"expected {math.degrees(expected_phase):.3f} deg"
            )
            compared += 1
        assert compared >= 5, "too few resolvable modes -- vacuous test"


# ---------------------------------------------------------------------------
# 2. Zero-amplitude reduction
# ---------------------------------------------------------------------------

class TestZeroAmplitudeReduction:

    def test_zero_amplitude_matches_1d_k2(self):
        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=50)

        love_1d, _, _ = get_love(model, forcing, numerics)
        love_3d, _, _ = get_love(
            model, forcing, numerics,
            mu_variable={CRUST_LAYER_INDEX: [(2, 0, 0.0)]},
        )
        k_idx = np.where((love_3d.n == 2) & (love_3d.m == 0))[0][0]
        rel = abs(love_3d.k[k_idx] - love_1d.k[0]) / abs(love_1d.k[0])
        assert rel < 1e-10, f"zero-amplitude k2 relative error {rel:.3e} >= 1e-10"


# ---------------------------------------------------------------------------
# 3. Linearity in the lateral amplitude
# ---------------------------------------------------------------------------

class TestLinearity:

    def test_order1_mode_scales_linearly(self):
        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=4, method="combination", Nrbase=15, perturbation_order=2,
        )
        base_entries = mu_variable_from_topography(lmax=4)[CRUST_LAYER_INDEX]

        # Determine which (n, m) modes are order-1 (reachable via one
        # forcing-rheology coupling step), independent of amplitude.
        numerics_g, model_g = set_boundary_indices(numerics, model)
        model_g = get_rheology(model_g, forcing)
        _, lateral = process_lateral_variations(
            model_g, forcing, mu_variable={CRUST_LAYER_INDEX: base_entries},
        )
        couplings = get_couplings(
            lateral.variations, forcing.n, forcing.m,
            perturbation_order=numerics.perturbation_order,
        )
        order1_nm = {
            (int(n), int(m))
            for n, m, o in zip(couplings.n_s, couplings.m_s, couplings.order)
            if o == 1
        }
        assert order1_nm, "no order-1 modes found -- test would be vacuous"

        results = {}
        for eps in (1e-3, 1e-2):
            entries = _scale_entries(base_entries, eps)
            love, _, _ = get_love(
                model, forcing, numerics,
                mu_variable={CRUST_LAYER_INDEX: entries},
            )
            results[eps] = love

        # get_couplings's "order" is a mode's *discovery* order (reachable
        # via one coupling step under the (n,m) triangle/parity selection
        # rules) -- an upper bound on, not always equal to, its true
        # leading response order, because get_active_modes collapses
        # (n, m, ST) to (n, m) keeping the minimum order and discarding ST
        # (couplings.py:143-147). Two distinct mechanisms produce this for
        # our rheology set (verified directly against coupling_coefficients,
        # not just inferred from the response):
        #  - (3,+/-2): its only order-1 *spheroidal* channel (rheology mode
        #    (3,+/-2) itself) has a genuinely near-zero coupling coefficient
        #    (~7.6e-17, an isolated selection-rule zero); its other order-1
        #    channels (rheology (2,+/-2), (4,+/-2)) are toroidal (k=0).
        #  - (5,+/-4): its only order-1 channel (rheology (4,+/-4)) has a
        #    significant coupling coefficient (0.632), but forcing degree 2
        #    + rheology degree 4 + response degree 5 has odd parity
        #    (2+4+5=11), which next_coupling assigns to the *toroidal*
        #    branch -- toroidal deformation carries no gravitational
        #    potential, so k=0 at that order regardless of the coefficient's
        #    size. Both modes' visible k response is therefore actually
        #    order-2, scaling ~eps**2 in this probe -- see
        #    docs/MARS_MODEL.md, "Lateral variations (TASK-016)", section 5.
        # Restrict the strict linearity check to the *dominant*
        # (largest-amplitude) order-1 modes, where this ambiguity does not
        # arise (measured: the top ~20 by amplitude all agree with linear
        # scaling to <0.3%).
        candidates = []
        for n_m, m_m in order1_nm:
            idx_lo = np.where((results[1e-3].n == n_m) & (results[1e-3].m == m_m))[0]
            idx_hi = np.where((results[1e-2].n == n_m) & (results[1e-2].m == m_m))[0]
            if len(idx_lo) == 0 or len(idx_hi) == 0:
                continue
            k_lo = abs(results[1e-3].k[idx_lo[0]])
            k_hi = abs(results[1e-2].k[idx_hi[0]])
            if k_lo < 1e-14:
                continue
            candidates.append((k_hi, n_m, m_m, k_lo, k_hi))
        candidates.sort(reverse=True)
        dominant = candidates[:10]
        assert len(dominant) >= 5, "too few order-1 modes with resolvable amplitude"

        checked = 0
        for _, n_m, m_m, k_lo, k_hi in dominant:
            ratio = k_hi / k_lo
            assert abs(ratio - 10.0) / 10.0 < 0.01, (
                f"mode (n={n_m}, m={m_m}): ratio {ratio:.4f} deviates from "
                f"linear (10.0) by more than 1%"
            )
            checked += 1
        assert checked > 0, "no order-1 mode had a comparable pair -- vacuous test"

    def test_forcing_mode_scaling_exponents(self):
        """D5: the forcing-mode (2,0) k2 shift is driven to FIRST order by
        the (4,0) harmonic alone (even parity 2+2+4=8, self-coupling the
        forcing mode back to itself), unlike every other harmonic (second
        order, e.g. (3,0), odd parity). Isolate each harmonic and fit the
        shift's scaling exponent between eps=1e-3 and eps=1e-2."""
        base_entries = mu_variable_from_topography(lmax=4)[CRUST_LAYER_INDEX]
        e40 = [e for e in base_entries if e[0] == 4 and e[1] == 0]
        e30 = [e for e in base_entries if e[0] == 3 and e[1] == 0]
        assert e40 and e30, "expected (4,0) and (3,0) harmonics in the base entries"

        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=15, perturbation_order=2)
        love_1d, _, _ = get_love(model, forcing, numerics)
        k2_1d = complex(love_1d.k[0])

        def exponent(base):
            shifts = {}
            for eps in (1e-3, 1e-2):
                scaled = _scale_entries(base, eps)
                love, _, _ = get_love(model, forcing, numerics, mu_variable={CRUST_LAYER_INDEX: scaled})
                k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
                shifts[eps] = abs(complex(love.k[k_idx]) - k2_1d)
            return math.log(shifts[1e-2] / shifts[1e-3]) / math.log(10.0)

        exp_40 = exponent(e40)
        exp_30 = exponent(e30)
        assert 0.9 <= exp_40 <= 1.1, f"(4,0) exponent {exp_40:.4f} not first-order"
        assert 1.8 <= exp_30 <= 2.2, f"(3,0) exponent {exp_30:.4f} not second-order"


# ---------------------------------------------------------------------------
# 4. JAX / NumPy coupled equivalence
# ---------------------------------------------------------------------------

class TestJaxEquivalence:

    def test_jax_matches_numpy_coupled(self):
        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=4, method="combination", Nrbase=15, perturbation_order=2,
        )
        numerics, model = set_boundary_indices(numerics, model)
        model = get_rheology(model, forcing)

        mu_variable = mu_variable_from_topography(lmax=4)
        model, lateral = process_lateral_variations(
            model, forcing, mu_variable=mu_variable,
        )
        couplings = get_couplings(
            lateral.variations, forcing.n, forcing.m,
            perturbation_order=numerics.perturbation_order,
        )

        y_ref, r_ref, _Y_ref, _aux_ref = _get_solution_coupled(
            model, forcing, numerics, couplings, lateral,
        )
        y_jax, r_jax, _Y_jax, _aux_jax = jax_get_solution_coupled_scan(
            model, forcing, numerics, couplings, lateral,
        )

        np.testing.assert_allclose(r_jax, r_ref)
        rel = _relative_error(y_jax, y_ref)
        assert rel < 1e-9, f"JAX/NumPy coupled relative error {rel:.3e} >= 1e-9"

        love_ref = extract_love_numbers(y_ref, model, forcing, couplings=couplings)
        love_jax = extract_love_numbers(y_jax, model, forcing, couplings=couplings)
        rel_k = _relative_error(love_jax.k, love_ref.k)
        assert rel_k < 1e-9, f"JAX/NumPy Love-number relative error {rel_k:.3e}"


# ---------------------------------------------------------------------------
# 5. Airy numbers sanity (D2 areoid-corrected)
# ---------------------------------------------------------------------------

class TestAiryNumbers:

    def test_max_dt_amplitude_in_areoid_corrected_band(self):
        diag = crustal_thickness_diagnostics(lmax=4)
        max_dt_km = diag["max_abs_dt_m"] / 1e3
        # Measured post-D2 (see docs/MARS_MODEL.md): 34.2 km. Band is the
        # design's original +/- 2 km around that measured value, tight
        # enough to catch a regression in the areoid correction itself.
        assert 32.2 < max_dt_km < 36.2, f"max|dt| = {max_dt_km:.2f} km out of band"

    def test_dt_below_shell_thickness(self):
        diag = crustal_thickness_diagnostics(lmax=4)
        assert diag["max_abs_dt_over_t0"] < 1.0

    def test_dmu_over_mu_below_elastic_positivity_bound(self):
        """The positivity guard the D2 areoid correction was meant to fix
        (pre-correction this was 1.017, i.e. violated)."""
        diag = crustal_thickness_diagnostics(lmax=4)
        assert diag["max_abs_dmu_over_mubar"] < 1.0, (
            f"max|d(mu)/mu_bar| = {diag['max_abs_dmu_over_mubar']:.4f} >= 1.0"
        )

    def test_degree_dichotomy(self):
        """Degree 1 dominant or at least comparable to degrees 2-3."""
        diag = crustal_thickness_diagnostics(lmax=4)
        rms = diag["degree_rms_km"]
        deg1 = rms[1]
        others_max = max(rms[2], rms[3])
        assert deg1 > 0.4 * others_max, (
            f"degree 1 ({deg1:.2f} km) unexpectedly small vs degrees 2-3 "
            f"(max {others_max:.2f} km)"
        )


# ---------------------------------------------------------------------------
# 6. Forcing-mode perturbation is small
# ---------------------------------------------------------------------------

class TestForcingModePerturbation:

    def test_k2_shift_much_smaller_than_observational_uncertainty(self):
        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=4, method="combination", Nrbase=15, perturbation_order=2,
        )
        love_1d, _, _ = get_love(model, forcing, numerics)
        k2_1d = complex(love_1d.k[0])

        mu_variable = mu_variable_from_topography(lmax=4)
        love_3d, _, _ = get_love(
            model, forcing, numerics, mu_variable=mu_variable,
        )
        k_idx = np.where((love_3d.n == 2) & (love_3d.m == 0))[0][0]
        k2_3d = complex(love_3d.k[k_idx])

        shift = abs(k2_3d - k2_1d)
        assert shift < 0.1 * MARS["k2_sigma"], (
            f"|k2_lateral - k2_1d| = {shift:.3e}, expected << "
            f"k2_sigma={MARS['k2_sigma']}"
        )


# ---------------------------------------------------------------------------
# 7. lmax=5 truncation-sensitivity spot check
# ---------------------------------------------------------------------------

class TestTruncationSensitivity:

    @pytest.mark.slow
    def test_lmax5_k2_shift_within_20_percent(self):
        """Cutoff sensitivity, as the design doc required: (4,0) -- the
        harmonic driving the forcing-mode shift to first order (D5) -- sits
        right at the lmax=4 truncation edge, so this is the sharpest
        available check of whether n_lv<=4 is an adequate cutoff."""
        model = build_mars_model()
        forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=4, method="combination", Nrbase=15, perturbation_order=2,
        )
        love_1d, _, _ = get_love(model, forcing, numerics)
        k2_1d = complex(love_1d.k[0])

        shifts = {}
        for lmax in (4, 5):
            mu_variable = mu_variable_from_topography(lmax=lmax)
            love, _, _ = get_love(model, forcing, numerics, mu_variable=mu_variable)
            k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
            shifts[lmax] = abs(complex(love.k[k_idx]) - k2_1d)

        rel_change = abs(shifts[5] - shifts[4]) / abs(shifts[4])
        assert rel_change < 0.20, (
            f"lmax=4->5 k2-shift relative change {rel_change:.2%} >= 20%"
        )
