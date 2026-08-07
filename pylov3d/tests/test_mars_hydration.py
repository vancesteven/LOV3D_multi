"""Tests for TASK-021 (pylov3d/mars_hydration.py).

Fast lane (default `pytest -q`, no `slow` marker): pure-Python
coefficient/geometry logic and single (uncoupled) 1D solves, all < 1 s.
Slow lane (`-m slow` or no `-m` filter): coupled solves through
`get_love_hydrated`, seconds to ~2.5 min total -- see
`TestRuntimeGuard` for the full-sweep timing assertion and the documented
`lmax` reduction (module docstring, "Runtime").
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pylov3d.love import get_love
from pylov3d.mars import LAYER_MU_CRUST, MARS, MARS_FORCING_TD, build_mars_model
from pylov3d.mars_hydration import (
    CRUST_REFERENCE_MU_GPA,
    K_CRUST,
    K_SERP_RATIO_CENTRAL,
    MU_SERP_RATIO_BRACKET,
    MU_SERP_RATIO_CENTRAL,
    RATIO_SCENARIOS,
    SIGMA_K2,
    build_hydrated_mars_model,
    crust_reference_sensitivity,
    detectability_summary,
    get_love_hydrated,
    hydration_dmu_over_mu_bar_real,
    hydration_forward_sweep,
    hydration_geometry_diagnostics,
    hydration_k2,
    hydration_lateral_variables,
    mean_softened_crust_moduli,
)
from pylov3d.mars_lateral import CRUST_LAYER_INDEX
from pylov3d.types import make_forcing, make_interior_model, make_numerics

FORCING = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
FAST_NUMERICS = make_numerics(n_layers=4, method="combination", Nrbase=30, perturbation_order=2)
SLOW_NUMERICS = make_numerics(n_layers=4, method="combination", Nrbase=15, perturbation_order=2)


# ---------------------------------------------------------------------------
# 1. Zero-amplitude reduction (fast: f_h=0 never reaches the coupled path)
# ---------------------------------------------------------------------------

class TestZeroAmplitudeReduction:

    def test_k2_matches_task011_baseline_exactly(self):
        love_ref, _, _ = get_love(build_mars_model(), FORCING, FAST_NUMERICS)
        k2_ref = complex(love_ref.k[0])

        r0 = hydration_k2(0.0, Nrbase=30)
        rel = abs(r0["k2_total"] - k2_ref) / abs(k2_ref)
        assert rel < 1e-12, f"f_h=0 relative k2 error {rel:.3e} >= 1e-12"
        assert r0["k2_lateral"] == 0.0
        assert r0["n_coupled_modes"] == 1

    def test_lateral_variables_empty_at_zero_amplitude(self):
        mu_variable, K_variable = hydration_lateral_variables(0.0)
        assert mu_variable == {}
        assert K_variable == {}
        assert hydration_dmu_over_mu_bar_real(0.0) == {}

    def test_softened_moduli_exact_at_zero_amplitude(self):
        mu0, Ks0 = mean_softened_crust_moduli(0.0)
        assert mu0 == LAYER_MU_CRUST
        assert Ks0 == K_CRUST


# ---------------------------------------------------------------------------
# 2. Sign: softening increases k2
# ---------------------------------------------------------------------------

class TestSofteningSign:

    def test_mean_term_increases_k2(self):
        base = hydration_k2(0.0, include_lateral=False)["k2_mean"].real
        for f_h in (0.1, 0.3, 0.5):
            k2_mean = hydration_k2(f_h, include_lateral=False)["k2_mean"].real
            assert k2_mean > base, f"mean k2 did not increase at f_h={f_h}"

    @pytest.mark.slow
    def test_lateral_term_increases_forced_mode_k2_at_validated_lmax(self):
        """Sign check at the validated lmax=4 config, NOT the module's
        reduced DEFAULT_LMAX=2 default (D4, review): at lmax=2 the
        forced (2,0) mode's lateral shift is purely second-order (missing
        the (4,0) first-order self-coupling channel -- module docstring,
        "Runtime" / docs/MARS_MODEL.md section 4), so its sign is not
        reliably diagnostic of "softening increases k2" there (review
        verified a full sign flip of the injected lateral field changes
        the lmax=2 result by only 0.6% and still passes a same-sign
        assertion -- i.e. a lmax=2-only version of this test is nearly
        vacuous). lmax=4 includes the first-order channel, so the sign
        check is meaningful here. Central ratio, f_h=0.3."""
        r = hydration_k2(0.3, lmax=4, Nrbase=30)
        assert r["k2_lateral"].real > 0.0


# ---------------------------------------------------------------------------
# 3. Linearity of the lateral contribution in f_h at small f_h
#    (TASK-016 pattern: dominant order-1 SPECTRUM modes scale linearly
#    with the rheology amplitude; NOT the forced (2,0) mode's own shift,
#    which is second-order at this lmax -- see module docstring)
# ---------------------------------------------------------------------------

class TestLateralLinearity:

    @pytest.mark.slow
    def test_order1_modes_scale_linearly_with_f_h(self):
        results = {}
        for f_h in (0.01, 0.02):
            model = build_hydrated_mars_model(f_h)
            mu_variable, K_variable = hydration_lateral_variables(f_h, lmax=2)
            love, _ = get_love_hydrated(
                model, FORCING, FAST_NUMERICS, mu_variable=mu_variable, K_variable=K_variable,
            )
            results[f_h] = love

        # (2,+-1)/(2,+-2) are dominant order-1 modes at lmax=2 (checked
        # directly against get_couplings when this test was designed);
        # the ratio between f_h=0.01 and f_h=0.02 should be close to 2.0.
        checked = 0
        for n_m, m_m in ((2, 1), (2, -1), (2, 2), (2, -2)):
            idx_lo = np.where((results[0.01].n == n_m) & (results[0.01].m == m_m))[0]
            idx_hi = np.where((results[0.02].n == n_m) & (results[0.02].m == m_m))[0]
            if len(idx_lo) == 0 or len(idx_hi) == 0:
                continue
            k_lo = abs(results[0.01].k[idx_lo[0]])
            k_hi = abs(results[0.02].k[idx_hi[0]])
            if k_lo < 1e-14:
                continue
            ratio = k_hi / k_lo
            assert abs(ratio - 2.0) / 2.0 < 0.02, (
                f"mode ({n_m},{m_m}): ratio {ratio:.4f} deviates from linear (2.0)"
            )
            checked += 1
        assert checked >= 2, "too few order-1 modes resolved -- test would be vacuous"


# ---------------------------------------------------------------------------
# 4. Mean-shift consistency: modified-mu0 model matches an independent
#    get_love call built by hand
# ---------------------------------------------------------------------------

class TestMeanShiftConsistency:

    def test_modified_mu0_matches_independent_model(self):
        f_h, mu_ratio, K_ratio = 0.25, MU_SERP_RATIO_CENTRAL, K_SERP_RATIO_CENTRAL
        mu0_soft, Ks0_soft = mean_softened_crust_moduli(f_h, mu_ratio, K_ratio)

        # Independent reconstruction: pull the exact geometry/densities out
        # of the plain reference model and rebuild by hand with the same
        # two crust moduli substituted, via a different code path than
        # build_hydrated_mars_model (make_interior_model directly, not
        # ._replace on build_mars_model()'s output).
        ref = build_mars_model()
        n = ref.n_layers
        R0_km = [float(x) for x in np.asarray(ref.R0[:n])]
        rho0 = [float(x) for x in np.asarray(ref.rho0[:n])]
        mu0_list = [float(x) for x in np.asarray(ref.mu0[:n])]
        Ks0_list = [float(x) for x in np.asarray(ref.Ks0[:n])]
        mu0_list[CRUST_LAYER_INDEX] = mu0_soft
        Ks0_list[CRUST_LAYER_INDEX] = Ks0_soft
        independent_model = make_interior_model(
            R0_km=R0_km, rho0=rho0, mu0=mu0_list, Ks0=Ks0_list,
            eta0=[None] * n,
        )
        love_independent, _, _ = get_love(independent_model, FORCING, FAST_NUMERICS)

        r = hydration_k2(f_h, mu_ratio=mu_ratio, K_ratio=K_ratio, include_lateral=False)
        assert r["k2_mean"] == complex(love_independent.k[0]), (
            "mean-softened k2 does not exactly match the independently "
            "rebuilt model"
        )

    def test_build_hydrated_model_only_touches_crust_layer(self):
        model = build_hydrated_mars_model(0.3)
        ref = build_mars_model()
        assert np.array_equal(np.asarray(model.R0), np.asarray(ref.R0))
        assert np.array_equal(np.asarray(model.rho0), np.asarray(ref.rho0))
        for i in range(model.n_layers):
            if i == CRUST_LAYER_INDEX:
                continue
            assert float(model.mu0[i]) == float(ref.mu0[i])
            assert float(model.Ks0[i]) == float(ref.Ks0[i])
        assert float(model.mu0[CRUST_LAYER_INDEX]) != float(ref.mu0[CRUST_LAYER_INDEX])

    def test_lateral_normalization_pin_at_10_mode(self):
        """D5 (review): pin the softened-mean-denominator normalization
        in hydration_lateral_variables -- computes the expected (1,0)
        entry independently (mean_softened_crust_moduli for mu0_soft,
        crustal_thickness_variation for dt), so a mutant that normalizes
        by the unsoftened LAYER_MU_CRUST instead (which the review found
        survives all other tests) is caught: the two denominators differ
        whenever f_h > 0, so the two formulas disagree numerically."""
        from pylov3d.mars_lateral import crustal_thickness_variation

        f_h, mu_ratio, K_ratio, lmax = 0.3, MU_SERP_RATIO_CENTRAL, K_SERP_RATIO_CENTRAL, 2
        dt = crustal_thickness_variation(lmax=lmax)
        mu0_soft, _ = mean_softened_crust_moduli(f_h, mu_ratio, K_ratio)
        mu_serp = mu_ratio * LAYER_MU_CRUST
        t0 = MARS["crust_thickness"]
        expected = f_h * (mu_serp - LAYER_MU_CRUST) * dt[(1, 0)] / (t0 * mu0_soft)

        # Sanity: the softened and unsoftened denominators must actually
        # differ here, or this test would not distinguish the mutant.
        assert mu0_soft != LAYER_MU_CRUST

        mu_variable, _ = hydration_lateral_variables(f_h, mu_ratio, K_ratio, lmax=lmax)
        entries = {(n, m): amp for n, m, amp in mu_variable[CRUST_LAYER_INDEX]}
        assert (1, 0) in entries
        assert entries[(1, 0)] == pytest.approx(complex(expected, 0.0), rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Property-bracket sanity: monotone in mu_serp ratio
# ---------------------------------------------------------------------------

class TestPropertyBracketMonotonicity:

    def test_k2_mean_monotone_decreasing_in_mu_ratio(self):
        """Softer serpentinite (lower mu_ratio) -> larger k2 increase; as
        mu_ratio -> 1 (no contrast) the shift -> 0."""
        f_h = 0.3
        ratios = (0.15, MU_SERP_RATIO_BRACKET[0], MU_SERP_RATIO_CENTRAL, MU_SERP_RATIO_BRACKET[1], 0.95)
        deltas = []
        for ratio in ratios:
            r = hydration_k2(f_h, mu_ratio=ratio, K_ratio=K_SERP_RATIO_CENTRAL, include_lateral=False)
            deltas.append(r["k2_mean"].real - MARS["k2"])
        assert all(deltas[i] > deltas[i + 1] for i in range(len(deltas) - 1)), (
            f"delta k2 not monotone decreasing in mu_ratio: {deltas}"
        )
        assert all(d > 0.0 for d in deltas), "softening should always increase k2 here"


# ---------------------------------------------------------------------------
# 5a. Reference-crust sensitivity (S1, review)
# ---------------------------------------------------------------------------

class TestCrustReferenceSensitivity:

    def test_self_referenced_baseline_not_global_constant(self):
        """S1 (review): delta_k2 must be self-referenced per row
        (k2_softened - k2_baseline for THAT row's own crust reference),
        not measured against the single global MARS["k2"]=0.169 -- an
        earlier draft did the latter, which conflates the hydration
        signal with the (different) baseline-k2 drift from swapping
        crust references, and even went negative for the peridotite row.
        All four self-referenced deltas must be positive (softening
        always raises k2 relative to its own row's baseline) and
        increasing with the crust reference's stiffness (a stiffer
        reference mixed toward the same fixed 14.4 GPa serpentinite
        modulus is a larger relative softening)."""
        rows = crust_reference_sensitivity()
        assert [r["label"] for r in rows] == list(CRUST_REFERENCE_MU_GPA.keys())
        assert all(r["delta_k2"] > 0.0 for r in rows), (
            f"expected all-positive self-referenced deltas: {rows}"
        )
        deltas = [r["delta_k2"] for r in rows]
        assert all(deltas[i] < deltas[i + 1] for i in range(len(deltas) - 1)), (
            f"delta_k2 not monotone increasing with crust-reference stiffness: {deltas}"
        )
        # Baseline k2 itself must drift across the row (mu_crust is not
        # freely adjustable without refitting) -- and bracket the shipped
        # model's fitted k2=0.169 near the middle of the reference range.
        baselines = [r["k2_baseline"] for r in rows]
        assert len(set(round(b, 6) for b in baselines)) == 4, "baselines should all differ"


# ---------------------------------------------------------------------------
# 6. Runtime guard
# ---------------------------------------------------------------------------

class TestRuntimeGuard:

    @pytest.mark.slow
    def test_full_grid_reduced_lmax_under_guard(self):
        """Full DEFAULT_F_H_GRID x RATIO_SCENARIOS sweep at the module's
        documented reduced-lmax default (lmax=2, Nrbase held at the
        validated 30) -- module docstring, "Runtime": the full lmax=4,
        Nrbase=30 config costs ~130-140 s/solve (TASK-016), which would
        put an 18-point sweep at ~40 min, over the spec's ~5 min guard;
        the spec explicitly permits reducing the grid and noting it.
        Bound set to 480s (not 300s): measured 154s on the development
        machine but 254s on a reviewer's machine (15% headroom against a
        300s cap -- too tight/flaky across machines); 480s keeps the
        guard meaningful (well under the ~40 min full-lmax cost) with
        real margin."""
        t0 = time.perf_counter()
        rows = hydration_forward_sweep()
        wall = time.perf_counter() - t0
        assert wall < 480.0, f"full reduced-lmax sweep took {wall:.1f}s >= 480s guard"
        assert len(rows) == 18  # 6 f_h values x 3 scenarios


# ---------------------------------------------------------------------------
# 7. Geometry clip-bound diagnostic (fast: no solver calls)
# ---------------------------------------------------------------------------

class TestGeometryDiagnostics:

    def test_bound_never_violated_over_swept_f_h(self):
        diag = hydration_geometry_diagnostics()
        for d in diag:
            assert not d["bound_violation"], (
                f"t_h exceeds the 50 km shell bound at f_h={d['f_h']} "
                f"(max_t_h_km={d['max_t_h_km']:.2f}); the linear/spectral "
                "approximation is no longer exact against the clip "
                "requirement over this range (see module docstring)"
            )
        assert diag[0]["max_t_h_km"] == 0.0  # f_h=0

    def test_max_t_h_increases_monotonically_with_f_h(self):
        diag = hydration_geometry_diagnostics()
        values = [d["max_t_h_km"] for d in diag]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------------------------
# 8. Detectability summary logic (fast: synthetic rows, no solver calls)
# ---------------------------------------------------------------------------

class TestDetectabilitySummary:

    def _synthetic_rows(self, k2_by_f_h: dict[float, float]) -> list[dict]:
        return [
            {"scenario": "central", "f_h": f_h, "k2_total": complex(k2, 0.0)}
            for f_h, k2 in k2_by_f_h.items()
        ]

    def test_crossing_detected_when_signal_exceeds_sigma(self):
        base = MARS["k2"]
        rows = self._synthetic_rows({
            0.0: base, 0.1: base + 0.1 * SIGMA_K2, 0.3: base + 2.0 * SIGMA_K2,
        })
        summary = detectability_summary(rows)
        assert summary["crossing_f_h"] == 0.3
        assert summary["precision_to_resolve_f_h_0p1"] == pytest.approx(0.1 * SIGMA_K2)

    def test_no_crossing_reported_as_none(self):
        base = MARS["k2"]
        rows = self._synthetic_rows({0.0: base, 0.1: base + 1e-6})
        summary = detectability_summary(rows)
        assert summary["crossing_f_h"] is None

    def test_missing_required_f_h_raises(self):
        rows = self._synthetic_rows({0.0: MARS["k2"]})
        with pytest.raises(ValueError):
            detectability_summary(rows)


# ---------------------------------------------------------------------------
# 9. get_love_hydrated regression guard: mu-only matches plain get_love
# ---------------------------------------------------------------------------

class TestGetLoveHydratedRegression:

    @pytest.mark.slow
    def test_mu_only_matches_plain_get_love(self):
        model = build_hydrated_mars_model(0.2)
        mu_variable, _ = hydration_lateral_variables(0.2, lmax=2)

        love_plain, _, _ = get_love(model, FORCING, SLOW_NUMERICS, mu_variable=mu_variable)
        love_wrapped, _ = get_love_hydrated(model, FORCING, SLOW_NUMERICS, mu_variable=mu_variable)

        assert np.array_equal(np.asarray(love_plain.n), np.asarray(love_wrapped.n))
        assert np.array_equal(np.asarray(love_plain.m), np.asarray(love_wrapped.m))
        assert np.allclose(np.asarray(love_plain.k), np.asarray(love_wrapped.k), rtol=0, atol=0)

    @pytest.mark.slow
    def test_K_variable_actually_changes_the_solution(self):
        """Guards against a no-op K injection (mirrors
        test_jax_coupled_scan.py::test_nonzero_K_amp_selection's own
        rationale: the default pipeline produces identically zero K_amp,
        so a silently-broken injection would pass every other test)."""
        model = build_hydrated_mars_model(0.3)
        mu_variable, K_variable = hydration_lateral_variables(0.3, lmax=2)
        assert K_variable, "expected nonempty K_variable at f_h=0.3 -- test setup is wrong"

        love_mu_only, _ = get_love_hydrated(model, FORCING, SLOW_NUMERICS, mu_variable=mu_variable)
        love_mu_and_K, _ = get_love_hydrated(
            model, FORCING, SLOW_NUMERICS, mu_variable=mu_variable, K_variable=K_variable,
        )
        assert not np.allclose(
            np.asarray(love_mu_only.k), np.asarray(love_mu_and_K.k), rtol=0, atol=0,
        ), "adding K_variable did not change the coupled solution"

    @pytest.mark.slow
    def test_empty_mu_variable_with_nonempty_K_variable_still_couples(self):
        """D3 (review): mu_ratio=1.0 makes hydration_lateral_variables'
        mu_variable exactly empty (mu_serp == mu_crust, coefficient is
        exactly 0.0) while K_variable stays nonempty (K_ratio != 1.0).
        get_love_hydrated must not silently collapse to the trivial
        (uncoupled, N=1) solve in this case -- found in review: the
        pre-fix code only passed mu_variable into
        process_lateral_variations, so its (n,m) union never saw K's
        modes when mu_variable was empty, discarding the whole K
        perturbation without warning."""
        model = build_hydrated_mars_model(0.3, mu_ratio=1.0, K_ratio=K_SERP_RATIO_CENTRAL)
        mu_variable, K_variable = hydration_lateral_variables(
            0.3, mu_ratio=1.0, K_ratio=K_SERP_RATIO_CENTRAL, lmax=2,
        )
        assert mu_variable == {}, "test setup expects mu_variable to be empty at mu_ratio=1.0"
        assert K_variable, "test setup expects nonempty K_variable"

        love, _ = get_love_hydrated(
            model, FORCING, SLOW_NUMERICS, mu_variable=mu_variable, K_variable=K_variable,
        )
        assert len(love.n) > 1, "collapsed to the trivial uncoupled solve -- K_variable was dropped"
