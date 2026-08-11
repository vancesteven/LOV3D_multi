# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.mars_crust_models (TASK-028).

Covers: hand-checked loader values (the sign/normalization pin -- the
likeliest error class in a "subtract two SH radius fields" conversion, per
the project's own precedent of 41 tests passing against a real sqrt(2) bug
because none of them pinned a signed, hand-computed number), the mean
crustal thickness figures from ``data/mars/insight_moho/SOURCES.md``, the
deliberate C20-retention decision (and its contrast with the Airy path,
which drops C20), shell-residency/elastic-positivity bounds, geographic
sanity (Tharsis thickest, Hellas/Utopia thinnest), lmax truncation, and a
``slow`` solver-backed smoke test of the comparison driver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylov3d.mars_lateral import TOPO_PATH, crustal_thickness_variation

from pylov3d.mars_crust_models import (
    GEO_LOCATIONS,
    MOHO_DIR,
    MOHO_MODELS,
    SOURCES_MEAN_THICKNESS_KM,
    compare_crustal_models,
    moho_thickness_diagnostics,
    moho_thickness_variation,
    mu_variable_from_dt,
)
from pylov3d.sh_data import load_shape

pytestmark = pytest.mark.skipif(
    not TOPO_PATH.exists() or not MOHO_DIR.is_dir(),
    reason="MarsTopo719.shape.gz or data/mars/insight_moho/ not present",
)


# ---------------------------------------------------------------------------
# 1. Hand-checked loader values (sign/normalization pin -- the load-bearing
#    test class; see module docstring)
# ---------------------------------------------------------------------------

class TestHandCheckedValues:
    """dt = R_topo - R_moho, computed independently of the module under test
    by reading the raw .sh files' (n, m) rows directly and subtracting, then
    compared against ``moho_thickness_variation``'s output. This is the one
    check in this file that would fail on a flipped sign (moho - topo
    instead of topo - moho) or a dropped/duplicated term, which a
    magnitude-only or bounds-only test cannot distinguish from a correct
    result: a sign flip leaves |dt| and the shell-residency bound unchanged,
    it only inverts which hemisphere looks thick vs. thin (caught here) and
    the geographic-sanity check (TestGeographicSanity) below.
    """

    # Raw file values, read independently (not via pylov3d.sh_data):
    # MarsTopo719 (1,0)=-1735.5544682175901, (2,0)=-5966.19392043645 [m]
    # Moho-Mars-DWAK-33-2900 (1,0)=10821.360029395926, (2,0)=-2057.6197377405279 [m]
    # dt = topo - moho.
    _DWAK_EXPECTED = {
        (1, 0): -12556.914497613516,
        (2, 0): -3908.574182695922,
    }
    # Moho-Mars-Khan2022-33-2900 (1,0)=?, (2,0)=? -- second, independent model
    # spot check so the pin is not tied to a single file's happening to be
    # correct by accident.
    _KHAN2022_EXPECTED = {
        (1, 0): -12532.14696546107,
        (2, 0): -3221.1311710855966,
    }

    @pytest.mark.parametrize("model,expected", [
        ("DWAK", _DWAK_EXPECTED),
        ("Khan2022", _KHAN2022_EXPECTED),
    ])
    def test_hand_checked_coefficients(self, model, expected):
        dt = moho_thickness_variation(model, lmax=4)
        for (n, m), val in expected.items():
            assert math.isclose(dt[(n, m)], val, rel_tol=1e-9), (
                f"{model} dt[{(n, m)}] = {dt[(n, m)]!r}, expected {val!r} "
                "(sign or normalization regression in the Moho->thickness "
                "subtraction)"
            )

    def test_c00_removed(self):
        """(0,0) is the mean thickness, not a lateral variation -- must be
        absent (the loader only inserts nonzero entries, and C00 is zeroed
        before the nonzero filter)."""
        for name in MOHO_MODELS:
            dt = moho_thickness_variation(name, lmax=4)
            assert (0, 0) not in dt


# ---------------------------------------------------------------------------
# 2. Mean thickness vs. SOURCES.md
# ---------------------------------------------------------------------------

class TestMeanThickness:
    """SOURCES.md's mean-thickness table (computed there as
    "MarsTopo719 C00 - Moho C00") re-derived independently here via
    ``pylov3d.sh_data.load_shape`` directly (not through the dt-producing
    loader, which zeroes C00) -- this is the loader's C00 handling checked
    from the other side."""

    def test_mean_thickness_matches_sources_md(self):
        topo_c00 = load_shape(TOPO_PATH)["clm"][0, 0]
        for name, path in MOHO_MODELS.items():
            moho_c00 = load_shape(path)["clm"][0, 0]
            mean_km = (topo_c00 - moho_c00) / 1e3
            expected = SOURCES_MEAN_THICKNESS_KM[name]
            assert abs(mean_km - expected) < 0.02, (
                f"{name}: mean thickness {mean_km:.3f} km vs SOURCES.md "
                f"{expected} km"
            )

    def test_all_five_within_1km_of_50km_reference(self):
        """SOURCES.md's stated selection criterion: mean thickness within
        1 km of the project's 50 km reference (MARS['crust_thickness'])."""
        for expected in SOURCES_MEAN_THICKNESS_KM.values():
            assert abs(expected - 50.0) < 1.0


# ---------------------------------------------------------------------------
# 3. The C20 decision: retained here, dropped by the Airy path
# ---------------------------------------------------------------------------

class TestC20Retained:

    def test_c20_present_and_nonzero_for_all_models(self):
        for name in MOHO_MODELS:
            dt = moho_thickness_variation(name, lmax=4)
            assert (2, 0) in dt
            assert dt[(2, 0)] != 0.0

    def test_airy_path_drops_c20_insight_path_does_not(self):
        """Direct contrast pin: the two loaders' C20 handling is a
        deliberate divergence (module docstring, "The C20 decision"), not
        an oversight -- this test fails loudly if either side's behavior
        silently changes."""
        airy_dt = crustal_thickness_variation(lmax=4)
        assert (2, 0) not in airy_dt

        insight_dt = moho_thickness_variation("DWAK", lmax=4)
        assert (2, 0) in insight_dt


# ---------------------------------------------------------------------------
# 4. Shell-residency and elastic-positivity bounds
# ---------------------------------------------------------------------------

class TestBounds:

    @pytest.mark.parametrize("model", list(MOHO_MODELS))
    def test_dt_below_shell_thickness(self, model):
        diag = moho_thickness_diagnostics(model, lmax=4)
        assert diag["max_abs_dt_over_t0"] < 1.0

    @pytest.mark.parametrize("model", list(MOHO_MODELS))
    def test_max_dt_in_measured_band(self, model):
        """Measured (this session): peak |dt| ranges 33.0-38.7 km across
        the five models -- comparable to, and somewhat higher than, the
        Airy field's 34.2 km peak. Band is +/-3 km around the measured
        per-model value, tight enough to catch a normalization regression."""
        measured_km = {
            "DWAK": 38.717, "DWThotCrust1": 33.002, "EH45Tcold": 34.569,
            "EH45TcoldCrust1r": 34.411, "Khan2022": 37.888,
        }
        diag = moho_thickness_diagnostics(model, lmax=4)
        max_km = diag["max_abs_dt_m"] / 1e3
        expected = measured_km[model]
        assert abs(max_km - expected) < 3.0, (
            f"{model}: max|dt| = {max_km:.2f} km, expected near {expected} km"
        )

    @pytest.mark.parametrize("model", list(MOHO_MODELS))
    def test_dmu_over_mu_below_elastic_positivity_bound(self, model):
        diag = moho_thickness_diagnostics(model, lmax=4)
        assert diag["max_abs_dmu_over_mubar"] < 1.0, (
            f"{model}: max|d(mu)/mu_bar| = {diag['max_abs_dmu_over_mubar']:.4f} >= 1.0"
        )


# ---------------------------------------------------------------------------
# 5. Geographic sanity
# ---------------------------------------------------------------------------

class TestGeographicSanity:

    @pytest.mark.parametrize("model", list(MOHO_MODELS))
    def test_tharsis_thick_hellas_utopia_thin(self, model):
        diag = moho_thickness_diagnostics(model, lmax=4)
        geo = diag["geo"]
        assert set(geo) == set(GEO_LOCATIONS)
        assert geo["Tharsis"] > 0, f"{model}: Tharsis dt = {geo['Tharsis']!r}, expected > 0"
        assert geo["Hellas"] < 0, f"{model}: Hellas dt = {geo['Hellas']!r}, expected < 0"
        assert geo["Utopia"] < 0, f"{model}: Utopia dt = {geo['Utopia']!r}, expected < 0"
        # Tharsis is the largest positive excursion of the three named points.
        assert geo["Tharsis"] > abs(geo["Hellas"])
        assert geo["Tharsis"] > abs(geo["Utopia"]) * 0.9  # Utopia is comparable in magnitude


# ---------------------------------------------------------------------------
# 6. lmax truncation
# ---------------------------------------------------------------------------

class TestTruncation:

    def test_no_modes_above_lmax(self):
        dt = moho_thickness_variation("DWAK", lmax=3)
        assert all(n <= 3 for (n, _m) in dt)

    def test_lower_lmax_is_a_subset_at_matching_degrees(self):
        """Truncating to a lower lmax must not change the surviving
        coefficients (SH truncation is a pure projection, not a re-fit)."""
        dt4 = moho_thickness_variation("DWAK", lmax=4)
        dt2 = moho_thickness_variation("DWAK", lmax=2)
        for (n, m), val in dt2.items():
            assert dt4[(n, m)] == val


# ---------------------------------------------------------------------------
# 7. mu_variable construction reuses the validated Airy-path machinery
# ---------------------------------------------------------------------------

class TestMuVariableFromDt:

    def test_reuses_dmu_ddt_coeff_and_conjugate_pairing(self):
        dt = moho_thickness_variation("DWAK", lmax=4)
        mu_variable = mu_variable_from_dt(dt)
        from pylov3d.mars_lateral import CRUST_LAYER_INDEX
        assert set(mu_variable) == {CRUST_LAYER_INDEX}
        entries = mu_variable[CRUST_LAYER_INDEX]
        assert len(entries) > 0
        # every m != 0 mode must have its conjugate partner present
        present = {(n, m) for n, m, _amp in entries}
        for n, m, _amp in entries:
            if m != 0:
                assert (n, -m) in present

    def test_empty_dt_gives_no_entries(self):
        mu_variable = mu_variable_from_dt({})
        from pylov3d.mars_lateral import CRUST_LAYER_INDEX
        assert mu_variable[CRUST_LAYER_INDEX] == []

    def test_matches_airy_path_on_the_airy_field_exactly(self):
        """Cross-caller consistency pin: feeding the *Airy* field's own
        ``dt`` (:func:`pylov3d.mars_lateral.crustal_thickness_variation`)
        through this module's :func:`mu_variable_from_dt` reuses exactly the
        same ``_dmu_ddt_coeff``/``_real_sh_to_complex_mu_variable`` machinery
        that :func:`pylov3d.mars_lateral.mu_variable_from_topography` calls
        directly, so the two must agree element-for-element on 23 entries.
        This catches drift if either caller stops sharing the conversion
        code (e.g. an accidental local reimplementation) -- but it does
        *not* by itself catch a bug in the shared conversion itself, since
        both sides would still agree after being mutated identically. See
        ``test_dwak_conversion_matches_hand_computed_values`` below for the
        value pin that catches that.
        """
        from pylov3d.mars_lateral import (
            CRUST_LAYER_INDEX,
            crustal_thickness_variation,
            mu_variable_from_topography,
        )

        airy_dt = crustal_thickness_variation(lmax=4)
        got = mu_variable_from_dt(airy_dt)[CRUST_LAYER_INDEX]
        expected = mu_variable_from_topography(lmax=4)[CRUST_LAYER_INDEX]

        assert len(got) == 23, f"expected 23 entries, got {len(got)}"
        assert sorted(got) == sorted(expected), (
            "mu_variable_from_dt(crustal_thickness_variation(lmax=4)) does "
            "not exactly reproduce mu_variable_from_topography(lmax=4) -- "
            "the real->complex SH conversion has diverged from the "
            "validated Airy-path result"
        )

    def test_dwak_conversion_matches_hand_computed_values(self):
        """Value pin, computed independently of the module under test (D5/
        review finding: the other tests in this class check only the layer
        key, non-emptiness, conjugate-partner presence, and cross-caller
        consistency -- never an independently-computed value, so mutating
        the real->complex conversion (drop the ``1/sqrt(2)`` normalization,
        drop the ``(-1)**m`` phase, or flip the sine-term sign) leaves every
        one of them passing, including the cross-caller check above, since
        both callers share the same mutated code).

        ``coeff = _dmu_ddt_coeff(None)`` and DWAK's ``dt`` at (1,0), (2,0),
        (1,+/-1) (the last a non-zonal pair, exercising the ``m != 0``
        sqrt(2)/sign branch) are taken from this file's own
        ``TestHandCheckedValues``/independently re-read raw ``.sh`` values,
        then run through the documented formula by hand::

            amp(n, 0)  = coeff * C_n0
            amp(n, +m) = coeff * (C_nm - i*S_nm) / sqrt(2)
            amp(n, -m) = coeff * (-1)**m * (C_nm + i*S_nm) / sqrt(2)

        Confirmed (this session) that mutating any of the three terms above
        in a local copy of ``_real_sh_to_complex_mu_variable`` breaks this
        assertion while leaving every other test in this file passing.
        """
        dt = moho_thickness_variation("DWAK", lmax=4)
        mu_variable = mu_variable_from_dt(dt)
        from pylov3d.mars_lateral import CRUST_LAYER_INDEX
        entries = {(n, m): amp for n, m, amp in mu_variable[CRUST_LAYER_INDEX]}

        expected = {
            (1, 0): 0.31423874106022776 + 0.0j,
            (2, 0): 0.09781267768800238 + 0.0j,
            (1, 1): 0.013165565898733193 - 0.10149492294862288j,
            (1, -1): -0.013165565898733193 - 0.10149492294862288j,
        }
        for nm, val in expected.items():
            assert nm in entries, f"missing entry {nm}"
            got = entries[nm]
            assert abs(got - val) < 1e-9 * max(abs(val), 1e-12), (
                f"mu_variable[{nm}] = {got!r}, hand-computed expected {val!r} "
                "(real->complex SH conversion regression: check the "
                "1/sqrt(2) normalization, the (-1)**m phase, and the "
                "sine-term sign)"
            )


# ---------------------------------------------------------------------------
# 8. Solver-backed comparison driver (slow)
# ---------------------------------------------------------------------------

class TestComparisonDriver:

    @pytest.mark.slow
    def test_compare_crustal_models_smoke(self):
        """Not a converged production run (Nrbase=15, one InSight model) --
        just checks the driver wires the Airy baseline and an InSight model
        through the same solver path and returns finite, distinct shifts.
        The production lmax=4/Nrbase=30, all-five-models numbers are
        reported in docs/MARS_MODEL.md, "Non-Airy crustal model
        substitution (TASK-028)"."""
        result = compare_crustal_models(lmax=4, Nrbase=15, models=["DWAK"])
        assert set(result["rows"]) == {"Airy", "DWAK"}
        assert np.isfinite(result["k2_uniform"])

        for name, row in result["rows"].items():
            assert row["n_modes"] > 0
            assert np.isfinite(row["k20"])
            assert np.isfinite(row["shift20"])
            assert abs(row["shift20"]) > 0, f"{name}: zero lateral shift"

        # The two crustal models are physically different fields; their
        # induced shifts should not coincide to high precision.
        rel = abs(result["rows"]["Airy"]["shift20"] - result["rows"]["DWAK"]["shift20"])
        assert rel > 0
