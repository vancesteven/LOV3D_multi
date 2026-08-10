# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for TASK-026 (pylov3d/mars_detectability.py).

Fast lane (default `pytest -q`, no `slow` marker): the amplitude->observable
formula, its degree-2 hand-check, the spectrum loader, and the
detectability table -- all pure-Python/array arithmetic plus one small
scipy.io.loadmat call, well under 1 s total. Slow lane
(`TestForcingOrderRobustness`): two reduced-grid coupled solves through
`pylov3d.mars_lateral.mars_lateral_love_spectrum` (~10-20 s each).
"""

from __future__ import annotations

import math

import pytest

from pylov3d.mars import MARS
from pylov3d.mars_detectability import (
    AU_M,
    DEFAULT_SPECTRUM_PATH,
    DEGREE2_HAND_CHECK_DC20,
    GM_SUN,
    GRAIL_K3,
    GRAIL_SIGMA_K3,
    MARS_ORBITAL_PERIOD_S,
    MARS_PERIHELION_M,
    MARS_SEMIMAJOR_AXIS_M,
    MARS_SIGMA_C30_SEASONAL,
    forcing_order_robustness_check,
    frequency_separation_factor,
    load_mars_lateral_spectrum,
    mars_off20_detectability_table,
    mars_required_precision,
    peak_legendre_factor,
    required_stokes_amplitude,
    solar_tide_amplitude_parameter,
)


# ---------------------------------------------------------------------------
# 1. The core relation and its degree-2 hand check
# ---------------------------------------------------------------------------

class TestDegree2HandCheck:

    def test_degree2_hand_check(self):
        """The generalized relation (eq. * in the module docstring),
        evaluated at n=n_forcing=2, m=m_forcing=0 with the real, measured
        Mars k2, must reproduce the independently (by-hand) computed
        constant -- this is the "if your machinery cannot reproduce the
        degree-2 case it is wrong" pin the task spec asks for."""
        dc20 = required_stokes_amplitude(
            MARS["k2"], GM_SUN, MARS["GM"], MARS["R"], MARS_SEMIMAJOR_AXIS_M,
            n_forcing=2, m_forcing=0,
        )
        assert dc20 == pytest.approx(DEGREE2_HAND_CHECK_DC20, rel=1e-12)

    def test_degree2_collapses_diagonal_formula_by_hand(self):
        """Recompute the same number from the raw formula text (module
        docstring eq. *), independently of solar_tide_amplitude_parameter/
        peak_legendre_factor, as an extra guard against both those helpers
        sharing a common bug."""
        xi = (GM_SUN / MARS["GM"]) * (MARS["R"] / MARS_SEMIMAJOR_AXIS_M) ** 3 / 5.0
        pbar20_equinox = math.sqrt(5.0) * 0.5  # Pbar_2^0(0) = sqrt(5)*P2(0), P2(0)=-1/2
        dc20 = MARS["k2"] * xi * pbar20_equinox
        assert dc20 == pytest.approx(DEGREE2_HAND_CHECK_DC20, rel=1e-12)

    def test_solar_tide_amplitude_parameter_matches_hand_value(self):
        xi = solar_tide_amplitude_parameter(
            GM_SUN, MARS["GM"], MARS["R"], MARS_SEMIMAJOR_AXIS_M, n_forcing=2,
        )
        assert xi == pytest.approx(2.0377939372960587e-09, rel=1e-9)

    def test_amplitude_parameter_positive_and_smaller_at_higher_forcing_degree(self):
        xi2 = solar_tide_amplitude_parameter(
            GM_SUN, MARS["GM"], MARS["R"], MARS_SEMIMAJOR_AXIS_M, n_forcing=2,
        )
        xi3 = solar_tide_amplitude_parameter(
            GM_SUN, MARS["GM"], MARS["R"], MARS_SEMIMAJOR_AXIS_M, n_forcing=3,
        )
        assert xi2 > 0.0
        assert xi3 > 0.0
        assert xi3 < xi2  # (R/d) << 1 for Mars-Sun, so higher n suppresses further

    def test_perihelion_amplitude_exceeds_mean_distance(self):
        xi_mean = solar_tide_amplitude_parameter(
            GM_SUN, MARS["GM"], MARS["R"], MARS_SEMIMAJOR_AXIS_M, n_forcing=2,
        )
        xi_peri = solar_tide_amplitude_parameter(
            GM_SUN, MARS["GM"], MARS["R"], MARS_PERIHELION_M, n_forcing=2,
        )
        assert xi_peri > xi_mean


class TestPeakLegendreFactor:

    def test_zonal_20_matches_hand_value(self):
        # Pbar_2^0(sin phi)=sqrt(5)*(3 sin^2 phi - 1)/2; |.| maximal at
        # phi=0 (equinox), value sqrt(5)/2.
        assert peak_legendre_factor(2, 0) == pytest.approx(math.sqrt(5.0) / 2.0, rel=1e-12)

    def test_sectoral_22_matches_hand_value(self):
        # N_22 = sqrt(5*2*1/24); Pbar_2^2(x) = N_22 * 3*(1-x^2), maximal
        # at x=0 (equinox).
        n22 = math.sqrt(5.0 * 2.0 * 1.0 / 24.0)
        assert peak_legendre_factor(2, 2) == pytest.approx(n22 * 3.0, rel=1e-12)

    def test_sectoral_exceeds_zonal(self):
        # Drives the "optimistic" bound's m_forcing=2 choice.
        assert peak_legendre_factor(2, 2) > peak_legendre_factor(2, 0)


class TestConstantsProvenance:

    def test_au_matches_iau_2012_resolution_b2(self):
        assert AU_M == 149_597_870_700.0

    def test_gm_sun_matches_iau_2015_resolution_b3(self):
        assert GM_SUN == pytest.approx(1.3271244e20, rel=1e-12)

    def test_mars_semimajor_axis_positive_and_au_scale(self):
        assert 1.0 * AU_M < MARS_SEMIMAJOR_AXIS_M < 2.0 * AU_M

    def test_mars_perihelion_less_than_semimajor_axis(self):
        assert 0.0 < MARS_PERIHELION_M < MARS_SEMIMAJOR_AXIS_M


# ---------------------------------------------------------------------------
# 2. The N=115 spectrum (MATLAB-cross-validated, TASK-016) and its
#    required-precision table
# ---------------------------------------------------------------------------

class TestSpectrumLoader:

    def test_default_spectrum_path_exists(self):
        assert DEFAULT_SPECTRUM_PATH.exists()

    def test_loads_115_modes_forcing_20(self):
        spectrum = load_mars_lateral_spectrum()
        assert len(spectrum["modes"]) == 115
        assert spectrum["forcing_n"] == 2
        assert spectrum["forcing_m"] == 0
        assert sum(1 for mode in spectrum["modes"] if mode["is_forcing"]) == 1

    def test_top_mode_matches_task016_spec_table(self):
        """Regression pin against the values quoted in
        docs/tasks/TASK-026-off-forcing-detectability.md (themselves from
        the MATLAB-cross-validated data/tests/mars/mars_lateral_cross_check.mat)."""
        spectrum = load_mars_lateral_spectrum()
        top = next(mode for mode in spectrum["modes"] if not mode["is_forcing"])
        assert top["n"] == 3
        assert top["m"] == 0
        assert abs(top["k"]) == pytest.approx(7.29e-5, rel=2e-3)

    def test_k2_shift_matches_task016_value(self):
        spectrum = load_mars_lateral_spectrum()
        assert spectrum["k2_shift"] == pytest.approx(5.517410e-5, rel=1e-3)


_TABLE = mars_off20_detectability_table()


class TestDetectabilityTable:

    @pytest.fixture
    def table(self):
        return _TABLE

    def test_114_non_forcing_rows(self, table):
        assert len(table) == 114

    def test_sorted_by_descending_k(self, table):
        magnitudes = [row["k_abs"] for row in table]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_optimistic_bound_looser_than_conservative(self, table):
        """"optimistic" (perihelion + sectoral, largest real signal) is
        the loosest, most detection-favorable required-precision bound,
        i.e. numerically *larger* than "conservative" (mean distance +
        zonal) -- see mars_required_precision's docstring."""
        for row in table[:10]:
            assert row["required_dC_optimistic"] > row["required_dC_conservative"]

    def test_no_mode_is_currently_detectable(self, table):
        """The task expects (and instructs not to tune away from) a
        negative result: every ratio (achieved sigma / required precision)
        must exceed 1, i.e. current achieved precision is coarser than
        what any mode requires."""
        for row in table:
            assert row["ratio_orbiter"] > 1.0
            assert row["ratio_grail"] > 1.0

    def test_top_mode_ratios_match_hand_computed_order_of_magnitude(self, table):
        top = table[0]
        assert top["n"] == 3 and top["m"] == 0
        assert top["ratio_orbiter"] == pytest.approx(28.5, rel=0.05)
        assert top["ratio_grail"] == pytest.approx(28.8, rel=0.05)

    def test_invalid_bound_raises(self):
        with pytest.raises(ValueError):
            mars_required_precision(1e-5, bound="nonsense")


class TestAchievedPrecisionConstants:

    def test_mars_seasonal_sigma_positive(self):
        assert MARS_SIGMA_C30_SEASONAL > 0.0
        assert MARS_SIGMA_C30_SEASONAL < 1e-9  # sanity: sub-nanoradian-scale Stokes coefficient

    def test_grail_k3_relative_uncertainty_matches_konopliv_2013_text(self):
        # Konopliv et al. (2013) state "about 25%" in prose; the Table 4
        # numbers (k3=0.0089+/-0.0021) imply 23.6% -- consistent.
        assert GRAIL_SIGMA_K3 / GRAIL_K3 == pytest.approx(0.236, rel=0.05)


# ---------------------------------------------------------------------------
# 3. Frequency separation
# ---------------------------------------------------------------------------

class TestFrequencySeparation:

    def test_factor_order_of_magnitude(self):
        factor = frequency_separation_factor()
        assert 1000.0 < factor < 2000.0

    def test_factor_matches_hand_value(self):
        assert frequency_separation_factor() == pytest.approx(1337.2, rel=1e-3)

    def test_orbital_period_matches_genova_2016(self):
        assert MARS_ORBITAL_PERIOD_S == pytest.approx(686.98 * 86400.0)


# ---------------------------------------------------------------------------
# 4. Forcing-order scope caveat (module docstring, sec. 2) -- slow
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestForcingOrderRobustness:

    def test_20_vs_22_forcing_same_order_of_magnitude(self):
        """The (2,0)-forced (given, MATLAB-validated) and (2,2)-forced
        (real, semidiurnal-dominant) spectra excite different modes, but
        this must not differ by more than roughly an order of magnitude
        in overall amplitude scale -- otherwise the module docstring's
        "order-of-magnitude, not mode-exact" caveat would be too weak a
        statement."""
        check = forcing_order_robustness_check(lmax=2, Nrbase=30)
        assert 0.1 < check["ratio"] < 10.0
        assert check[0]["n"] == 3 and check[0]["m"] == 0
        assert check[2]["n"] == 3 and check[2]["m"] == 2
