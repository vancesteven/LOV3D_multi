# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for TASK-026 (pylov3d/mars_detectability_k2m.py).

Fast lane: the pre-computed diagonal-splitting table (pure arithmetic
against hardcoded, this-session-computed constants). Slow lane
(`TestRecomputeK2mDiagonalShift`): reruns the two coupled solves that
produced those constants (~130-140 s each) to confirm they still
reproduce.
"""

from __future__ import annotations

import pytest

from pylov3d.mars import MARS
from pylov3d.mars_detectability_k2m import (
    GRAIL_K2M,
    GRAIL_K2M_GSFC,
    GRAIL_K2M_SIGMA,
    GRAIL_K2M_SIGMA_GSFC,
    MARS_K21_FORCING,
    MARS_K22_FORCING,
    MARS_K2M_LMAX2_DELTA,
    MAQUIS_CURRENT_RESOLUTION_DO,
    MAQUIS_SEASONAL_SIGNAL_UGAL,
    MAQUIS_SECULAR_GOAL_UGAL_PER_YR,
    MAQUIS_TARGET_RESOLUTION_DO,
    MOON_XI_2,
    grail_k2m_cross_analysis_disagreement,
    mars_diagonal_k2m_table,
    moon_mars_xi_ratio,
    recompute_k2m_diagonal_shift,
)


_TABLE = mars_diagonal_k2m_table()


class TestDiagonalSplittingTable:

    @pytest.fixture
    def table(self):
        return _TABLE

    def test_three_orders(self, table):
        assert [row["m"] for row in table] == [0, 1, 2]

    def test_m0_matches_task016_validated_shift(self, table):
        """m=0's delta must equal the MATLAB-cross-validated TASK-016
        k2_shift (5.517410e-5) -- this module must not silently
        recompute or override that already-validated number."""
        row = table[0]
        assert row["delta"] == pytest.approx(5.517410e-5, rel=1e-3)

    def test_baseline_is_uniform_k2(self, table):
        for row in table:
            assert row["k2m"] - row["delta"] == pytest.approx(MARS["k2"], abs=1e-12)

    def test_all_deltas_positive_and_distinct(self, table):
        """Regression pin: all three orders shift the same sign (module
        docstring numbers), and are not degenerate with one another --
        i.e. there is an actual, nonzero order-dependent splitting to
        look for, not three copies of the same number."""
        deltas = [row["delta"] for row in table]
        assert all(d > 0 for d in deltas)
        assert len(set(round(d, 10) for d in deltas)) == 3

    def test_no_mode_currently_detectable_at_grail_lunar_precision(self, table):
        """The task expects (and instructs not to tune away from) a
        negative result here too: GRAIL's own best lunar individual-order
        precision must not be tight enough to resolve any predicted
        Mars order, i.e. every ratio must exceed 1."""
        for row in table:
            assert row["ratio_grail"] > 1.0

    def test_ratios_match_hand_computed_values(self, table):
        expected = {0: 8.16, 1: 11.95, 2: 8.24}
        for row in table:
            assert row["ratio_grail"] == pytest.approx(expected[row["m"]], rel=0.02)

    def test_gsfc_ratios_match_hand_computed_values(self, table):
        """The independent GSFC/GRGM660PRIM analysis (Williams et al.
        2014, Table 4) requires a markedly *larger* improvement than the
        JPL/GL0660B analysis above -- 41/16/12x, not 8/12/8x (module
        docstring, "Achieved precision")."""
        expected = {0: 41.3, 1: 15.8, 2: 12.4}
        for row in table:
            assert row["ratio_grail_gsfc"] == pytest.approx(expected[row["m"]], rel=0.02)

    def test_gsfc_stricter_than_jpl_at_every_order(self, table):
        for row in table:
            assert row["ratio_grail_gsfc"] > row["ratio_grail"]


class TestGrailCrossAnalysisDisagreement:
    """The reproducibility-floor finding: the JPL and GSFC analyses of
    the SAME GRAIL data disagree with each other by more than either
    paper's own formal uncertainty, and by more than the Mars splitting
    this module predicts at m=1 and m=2 (module docstring, "Achieved
    precision")."""

    @pytest.fixture
    def rows(self):
        return grail_k2m_cross_analysis_disagreement()

    def test_three_orders(self, rows):
        assert [row["m"] for row in rows] == [0, 1, 2]

    def test_k22_disagreement_is_27x_the_mars_splitting(self, rows):
        row = next(r for r in rows if r["m"] == 2)
        assert row["abs_diff"] == pytest.approx(9.1e-4, rel=0.02)
        assert row["ratio_to_mars_delta"] == pytest.approx(27.0, rel=0.05)

    def test_k21_disagreement_is_11x_the_mars_splitting(self, rows):
        row = next(r for r in rows if r["m"] == 1)
        assert row["abs_diff"] == pytest.approx(2.25e-4, rel=0.02)
        assert row["ratio_to_mars_delta"] == pytest.approx(11.0, rel=0.1)

    def test_m2_disagreement_exceeds_formal_jpl_sigma(self, rows):
        """The clearest instance of the point this comparison makes:
        m=2's cross-analysis disagreement (9.1e-4) exceeds even its own
        formal per-analysis uncertainty (2.8e-4) by more than 3x -- the
        formal sigma understates the real floor on individual-order k2m
        reproducibility. m=1's disagreement (2.25e-4) is close to but
        just under its own formal sigma (2.5e-4); m=0's (8.5e-5) is well
        under (4.5e-4). This test only asserts what m=2 actually shows,
        rather than overclaiming the same pattern for all three orders."""
        row = next(r for r in rows if r["m"] == 2)
        assert row["abs_diff"] > GRAIL_K2M_SIGMA[2]

    def test_disagreement_scatter_is_order_1em3_not_1em4(self, rows):
        """module docstring's summary characterization: demonstrated
        reproducibility on individual-order k2m is nearer 1e-3 (the
        largest disagreement, m=2, 9.1e-4) than the formal 2.5-4.5e-4
        band either analysis quotes alone."""
        max_diff = max(row["abs_diff"] for row in rows)
        assert max_diff > 5e-4


class TestTruncationSensitivity:
    """lmax=2 vs lmax=4 -- module docstring, "Truncation sensitivity":
    not converged, and the ordering among m reverses."""

    def test_lmax2_deltas_match_hand_computed_values(self):
        expected = {0: 3.079e-05, 1: 2.755e-05, 2: 1.948e-05}
        for m, delta in MARS_K2M_LMAX2_DELTA.items():
            assert delta == pytest.approx(expected[m], rel=1e-3)

    def test_lmax4_deltas_differ_substantially_from_lmax2(self):
        lmax4 = {0: 5.517e-05, 1: 2.091e-05, 2: 3.400e-05}
        for m, delta_lmax2 in MARS_K2M_LMAX2_DELTA.items():
            rel_change = abs(lmax4[m] - delta_lmax2) / delta_lmax2
            assert rel_change > 0.2  # every order moves by >20% between lmax 2 and 4

    def test_ordering_among_m_reverses_between_lmax2_and_lmax4(self):
        """At lmax=2, m=1 splits more than m=2; at lmax=4 (the shipped,
        proposal-cited values), the reverse. Neither ordering should be
        relied upon until TASK-027's convergence study completes."""
        lmax4 = {0: 5.517e-05, 1: 2.091e-05, 2: 3.400e-05}
        assert MARS_K2M_LMAX2_DELTA[1] > MARS_K2M_LMAX2_DELTA[2]
        assert lmax4[2] > lmax4[1]


class TestGrailConstantsProvenance:

    def test_grail_k2m_keys(self):
        assert set(GRAIL_K2M) == {0, 1, 2}
        assert set(GRAIL_K2M_SIGMA) == {0, 1, 2}

    def test_grail_k2m_close_to_each_other(self):
        """Konopliv et al. (2013) Table 4: the three GRAIL solutions are
        "almost equal" (Wörner et al. 2023's own characterization,
        retrieved this session) -- sanity check that constant, not a
        typo'd digit."""
        values = list(GRAIL_K2M.values())
        assert max(values) - min(values) < 0.001

    def test_grail_sigma_positive(self):
        for sigma in GRAIL_K2M_SIGMA.values():
            assert sigma > 0.0


class TestMaquisConstantsProvenance:

    def test_resolution_target_exceeds_current(self):
        assert MAQUIS_TARGET_RESOLUTION_DO > MAQUIS_CURRENT_RESOLUTION_DO

    def test_seasonal_signal_far_exceeds_secular_goal(self):
        # Sanity: the seasonal (CO2 + Phobos/Deimos tide) signal MaQuIs
        # quotes (230 uGal) is orders of magnitude larger than its
        # secular-change detection goal (0.01 uGal/yr) -- different
        # observables, both retrieved this session, not meant to be
        # divided into each other; this only guards against a transposed
        # digit.
        assert MAQUIS_SEASONAL_SIGNAL_UGAL / MAQUIS_SECULAR_GOAL_UGAL_PER_YR > 1000.0


class TestMoonMarsXiRatio:
    """Quantifying the Moon-vs-Mars gap (module docstring) -- the
    instrumental requirement is ~3.8 orders of magnitude, not the "one
    order of magnitude" the Love-number-space ratios alone suggest."""

    def test_moon_xi_2_matches_hand_value(self):
        assert MOON_XI_2 == pytest.approx(1.5007e-06, rel=1e-3)

    def test_ratio_matches_hand_value(self):
        assert moon_mars_xi_ratio() == pytest.approx(736.0, rel=1e-2)

    def test_stokes_space_requirement_is_orders_of_magnitude_worse(self):
        """Combining the Love-number-space ratio (8.2, 12.0, 8.2 -- JPL)
        with the xi_2 ratio gives the Stokes-coefficient-space
        requirement (~6.0e3, 8.8e3, 6.1e3), not just one order of
        magnitude beyond GRAIL."""
        ratio = moon_mars_xi_ratio()
        love_number_ratios = {0: 8.156, 1: 11.954, 2: 8.236}
        expected_stokes = {0: 6.0e3, 1: 8.8e3, 2: 6.1e3}
        for m, love_ratio in love_number_ratios.items():
            stokes_ratio = love_ratio * ratio
            assert stokes_ratio == pytest.approx(expected_stokes[m], rel=0.02)
            # "3.8 orders of magnitude" beyond GRAIL, not the "one order
            # of magnitude" the Love-number ratio alone would suggest.
            assert stokes_ratio > 100.0 * love_ratio


@pytest.mark.slow
class TestRecomputeK2mDiagonalShift:

    def test_m1_reproduces_hardcoded_constant(self):
        k21 = recompute_k2m_diagonal_shift(1, lmax=4, Nrbase=30)
        assert k21.real == pytest.approx(MARS_K21_FORCING, rel=1e-8)
        assert abs(k21.imag) < 1e-10

    def test_m2_reproduces_hardcoded_constant(self):
        k22 = recompute_k2m_diagonal_shift(2, lmax=4, Nrbase=30)
        assert k22.real == pytest.approx(MARS_K22_FORCING, rel=1e-8)
        assert abs(k22.imag) < 1e-10
