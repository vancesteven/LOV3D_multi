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
    GRAIL_K2M_SIGMA,
    MARS_K21_FORCING,
    MARS_K22_FORCING,
    MAQUIS_CURRENT_RESOLUTION_DO,
    MAQUIS_SEASONAL_SIGNAL_UGAL,
    MAQUIS_SECULAR_GOAL_UGAL_PER_YR,
    MAQUIS_TARGET_RESOLUTION_DO,
    mars_diagonal_k2m_table,
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
