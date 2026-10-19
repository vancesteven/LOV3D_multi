# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pylov3d.mars_joint_constraints import (
    RHO_CRUST_KG_M3,
    RHO_SERPENTINITE_KG_M3,
    hydrated_solid_state,
    mixed_density,
)


def test_mixed_density_endpoints():
    assert mixed_density(0.0) == pytest.approx(RHO_CRUST_KG_M3)
    assert mixed_density(1.0) == pytest.approx(RHO_SERPENTINITE_KG_M3["central"])


def test_mixed_density_monotone_for_serpentinite():
    vals = [mixed_density(f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_hydrated_state_positive_and_finite():
    s = hydrated_solid_state(0.5, "low", "reuss", "central")
    assert s.K_pa > 0
    assert s.mu_pa > 0
    assert s.rho_kg_m3 > 0
    assert s.vp_m_s > s.vs_m_s > 0
    assert s.chi2_seismic >= 0


def test_density_scenario_ordering():
    lows = hydrated_solid_state(0.5, "low", "reuss", "low")
    mids = hydrated_solid_state(0.5, "low", "reuss", "central")
    highs = hydrated_solid_state(0.5, "low", "reuss", "high")
    assert lows.rho_kg_m3 < mids.rho_kg_m3 < highs.rho_kg_m3
