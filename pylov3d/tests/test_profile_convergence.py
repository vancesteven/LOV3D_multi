# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pylov3d.constants import MAX_LAYERS
from pylov3d.profile_convergence import (
    love_number_convergence,
    successive_k2_differences,
    synthetic_mars_like_shells,
)
from pylov3d.profile_reduction import reduced_shells_to_interior_model
from pylov3d.types import make_forcing


@pytest.fixture(scope="module")
def mars_like_entries():
    shells = synthetic_mars_like_shells(64)
    forcing = make_forcing(Td=44387.62, n=2, m=0, F=1.0)
    return love_number_convergence(
        shells, forcing, layer_counts=[6, 10, 14, MAX_LAYERS]
    )


class TestLiquidCoreConversion:
    """A multi-shell fluid core must convert as liquid core, not ocean."""

    def test_leading_fluid_run_is_core_not_ocean(self):
        shells = synthetic_mars_like_shells(16)
        model = reduced_shells_to_interior_model(shells)
        fluid = np.asarray(shells.mu_Pa) <= 1.0
        n_core = int(np.argmin(fluid)) if not fluid.all() else fluid.size
        assert n_core > 1  # the fixture must actually exercise a multi-shell core
        ocean = np.asarray(model.ocean[: model.n_layers])
        assert np.all(ocean[:n_core] == 0)
        mu = np.asarray(model.mu0[: model.n_layers])
        assert np.all(mu[:n_core] == 0.0)

    def test_interior_fluid_shell_still_flagged_ocean(self):
        shells = synthetic_mars_like_shells(16)
        mu = np.asarray(shells.mu_Pa, dtype=float).copy()
        mu[10] = 0.0  # internal ocean well above the core
        shells = type(shells)(
            shells.outer_radius_m, shells.rho_kgm3, shells.K_Pa, mu, shells.metadata
        )
        model = reduced_shells_to_interior_model(shells)
        assert int(model.ocean[10]) == 1


class TestLoveNumberConvergence:

    def test_entries_report_exact_mass_closure(self, mars_like_entries):
        for e in mars_like_entries:
            assert abs(e.reduction.mass_relative_change) < 1e-14

    def test_k2_physically_sane(self, mars_like_entries):
        for e in mars_like_entries:
            k2 = e.k2
            assert abs(k2.imag) < 1e-10  # elastic
            assert 0.0 < k2.real < 1.0
            assert e.h2.real > k2.real

    def test_k2_converges_with_layer_count(self, mars_like_entries):
        diffs = successive_k2_differences(mars_like_entries)
        # Successive change must shrink and end well-resolved.
        assert diffs[-1] < diffs[0]
        assert diffs[-1] < 1e-3

    def test_layer_counts_above_static_limit_are_refused(self):
        shells = synthetic_mars_like_shells(32)
        forcing = make_forcing(Td=44387.62, n=2, m=0, F=1.0)
        with pytest.raises(ValueError, match="MAX_LAYERS"):
            love_number_convergence(
                shells, forcing, layer_counts=[8, MAX_LAYERS + 1]
            )

    def test_convergence_diagnostic_requires_two_counts(self, mars_like_entries):
        with pytest.raises(ValueError, match="two layer counts"):
            successive_k2_differences(mars_like_entries[:1])
