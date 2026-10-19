# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Science regressions for the Mars serpentinite connectivity experiment.

These tests pin the interpretation of
``scripts/mars_serpentinite_connectivity_sensitivity.py``:

* Voigt must exactly reproduce the existing TASK-021 mean-mixing law;
* Reuss <= Hill <= Voigt in effective stiffness for a weak hydrated phase;
* increased compliance must not reduce the magnitude of the hydration k2
  signal for the central case.
"""

import numpy as np

from pylov3d.mars_hydration import (
    K_SERP_RATIO_CENTRAL,
    MU_SERP_RATIO_CENTRAL,
    mean_softened_crust_moduli,
)
from scripts.mars_serpentinite_connectivity_sensitivity import (
    mixed_moduli,
    solve_k2,
)


def test_voigt_reproduces_task021_mean_mixing():
    for f_h in (0.0, 0.1, 0.5, 1.0):
        expected_mu, expected_K = mean_softened_crust_moduli(
            f_h,
            MU_SERP_RATIO_CENTRAL,
            K_SERP_RATIO_CENTRAL,
        )
        got_mu, got_K = mixed_moduli(
            f_h,
            MU_SERP_RATIO_CENTRAL,
            K_SERP_RATIO_CENTRAL,
            "voigt",
        )
        np.testing.assert_allclose(got_mu, expected_mu, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(got_K, expected_K, rtol=0.0, atol=0.0)


def test_connectivity_stiffness_ordering_central_case():
    for f_h in (0.1, 0.5, 0.9):
        mu_v, K_v = mixed_moduli(f_h, MU_SERP_RATIO_CENTRAL, K_SERP_RATIO_CENTRAL, "voigt")
        mu_h, K_h = mixed_moduli(f_h, MU_SERP_RATIO_CENTRAL, K_SERP_RATIO_CENTRAL, "hill")
        mu_r, K_r = mixed_moduli(f_h, MU_SERP_RATIO_CENTRAL, K_SERP_RATIO_CENTRAL, "reuss")
        assert mu_r <= mu_h <= mu_v
        assert K_r <= K_h <= K_v


def test_weaker_connectivity_increases_hydration_k2_signal():
    """Cheap coarse-grid science monotonicity check, not a precision anchor."""
    f_h = 0.1
    nrbase = 10
    mu0, K0 = mean_softened_crust_moduli(0.0)
    k0 = solve_k2(mu0, K0, nrbase)

    signals = []
    for law in ("voigt", "hill", "reuss"):
        mu_eff, K_eff = mixed_moduli(
            f_h,
            MU_SERP_RATIO_CENTRAL,
            K_SERP_RATIO_CENTRAL,
            law,
        )
        signals.append(abs(solve_k2(mu_eff, K_eff, nrbase) - k0))

    assert signals[0] < signals[1] < signals[2]
