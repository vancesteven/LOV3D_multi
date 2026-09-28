# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Transcription guards and k2 for the Berne et al. 2026 ED Table 2 model."""

import math

import pytest

from pylov3d.berne2026 import (
    BERNE2026_ANNUAL_TD,
    BERNE2026_ED_TABLE2,
    build_berne2026_reference_model,
)
from pylov3d.love import get_love
from pylov3d.mars import MARS
from pylov3d.types import make_forcing, make_numerics


def _bulk():
    bounds = [0.0] + [r * 1e3 for r, _, _, _ in BERNE2026_ED_TABLE2]
    R = bounds[-1]
    M = I = 0.0
    for i, (_, rho, _, _) in enumerate(BERNE2026_ED_TABLE2):
        M += (4 * math.pi / 3) * rho * (bounds[i + 1] ** 3 - bounds[i] ** 3)
        I += (8 * math.pi / 15) * rho * (bounds[i + 1] ** 5 - bounds[i] ** 5)
    return M / ((4 * math.pi / 3) * R**3), I / (M * R * R)


class TestTranscription:
    """Analytic pins guarding the 2026-09-28 image transcription."""

    def test_mean_density_pin(self):
        mean_rho, _ = _bulk()
        assert mean_rho == pytest.approx(3995.9244569, rel=1e-9)

    def test_mean_moment_pin(self):
        _, moi = _bulk()
        assert moi == pytest.approx(0.3701477946, rel=1e-9)

    def test_documented_misfits_to_observations(self):
        """The coarse 3-layer model misses observed bulk Mars by 1.6-1.9%.
        If a retranscription ever brings these inside observational error,
        the berne2026 docstring's misfit discussion must be revisited."""
        mean_rho, moi = _bulk()
        assert 0.010 < abs(mean_rho / 3933.9637 - 1.0) < 0.025
        assert 0.010 < abs(moi / MARS["MoI_factor"] - 1.0) < 0.025


class TestK2:

    @pytest.fixture(scope="class")
    def k2(self):
        love, _, _ = get_love(
            build_berne2026_reference_model(),
            make_forcing(BERNE2026_ANNUAL_TD, 2, 0, 1.0),
            make_numerics(n_layers=3, method="variable", Nrbase=200),
        )
        return complex(love.k[0])

    def test_k2_regression_pin(self, k2):
        assert abs(k2.imag) < 1e-12  # elastic
        assert k2.real == pytest.approx(0.1763155324, rel=1e-6)

    def test_k2_within_mars_observation(self, k2):
        """Caption says the model is constrained by degree-2 tidal
        deformation; 0.1763 sits within 2 sigma of 0.169 +/- 0.006."""
        assert abs(k2.real - MARS["k2"]) < 2.0 * MARS["k2_sigma"]
