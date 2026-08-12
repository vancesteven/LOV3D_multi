# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for the TASK-031 Moon lateral-crust pipeline."""

import numpy as np
import pytest

from pylov3d.love import get_love
from pylov3d.mapping import latlon_grid, sh_to_latlon
from pylov3d.mars_lateral import complex_sh_synthesis
from pylov3d.moon import LAYER_RADII_KM, build_moon_model, moon_forcing, moon_numerics
from pylov3d.moon_lateral import (
    AIRY_FACTOR,
    CRUST_LAYER_INDEX,
    CRUST_THICKNESS_M,
    MANTLE_LAYER_INDEX,
    WEBER_CRUST_SHELL_THICKNESS_M,
    _dmu_ddt_coeff,
    crustal_thickness_diagnostics,
    crustal_thickness_variation,
    mu_variable_from_topography,
)


def test_constants_follow_model_and_adopted_mean():
    assert CRUST_LAYER_INDEX == len(LAYER_RADII_KM) - 1 == 9
    assert MANTLE_LAYER_INDEX == 8
    assert WEBER_CRUST_SHELL_THICKNESS_M == pytest.approx(34e3)
    assert CRUST_THICKNESS_M == pytest.approx(40e3)
    assert AIRY_FACTOR == pytest.approx(2800.0 / (3220.0 - 2800.0))


def test_degree_one_translation_is_removed():
    dt = crustal_thickness_variation(lmax=4)
    assert not any(n == 1 for n, _m in dt)


def test_default_removes_only_zonal_degree_two():
    dt = crustal_thickness_variation(lmax=4)
    assert (2, 0) not in dt
    assert any(n == 2 and m != 0 for n, m in dt)


def test_c20_retention_is_explicit_sensitivity_option():
    without_c20 = crustal_thickness_variation(lmax=4)
    with_c20 = crustal_thickness_variation(lmax=4, include_c20=True)
    assert (2, 0) not in without_c20
    assert with_c20[(2, 0)] == pytest.approx(-3403.927969554922)


def test_default_field_stays_barely_inside_linear_domain():
    diag = crustal_thickness_diagnostics(lmax=4)
    assert diag["max_abs_dt_m"] / 1e3 == pytest.approx(32.63, abs=0.05)
    assert 0.98 < diag["max_abs_dmu_over_mubar"] < 1.0


def test_c20_retention_crosses_rigidity_positivity_bound():
    diag = crustal_thickness_diagnostics(lmax=4, include_c20=True)
    assert diag["max_abs_dmu_over_mubar"] > 1.0


def test_nonphysical_c20_field_is_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        mu_variable_from_topography(lmax=4, include_c20=True)


def test_real_to_complex_field_round_trip():
    dt = crustal_thickness_variation(lmax=4)
    entries = mu_variable_from_topography(lmax=4)[CRUST_LAYER_INDEX]
    scaled_entries = [(n, m, amp / _dmu_ddt_coeff()) for n, m, amp in entries]
    lat, lon = latlon_grid(nlat=40, nlon=80)
    expected = sh_to_latlon(dt, nlat=40, nlon=80).z
    actual = complex_sh_synthesis(scaled_entries, lat, lon)
    np.testing.assert_allclose(actual.imag, 0.0, atol=1e-10)
    np.testing.assert_allclose(actual.real, expected, rtol=1e-11, atol=1e-8)


@pytest.mark.slow
def test_zero_amplitude_reduces_to_uniform_moon():
    model = build_moon_model()
    forcing = moon_forcing()
    numerics = moon_numerics(Nrbase=15)
    uniform, _, _ = get_love(model, forcing, numerics)
    lateral, _, _ = get_love(
        model,
        forcing,
        numerics,
        mu_variable={CRUST_LAYER_INDEX: [(2, 0, 0.0)]},
    )
    idx = np.where((lateral.n == 2) & (lateral.m == 0))[0][0]
    assert lateral.k[idx] == pytest.approx(uniform.k[0], rel=1e-10)
