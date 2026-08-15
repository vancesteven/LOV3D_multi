# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for the TASK-031 Moon lateral-crust pipeline."""

from pathlib import Path

import numpy as np
import pytest

from pylov3d.couplings import coupling_coefficients
from pylov3d.love import get_love
from pylov3d.mapping import latlon_grid, sh_to_latlon
from pylov3d.mars_lateral import complex_sh_synthesis
from pylov3d.moon import (
    LAYER_MU,
    LAYER_RADII_KM,
    build_moon_model,
    moon_forcing,
    moon_numerics,
)
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


REPO_ROOT = Path(__file__).resolve().parents[2]
SPECTRUM_PATH = REPO_ROOT / "docs/figures/proposal/moon_lateral_spectrum.npz"
K2M_PATH = REPO_ROOT / "docs/figures/proposal/moon_k2m_vs_grail.npz"
MU_VARIABLE_PATH = REPO_ROOT / "data/moon/moon_mu_variable_lateral.npz"


def test_committed_dichotomy_spectrum_artifact():
    with np.load(SPECTRUM_PATH) as data:
        n = data["n"]
        m = data["m"]
        k = data["k"]
        assert not bool(data["degree_one_removed"])
        assert int(data["mode_count"]) == len(n) == 115
        assert complex(data["delta_k2"]) == pytest.approx(2.14124e-6, rel=1e-5)

        off_forcing = ~((n == 2) & (m == 0))
        index = np.flatnonzero(off_forcing)[np.argmax(np.abs(k[off_forcing]))]
        assert (int(n[index]), abs(int(m[index]))) == (3, 1)
        assert abs(k[index]) == pytest.approx(6.37279e-6, rel=1e-5)


def test_committed_k2m_artifact_is_a_three_tier_null():
    with np.load(K2M_PATH) as k2m, np.load(SPECTRUM_PATH) as spectrum:
        delta = k2m["delta_k2m"]
        sigma = k2m["grail_k2m_sigma"]
        np.testing.assert_allclose(
            delta, [2.1412e-6, 1.0606e-6, 1.9250e-6], rtol=5e-5,
        )
        assert np.all(np.abs(delta) / sigma < 0.01)
        assert delta[0] == pytest.approx(
            complex(spectrum["delta_k2"]).real, rel=1e-9,
        )


def test_committed_mu_variable_artifact_has_degree_one_and_real_symmetry():
    with np.load(MU_VARIABLE_PATH) as data:
        n = data["n"]
        m = data["m"]
        amp = data["amp_real"] + 1j * data["amp_imag"]
        assert len(n) == 23
        assert set(zip(n[n == 1], m[n == 1])) == {(1, -1), (1, 0), (1, 1)}

        field = {(int(nn), int(mm)): value for nn, mm, value in zip(n, m, amp)}
        for (nn, mm), value in field.items():
            if mm > 0:
                assert field[(nn, -mm)] == (-1) ** mm * np.conj(value)


def test_degree_one_sectoral_channel_to_dominant_mode_is_strong():
    coeffs = coupling_coefficients(3, 1, 2, 0, 1, 1)
    assert np.max(np.abs(coeffs[:26])) == pytest.approx(
        0.7171371656, rel=1e-9,
    )


def test_degree_one_zonal_channel_cannot_touch_forcing_mode_at_first_order():
    """The +52% delta-k20 rise is not a direct first-order dt(1,0) path.

    Parity blocks that zonal channel, leaving sectoral degree-1 terms and
    second-order paths to carry the forcing-mode change.
    """
    coeffs = coupling_coefficients(2, 0, 2, 0, 1, 0)
    assert np.array_equal(coeffs[:26], np.zeros(26))


def test_degree_one_sectoral_channel_to_degree_two_order_one_is_pinned():
    coeffs = coupling_coefficients(2, 1, 2, 0, 1, 1)
    assert np.max(np.abs(coeffs[:26])) == pytest.approx(
        0.7071067812, rel=1e-9,
    )


def test_degree_one_flag_changes_only_the_three_dichotomy_coefficients():
    retained = crustal_thickness_variation(lmax=4)
    removed = crustal_thickness_variation(lmax=4, include_degree1=False)
    degree_one = {(1, -1), (1, 0), (1, 1)}
    assert set(retained) - set(removed) == degree_one
    assert set(removed) - set(retained) == set()
    assert all(retained[key] == removed[key] for key in removed)


def test_constants_follow_model_and_adopted_mean():
    assert CRUST_LAYER_INDEX == len(LAYER_RADII_KM) - 1 == 9
    assert MANTLE_LAYER_INDEX == 8
    assert WEBER_CRUST_SHELL_THICKNESS_M == pytest.approx(34e3)
    assert CRUST_THICKNESS_M == pytest.approx(40e3)
    assert AIRY_FACTOR == pytest.approx(2800.0 / (3220.0 - 2800.0))


def test_rigidity_coefficient_is_fixed_shell_voigt_average():
    mu_crust = LAYER_MU[CRUST_LAYER_INDEX]
    mu_mantle = LAYER_MU[MANTLE_LAYER_INDEX]
    expected = (mu_crust - mu_mantle) / (CRUST_THICKNESS_M * mu_crust)
    assert _dmu_ddt_coeff() == pytest.approx(expected, rel=1e-15)


def test_rigidity_unity_crossing_precedes_shell_fullness():
    """Contrast reaches unity at 32.95 km, below the 40 km shell."""
    unity_crossing_m = 1.0 / abs(_dmu_ddt_coeff())
    assert unity_crossing_m / 1e3 == pytest.approx(32.95, abs=0.01)
    assert unity_crossing_m < CRUST_THICKNESS_M


@pytest.mark.parametrize(
    ("lmax", "expected_margin"),
    [(4, 0.989844), (5, 1.050508), (6, 1.081111)],
)
def test_reported_rigidity_margins(lmax, expected_margin):
    """Margins for the default (dichotomy-retaining) field.

    Degree-1 partially cancels the high-degree extremes: the old
    degree-1-removed margins were 0.9902 / 1.1531 / 1.2897.
    """
    diag = crustal_thickness_diagnostics(lmax=lmax)
    assert diag["max_abs_dmu_over_mubar"] == pytest.approx(
        expected_margin, abs=5e-5,
    )


@pytest.mark.parametrize(
    ("lmax", "expected_margin"),
    [(4, 0.9902), (5, 1.1531), (6, 1.2897)],
)
def test_reported_rigidity_margins_degree1_removed(lmax, expected_margin):
    """The pre-2026-08-14 field is still reproducible via the flag."""
    diag = crustal_thickness_diagnostics(lmax=lmax, include_degree1=False)
    assert diag["max_abs_dmu_over_mubar"] == pytest.approx(
        expected_margin, abs=5e-5,
    )


@pytest.mark.parametrize(
    ("lmax", "message"),
    [(5, "non-positive"), (6, "non-positive")],
)
def test_nonphysical_high_degree_fields_are_rejected(lmax, message):
    """With degree-1 retained, lmax=6 stays under the 40 km shell
    (max|dt|/T = 0.89) so the rigidity-positivity guard binds, not the
    thickness guard that bound the degree-1-removed field."""
    with pytest.raises(ValueError, match=message):
        mu_variable_from_topography(lmax=lmax)


def test_degree_one_dichotomy_is_retained_by_default():
    """PI decision 2026-08-14: the nearside-farside dichotomy is physics,
    not a frame artifact, per the original TASK-031 plan."""
    dt = crustal_thickness_variation(lmax=4)
    assert any(n == 1 for n, _m in dt)
    # and it is the dominant m=1 term of the field, not a trace residue
    assert abs(dt[(1, 1)]) > 1e3


def test_degree_one_removal_is_explicit_option():
    dt = crustal_thickness_variation(lmax=4, include_degree1=False)
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
