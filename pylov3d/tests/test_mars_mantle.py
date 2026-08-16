# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Pure-contract tests for the TASK-043 Mars thermal-template pilot."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pylov3d.mars_lateral import (
    CRUST_LAYER_INDEX,
    _real_sh_to_complex_mu_variable,
    complex_sh_synthesis,
)
from pylov3d.mars_mantle import (
    BETA_MU_PER_K,
    EXPECTED_L2_ACTIVE_MODES,
    UPPER_MANTLE_LAYER_INDEX,
    active_modes_for_mu_variable,
    area_weighted_mean,
    jacobian_distinguishability,
    l2_mode_closure,
    load_plesa_s1,
    merge_mantle_crust_mu_variable,
    positivity_diagnostics,
    project_temperature_real_sh,
    remove_area_weighted_mean,
    thermal_fractional_coefficients,
    thermal_mu_variable,
    unit_rms_coefficients,
    unit_rms_coefficients_by_cutoff,
)
from pylov3d.mapping import fully_normalized_legendre


def _synthesize_real(
    coefficients: dict[tuple[int, int], float],
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    """Independent real-basis synthesis used to test the projector."""

    lmax = max(n for n, _m in coefficients)
    P = fully_normalized_legendre(lmax, np.sin(np.radians(lat_deg)))
    lon_rad = np.radians(lon_deg)
    field = np.zeros((lat_deg.size, lon_deg.size))
    for (n, m), amplitude in coefficients.items():
        mm = abs(m)
        angular = np.cos(mm * lon_rad) if m >= 0 else np.sin(mm * lon_rad)
        field += amplitude * P[n, mm, :, None] * angular[None, :]
    return field


class TestPlesaS1Contract:

    def test_loads_complete_grid_independent_of_row_order(self, tmp_path):
        lat = np.repeat(np.arange(90, -90, -1), 360)
        lon = np.tile(np.arange(360), 180)
        rows = np.column_stack(
            [
                lon,
                lat,
                50.0 + 0.01 * lon,
                20.0 + 0.02 * lat,
                100.0 + 0.03 * lon,
                150.0 + 0.04 * lon,
                1000.0 + 0.5 * lat + 0.1 * lon,
                200.0 + 0.05 * lat,
            ]
        )
        # The archived north-pole row starts at a rotated longitude.  A full
        # deterministic roll proves the loader does not assume row-major order.
        rows = np.roll(rows, 90, axis=0)
        path = tmp_path / "plesa_s1.txt"
        np.savetxt(path, rows, fmt="%.10g", header="synthetic S1\nsecond comment")

        grid = load_plesa_s1(path)

        np.testing.assert_array_equal(grid.lon_e_deg, np.arange(360.0))
        np.testing.assert_array_equal(grid.lat_deg, np.arange(90.0, -90.0, -1.0))
        assert grid.temperature_150km_k.shape == (180, 360)
        assert grid.temperature_150km_k[0, 0] == pytest.approx(1045.0)
        assert grid.temperature_150km_k[-1, -1] == pytest.approx(991.4)
        assert grid.elastic_thickness_1e14_km[10, 20] == pytest.approx(100.6)
        assert grid.elastic_thickness_1e17_km[10, 20] == pytest.approx(150.8)

    def test_rejects_wrong_shape(self, tmp_path):
        path = tmp_path / "not_s1.txt"
        np.savetxt(path, np.zeros((4, 8)))
        with pytest.raises(ValueError, match="must have shape"):
            load_plesa_s1(path)


class TestProjectionAndThermalMapping:

    def test_area_weighted_mean_removal(self):
        lat = np.array([75.0, 25.0, -25.0, -75.0])
        field = 400.0 + np.sin(np.radians(lat))[:, None] * np.ones((1, 8))
        assert area_weighted_mean(field, lat) == pytest.approx(400.0)
        centered = remove_area_weighted_mean(field, lat)
        assert area_weighted_mean(centered, lat) == pytest.approx(0.0, abs=1e-13)

    def test_projection_recovers_real_4pi_coefficients(self):
        lat = np.linspace(87.5, -87.5, 36)
        lon = np.arange(0.0, 360.0, 5.0)
        expected = {
            (1, 0): 45.0,
            (1, 1): -12.0,
            (1, -1): 8.0,
            (2, 0): 31.0,
            (2, 1): 4.0,
            (2, -1): -7.0,
            (2, 2): 19.0,
            (2, -2): -11.0,
            (3, 0): 3.5,
            (4, -3): -2.25,
        }
        temperature = 1000.0 + _synthesize_real(expected, lat, lon)

        recovered = project_temperature_real_sh(temperature, lat, lon, lmax=4)

        for mode, value in expected.items():
            assert recovered[mode] == pytest.approx(value, rel=1e-11, abs=1e-11)
        for mode, value in recovered.items():
            if mode not in expected:
                assert value == pytest.approx(0.0, abs=1e-11)
        assert (0, 0) not in recovered

    def test_each_cutoff_is_independently_unit_rms(self):
        coefficients = {
            (1, 0): 3.0,
            (1, 1): 4.0,
            (2, 0): 12.0,
            (3, -2): 5.0,
            (4, 4): 7.0,
        }
        by_cutoff = unit_rms_coefficients_by_cutoff(coefficients, (1, 2, 3, 4))
        for cutoff, template in by_cutoff.items():
            assert max(n for n, _m in template) <= cutoff
            assert sum(value * value for value in template.values()) == pytest.approx(1.0)
        assert by_cutoff[1][(1, 0)] == pytest.approx(3.0 / 5.0)
        assert by_cutoff[2][(2, 0)] == pytest.approx(12.0 / 13.0)

    def test_degree_zero_and_zero_rms_are_rejected(self):
        with pytest.raises(ValueError, match="degree-zero"):
            unit_rms_coefficients({(0, 0): 1.0, (1, 0): 1.0}, 1)
        with pytest.raises(ValueError, match="zero RMS"):
            unit_rms_coefficients({(1, 0): 0.0}, 1)

    def test_thermal_mapping_rejects_non_unit_template(self):
        with pytest.raises(ValueError, match="unit RMS"):
            thermal_fractional_coefficients({(1, 0): 2.0}, amplitude_k=300.0)

    def test_beta_times_temperature_mapping_and_solver_convention(self):
        template = unit_rms_coefficients({(1, 0): 3.0, (2, 2): 4.0}, 2)
        fractional = thermal_fractional_coefficients(template, amplitude_k=300.0)
        expected_rms = abs(BETA_MU_PER_K) * 300.0
        actual_rms = math.sqrt(sum(value * value for value in fractional.values()))
        assert actual_rms == pytest.approx(0.0577455)
        assert actual_rms == pytest.approx(expected_rms)

        mu_variable = thermal_mu_variable(template, amplitude_k=300.0)
        assert set(mu_variable) == {UPPER_MANTLE_LAYER_INDEX}
        entries = mu_variable[UPPER_MANTLE_LAYER_INDEX]
        nm = {(n, m) for n, m, _a in entries}
        assert (2, 2) in nm and (2, -2) in nm

        lat = np.linspace(-90.0, 90.0, 31)
        lon = np.linspace(0.0, 360.0, 60, endpoint=False)
        from_solver_basis = complex_sh_synthesis(entries, lat, lon)
        expected_grid = _synthesize_real(fractional, lat, lon)
        np.testing.assert_allclose(from_solver_basis.real, expected_grid, rtol=1e-12, atol=1e-12)
        assert np.max(np.abs(from_solver_basis.imag)) < 1e-12


class TestPositivity:

    def test_small_thermal_field_passes_both_guards(self):
        template = unit_rms_coefficients({(1, 0): 1.0, (2, 1): -0.2}, 2)
        fractional = thermal_fractional_coefficients(template, amplitude_k=300.0)
        diagnostic = positivity_diagnostics(fractional, nlat=91, nlon=180)
        assert diagnostic.minimum_mu_factor > 0.8
        assert diagnostic.grid_passes
        assert diagnostic.coefficient_passes
        assert diagnostic.passes
        assert diagnostic.maximum_imaginary_residual < 1e-12

    def test_nonpositive_field_fails_grid_and_conservative_guards(self):
        diagnostic = positivity_diagnostics({(1, 0): 1.0}, nlat=91, nlon=180)
        assert diagnostic.minimum_mu_factor < 0.0
        assert not diagnostic.grid_passes
        assert diagnostic.coefficient_upper_bound == pytest.approx(math.sqrt(3.0))
        assert not diagnostic.coefficient_passes
        assert not diagnostic.passes

    def test_empty_lateral_field_has_unit_margin(self):
        diagnostic = positivity_diagnostics({}, epsilon=1e-5, nlat=5, nlon=8)
        assert diagnostic.minimum_mu_factor == pytest.approx(1.0)
        assert diagnostic.coefficient_upper_bound == 0.0
        assert diagnostic.passes


class TestTwoLayerClosure:

    @staticmethod
    def _full_l2_real(scale: float) -> dict[tuple[int, int], float]:
        coefficients = {}
        for n in (1, 2):
            coefficients[(n, 0)] = scale
            for m in range(1, n + 1):
                coefficients[(n, m)] = scale * (1.0 + 0.1 * m)
                coefficients[(n, -m)] = scale * (1.0 - 0.1 * m)
        return coefficients

    def test_merge_keeps_layers_distinct_and_l2_closure_is_43(self):
        mantle = {
            UPPER_MANTLE_LAYER_INDEX: _real_sh_to_complex_mu_variable(self._full_l2_real(0.02))
        }
        crust = {
            CRUST_LAYER_INDEX: _real_sh_to_complex_mu_variable(self._full_l2_real(0.04))
        }

        merged = merge_mantle_crust_mu_variable(mantle, crust)
        closure = l2_mode_closure(merged)

        assert set(merged) == {UPPER_MANTLE_LAYER_INDEX, CRUST_LAYER_INDEX}
        assert merged[UPPER_MANTLE_LAYER_INDEX] is not mantle[UPPER_MANTLE_LAYER_INDEX]
        assert len(closure) == EXPECTED_L2_ACTIVE_MODES
        np.testing.assert_array_equal(np.unique(closure[:, 0]), np.arange(7))

    def test_two_layers_do_not_double_mode_count(self):
        entries = _real_sh_to_complex_mu_variable(self._full_l2_real(0.02))
        one_layer = {UPPER_MANTLE_LAYER_INDEX: entries}
        two_layers = {
            UPPER_MANTLE_LAYER_INDEX: entries,
            CRUST_LAYER_INDEX: [(n, m, 2.0 * amplitude) for n, m, amplitude in entries],
        }
        assert len(active_modes_for_mu_variable(one_layer)) == EXPECTED_L2_ACTIVE_MODES
        assert len(active_modes_for_mu_variable(two_layers)) == EXPECTED_L2_ACTIVE_MODES

    def test_zero_amplitudes_reduce_to_forcing_mode(self):
        active = active_modes_for_mu_variable(
            {UPPER_MANTLE_LAYER_INDEX: [(2, 0, 0.0)]}
        )
        np.testing.assert_array_equal(active, np.array([[2, 0, 0]]))

    def test_l2_guard_rejects_higher_degree(self):
        with pytest.raises(ValueError, match="above degree 2"):
            l2_mode_closure({UPPER_MANTLE_LAYER_INDEX: [(3, 0, 0.1)]})


class TestJacobianDistinguishability:

    def test_parallel_jacobians_warn_and_fail_orthogonal_gate(self):
        metrics = jacobian_distinguishability(
            np.array([2.0, -1.0]),
            np.array([4.0, -2.0]),
            np.eye(2),
            max_abs_thermal_amplitude=1000.0,
        )
        assert metrics.correlation == pytest.approx(1.0)
        assert metrics.orthogonal_norm_per_unit == pytest.approx(0.0, abs=1e-14)
        assert metrics.correlation_warning
        assert not metrics.passes_one_sigma

    def test_orthogonal_signal_uses_whitened_one_sigma_decision(self):
        covariance = np.diag([4.0, 0.25])
        metrics = jacobian_distinguishability(
            np.array([2.0, 0.0]),
            np.array([0.0, 0.5]),
            covariance,
            max_abs_thermal_amplitude=1.1,
        )
        np.testing.assert_allclose(metrics.thermal_whitened, [1.0, 0.0])
        np.testing.assert_allclose(metrics.crust_whitened, [0.0, 1.0])
        assert metrics.correlation == pytest.approx(0.0)
        assert metrics.max_orthogonal_sigma == pytest.approx(1.1)
        assert not metrics.correlation_warning
        assert metrics.passes_one_sigma

    def test_high_absolute_correlation_warns_but_is_not_decision_rule(self):
        metrics = jacobian_distinguishability(
            np.array([1.0, 0.0]),
            np.array([0.96, math.sqrt(1.0 - 0.96**2)]),
            np.eye(2),
            max_abs_thermal_amplitude=10.0,
        )
        assert metrics.correlation == pytest.approx(0.96)
        assert metrics.correlation_warning
        assert metrics.passes_one_sigma

    @pytest.mark.parametrize(
        "covariance,match",
        [
            (np.array([[1.0, 2.0], [0.0, 1.0]]), "symmetric"),
            (np.array([[1.0, 0.0], [0.0, 0.0]]), "positive definite"),
        ],
    )
    def test_invalid_covariance_is_rejected(self, covariance, match):
        with pytest.raises(ValueError, match=match):
            jacobian_distinguishability(
                np.ones(2), np.array([1.0, -1.0]), covariance, 1.0
            )
