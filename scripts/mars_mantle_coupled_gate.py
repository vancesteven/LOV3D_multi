"""Coupled L=2 Jacobian evidence for the TASK-043 driver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pylov3d.love import get_love
from pylov3d.mars import MARS, MARS_FORCING_TD, build_mars_model
from pylov3d.mars_detectability import (
    GM_SUN,
    MARS_SEMIMAJOR_AXIS_M,
    MARS_SIGMA_C20_SEASONAL,
    MARS_SIGMA_C30_SEASONAL,
    peak_legendre_factor,
    sh_basis_norm,
    solar_tide_amplitude_parameter,
)
from pylov3d.mars_lateral import mu_variable_from_topography
from pylov3d.mars_mantle import (
    EXPECTED_L2_ACTIVE_MODES,
    jacobian_distinguishability,
    l2_mode_closure,
    merge_mantle_crust_mu_variable,
    positivity_diagnostics,
    thermal_fractional_coefficients,
    thermal_mu_variable,
    unit_rms_coefficients,
)
from pylov3d.types import make_forcing, make_numerics


CRUST_LAYER = 3


def _forcing():
    return make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)


def _numerics(nrbase: int):
    return make_numerics(
        n_layers=4,
        method="combination",
        Nrbase=nrbase,
        perturbation_order=2,
    )


def _load_l2_template(path: Path) -> dict[tuple[int, int], float]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"l2_n", "l2_m", "l2_unit_coefficient"}
        if not required.issubset(archive.files):
            raise ValueError(f"template artifact lacks {sorted(required - set(archive.files))}")
        coefficients = {
            (int(n), int(m)): float(value)
            for n, m, value in zip(
                archive["l2_n"], archive["l2_m"], archive["l2_unit_coefficient"]
            )
        }
    return unit_rms_coefficients(coefficients, cutoff=2)


def _scaled_crust_mu_variable(scale: float):
    base = mu_variable_from_topography(lmax=2)
    return {
        CRUST_LAYER: [
            (n, m, complex(amplitude) * float(scale))
            for n, m, amplitude in base[CRUST_LAYER]
        ]
    }


def _combined_mu_variable(template, amplitude_k: float, crust_scale: float):
    return merge_mantle_crust_mu_variable(
        thermal_mu_variable(template, amplitude_k),
        _scaled_crust_mu_variable(crust_scale),
    )


def _mode_value(love, degree: int, order: int) -> complex:
    n = np.asarray(love.n, dtype=int)
    m = np.asarray(love.m, dtype=int)
    indices = np.flatnonzero((n == degree) & (m == order))
    if indices.size != 1:
        raise RuntimeError(f"expected one response mode ({degree}, {order}), got {indices.size}")
    return complex(np.asarray(love.k)[indices[0]])


def _signed_stokes_observables(love) -> tuple[np.ndarray, list[str], float]:
    """Pack signed C20 and degree-3 C/S characteristic amplitudes."""
    xi = solar_tide_amplitude_parameter(
        GM_SUN, MARS["GM"], MARS["R"], MARS_SEMIMAJOR_AXIS_M, n_forcing=2
    )
    scale = xi * peak_legendre_factor(2, 0)
    values = [float(np.real(_mode_value(love, 2, 0))) * scale]
    channels = ["C20"]
    max_conjugacy_residual = 0.0
    for m in range(0, 4):
        positive = _mode_value(love, 3, m)
        basis_correction = sh_basis_norm(0) / sh_basis_norm(m)
        if m == 0:
            values.append(float(np.real(positive)) * scale)
            channels.append("C30")
        else:
            negative = _mode_value(love, 3, -m)
            expected_negative = ((-1) ** m) * np.conj(positive)
            max_conjugacy_residual = max(
                max_conjugacy_residual, abs(negative - expected_negative)
            )
            values.extend(
                [
                    float(np.real(positive)) * scale * basis_correction,
                    -float(np.imag(positive)) * scale * basis_correction,
                ]
            )
            channels.extend([f"C3{m}", f"S3{m}"])
    return np.asarray(values, dtype=float), channels, max_conjugacy_residual


def _solve_pilot_response(template, amplitude_k: float, crust_scale: float, nrbase: int):
    love, _radial, _model = get_love(
        build_mars_model(),
        _forcing(),
        _numerics(nrbase),
        mu_variable=_combined_mu_variable(template, amplitude_k, crust_scale),
    )
    return _signed_stokes_observables(love)


def _central_jacobian(template, parameter: str, step: float, nrbase: int):
    if parameter == "thermal":
        plus = _solve_pilot_response(template, step, 1.0, nrbase)
        minus = _solve_pilot_response(template, -step, 1.0, nrbase)
    elif parameter == "crust":
        plus = _solve_pilot_response(template, 0.0, 1.0 + step, nrbase)
        minus = _solve_pilot_response(template, 0.0, 1.0 - step, nrbase)
    else:
        raise ValueError(f"unknown parameter {parameter!r}")
    if plus[1] != minus[1]:
        raise RuntimeError("observable channel order changed between central-difference solves")
    return (plus[0] - minus[0]) / (2.0 * step), plus[1], max(plus[2], minus[2])


def run_pilot(
    template_path: Path,
    output: Path,
    nrbase: int,
    thermal_step_k: float,
    crust_step: float,
    max_amplitude_k: float,
    convergence_tolerance: float,
) -> None:
    """Run only the gate-3--5 L=2 central-difference pilot."""
    if nrbase < 1:
        raise ValueError("nrbase must be positive")
    if thermal_step_k <= 0.0 or crust_step <= 0.0:
        raise ValueError("central-difference steps must be positive")
    if not (0.0 < crust_step < 1.0):
        raise ValueError("crust step must keep both scale factors positive")
    if max_amplitude_k <= 0.0 or convergence_tolerance <= 0.0:
        raise ValueError("amplitude and convergence tolerance must be positive")

    template = _load_l2_template(template_path)
    positivity = positivity_diagnostics(
        thermal_fractional_coefficients(template, max_amplitude_k)
    )
    if not positivity.passes:
        raise SystemExit("TASK-043 positivity gate failed; stop before coupled pilot")
    closure = l2_mode_closure(_combined_mu_variable(template, max_amplitude_k, 1.0))
    if len(closure) != EXPECTED_L2_ACTIVE_MODES:
        raise RuntimeError(
            f"L=2 two-layer closure N={len(closure)}, expected {EXPECTED_L2_ACTIVE_MODES}"
        )

    thermal_jacobian, channels, conjugacy_t = _central_jacobian(
        template, "thermal", thermal_step_k, nrbase
    )
    thermal_half, channels_half, conjugacy_th = _central_jacobian(
        template, "thermal", thermal_step_k / 2.0, nrbase
    )
    crust_jacobian, channels_c, conjugacy_c = _central_jacobian(
        template, "crust", crust_step, nrbase
    )
    crust_half, channels_ch, conjugacy_ch = _central_jacobian(
        template, "crust", crust_step / 2.0, nrbase
    )
    if not (channels == channels_half == channels_c == channels_ch):
        raise RuntimeError("observable channel order changed across Jacobian cases")

    sigma = np.array(
        [MARS_SIGMA_C20_SEASONAL]
        + [MARS_SIGMA_C30_SEASONAL] * (len(channels) - 1),
        dtype=float,
    )
    covariance = np.diag(sigma**2)
    metrics = jacobian_distinguishability(
        thermal_jacobian, crust_jacobian, covariance, max_amplitude_k
    )
    metrics_half = jacobian_distinguishability(
        thermal_half, crust_half, covariance, max_amplitude_k
    )
    thermal_step_error = float(
        np.linalg.norm(metrics.thermal_whitened - metrics_half.thermal_whitened)
        / np.linalg.norm(metrics_half.thermal_whitened)
    )
    crust_step_error = float(
        np.linalg.norm(metrics.crust_whitened - metrics_half.crust_whitened)
        / np.linalg.norm(metrics_half.crust_whitened)
    )
    correlation_step_error = abs(metrics.correlation - metrics_half.correlation)
    convergence_pass = bool(
        max(thermal_step_error, crust_step_error, correlation_step_error)
        < convergence_tolerance
    )
    max_conjugacy_residual = max(conjugacy_t, conjugacy_th, conjugacy_c, conjugacy_ch)
    conjugacy_pass = bool(max_conjugacy_residual < 1e-10)

    metadata = {
        "task": "TASK-043 gates 3-5",
        "lmax": 2,
        "N": int(len(closure)),
        "nrbase": nrbase,
        "method": "combination",
        "perturbation_order": 2,
        "thermal_step_k": thermal_step_k,
        "crust_scale_step": crust_step,
        "max_abs_thermal_amplitude_k": max_amplitude_k,
        "beta_mu_per_k": -1.92485e-4,
        "temperature_depth_km": 150,
        "covariance_scenario": (
            "diagonal seasonal-gravity analogue: sigma(C20)=1.6e-11 and "
            "sigma(all degree-3 C/S)=1.1e-11"
        ),
        "covariance_caveat": (
            "No mission covariance for this complete signed vector exists in "
            "the repository; this is a scenario benchmark, not mission detectability."
        ),
        "observable_geometry": "mean Mars-Sun distance, (2,0) forcing peak factor",
        "thermal_step_error": thermal_step_error,
        "crust_step_error": crust_step_error,
        "correlation_step_error": correlation_step_error,
        "convergence_tolerance": convergence_tolerance,
        "convergence_pass": convergence_pass,
        "conjugacy_pass": conjugacy_pass,
        "positivity_pass": bool(positivity.passes),
        "positivity_minimum_mu_factor": positivity.minimum_mu_factor,
        "positivity_coefficient_margin": positivity.coefficient_margin,
        "whitened_correlation": metrics_half.correlation,
        "correlation_warning": bool(metrics_half.correlation_warning),
        "orthogonal_sigma_at_max_amplitude": metrics_half.max_orthogonal_sigma,
        "passes_one_sigma": bool(metrics_half.passes_one_sigma),
        "gate_3_5_pass": bool(
            positivity.passes
            and conjugacy_pass
            and convergence_pass
            and metrics_half.passes_one_sigma
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_modes = sorted(template)
    np.savez_compressed(
        output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        channel=np.asarray(channels),
        sigma=sigma,
        covariance=covariance,
        thermal_jacobian=thermal_jacobian,
        thermal_jacobian_halfstep=thermal_half,
        crust_jacobian=crust_jacobian,
        crust_jacobian_halfstep=crust_half,
        thermal_whitened=metrics_half.thermal_whitened,
        crust_whitened=metrics_half.crust_whitened,
        orthogonal_thermal=metrics_half.orthogonal_thermal,
        l2_template_n=np.asarray([n for n, _m in ordered_modes], dtype=int),
        l2_template_m=np.asarray([m for _n, m in ordered_modes], dtype=int),
        l2_template_unit_coefficient=np.asarray(
            [template[mode] for mode in ordered_modes], dtype=float
        ),
        closure_n=closure[:, 0],
        closure_m=closure[:, 1],
        closure_order=closure[:, 2],
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"channels: {channels}")
    print(f"archived: {output}")
    if not metadata["gate_3_5_pass"]:
        print("STOP: TASK-043 gates 3-5 failed; do not escalate to L=3/L=4")
