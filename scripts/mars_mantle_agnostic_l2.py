#!/usr/bin/env python3
"""TASK-044: the agnostic L=2 residual test.

TASK-043's single-degree-of-freedom thermal template hid inside the crust
nuisance at L=2 (whitened correlation 0.972940; thermal component orthogonal
to crust only 0.050187 sigma at the declared 300 K bound). Per the verified
TASK-042 design, the next scientific action is to drop the thermal prior
entirely and ask whether *any* laterally varying upper-mantle rigidity
pattern at L=2 -- not just the Plesa thermal shape -- produces an observable
response orthogonal to the crust nuisance at positivity-admissible amplitude.

This driver reuses the TASK-043 pilot's solve/observable machinery
(``scripts/mars_mantle_coupled_gate.py``) and its committed crust Jacobian
artifact rather than recomputing them, computes central-difference response
Jacobians for each of the five real L=2 upper-mantle basis coefficients
(C20, C21, S21, C22, S22), whitens and projects them orthogonal to the crust
Jacobian, and reports the SVD decomposition, positivity-bounded amplitudes,
and the resulting sigma-level decision metric.

The diagonal seasonal-gravity covariance reused from the TASK-043 artifact is
a scenario benchmark, not a mission covariance; every conclusion drawn here
is conditional on it. No L=3 or L=4 pattern is solved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.love import get_love
from pylov3d.mars import build_mars_model
from pylov3d.mars_mantle import (
    BETA_MU_PER_K,
    merge_mantle_crust_mu_variable,
    positivity_diagnostics,
    thermal_mu_variable,
)
from scripts.mars_mantle_coupled_gate import (
    _forcing,
    _numerics,
    _scaled_crust_mu_variable,
    _signed_stokes_observables,
)

PILOT_ARTIFACT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "mars_mantle_thermal_l2_pilot.npz"
)
PILOT_ARTIFACT_SHA256 = (
    "09022def88d241318c63271ded6eee5ecddf8c5ca164d1de345b337fd63f65ee"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "mars_mantle_agnostic_l2.npz"
)

# The five real L=2 upper-mantle basis coefficients in the repository's real,
# 4pi-normalized basis (no Condon-Shortley phase; positive m = cosine, negative
# m = sine). Each pattern below is a single unit coefficient, which already
# has unit spherical RMS in this basis (see mars_mantle.unit_rms_coefficients).
BASIS_ORDER = ["C20", "C21", "S21", "C22", "S22"]
BASIS_MODES = [(2, 0), (2, 1), (2, -1), (2, 2), (2, -2)]

# Fractional delta-mu/mu step, matching the magnitude of TASK-043's thermal
# central-difference step (|beta_mu| * 10 K) so the mantle-basis Jacobians
# are evaluated in the same near-linear regime that pilot demonstrated
# (thermal half-step error 3.12e-6 at that magnitude).
DEFAULT_STEP = abs(BETA_MU_PER_K) * 10.0

POSITIVITY_EPSILON = 1e-6
POSITIVITY_NLAT = 361
POSITIVITY_NLON = 720


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pilot_artifact(path: Path) -> dict:
    observed = _sha256(path)
    if observed != PILOT_ARTIFACT_SHA256:
        raise ValueError(
            f"pilot artifact SHA-256 mismatch: expected {PILOT_ARTIFACT_SHA256}, got {observed}"
        )
    with np.load(path, allow_pickle=False) as archive:
        return {
            "sha256": observed,
            "channel": [str(c) for c in archive["channel"]],
            "sigma": np.asarray(archive["sigma"], dtype=float),
            "covariance": np.asarray(archive["covariance"], dtype=float),
            "crust_jacobian": np.asarray(archive["crust_jacobian"], dtype=float),
            "crust_jacobian_halfstep": np.asarray(
                archive["crust_jacobian_halfstep"], dtype=float
            ),
            "metadata_json": json.loads(str(archive["metadata_json"])),
        }


def _mantle_basis_mu_variable(mode: tuple[int, int], amplitude: float):
    """Single-mode upper-mantle mu_variable, fractional-delta-mu/mu = amplitude."""

    template = {mode: 1.0}
    return thermal_mu_variable(template, amplitude, beta_mu_per_k=1.0)


def _combined_basis_mu_variable(mode: tuple[int, int], amplitude: float):
    return merge_mantle_crust_mu_variable(
        _mantle_basis_mu_variable(mode, amplitude),
        _scaled_crust_mu_variable(1.0),
    )


def _solve_basis_response(mode: tuple[int, int], amplitude: float, nrbase: int):
    love, _radial, _model = get_love(
        build_mars_model(),
        _forcing(),
        _numerics(nrbase),
        mu_variable=_combined_basis_mu_variable(mode, amplitude),
    )
    return _signed_stokes_observables(love)


def _central_basis_jacobian(mode: tuple[int, int], step: float, nrbase: int):
    plus = _solve_basis_response(mode, step, nrbase)
    minus = _solve_basis_response(mode, -step, nrbase)
    if plus[1] != minus[1]:
        raise RuntimeError("observable channel order changed between central-difference solves")
    return (plus[0] - minus[0]) / (2.0 * step), plus[1], max(plus[2], minus[2])


def _positivity_bounds(
    unit_coefficients: dict[tuple[int, int], float],
    epsilon: float,
    nlat: int,
    nlon: int,
) -> dict:
    """Analytic positivity-admissible amplitude bounds for a unit-RMS pattern.

    ``min(1 + A * pattern) > 0`` is linear in the signed amplitude A, so the
    grid-based bound is found directly from the pattern's sampled minimum and
    maximum rather than by search: a positive A is limited by the pattern's
    minimum (most negative excursion), and a negative A by the pattern's
    maximum. The conservative coefficient guard sum_i |a_i| * sqrt(2n_i + 1)
    (TASK-043 gate 3's method, a direct consequence of the 4pi addition
    theorem |Y_nm| <= sqrt(2n+1)) is sign-symmetric and bounds both directions
    identically. Both bounds are reported; the smaller (more conservative) is
    used per sign, and the symmetric amplitude usable for both signs is the
    minimum of the two signed bounds.
    """

    diagnostic = positivity_diagnostics(
        unit_coefficients, epsilon=epsilon, nlat=nlat, nlon=nlon
    )
    pattern_min = diagnostic.minimum_mu_factor - 1.0
    pattern_max = diagnostic.maximum_mu_factor - 1.0
    coefficient_bound = diagnostic.coefficient_upper_bound

    grid_bound_plus = (1.0 - epsilon) / (-pattern_min) if pattern_min < 0.0 else float("inf")
    grid_bound_minus = (1.0 - epsilon) / pattern_max if pattern_max > 0.0 else float("inf")
    coefficient_amplitude = (
        (1.0 - epsilon) / coefficient_bound if coefficient_bound > 0.0 else float("inf")
    )

    amplitude_plus = min(grid_bound_plus, coefficient_amplitude)
    amplitude_minus = min(grid_bound_minus, coefficient_amplitude)
    return {
        "grid_bound_plus": grid_bound_plus,
        "grid_bound_minus": grid_bound_minus,
        "coefficient_bound_amplitude": coefficient_amplitude,
        "coefficient_upper_bound": coefficient_bound,
        "amplitude_plus": amplitude_plus,
        "amplitude_minus": amplitude_minus,
        "amplitude_symmetric": min(amplitude_plus, amplitude_minus),
        "max_imaginary_residual": diagnostic.maximum_imaginary_residual,
    }


def run(
    output: Path,
    nrbase: int,
    step: float,
    convergence_tolerance: float,
    context_rms_amplitude: float,
) -> None:
    if nrbase < 1:
        raise ValueError("nrbase must be positive")
    if step <= 0.0:
        raise ValueError("central-difference step must be positive")
    if convergence_tolerance <= 0.0 or context_rms_amplitude <= 0.0:
        raise ValueError("convergence tolerance and context amplitude must be positive")

    pilot = _load_pilot_artifact(PILOT_ARTIFACT)

    jacobians = []
    jacobians_half = []
    half_step_errors = {}
    channels_ref = None
    max_conjugacy_residual = 0.0
    for mode, label in zip(BASIS_MODES, BASIS_ORDER):
        full, channels, conj_full = _central_basis_jacobian(mode, step, nrbase)
        half, channels_h, conj_half = _central_basis_jacobian(mode, step / 2.0, nrbase)
        if channels_ref is None:
            channels_ref = channels
        if channels != channels_ref or channels_h != channels_ref:
            raise RuntimeError("observable channel order changed across basis Jacobian cases")
        if channels != pilot["channel"]:
            raise RuntimeError("basis-Jacobian channel order does not match the pilot artifact")
        max_conjugacy_residual = max(max_conjugacy_residual, conj_full, conj_half)
        error = float(np.linalg.norm(full - half) / np.linalg.norm(half))
        half_step_errors[label] = error
        jacobians.append(full)
        jacobians_half.append(half)

    mantle_jacobian = np.column_stack(jacobians)  # (8, 5), full step
    mantle_jacobian_half = np.column_stack(jacobians_half)  # (8, 5), half step
    step_converged = bool(max(half_step_errors.values()) < convergence_tolerance)
    conjugacy_pass = bool(max_conjugacy_residual < 1e-10)

    crust_jacobian = pilot["crust_jacobian"]
    covariance = pilot["covariance"]
    sigma = pilot["sigma"]
    channels = pilot["channel"]

    chol = np.linalg.cholesky(covariance)
    crust_w = np.linalg.solve(chol, crust_jacobian)
    crust_norm_sq = float(np.dot(crust_w, crust_w))
    mantle_w = np.linalg.solve(chol, mantle_jacobian_half)  # use the converged half-step version
    projection = np.outer(crust_w, crust_w) / crust_norm_sq
    mantle_orth = mantle_w - projection @ mantle_w  # (8, 5)

    U, S, Vt = np.linalg.svd(mantle_orth, full_matrices=False)

    epsilon = POSITIVITY_EPSILON
    per_direction = []
    for i in range(len(S)):
        mode_coefficients = {mode: float(Vt[i, j]) for j, mode in enumerate(BASIS_MODES)}
        label_coefficients = {
            label: float(Vt[i, j]) for j, label in enumerate(BASIS_ORDER)
        }
        bounds = _positivity_bounds(
            mode_coefficients, epsilon, POSITIVITY_NLAT, POSITIVITY_NLON
        )
        amplitude = bounds["amplitude_symmetric"]
        metric_sigma = float(amplitude * S[i])
        context_sigma = float(context_rms_amplitude * S[i])
        # Convention-resolved metrics (TASK-045 verification finding: the
        # committed verdict sat 1.4% under the gate using the doubly
        # conservative guard+symmetric amplitude, and the raw grid bounds
        # were not archived, so the convention sensitivity could not be
        # assessed from the artifact.  All three conventions are now
        # archived; the sign-respecting grid bound is the "largest
        # physically admitted amplitude" reading of TASK-042's stop rule.)
        grid_symmetric = min(bounds["grid_bound_plus"], bounds["grid_bound_minus"])
        grid_best_sign = max(bounds["grid_bound_plus"], bounds["grid_bound_minus"])
        per_direction.append(
            {
                "index": i,
                "singular_value": float(S[i]),
                "coefficients": label_coefficients,
                "bounds": bounds,
                "amplitude_used": amplitude,
                "decision_metric_sigma": metric_sigma,
                "metric_sigma_grid_symmetric": float(grid_symmetric * S[i]),
                "metric_sigma_grid_best_sign": float(grid_best_sign * S[i]),
                "context_metric_sigma": context_sigma,
            }
        )

    decision_index = int(np.argmax([d["decision_metric_sigma"] for d in per_direction]))
    decision = per_direction[decision_index]
    top = per_direction[0]

    gate_pass = bool(decision["decision_metric_sigma"] >= 1.0)

    metadata = {
        "task": "TASK-044 agnostic L=2 residual test",
        "lmax": 2,
        "nrbase": nrbase,
        "method": "combination",
        "perturbation_order": 2,
        "basis": "real 4pi normalized, no Condon-Shortley; +m cosine/-m sine; single unit coefficient per basis pattern (unit spherical RMS)",
        "basis_order": BASIS_ORDER,
        "observable_channels": channels,
        "mantle_step": step,
        "mantle_step_rationale": (
            "matches TASK-043's thermal central-difference step magnitude "
            "(|beta_mu|*10 K = 1.92485e-3) so mantle-basis Jacobians are "
            "evaluated in the same near-linear regime demonstrated there"
        ),
        "half_step_errors": half_step_errors,
        "convergence_tolerance": convergence_tolerance,
        "step_converged": step_converged,
        "conjugacy_pass": conjugacy_pass,
        "max_conjugacy_residual": max_conjugacy_residual,
        "crust_jacobian_provenance": "reused",
        "crust_jacobian_source_artifact": str(PILOT_ARTIFACT.relative_to(REPO_ROOT)),
        "crust_jacobian_source_sha256": pilot["sha256"],
        "crust_jacobian_reuse_rationale": (
            "mantle-basis half-step error is below the convergence tolerance at "
            "the chosen step, so mixing the reused crust Jacobian (TASK-043's "
            "crust_scale_step=0.05, Nrbase=30, crust_step_error=1.30e-4) with "
            "the newly computed mantle basis Jacobians (also Nrbase=30) does "
            "not require recomputation"
        ),
        "covariance_scenario": pilot["metadata_json"]["covariance_scenario"],
        "covariance_caveat": (
            "No mission covariance for this complete signed vector exists in "
            "the repository; this diagonal seasonal-gravity analogue is a "
            "scenario benchmark, not a mission-detectability claim. Every "
            "conclusion below is conditional on it."
        ),
        "positivity_epsilon": epsilon,
        "positivity_grid_nlat": POSITIVITY_NLAT,
        "positivity_grid_nlon": POSITIVITY_NLON,
        "singular_values": [float(s) for s in S],
        "top_direction_coefficients": top["coefficients"],
        "top_direction_amplitude_plus": top["bounds"]["amplitude_plus"],
        "top_direction_amplitude_minus": top["bounds"]["amplitude_minus"],
        "top_direction_amplitude_symmetric_used": top["amplitude_used"],
        "top_direction_decision_metric_sigma": top["decision_metric_sigma"],
        "per_direction_decision_metric_sigma": [
            d["decision_metric_sigma"] for d in per_direction
        ],
        "decision_direction_index": decision_index,
        "decision_metric_sigma": decision["decision_metric_sigma"],
        "decision_amplitude_used": decision["amplitude_used"],
        "top_direction_is_best_product": bool(decision_index == 0),
        "context_rms_amplitude": context_rms_amplitude,
        "context_metric_sigma_at_decision_direction": decision["context_metric_sigma"],
        "context_metric_sigma_at_top_direction": top["context_metric_sigma"],
        "context_is_not_the_gate": True,
        "gate_pass_one_sigma": gate_pass,
        "no_l3_l4_solve": True,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        channel=np.asarray(channels),
        sigma=sigma,
        covariance=covariance,
        basis_order=np.asarray(BASIS_ORDER),
        basis_n=np.asarray([n for n, _m in BASIS_MODES], dtype=int),
        basis_m=np.asarray([m for _n, m in BASIS_MODES], dtype=int),
        mantle_jacobian=mantle_jacobian,
        mantle_jacobian_halfstep=mantle_jacobian_half,
        crust_jacobian=crust_jacobian,
        crust_jacobian_halfstep=pilot["crust_jacobian_halfstep"],
        crust_jacobian_source_sha256=np.array(pilot["sha256"]),
        singular_values=S,
        singular_directions=Vt,
        singular_response_directions=U,
        positivity_amplitude_plus=np.asarray(
            [d["bounds"]["amplitude_plus"] for d in per_direction]
        ),
        positivity_amplitude_minus=np.asarray(
            [d["bounds"]["amplitude_minus"] for d in per_direction]
        ),
        positivity_amplitude_symmetric=np.asarray(
            [d["amplitude_used"] for d in per_direction]
        ),
        positivity_grid_bound_plus=np.asarray(
            [d["bounds"]["grid_bound_plus"] for d in per_direction]
        ),
        positivity_grid_bound_minus=np.asarray(
            [d["bounds"]["grid_bound_minus"] for d in per_direction]
        ),
        positivity_coefficient_bound_amplitude=np.asarray(
            [d["bounds"]["coefficient_bound_amplitude"] for d in per_direction]
        ),
        decision_metric_sigma_per_direction=np.asarray(
            [d["decision_metric_sigma"] for d in per_direction]
        ),
        metric_sigma_grid_symmetric_per_direction=np.asarray(
            [d["metric_sigma_grid_symmetric"] for d in per_direction]
        ),
        metric_sigma_grid_best_sign_per_direction=np.asarray(
            [d["metric_sigma_grid_best_sign"] for d in per_direction]
        ),
        threshold_rms_amplitude_for_1sigma=np.asarray(
            [1.0 / d["singular_value"] if d["singular_value"] > 0 else np.inf
             for d in per_direction]
        ),
        context_metric_sigma_per_direction=np.asarray(
            [d["context_metric_sigma"] for d in per_direction]
        ),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"archived: {output}")
    output_sha256 = _sha256(output)
    print(f"artifact SHA-256: {output_sha256}")
    if not gate_pass:
        print(
            "Guard-convention metric does not clear 1 sigma; see the "
            "convention-resolved metrics before reading this as a verdict."
        )
    top_grid_best = per_direction[0]["metric_sigma_grid_best_sign"]
    top_grid_sym = per_direction[0]["metric_sigma_grid_symmetric"]
    print(
        "convention-resolved top-direction metrics: "
        f"guard+symmetric {per_direction[0]['decision_metric_sigma']:.6f} sigma, "
        f"grid symmetric {top_grid_sym:.6f} sigma, "
        f"grid best-sign {top_grid_best:.6f} sigma"
    )
    print(
        "threshold framing: RMS delta-mu/mu for 1 sigma = "
        f"{1.0/per_direction[0]['singular_value']:.6f}; "
        "physically flavored amplitudes (thermal ceiling 0.0577 at 300 K, "
        f"10% RMS context {per_direction[0]['context_metric_sigma']:.6f} sigma) "
        "sit far below it — the stop decision rests on physical amplitude "
        "grounds, conditional on the scenario covariance, not on the "
        "mathematical positivity bound alone."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nrbase", type=int, default=30)
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--context-rms-amplitude", type=float, default=0.10)
    args = parser.parse_args(argv)
    run(args.output, args.nrbase, args.step, args.tolerance, args.context_rms_amplitude)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
