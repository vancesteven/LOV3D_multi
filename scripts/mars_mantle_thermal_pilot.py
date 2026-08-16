#!/usr/bin/env python3
"""Run bounded evidence stages for the TASK-043 Mars thermal pilot.

The initial ``kernel`` stage is deliberately one-dimensional. It archives a
degree-2 deviatoric strain-energy proxy and independently checks its layer
ordering with finite differences of k2. No coupled solve is run by this stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.energy import compute_stress_strain
from pylov3d.grid import set_boundary_indices
from pylov3d.love import get_love
from pylov3d.mars import MARS_FORCING_TD, build_mars_model
from pylov3d.mars_lateral import _real_sh_to_complex_mu_variable, complex_sh_synthesis
from pylov3d.mars_mantle import (
    area_weighted_mean,
    load_plesa_s1,
    project_temperature_real_sh,
    unit_rms_coefficients,
)
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.types import make_forcing, make_numerics


DEFAULT_KERNEL_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "mars_1d_shear_kernel.npz"
)
DEFAULT_SOURCE_INPUT = (
    REPO_ROOT / "data" / "mars" / "plesa2018" / "grl58258-sup-0002-data"
)
DEFAULT_TEMPLATE_OUTPUT = (
    REPO_ROOT
    / "data"
    / "mars"
    / "plesa2018"
    / "plesa2018_t150_l4_template.npz"
)
UPPER_MANTLE_LAYER = 2
CRUST_LAYER = 3


def _coefficient_arrays(coefficients: dict[tuple[int, int], float]):
    modes = sorted(coefficients)
    return (
        np.asarray([n for n, _m in modes], dtype=int),
        np.asarray([m for _n, m in modes], dtype=int),
        np.asarray([coefficients[mode] for mode in modes], dtype=float),
    )


def run_projection(source: Path, output: Path, lmax: int = 4) -> None:
    """Run gate 2 and archive only the low-degree derived coefficients."""
    if lmax != 4:
        raise ValueError("TASK-043 registers the source projection at lmax=4")
    grid = load_plesa_s1(source)
    temperature = grid.temperature_150km_k
    source_mean = area_weighted_mean(temperature, grid.lat_deg)
    centered = temperature - source_mean
    coefficients = project_temperature_real_sh(
        temperature, grid.lat_deg, grid.lon_e_deg, lmax=lmax
    )
    l2_unit = unit_rms_coefficients(coefficients, cutoff=2)

    entries_l4 = _real_sh_to_complex_mu_variable(coefficients)
    reconstruction_l4 = complex_sh_synthesis(
        entries_l4, grid.lat_deg, grid.lon_e_deg
    ).real
    weights = np.cos(np.radians(grid.lat_deg))[:, None]
    residual = centered - reconstruction_l4
    centered_rms = float(
        np.sqrt(np.sum(weights * centered**2) / (np.sum(weights) * centered.shape[1]))
    )
    residual_rms = float(
        np.sqrt(np.sum(weights * residual**2) / (np.sum(weights) * residual.shape[1]))
    )

    degree_rms = np.array(
        [
            np.sqrt(
                sum(value**2 for (degree, _m), value in coefficients.items() if degree == n)
            )
            for n in range(1, lmax + 1)
        ],
        dtype=float,
    )
    n, m, coefficient_k = _coefficient_arrays(coefficients)
    l2_n, l2_m, l2_unit_coefficient = _coefficient_arrays(l2_unit)
    metadata = {
        "task": "TASK-043 gate 2",
        "source_product": "Plesa et al. (2018) Data Set S1",
        "source_filename": "grl58258-sup-0002-data",
        "source_depth_km": 150,
        "source_longitude_convention": "0..359 degrees east",
        "source_latitude_order": "90..-89 degrees",
        "source_temperature_units": "K",
        "source_size_bytes": 7_841_309,
        "source_md5": "47bab533418619fa8da74c99e9a4e6d1",
        "source_sha256": (
            "88c80be18a4a4bef411c18218ead2f8019c8bf33e96ec71e1a42a5788b9ed1ee"
        ),
        "basis": "real 4pi normalized, no Condon-Shortley; +m cosine/-m sine",
        "projection": "cos(latitude)-weighted least squares",
        "mean_removal": "cos(latitude)-weighted",
        "lmax": lmax,
        "source_driven_design_change": (
            "Data Set S1 contains T at 150 km, not the provisional 400 km slice"
        ),
        "raw_data_redistributed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        n=n,
        m=m,
        coefficient_k=coefficient_k,
        l2_n=l2_n,
        l2_m=l2_m,
        l2_unit_coefficient=l2_unit_coefficient,
        degree=np.arange(1, lmax + 1, dtype=int),
        degree_rms_k=degree_rms,
        source_area_weighted_mean_k=np.array(source_mean),
        source_min_k=np.array(float(np.min(temperature))),
        source_max_k=np.array(float(np.max(temperature))),
        source_centered_rms_k=np.array(centered_rms),
        l4_projection_residual_rms_k=np.array(residual_rms),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"source mean/min/max K: {source_mean:.6f} / {np.min(temperature):.6f} / {np.max(temperature):.6f}")
    print(f"centered RMS K: {centered_rms:.6f}")
    print(f"degree RMS K: {degree_rms}")
    print(f"L4 residual RMS K: {residual_rms:.6f}")
    print(f"archived: {output}")


def _forcing():
    return make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)


def _numerics(nrbase: int):
    return make_numerics(
        n_layers=4,
        method="combination",
        Nrbase=nrbase,
        perturbation_order=2,
    )


def _k2_for_layer_scale(layer: int, factor: float, nrbase: int) -> float:
    model = build_mars_model()
    model = model._replace(mu0=model.mu0.at[layer].set(model.mu0[layer] * factor))
    love, _radial, _normalized = get_love(model, _forcing(), _numerics(nrbase))
    return float(np.real(np.asarray(love.k)[0]))


def _finite_difference(layer: int, epsilon: float, nrbase: int) -> float:
    """Central derivative of real k2 with respect to fractional layer mu."""
    plus = _k2_for_layer_scale(layer, 1.0 + epsilon, nrbase)
    minus = _k2_for_layer_scale(layer, 1.0 - epsilon, nrbase)
    return (plus - minus) / (2.0 * epsilon)


def _layer_interval_integrals(
    r: np.ndarray,
    weighted_profile: np.ndarray,
    boundary_radii: np.ndarray,
) -> np.ndarray:
    """Trapezoidal integrals, assigning intervals by their radial midpoint."""
    mids = 0.5 * (r[:-1] + r[1:])
    pieces = 0.5 * (weighted_profile[:-1] + weighted_profile[1:]) * np.diff(r)
    result = np.zeros(len(boundary_radii), dtype=float)
    lower = 0.0
    for layer, upper in enumerate(boundary_radii):
        mask = (mids > lower) & (mids <= upper)
        result[layer] = float(np.sum(pieces[mask]))
        lower = float(upper)
    return result


def kernel_record(nrbase: int, epsilon: float) -> dict[str, np.ndarray | float | int]:
    """Compute the gate-1 record at one radial resolution."""
    forcing = _forcing()
    numerics = _numerics(nrbase)
    numerics, model = set_boundary_indices(numerics, build_mars_model())
    model = get_rheology(model, forcing)
    y, r, _fundamental, aprop_aux = get_solution(model, forcing, numerics)
    _u, stress, strain = compute_stress_strain(
        y, r, aprop_aux, model, forcing, numerics
    )

    # With LOV3D's stress sign convention this contraction is negative for
    # the elastic Mars reference solution. Negating it yields the positive
    # deviatoric strain-energy weight used only as a radial sensitivity proxy.
    shear_contraction = np.real(
        np.sum(np.conj(stress[:, 1:]) * strain[:, 1:], axis=1)
    )
    weighted_profile = -shear_contraction * np.asarray(r) ** 2
    if np.min(weighted_profile) < -1e-12 * np.max(np.abs(weighted_profile)):
        raise RuntimeError("deviatoric energy proxy has a material negative region")
    weighted_profile = np.maximum(weighted_profile, 0.0)

    boundary_radii = np.asarray(model.R[: model.n_layers], dtype=float)
    layer_integrals = _layer_interval_integrals(
        np.asarray(r, dtype=float), weighted_profile, boundary_radii
    )
    solid_total = float(np.sum(layer_integrals[1:]))
    layer_fractions = layer_integrals / solid_total

    derivatives = np.array(
        [
            _finite_difference(UPPER_MANTLE_LAYER, epsilon, nrbase),
            _finite_difference(CRUST_LAYER, epsilon, nrbase),
        ],
        dtype=float,
    )
    derivatives_half = np.array(
        [
            _finite_difference(UPPER_MANTLE_LAYER, epsilon / 2.0, nrbase),
            _finite_difference(CRUST_LAYER, epsilon / 2.0, nrbase),
        ],
        dtype=float,
    )

    return {
        "nrbase": nrbase,
        "nr": int(numerics.Nr),
        "epsilon": epsilon,
        "r_normalized": np.asarray(r, dtype=float),
        "r_km": np.asarray(r, dtype=float) * float(model.R0[model.n_layers - 1]),
        "shear_energy_weight": weighted_profile,
        "stress_real": np.real(stress),
        "stress_imag": np.imag(stress),
        "strain_real": np.real(strain),
        "strain_imag": np.imag(strain),
        "boundary_radii_normalized": boundary_radii,
        "boundary_radii_km": np.asarray(model.R0[: model.n_layers], dtype=float),
        "mu0_pa": np.asarray(model.mu0[: model.n_layers], dtype=float),
        "layer_integrals": layer_integrals,
        "layer_fractions": layer_fractions,
        "dk2_dfraction_layers_2_3": derivatives,
        "dk2_dfraction_layers_2_3_halfstep": derivatives_half,
    }


def run_kernel(
    output: Path,
    nrbase_values: list[int],
    epsilon: float,
    tolerance: float,
) -> None:
    if len(nrbase_values) < 2:
        raise ValueError("kernel evidence needs at least two Nrbase values")
    if sorted(nrbase_values) != nrbase_values or len(set(nrbase_values)) != len(nrbase_values):
        raise ValueError("Nrbase values must be unique and increasing")
    if not (0.0 < epsilon < 0.1):
        raise ValueError("epsilon must lie between 0 and 0.1")

    records = [kernel_record(nrbase, epsilon) for nrbase in nrbase_values]
    finest = records[-1]
    prior = records[-2]

    fractions = np.vstack([record["layer_fractions"] for record in records])
    derivatives = np.vstack(
        [record["dk2_dfraction_layers_2_3"] for record in records]
    )
    derivatives_half = np.vstack(
        [record["dk2_dfraction_layers_2_3_halfstep"] for record in records]
    )
    fraction_convergence = np.max(
        np.abs(fractions[-1, 1:] - fractions[-2, 1:])
        / np.maximum(np.abs(fractions[-1, 1:]), np.finfo(float).tiny)
    )
    derivative_convergence = np.max(
        np.abs(derivatives[-1] - derivatives[-2])
        / np.maximum(np.abs(derivatives[-1]), np.finfo(float).tiny)
    )
    halfstep_convergence = np.max(
        np.abs(derivatives[-1] - derivatives_half[-1])
        / np.maximum(np.abs(derivatives_half[-1]), np.finfo(float).tiny)
    )

    energy_ordering_pass = bool(
        finest["layer_integrals"][UPPER_MANTLE_LAYER]
        > finest["layer_integrals"][CRUST_LAYER]
    )
    derivative_ordering_pass = bool(abs(derivatives[-1, 0]) > abs(derivatives[-1, 1]))
    convergence_pass = bool(
        max(fraction_convergence, derivative_convergence, halfstep_convergence)
        < tolerance
    )
    gate_pass = energy_ordering_pass and derivative_ordering_pass and convergence_pass

    metadata = {
        "task": "TASK-043 gate 1",
        "description": (
            "1D degree-2 deviatoric strain-energy radial proxy, corroborated "
            "by central finite differences of k2"
        ),
        "forcing": {"n": 2, "m": 0, "F": 1.0, "Td_s": MARS_FORCING_TD},
        "method": "combination",
        "layers": {"upper_mantle": UPPER_MANTLE_LAYER, "crust": CRUST_LAYER},
        "epsilon": epsilon,
        "tolerance": tolerance,
        "fraction_convergence": float(fraction_convergence),
        "derivative_convergence": float(derivative_convergence),
        "halfstep_convergence": float(halfstep_convergence),
        "energy_ordering_pass": energy_ordering_pass,
        "derivative_ordering_pass": derivative_ordering_pass,
        "convergence_pass": convergence_pass,
        "gate_pass": gate_pass,
        "caveat": (
            "The energy profile is a radial sensitivity proxy, not an "
            "absolutely normalized Frechet kernel; finite-difference k2 "
            "derivatives provide the independent layer-ordering check."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        nrbase=np.asarray(nrbase_values, dtype=int),
        nr=np.asarray([record["nr"] for record in records], dtype=int),
        layer_fractions_by_resolution=fractions,
        dk2_dfraction_layers_2_3_by_resolution=derivatives,
        dk2_dfraction_layers_2_3_halfstep_by_resolution=derivatives_half,
        r_normalized=finest["r_normalized"],
        r_km=finest["r_km"],
        shear_energy_weight=finest["shear_energy_weight"],
        stress_real=finest["stress_real"],
        stress_imag=finest["stress_imag"],
        strain_real=finest["strain_real"],
        strain_imag=finest["strain_imag"],
        boundary_radii_normalized=finest["boundary_radii_normalized"],
        boundary_radii_km=finest["boundary_radii_km"],
        mu0_pa=finest["mu0_pa"],
        layer_integrals=finest["layer_integrals"],
        layer_fractions=finest["layer_fractions"],
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"upper-mantle/crust energy ratio: {finest['layer_integrals'][2] / finest['layer_integrals'][3]:.6g}")
    print(f"upper-mantle/crust |dk2/dln(mu)| ratio: {abs(derivatives[-1, 0] / derivatives[-1, 1]):.6g}")
    print(f"archived: {output}")
    if not gate_pass:
        raise SystemExit("TASK-043 gate 1 failed; stop before coupled pilot")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    project = subparsers.add_parser("project", help="run gate-2 source projection")
    project.add_argument("--source", type=Path, default=DEFAULT_SOURCE_INPUT)
    project.add_argument("--output", type=Path, default=DEFAULT_TEMPLATE_OUTPUT)
    kernel = subparsers.add_parser("kernel", help="run gate-1 1D evidence")
    kernel.add_argument("--output", type=Path, default=DEFAULT_KERNEL_OUTPUT)
    kernel.add_argument("--nrbase", type=int, nargs="+", default=[50, 100, 200])
    kernel.add_argument("--epsilon", type=float, default=1e-3)
    kernel.add_argument("--tolerance", type=float, default=0.01)

    args = parser.parse_args(argv)
    if args.stage == "project":
        run_projection(args.source, args.output)
    elif args.stage == "kernel":
        run_kernel(args.output, args.nrbase, args.epsilon, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
