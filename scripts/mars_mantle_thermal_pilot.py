#!/usr/bin/env python3
"""Run the bounded evidence stages for TASK-043.

``project`` registers the external temperature map without redistributing it;
``kernel`` runs only the one-dimensional radial gate; and ``pilot`` runs the
eight central-difference L=2 cases. The pilot archives a negative scientific
result normally, while printing the designed stop decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.mars_lateral import _real_sh_to_complex_mu_variable, complex_sh_synthesis
from pylov3d.mars_mantle import (
    area_weighted_mean,
    load_plesa_s1,
    project_temperature_real_sh,
    unit_rms_coefficients,
)
from scripts.mars_mantle_coupled_gate import run_pilot
from scripts.mars_mantle_radial_gate import run_kernel


DEFAULT_KERNEL_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "mars_1d_shear_kernel.npz"
)
DEFAULT_SOURCE_INPUT = (
    REPO_ROOT / "data" / "mars" / "plesa2018" / "grl58258-sup-0002-data"
)
DEFAULT_TEMPLATE_OUTPUT = (
    REPO_ROOT / "data" / "mars" / "plesa2018" / "plesa2018_t150_l4_template.npz"
)
DEFAULT_PILOT_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "mars_mantle_thermal_l2_pilot.npz"
)


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
    reconstruction_l4 = complex_sh_synthesis(
        _real_sh_to_complex_mu_variable(coefficients), grid.lat_deg, grid.lon_e_deg
    ).real
    weights = np.cos(np.radians(grid.lat_deg))[:, None]
    denominator = np.sum(weights) * centered.shape[1]
    centered_rms = float(np.sqrt(np.sum(weights * centered**2) / denominator))
    residual_rms = float(
        np.sqrt(np.sum(weights * (centered - reconstruction_l4) ** 2) / denominator)
    )
    degree_rms = np.array(
        [
            np.sqrt(sum(v**2 for (degree, _m), v in coefficients.items() if degree == n))
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
    pilot = subparsers.add_parser("pilot", help="run gate-3--5 L=2 evidence")
    pilot.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_OUTPUT)
    pilot.add_argument("--output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    pilot.add_argument("--nrbase", type=int, default=30)
    pilot.add_argument("--thermal-step-k", type=float, default=10.0)
    pilot.add_argument("--crust-step", type=float, default=0.05)
    pilot.add_argument("--max-amplitude-k", type=float, default=300.0)
    pilot.add_argument("--tolerance", type=float, default=0.01)

    args = parser.parse_args(argv)
    if args.stage == "project":
        run_projection(args.source, args.output)
    elif args.stage == "kernel":
        run_kernel(args.output, args.nrbase, args.epsilon, args.tolerance)
    elif args.stage == "pilot":
        run_pilot(
            args.template,
            args.output,
            args.nrbase,
            args.thermal_step_k,
            args.crust_step,
            args.max_amplitude_k,
            args.tolerance,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
