#!/usr/bin/env python3
"""TASK-046 raw-grid rheology-spectrum parity against native MATLAB.

This diagnostic bypasses the radial solver and reproduces the exact
``Consistency_test_Energy.m`` raw lat/lon rheology path using the Python
port of MATLAB's ``SPH_LatLon`` / ``LatLon_SPH`` transform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_grids,
)
from pylov3d.matlab_sph import (
    filter_rheology_modes,
    maxwell_rheology_from_fractional_grid,
)
from pylov3d.rheology import get_rheology

# Native MATLAB raw-grid anchor from scripts/io_matlab_raw_grid_closure_diagnostic.m
MATLAB = {
    (2, -2): complex(-9.60735661e-08, +5.65536188e-08),
    (2, 0): complex(+2.85014363e-07, -1.57998590e-07),
    (2, 2): complex(-9.69543634e-08, +5.50299311e-08),
    (4, -2): complex(+9.10944254e-10, -1.88740912e-09),
    (4, 0): complex(-1.69998059e-09, +4.52976873e-09),
    (4, 2): complex(+9.40625838e-10, -1.87279354e-09),
}


def relerr(a: complex, b: complex) -> float:
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


def main() -> int:
    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(10)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)

    # The native test constructs the full fractional fields on an lmax=199
    # equiangular grid (source l_max=100 -> map lmax=2*l_max-1).
    _lat, _lon, dmu, deta = io_mu_eta_grids(l_max=100)
    map_lmax = 199
    ilayer = IO_ASTHENOSPHERE_LAYER_INDEX
    mu00, modes = maxwell_rheology_from_fractional_grid(
        dmu,
        deta,
        mu_mean=float(model.mu[ilayer]),
        maxwell_mean=float(model.MaxTime[ilayer]),
        lmax=map_lmax,
    )
    kept = filter_rheology_modes(
        modes,
        cutoff=numerics.rheology_cutoff,
        minimum_log_value=-13.0,
    )
    kept_map = {(n, m): amp for n, m, amp, _lr, _li in kept}

    print("TASK-046 exact MATLAB raw-grid SH transform parity")
    print(f"complex mean muC: {mu00.real:+.12e}{mu00.imag:+.12e}i")
    print(f"retained modes: {len(kept_map)} {sorted(kept_map)}")
    worst = 0.0
    for nm, target in MATLAB.items():
        got = kept_map.get(nm, 0j)
        err = relerr(got, target)
        worst = max(worst, err)
        print(
            f"  {nm}: Python={got.real:+.10e}{got.imag:+.10e}i  "
            f"MATLAB={target.real:+.10e}{target.imag:+.10e}i  relerr={err:.3e}"
        )
    extras = sorted(set(kept_map) - set(MATLAB))
    missing = sorted(set(MATLAB) - set(kept_map))
    print(f"extra retained modes: {extras}")
    print(f"missing retained modes: {missing}")
    print(f"worst retained coefficient relerr: {worst:.3e}")
    print("\nInterpretation:")
    print("  If this reaches near-floating-point parity, the remaining production")
    print("  floor is caused by the current GL/SciPy coefficient path rather than")
    print("  the LOV3D solver. If it does not, the MATLAB SH port itself needs")
    print("  refinement before touching process_lateral_variations().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
