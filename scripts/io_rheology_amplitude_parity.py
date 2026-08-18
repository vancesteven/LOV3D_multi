#!/usr/bin/env python3
"""TASK-046 cheap diagnostic for raw-grid MATLAB rheology-amplitude parity.

The authoritative MATLAB raw-grid path treats mu_latlon.z and eta_latlon.z as
complete multiplicative fields, including their non-unit spherical means.
Python's coefficient API carries only lateral residuals and drops (0,0).

This diagnostic factors each physical field into mean x mean-normalized
residual, folds the means into the scalar layer mu and Maxwell time, and sends
only the normalized residuals through process_lateral_variations.  Algebraically
this reproduces the raw-grid Maxwell field without requiring a new grid API.
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
    _io_dmu_deta,
    _io_z_pattern,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
)
from pylov3d.rheology import _sh_analysis, _sh_grid, get_rheology, process_lateral_variations

LAYER = IO_ASTHENOSPHERE_LAYER_INDEX
Y00 = 1.0 / np.sqrt(4.0 * np.pi)

# From scripts/io_matlab_raw_grid_closure_diagnostic.m / raw-grid MATLAB path.
MATLAB_MUC = {
    (2, -2): -9.60735661e-08 + 5.65536188e-08j,
    (2, 0):  +2.85014363e-07 - 1.57998590e-07j,
    (2, 2):  -9.69543634e-08 + 5.50299311e-08j,
    (4, -2): +9.10944254e-10 - 1.88740912e-09j,
    (4, 0):  -1.69998059e-09 + 4.52976873e-09j,
    (4, 2):  +9.40625838e-10 - 1.87279354e-09j,
}


def _entries(sh: dict[tuple[int, int], complex], rel=1e-12):
    vals = [abs(v) for (n, _m), v in sh.items() if n > 0]
    vmax = max(vals, default=0.0)
    return [(n, m, a) for (n, m), a in sorted(sh.items()) if n > 0 and abs(a) > rel * vmax]


def main() -> int:
    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(10)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)

    theta, phi, weights = _sh_grid(30)
    z = _io_z_pattern(theta[:, None], phi[None, :])
    dmu, deta = _io_dmu_deta(z)
    mu_full = 1.0 + dmu
    eta_full = 1.0 + deta

    mu_full_sh = _sh_analysis(mu_full.astype(complex), theta, phi, weights, 30)
    eta_full_sh = _sh_analysis(eta_full.astype(complex), theta, phi, weights, 30)
    mu_mean = float((mu_full_sh[(0, 0)] * Y00).real)
    eta_mean = float((eta_full_sh[(0, 0)] * Y00).real)

    mu_norm = mu_full / mu_mean - 1.0
    eta_norm = eta_full / eta_mean - 1.0
    mu_sh = _sh_analysis(mu_norm.astype(complex), theta, phi, weights, 30)
    eta_sh = _sh_analysis(eta_norm.astype(complex), theta, phi, weights, 30)

    mu_variable = {LAYER: _entries(mu_sh)}
    eta_variable = {LAYER: _entries(eta_sh)}

    mu_arr = np.asarray(model.mu, dtype=float).copy()
    max_arr = np.asarray(model.MaxTime, dtype=float).copy()
    mu_arr[LAYER] *= mu_mean
    max_arr[LAYER] *= eta_mean / mu_mean
    model = model._replace(mu=model.mu.at[LAYER].set(mu_arr[LAYER]),
                           MaxTime=model.MaxTime.at[LAYER].set(max_arr[LAYER]))

    model2, lateral = process_lateral_variations(
        model, forcings,
        mu_variable=mu_variable,
        eta_variable=eta_variable,
        rheology_cutoff=numerics.rheology_cutoff,
    )

    got = {
        tuple(map(int, nm)): complex(lateral.muC_amp[LAYER, j])
        for j, nm in enumerate(np.asarray(lateral.variations))
    }

    print("TASK-046 Io rheology amplitude parity")
    print(f"mu full-field mean factor:  {mu_mean:.12e}")
    print(f"eta full-field mean factor: {eta_mean:.12e}")
    print(f"retained modes: {len(got)} {sorted(got)}")
    worst = 0.0
    for nm in sorted(MATLAB_MUC):
        g = got.get(nm, 0j)
        ref = MATLAB_MUC[nm]
        err = abs(g-ref) / max(abs(ref), np.finfo(float).tiny)
        worst = max(worst, err)
        print(f"  {nm}: Python={g.real:+.10e}{g.imag:+.10e}i  MATLAB={ref.real:+.10e}{ref.imag:+.10e}i  relerr={err:.3e}")
    print(f"worst retained-coefficient relerr: {worst:.3e}")
    return 0 if len(got) == 6 and worst < 5e-3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
