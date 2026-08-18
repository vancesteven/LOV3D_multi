#!/usr/bin/env python3
"""Diagnose TASK-046 Io viscoelastic rheology-spectrum parity.

This script isolates the active-mode discrepancy without running the radial
solver. It compares the current Python viscoelastic processing with the
MATLAB-faithful raw-grid/work-grid path established during TASK-046:

* degree-30 working grid for viscoelastic rheology;
* nonlinear Maxwell complex shear modulus evaluated on that grid;
* re-analysis through degree 59; and
* real/imaginary amplitude filtering within ``rheology_cutoff`` decades of
  the strongest nonzero component, matching ``get_rheology.m``.

The authoritative native-MATLAB raw-grid diagnostic retains six rheology
modes and gives perturbation-order-2 solution closures [43, 41, 41] for the
three Io eccentricity-tide forcing components. The earlier [125,125,125]
coefficient-path result is retired because it mixed Python and MATLAB SH
coefficient conventions before the nonlinear grid transform.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_variable,
)
from pylov3d.rheology import (
    _ensure_conjugate_pairs,
    _sh_analysis,
    _sh_grid,
    _sh_synthesis,
    _unify_modes,
    get_rheology,
    process_lateral_variations,
)

TARGET_RETAINED_RHEOLOGY_MODES = 6
TARGET_ACTIVE_COUNTS = [43, 41, 41]


def matlab_work_spectrum(model, mu_variable, eta_variable, cutoff=2.0):
    ilayer = IO_ASTHENOSPHERE_LAYER_INDEX
    mu_modes = _ensure_conjugate_pairs(mu_variable.get(ilayer, []))
    eta_modes = _ensure_conjugate_pairs(eta_variable.get(ilayer, []))
    nm_list, mu_map, eta_map, _ = _unify_modes(mu_modes, eta_modes, None)

    working_lmax = 30
    analysis_lmax = 2 * working_lmax - 1
    theta, phi, weights = _sh_grid(working_lmax)

    mu_coeffs = [(n, m, mu_map[(n, m)]) for n, m in nm_list if abs(mu_map[(n, m)]) > 0]
    eta_coeffs = [(n, m, eta_map[(n, m)]) for n, m in nm_list if abs(eta_map[(n, m)]) > 0]

    mu_field = _sh_synthesis(mu_coeffs, theta, phi)
    eta_field = _sh_synthesis(eta_coeffs, theta, phi)
    maxwell_field = np.real(eta_field) / np.real(mu_field)

    mu_i = float(model.mu[ilayer])
    maxwell_mean = float(model.MaxTime[ilayer])
    cmu_field = mu_i * mu_field / (1.0 - 1j / (maxwell_field * maxwell_mean))
    cmu_sh = _sh_analysis(cmu_field, theta, phi, weights, analysis_lmax)

    y00 = 1.0 / np.sqrt(4.0 * np.pi)
    mu00 = cmu_sh[(0, 0)] * y00

    rows = []
    for (n, m), amp in cmu_sh.items():
        if n == 0:
            continue
        rr = abs(amp.real * y00 / max(abs(mu00.real), np.finfo(float).tiny))
        ii = abs(amp.imag * y00 / max(abs(mu00.imag), np.finfo(float).tiny))
        lr = np.log10(max(rr, np.finfo(float).tiny))
        li = np.log10(max(ii, np.finfo(float).tiny))
        rows.append((n, m, amp, lr, li))

    max_log = max(max(r[3], r[4]) for r in rows)
    kept = [
        (n, m, amp, lr, li)
        for n, m, amp, lr, li in rows
        if (lr - max_log >= -cutoff) or (li - max_log >= -cutoff)
    ]
    variations = np.asarray(sorted((n, m) for n, m, *_ in kept), dtype=int)
    return mu00, kept, variations


def closure_counts(variations, forcings, perturbation_order):
    return [
        len(
            get_couplings(
                variations,
                f.n,
                f.m,
                perturbation_order=perturbation_order,
            ).n_s
        )
        for f in forcings
    ]


def main():
    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(10)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)
    mu_variable, eta_variable, _ = io_mu_eta_variable()

    _, lateral_current = process_lateral_variations(
        model,
        forcings,
        mu_variable=mu_variable,
        eta_variable=eta_variable,
        rheology_cutoff=numerics.rheology_cutoff,
    )
    current_counts = closure_counts(
        lateral_current.variations, forcings, numerics.perturbation_order
    )

    mu00, kept, variations = matlab_work_spectrum(
        model, mu_variable, eta_variable, cutoff=numerics.rheology_cutoff
    )
    parity_counts = closure_counts(
        variations, forcings, numerics.perturbation_order
    )

    print("TASK-046 Io rheology-spectrum diagnostic")
    print(f"current Python retained rheology modes: {len(lateral_current.variations)}")
    print(f"current Python active solution counts: {current_counts}")
    print(f"MATLAB-work retained rheology modes: {len(variations)}")
    print(f"MATLAB-work active solution counts: {parity_counts}")
    print(f"MATLAB-work retained degree range: {variations[:,0].min()}..{variations[:,0].max()}")
    print(f"complex mean muC: {mu00.real:+.12e}{mu00.imag:+.12e}i")
    print("strongest retained modes (sorted by |muC_nm|):")
    for n, m, amp, lr, li in sorted(kept, key=lambda x: abs(x[2]), reverse=True)[:20]:
        print(
            f"  ({n:2d},{m:+3d})  muC={amp.real:+.6e}{amp.imag:+.6e}i  "
            f"logR={lr:+.3f} logI={li:+.3f}"
        )
    print(
        "native MATLAB raw-grid target: "
        f"{TARGET_RETAINED_RHEOLOGY_MODES} rheology modes, "
        f"active counts {TARGET_ACTIVE_COUNTS}"
    )

    reference_ok = (
        len(variations) == TARGET_RETAINED_RHEOLOGY_MODES
        and parity_counts == TARGET_ACTIVE_COUNTS
    )
    current_ok = (
        len(lateral_current.variations) == TARGET_RETAINED_RHEOLOGY_MODES
        and current_counts == TARGET_ACTIVE_COUNTS
    )
    print(f"MATLAB-work/raw-grid reference reproduction: {'PASS' if reference_ok else 'FAIL'}")
    print(f"general Python processor parity: {'PASS' if current_ok else 'OPEN'}")
    return 0 if (reference_ok and current_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
