#!/usr/bin/env python3
"""TASK-046 strict MATLAB/Python solver parity on identical coefficients.

Generate the MATLAB artifact first with

    /Applications/MATLAB_R2025b.app/bin/matlab -batch \
      "run('scripts/io_matlab_identical_coefficients_anchor.m')"

The artifact contains a six-mode lateral complex-rheology field whose +/-m
pairs have been symmetrized.  Both codes then apply those exact coefficients to
the already-validated uniform complex rheology background.  This deliberately
removes raw-grid SH quadrature/normalization from the comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
)
from pylov3d.love import extract_love_numbers
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.types import LateralRheology
from scripts.io_energy_gate_bc_multibasis import (
    forcing_mode_k,
    love_energy_estimate,
    monopole_direct_energy,
)

ANCHOR = REPO_ROOT / "data" / "tests" / "io" / "io_identical_coefficients_anchor.mat"
LAYER = IO_ASTHENOSPHERE_LAYER_INDEX


def relerr(a, b):
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


def main() -> int:
    if not ANCHOR.exists():
        raise SystemExit(
            f"missing {ANCHOR}\nRun scripts/io_matlab_identical_coefficients_anchor.m in MATLAB first."
        )

    d = loadmat(ANCHOR, squeeze_me=True)
    rv = np.atleast_2d(np.asarray(d["rv_sym"]))
    if rv.shape[1] < 4:
        raise SystemExit(f"unexpected rv_sym shape {rv.shape}")

    variations = np.asarray(np.rint(rv[:, :2]), dtype=int)
    coeff = np.asarray(rv[:, 3], dtype=complex)

    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(50)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)

    muC_amp = np.zeros((model.n_layers, len(variations)), dtype=complex)
    K_amp = np.zeros_like(muC_amp)
    muC_amp[LAYER, :] = coeff
    uniform = np.ones(model.n_layers, dtype=bool)
    uniform[LAYER] = False
    lateral = LateralRheology(
        variations=variations,
        muC_amp=muC_amp,
        K_amp=K_amp,
        uniform=uniform,
    )

    results = []
    couplings = []
    for f in forcings:
        c = get_couplings(
            variations,
            f.n,
            f.m,
            perturbation_order=numerics.perturbation_order,
        )
        y, r, _Y, aprop = get_solution(
            model, f, numerics, couplings=c, lateral=lateral,
        )
        love = extract_love_numbers(y, model, f, couplings=c)
        results.append((y, r, aprop, love))
        couplings.append(c)

    counts = [len(c.n_s) for c in couplings]
    k_py = [forcing_mode_k(x[3], f) for x, f in zip(results, forcings)]
    e_direct = monopole_direct_energy(
        results,
        forcings,
        model,
        numerics,
        [c.n_s for c in couplings],
        [c.m_s for c in couplings],
        couplings_list=couplings,
        lateral=lateral,
    )
    e_love = (5.0 / float(model.Gg)) * love_energy_estimate(
        [x[3] for x in results], forcings
    )

    k_ml = np.asarray(d["k_strict"]).ravel().astype(complex)
    counts_ml = np.asarray(d["counts_strict"]).ravel().astype(int).tolist()
    e_direct_ml = float(np.real_if_close(np.asarray(d["E_direct_strict"]).squeeze()))
    e_love_ml = float(np.real_if_close(np.asarray(d["E_love_strict"]).squeeze()))

    print("TASK-046 identical-coefficient MATLAB/Python solver parity, Nrbase=50")
    print(f"mode counts Python/MATLAB: {counts} / {counts_ml}")
    worst_k = 0.0
    for f, kp, km in zip(forcings, k_py, k_ml):
        err = relerr(kp, km)
        worst_k = max(worst_k, err)
        print(
            f"m={f.m:+d} Python={kp.real:+.12e}{kp.imag:+.12e}i  "
            f"MATLAB={km.real:+.12e}{km.imag:+.12e}i  relerr={err:.3e}"
        )
    ed_err = relerr(e_direct, e_direct_ml)
    el_err = relerr(e_love, e_love_ml)
    print(f"direct energy Python/MATLAB: {e_direct:.12e} / {e_direct_ml:.12e}  relerr={ed_err:.3e}")
    print(f"Love energy   Python/MATLAB: {e_love:.12e} / {e_love_ml:.12e}  relerr={el_err:.3e}")
    print(f"Python direct/Love mismatch: {100*abs(e_direct-e_love)/abs(e_love):.8f}%")
    print(f"worst k relerr: {worst_k:.3e}")

    # This is a strict solver-parity fixture.  The 1e-8 ceiling is intentionally
    # much tighter than the 1% raw-grid transform-floor gate while allowing
    # normal cross-language floating-point/library differences.
    ok = (
        counts == counts_ml
        and worst_k < 1e-8
        and ed_err < 1e-8
        and el_err < 1e-8
    )
    print(f"strict identical-coefficient solver parity: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
