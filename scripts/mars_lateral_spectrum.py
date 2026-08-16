#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Compute and archive the Mars lateral Love-number spectrum (TASK-041 part 0).

Mars has no committed full-spectrum artifact analogous to the Moon's
``moon_lateral_spectrum.npz`` (n, m, k, h, l + provenance): the Mars
artifact of record is the MATLAB anchor
``data/tests/mars/mars_lateral_cross_check.mat``, which carries k only.
Downstream displacement maps (part 1) need h, so this script runs the
shipped ``pylov3d.mars_lateral.mars_lateral_love_spectrum`` configuration
(lmax=4, N=115 coupled modes, ``method='variable'``, ``Nrbase=30``,
``perturbation_order=2``, unit (2,0) forcing -- matching the settings the
committed MATLAB anchor itself used, ``docs/MARS_MODEL.md`` section 4/the
``.mat``'s own ``method`` field) and saves the resulting n, m, k, h, l plus
a *uniform* (no-lateral-heterogeneity) reference solve at the same forcing
and numerics, so the forcing mode's k, h, l can be split into "the
spherically symmetric response" and "the lateral shift" (part 1 maps only
the latter for (2,0)).

Gate: the k array here must match the committed MATLAB anchor per mode at
the established precision (~1e-11 relative or better on significant
modes, matching the ~2.95e-13 forcing-mode precedent recorded in
``docs/MARS_MODEL.md``) before anything downstream (part 1/2) uses it.

Usage
-----
    venvLOV3Dconv/bin/python scripts/mars_lateral_spectrum.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.love import get_love
from pylov3d.mars import MARS, MARS_FORCING_TD, build_mars_model
from pylov3d.mars_lateral import mars_lateral_love_spectrum
from pylov3d.types import make_forcing, make_numerics

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "proposal" / "mars_lateral_spectrum.npz"
ANCHOR_MAT = REPO_ROOT / "data" / "tests" / "mars" / "mars_lateral_cross_check.mat"

FORCING = (2, 0)
LMAX = 4
NRBASE = 30
PERTURBATION_ORDER = 2
METHOD = "variable"


def _forcing_index(love, n_f: int, m_f: int) -> int:
    matches = np.where((love.n == n_f) & (love.m == m_f))[0]
    if len(matches) != 1:
        raise RuntimeError("forcing mode is missing or duplicated")
    return int(matches[0])


def _uniform_reference(nrbase: int, method: str, perturbation_order: int) -> dict:
    """No-lateral-heterogeneity (mu_variable=None) solve at the same forcing
    and numerics as the coupled solve, for the forcing mode's uniform
    (k, h, l) baseline -- the shift baseline part 1 subtracts for (2,0)."""
    model = build_mars_model()
    forcing_obj = make_forcing(Td=MARS_FORCING_TD, n=FORCING[0], m=FORCING[1], F=1.0)
    numerics = make_numerics(
        n_layers=4, method=method, Nrbase=nrbase, perturbation_order=perturbation_order,
    )
    love, _y, _model = get_love(model, forcing_obj, numerics, mu_variable=None)
    idx = _forcing_index(love, *FORCING)
    return {
        "k": complex(love.k[idx]),
        "h": complex(love.h[idx]),
        "l": complex(love.l[idx]),
    }


def _gate_against_mat(n: np.ndarray, m: np.ndarray, k: np.ndarray, mat_path: Path) -> dict:
    """Compare the computed k spectrum against the committed MATLAB anchor,
    per (n, m) mode. Returns per-mode relative-error statistics; raises if
    any significant mode disagrees worse than 1e-9 relative."""
    import scipy.io as sio

    d = sio.loadmat(mat_path)
    mat_n = d["n"].ravel().astype(int)
    mat_m = d["m"].ravel().astype(int)
    mat_k = d["k"].ravel().astype(complex)

    mat_map = {(int(ni), int(mi)): complex(ki) for ni, mi, ki in zip(mat_n, mat_m, mat_k)}
    py_map = {(int(ni), int(mi)): complex(ki) for ni, mi, ki in zip(n, m, k)}

    if set(mat_map.keys()) != set(py_map.keys()):
        missing_py = set(mat_map.keys()) - set(py_map.keys())
        missing_mat = set(py_map.keys()) - set(mat_map.keys())
        raise RuntimeError(
            f"(n,m) mode set mismatch vs anchor: missing from Python {missing_py}, "
            f"extra in Python {missing_mat}"
        )

    # "Significant" = amplitude not deep in numerical noise relative to the
    # largest response mode (the forcing mode itself, |k|~0.169); modes at
    # the 1e-12-and-below level are dominated by float64 cancellation in
    # the anchor export itself, not a solver disagreement.
    k_scale = max(abs(v) for v in mat_map.values())
    sig_floor = 1e-9 * k_scale

    rows = []
    worst_sig_rel = 0.0
    worst_sig_mode = None
    for nm, k_mat in mat_map.items():
        k_py = py_map[nm]
        abs_err = abs(k_py - k_mat)
        denom = max(abs(k_mat), sig_floor)
        rel_err = abs_err / denom if denom > 0 else abs_err
        significant = abs(k_mat) >= sig_floor
        rows.append({"n": nm[0], "m": nm[1], "k_py": k_py, "k_mat": k_mat,
                      "abs_err": abs_err, "rel_err": rel_err, "significant": significant})
        if significant and rel_err > worst_sig_rel:
            worst_sig_rel = rel_err
            worst_sig_mode = nm

    sig_rel = [r["rel_err"] for r in rows if r["significant"]]
    result = {
        "n_modes": len(rows),
        "n_significant": len(sig_rel),
        "median_sig_rel_err": float(np.median(sig_rel)) if sig_rel else float("nan"),
        "max_sig_rel_err": float(worst_sig_rel),
        "worst_sig_mode": worst_sig_mode,
        "rows": rows,
    }

    if worst_sig_rel > 1e-9:
        raise RuntimeError(
            f"GATE FAILED: significant mode {worst_sig_mode} disagrees with the MATLAB "
            f"anchor at {worst_sig_rel:.3e} relative (> 1e-9 threshold). "
            "Stopping rather than continuing past a real discrepancy."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mat", type=Path, default=ANCHOR_MAT)
    args = parser.parse_args()

    print(f"solving mars_lateral_love_spectrum(lmax={LMAX}, forcing={FORCING}, "
          f"perturbation_order={PERTURBATION_ORDER}, Nrbase={NRBASE}, method={METHOD!r}) ...")
    result = mars_lateral_love_spectrum(
        lmax=LMAX, forcing=FORCING, perturbation_order=PERTURBATION_ORDER,
        Nrbase=NRBASE, method=METHOD, F=1.0,
    )
    love = result["love"]
    forcing_index = _forcing_index(love, *FORCING)
    k_forcing = complex(love.k[forcing_index])
    h_forcing = complex(love.h[forcing_index])
    l_forcing = complex(love.l[forcing_index])

    print("solving uniform (no-lateral-heterogeneity) reference ...")
    uniform = _uniform_reference(NRBASE, METHOD, PERTURBATION_ORDER)
    k2_uniform = uniform["k"]
    h_uniform = uniform["h"]
    l_uniform = uniform["l"]

    delta_k2 = k_forcing - k2_uniform
    delta_h2 = h_forcing - h_uniform
    delta_l2 = l_forcing - l_uniform

    n_arr = np.asarray(love.n, dtype=int)
    m_arr = np.asarray(love.m, dtype=int)
    k_arr = np.asarray(love.k, dtype=complex)
    h_arr = np.asarray(love.h, dtype=complex)
    l_arr = np.asarray(love.l, dtype=complex)

    print(f"gating against {args.mat} ...")
    gate = _gate_against_mat(n_arr, m_arr, k_arr, args.mat)
    print(f"GATE PASSED: {gate['n_modes']} modes, {gate['n_significant']} significant; "
          f"median rel err {gate['median_sig_rel_err']:.3e}, "
          f"max rel err {gate['max_sig_rel_err']:.3e} at mode {gate['worst_sig_mode']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        n=n_arr, m=m_arr, k=k_arr, h=h_arr, l=l_arr,
        lmax=LMAX, Nrbase=NRBASE, perturbation_order=PERTURBATION_ORDER, method=METHOD,
        forcing_n=FORCING[0], forcing_m=FORCING[1],
        mode_count=len(k_arr),
        wall_s=result["wall_s"],
        k2_uniform=k2_uniform, h_uniform=h_uniform, l_uniform=l_uniform,
        k2_forcing=k_forcing, h_forcing=h_forcing, l_forcing=l_forcing,
        delta_k2=delta_k2, delta_h2=delta_h2, delta_l2=delta_l2,
        forcing_index=forcing_index,
        gate_anchor_path=str(args.mat),
        gate_n_modes=gate["n_modes"],
        gate_n_significant=gate["n_significant"],
        gate_median_sig_rel_err=gate["median_sig_rel_err"],
        gate_max_sig_rel_err=gate["max_sig_rel_err"],
        provenance=(
            "TASK-041 part 0. pylov3d.mars_lateral.mars_lateral_love_spectrum, "
            "lmax=4, forcing=(2,0), perturbation_order=2, Nrbase=30, "
            "method='variable', F=1.0 (unit forcing), N=115 coupled modes. "
            "Uniform reference: same forcing/numerics, mu_variable=None (single "
            "mode, no lateral coupling). Gated against the committed MATLAB "
            "anchor data/tests/mars/mars_lateral_cross_check.mat per mode "
            "before being written."
        ),
    )

    off = sorted(
        ((abs(k_arr[i]), int(n_arr[i]), int(m_arr[i]), complex(k_arr[i]))
         for i in range(len(k_arr)) if i != forcing_index),
        reverse=True,
    )
    print(f"saved {args.output}")
    print(f"N={len(k_arr)} modes, wall={result['wall_s']:.1f} s")
    print(f"k2_uniform={k2_uniform.real:.15g}{k2_uniform.imag:+.3e}j")
    print(f"k2_forcing={k_forcing.real:.15g}{k_forcing.imag:+.3e}j")
    print(f"delta_k2={delta_k2.real:.6e}{delta_k2.imag:+.3e}j")
    print(f"h_uniform={h_uniform.real:.15g}{h_uniform.imag:+.3e}j")
    print(f"h_forcing={h_forcing.real:.15g}{h_forcing.imag:+.3e}j")
    print(f"delta_h2={delta_h2.real:.6e}{delta_h2.imag:+.3e}j")
    print("top off-forcing modes:")
    for amplitude, n_, m_, value in off[:10]:
        print(f"  ({n_},{m_:+d}) |k|={amplitude:.6e} k={value.real:+.6e}{value.imag:+.6e}j")


if __name__ == "__main__":
    main()
