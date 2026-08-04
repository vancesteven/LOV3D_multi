#!/usr/bin/env python
"""F5 — validation_pedigree: coupled 3D solver vs. independent perturbation theory.

One-to-one validation scatter for the five Weber-Moon MATLAB/Qin cases,
reusing the fixtures/helpers of
``pylov3d/tests/test_matlab_validation_ocean.py`` directly (imported, not
duplicated): ``MOON_CASES``, ``_load_qin_reference``,
``_build_weber_moon_model``, ``_HOST_LAYERS``, ``_p2p_to_mu_variable``. For
each case, runs the pylov3d coupled-solver Love-number spectrum exactly as
``TestMoonCoupledOceanValidation.test_lateral_love_spectra_ocean`` does
(same amplitude node, same forcing-mode-deviation convention: subtract
``k2_uniform`` for the forcing mode, divide by sqrt(2) for the appropriate
m>0/m==0 reference rows), and plots ``|k_pylov3d|`` vs. ``|k_reference|``
on a log-log 1:1 comparison, colored by perturbation order.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig5_validation_pedigree.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np

from common import CATEGORICAL, INK, INK_GRAY, OUT_DIR, apply_style, save_fig, style_axes

from pylov3d.types import make_forcing, make_numerics
from pylov3d.tests.test_matlab_validation_ocean import (
    MOON_CASES,
    _HOST_LAYERS,
    _build_weber_moon_model,
    _load_qin_reference,
    _p2p_to_mu_variable,
)

ORDER_COLORS = {1: CATEGORICAL[0], 2: CATEGORICAL[1], 3: CATEGORICAL[2]}
CASE_MARKERS = {
    "moon_3D_LM_10": "o",
    "moon_3D_LM_20": "s",
    "moon_3D_UM_10": "^",
    "moon_3D_UM_11": "D",
    "moon_3D_UM_20": "v",
}


def _run_case(stem: str, n_lv: int, m_lv: int) -> list[dict]:
    """Reproduces TestMoonCoupledOceanValidation.test_lateral_love_spectra_ocean's
    solve + mode-matching + forcing-mode-deviation convention, collecting
    every compared mode instead of asserting on it."""
    from pylov3d.couplings import get_couplings
    from pylov3d.grid import set_boundary_indices
    from pylov3d.rheology import get_rheology, process_lateral_variations
    from pylov3d.solver import _get_solution_coupled
    from pylov3d.love import extract_love_numbers

    ref = _load_qin_reference(stem)
    idx_amp = min(4, len(ref["amp"]) - 1)
    p2p_percent = float(ref["amp"][idx_amp])
    entries = _p2p_to_mu_variable(n_lv, m_lv, p2p_percent)

    raw_model = _build_weber_moon_model()
    forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=10, method="variable", Nrbase=50, perturbation_order=2)
    numerics, model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(model, forcing)

    host = "UM" if "_UM_" in stem else "LM"
    mu_variable = {lay: list(entries) for lay in _HOST_LAYERS[host]}

    model, lateral = process_lateral_variations(model, forcing, mu_variable=mu_variable)
    couplings = get_couplings(lateral.variations, forcing.n, forcing.m, perturbation_order=numerics.perturbation_order)
    y_sol, _r, _Y, _aux = _get_solution_coupled(model, forcing, numerics, couplings, lateral)
    love = extract_love_numbers(y_sol, model, forcing, couplings=couplings)

    rows = []
    for i in range(len(ref["n"])):
        n_m, m_m = int(ref["n"][i]), int(ref["m"][i])
        if m_m < 0:
            continue
        idx_py = np.where((love.n == n_m) & (love.m == m_m))[0]
        if len(idx_py) == 0:
            continue
        k_py = love.k[idx_py[0]]
        k_ref = ref["k"][i, idx_amp]
        if forcing.m == 0 and m_m > 0:
            k_ref /= math.sqrt(2.0)
        elif forcing.m != 0 and m_m == 0:
            k_ref /= math.sqrt(2.0)
        is_forcing_mode = n_m == forcing.n and m_m == forcing.m
        if is_forcing_mode:
            k_py = k_py - ref["k2_uniform"]
        if abs(k_ref) < 1e-7 * ref["k2_uniform"]:
            continue
        rel_error = abs(k_py - k_ref) / abs(k_ref)
        order_m = int(ref["order"][i])
        rows.append(
            {
                "stem": stem, "n": n_m, "m": m_m, "order": order_m,
                "k_py": abs(complex(k_py)), "k_ref": abs(k_ref),
                "rel_error": float(rel_error), "is_forcing_mode": is_forcing_mode,
            }
        )
    return rows


def collect_all() -> list[dict]:
    rows = []
    for stem, n_lv, m_lv in MOON_CASES:
        print(f"[run] {stem} (n_lv={n_lv}, m_lv={m_lv}) ...")
        case_rows = _run_case(stem, n_lv, m_lv)
        print(f"  -> {len(case_rows)} modes compared")
        rows.extend(case_rows)
    return rows


def make_figure(rows: list[dict]):
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    style_axes(ax, grid=True)

    orders_present = sorted({r["order"] for r in rows})
    for order in orders_present:
        sub = [r for r in rows if r["order"] == order]
        xs = [r["k_ref"] for r in sub]
        ys = [r["k_py"] for r in sub]
        ax.scatter(
            xs, ys, s=20, color=ORDER_COLORS.get(order, INK_GRAY),
            marker="o", edgecolors="white", linewidths=0.4, alpha=0.9,
            label=f"perturbation order {order}", zorder=3,
        )

    all_vals = [r["k_ref"] for r in rows] + [r["k_py"] for r in rows]
    lo, hi = min(all_vals) * 0.5, max(all_vals) * 2.0
    ax.plot([lo, hi], [lo, hi], color=INK_GRAY, linewidth=0.8, linestyle="--", zorder=1, label="1:1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$|k_{\rm reference}|$ (MATLAB / Qin perturbation theory)")
    ax.set_ylabel(r"$|k_{\rm pylov3d}|$ (coupled 3D solver)")
    ax.set_title("Ocean-bearing (Weber Moon) validation", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.3, labelspacing=0.3)

    worst = max(rows, key=lambda r: r["rel_error"])
    order1 = [r["rel_error"] for r in rows if r["order"] == 1]
    order1_typ = float(np.median(order1)) if order1 else float("nan")

    ax.text(
        0.98, 0.04,
        f"worst rel. error: {worst['rel_error']:.2%}\n"
        f"({worst['stem']}, n={worst['n']}, m={worst['m']}, order {worst['order']})\n"
        f"typical order-1 agreement: {order1_typ:.1e} rel.",
        transform=ax.transAxes, fontsize=7, color=INK, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5, alpha=0.9),
    )

    return fig, worst, order1_typ


def main():
    rows = collect_all()
    fig, worst, order1_typ = make_figure(rows)
    pdf_path, png_path = save_fig(fig, "fig5_validation_pedigree", OUT_DIR)
    plt.close(fig)

    print(f"\n[summary] n_points={len(rows)}")
    print(f"[summary] worst relative error: {worst['rel_error']:.4%} "
          f"({worst['stem']}, n={worst['n']}, m={worst['m']}, order {worst['order']})")
    print(f"[summary] typical order-1 relative error (median): {order1_typ:.3e}")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
