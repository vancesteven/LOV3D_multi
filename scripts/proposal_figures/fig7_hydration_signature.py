#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F7 -- hydration_signature: TASK-021 hydration-front tidal signature.

Two panels, same style system as figs 1-6. Left: Delta-k2(f_h) vs the
global hydrated-fraction parameter f_h, central serpentinite-property
ratio (solid) with the literature bracket shaded, against a horizontal
line at the current observational 1-sigma (sigma_k2 = 0.006). Right: map
of the hydration-induced lateral rigidity perturbation delta_mu/mu_bar at
f_h=0.3, central ratio (diverging about 0, same convention as fig6's left
panel). All computed live via ``pylov3d.mars_hydration``
(``hydration_forward_sweep``, ``hydration_dmu_over_mu_bar_real``) -- no
``pylov3d`` module modified to produce this.

Runtime: the sweep is the module's documented reduced-grid default
(lmax=2, Nrbase=30 -- see ``pylov3d/mars_hydration.py`` module docstring,
"Runtime"); ~150 s wall (15 coupled solves at ~10 s each plus 18 fast 1D
solves).

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig7_hydration_signature.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np

from common import CATEGORICAL, DIVERGING_CMAP, INK, INK_GRAY, OUT_DIR, apply_style, save_fig, style_axes

from pylov3d.mapping import sh_to_latlon
from pylov3d.mars_hydration import (
    DEFAULT_LMAX,
    MU_SERP_RATIO_CENTRAL,
    SIGMA_K2,
    detectability_summary,
    hydration_dmu_over_mu_bar_real,
    hydration_forward_sweep,
)

SCENARIO_COLOR = CATEGORICAL[0]  # central curve + band


def make_figure(rows: list[dict], dmu_grid) -> plt.Figure:
    apply_style()
    fig, (ax_curve, ax_map) = plt.subplots(
        1, 2, figsize=(7.6, 3.6), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1.0]},
    )

    # ---- left: Delta-k2(f_h), central + bracket band -----------------
    by_scenario: dict[str, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], []).append(r)
    for rs in by_scenario.values():
        rs.sort(key=lambda r: r["f_h"])

    f_h = np.array([r["f_h"] for r in by_scenario["central"]])
    k2_base = by_scenario["central"][0]["k2_total"].real
    d_central = np.array([r["k2_total"].real - k2_base for r in by_scenario["central"]])
    d_low = np.array([r["k2_total"].real - k2_base for r in by_scenario["low"]])
    d_high = np.array([r["k2_total"].real - k2_base for r in by_scenario["high"]])

    ax_curve.fill_between(
        f_h, d_low, d_high, color=SCENARIO_COLOR, alpha=0.18, linewidth=0,
        label="serpentinite property bracket",
    )
    ax_curve.plot(f_h, d_central, color=SCENARIO_COLOR, linewidth=1.6, marker="o", markersize=3.2,
                  label="central ratio")
    ax_curve.axhline(SIGMA_K2, color=INK_GRAY, linewidth=0.9, linestyle="--", zorder=1)
    ax_curve.annotate(
        "current observational 1$\\sigma$ ($\\sigma_{k2}=0.006$)",
        xy=(f_h[-1], SIGMA_K2), xytext=(-2, -4), textcoords="offset points",
        ha="right", va="top", fontsize=6.5, color=INK_GRAY,
    )
    ax_curve.set_xlabel("Hydrated fraction $f_h$")
    ax_curve.set_ylabel("$\\Delta k_2$ (total: mean + lateral)")
    ax_curve.set_title("Hydration-front tidal signature", fontsize=8.5)
    style_axes(ax_curve, grid=True)
    ax_curve.legend(loc="center left", fontsize=6.3, bbox_to_anchor=(0.02, 0.55))

    # ---- right: delta_mu/mu_bar map at f_h=0.3, central ratio --------
    vmax = float(np.max(np.abs(dmu_grid.z)))
    mesh = ax_map.pcolormesh(
        dmu_grid.lon, dmu_grid.lat, dmu_grid.z, cmap=DIVERGING_CMAP,
        vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True,
    )
    ax_map.set_xlim(dmu_grid.lon.min(), dmu_grid.lon.max())
    ax_map.set_ylim(-90, 90)
    ax_map.set_xticks(np.arange(-180, 181, 90))
    ax_map.set_yticks(np.arange(-90, 91, 30))
    ax_map.set_xlabel("Longitude [deg]")
    ax_map.set_ylabel("Latitude [deg]")
    style_axes(ax_map, grid=False)
    ax_map.set_title(
        "Lateral $\\delta\\mu/\\mu_{crust}$, $f_h=0.3$, central ratio", fontsize=8.5,
    )
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.10, fraction=0.035)
    cbar.set_label("%", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    ax_map.annotate(
        f"lateral term only (mean handled separately); $\\ell\\leq${DEFAULT_LMAX}",
        xy=(0, 88), xytext=(0, 0), textcoords="offset points",
        ha="center", va="top", fontsize=6.0, color=INK,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.85),
    )

    fig.suptitle(
        "Mars crustal hydration front (TASK-021): predicted tidal signature",
        fontsize=9, y=1.03,
    )
    return fig


def main():
    rows = hydration_forward_sweep()
    summary = detectability_summary(rows, scenario="central")

    dmu_real = hydration_dmu_over_mu_bar_real(0.3, mu_ratio=MU_SERP_RATIO_CENTRAL, lmax=DEFAULT_LMAX)
    dmu_pct = {nm: v * 100.0 for nm, v in dmu_real.items()}
    dmu_grid = sh_to_latlon(dmu_pct, nlat=180, nlon=360)

    fig = make_figure(rows, dmu_grid)
    pdf_path, png_path = save_fig(fig, "fig7_hydration_signature", OUT_DIR)
    plt.close(fig)

    print(f"[fit] crossing_f_h={summary['crossing_f_h']}, "
          f"precision_to_resolve_f_h_0.1={summary['precision_to_resolve_f_h_0p1']:.3e}")
    for f_h_str, delta in sorted(summary["delta_k2_by_f_h"].items()):
        print(f"[fit] f_h={f_h_str}: |Delta k2|={delta:.4e}")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
