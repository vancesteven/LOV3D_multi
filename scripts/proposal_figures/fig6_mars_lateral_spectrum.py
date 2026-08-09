#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F6 -- mars_lateral_spectrum: TASK-016 Airy lateral-rigidity Love spectrum.

Two panels: left -- the input crust-shell rigidity perturbation
d(mu)/mu_bar (%), from Airy-compensated MarsTopo719 truncated to degree 4
(``pylov3d.mars_lateral.dmu_over_mu_real``); right -- the resulting coupled
Love-number spectrum |k_nm| (``pylov3d.mars_lateral.mars_lateral_love_spectrum``),
modes labeled (n, m) and colored by perturbation order using the same
Okabe-Ito order colors as fig5 (plus a 4th color for the order-0 forcing
mode itself, which fig5's Moon comparison did not need to show). At
perturbation_order=2 the real (both-cosine-and-sine) MarsTopo719 rheology
activates N=115 coupled modes -- far more than fit in a readable x-axis, so
the right panel shows the forcing mode plus the largest-|k| modes per
order, capped at a fixed count, with the true N annotated.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig6_mars_lateral_spectrum.py
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

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.mapping import sh_to_latlon
from pylov3d.mars import MARS_FORCING_TD, build_mars_model
from pylov3d.mars_lateral import (
    crustal_thickness_diagnostics,
    dmu_over_mu_real,
    mars_lateral_love_spectrum,
    mu_variable_from_topography,
)
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.types import make_forcing, make_numerics

# fig5's order colors (1, 2), plus order 0 (the forcing mode itself, not
# present in fig5's Moon comparison) in the 4th Okabe-Ito categorical slot.
ORDER_COLORS = {0: CATEGORICAL[3], 1: CATEGORICAL[0], 2: CATEGORICAL[1]}
ORDER_LABELS = {0: "order 0 (forcing)", 1: "order 1 (direct)", 2: "order 2"}
N_SHOW_PER_ORDER = {0: 1, 1: 14, 2: 10}


def compute_order_map(lmax: int = 4, perturbation_order: int = 2) -> dict:
    """(n, m) -> discovery order, independent of amplitude (get_couplings)."""
    model = build_mars_model()
    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=4, method="combination", Nrbase=15,
        perturbation_order=perturbation_order,
    )
    numerics, model = set_boundary_indices(numerics, model)
    model = get_rheology(model, forcing)
    mu_variable = mu_variable_from_topography(lmax=lmax)
    _, lateral = process_lateral_variations(model, forcing, mu_variable=mu_variable)
    couplings = get_couplings(
        lateral.variations, forcing.n, forcing.m, perturbation_order=perturbation_order,
    )
    return {
        (int(n), int(m)): int(o)
        for n, m, o in zip(couplings.n_s, couplings.m_s, couplings.order)
    }


def select_modes(love, order_map: dict) -> list[dict]:
    """Pick a readable subset: forcing mode + largest-|k| per order.

    Pair-aware (Nit2): +/-m modes of the same (n, |m|) are grouped and
    picked/dropped together, never split -- for a real input field they
    have identical |k| by the rotational-covariance relation
    (TestRotationCovariance), so splitting one out would misleadingly
    suggest an asymmetry that isn't there.
    """
    rows = []
    for i in range(len(love.n)):
        n_m, m_m = int(love.n[i]), int(love.m[i])
        order = order_map.get((n_m, m_m), -1)
        rows.append({"n": n_m, "m": m_m, "k": abs(complex(love.k[i])), "order": order})

    shown = []
    for order, n_show in N_SHOW_PER_ORDER.items():
        groups: dict[tuple[int, int], list[dict]] = {}
        for r in rows:
            if r["order"] != order:
                continue
            groups.setdefault((r["n"], abs(r["m"])), []).append(r)
        ranked = sorted(groups.values(), key=lambda g: -max(x["k"] for x in g))
        picked: list[dict] = []
        for g in ranked:
            if len(picked) >= n_show:
                break
            picked.extend(g)
        shown.extend(picked)
    shown.sort(key=lambda r: (r["order"], -r["k"], r["m"]))
    return shown


def make_figure(dmu_grid, shown_modes, n_total, max_dt_km, wall_s):
    apply_style()
    fig, (ax_map, ax_spec) = plt.subplots(
        1, 2, figsize=(7.6, 3.8), gridspec_kw={"wspace": 0.6, "width_ratios": [1.0, 1.15]},
    )

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
    ax_map.set_title("Crust rigidity perturbation $\\delta\\mu/\\bar\\mu$", fontsize=8.5)
    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.10, fraction=0.035)
    cbar.set_label("%", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    ax_map.annotate(
        f"Airy, $\\ell\\leq4$; max|$\\delta t$|={max_dt_km:.1f} km",
        xy=(0, 88), xytext=(0, 0), textcoords="offset points",
        ha="center", va="top", fontsize=6.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.85),
    )

    x = np.arange(len(shown_modes))
    k_vals = np.array([m["k"] for m in shown_modes])
    colors = [ORDER_COLORS.get(m["order"], INK_GRAY) for m in shown_modes]
    ymin = max(k_vals.min() * 0.5, 1e-30)
    ax_spec.vlines(x, ymin, k_vals, color=colors, linewidth=1.0, zorder=1)
    ax_spec.scatter(x, k_vals, c=colors, s=14, zorder=2, edgecolor="white", linewidth=0.3)
    ax_spec.set_yscale("log")
    ax_spec.set_xticks(x)
    ax_spec.set_xticklabels(
        [f"({m['n']},{m['m']})" for m in shown_modes], rotation=90, fontsize=5.5,
    )
    ax_spec.set_ylabel("$|k_{nm}|$")
    ax_spec.yaxis.set_label_coords(-0.16, 0.5)
    style_axes(ax_spec, grid=True)
    ax_spec.set_title(
        f"Coupled Love spectrum ({len(shown_modes)} of {n_total} modes)", fontsize=8.5,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color=ORDER_COLORS[o], linestyle="", label=ORDER_LABELS[o])
        for o in (0, 1, 2)
    ]
    ax_spec.legend(handles=handles, loc="upper right", fontsize=6)

    fig.suptitle(
        "Mars degree-2 tide + Airy lateral crust rigidity (TASK-016), "
        f"wall={wall_s:.0f}s",
        fontsize=9, y=1.03,
    )
    return fig


def main():
    diag = crustal_thickness_diagnostics(lmax=4)
    max_dt_km = diag["max_abs_dt_m"] / 1e3

    dmu_real = dmu_over_mu_real(lmax=4)
    dmu_pct = {nm: v * 100.0 for nm, v in dmu_real.items()}
    dmu_grid = sh_to_latlon(dmu_pct, nlat=180, nlon=360)

    result = mars_lateral_love_spectrum()
    love = result["love"]
    order_map = compute_order_map(lmax=4, perturbation_order=2)
    shown = select_modes(love, order_map)

    fig = make_figure(dmu_grid, shown, len(love.n), max_dt_km, result["wall_s"])
    pdf_path, png_path = save_fig(fig, "fig6_mars_lateral_spectrum", OUT_DIR)
    plt.close(fig)

    k_idx = [i for i in range(len(love.n)) if love.n[i] == 2 and love.m[i] == 0][0]
    print(f"[fit] N modes={len(love.n)}, wall={result['wall_s']:.1f}s, "
          f"max|dt|={max_dt_km:.2f} km, k2={love.k[k_idx]!r}")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
