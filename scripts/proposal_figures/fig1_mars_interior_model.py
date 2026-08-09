#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F1 — mars_interior_model: TASK-011 fitted 1-D Mars profile + fit residuals.

Two-panel vertical profile (radius on the y-axis, 0 at bottom, surface
3389.5 km at top -- planetary-science convention) of the TASK-011 fitted
model (``pylov3d.mars.build_mars_model()``): left panel density rho(r),
right panel shear modulus mu(r), both step profiles, layers shaded and
named. A bottom strip shows the normalized fit residuals
(achieved - target) / sigma for mass, mean MoI, and k2 against a +/-1 sigma
reference band -- all computed live via ``pylov3d.mars.mars_moi_factor``
and ``pylov3d.love.get_love`` (never hardcoded); targets/sigmas are the
same ``pylov3d.mars_mc.MARS_CONSTRAINTS`` used by the F2 posterior run, so
the two figures cite one source of truth.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig1_mars_interior_model.py
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

from common import CATEGORICAL, INK, INK_GRAY, OUT_DIR, apply_style, save_fig, style_axes

from pylov3d.mars import (
    LAYER_RADII_KM,
    MARS,
    MARS_FORCING_TD,
    _mass_and_moi,
    build_mars_model,
    mars_moi_factor,
)
from pylov3d.mars_mc import MARS_CONSTRAINTS
from pylov3d.love import get_love
from pylov3d.types import make_forcing, make_numerics

LAYER_NAMES = ["liquid core", "lower mantle", "upper mantle", "crust"]


def _step_arrays(boundaries_km, values):
    """Piecewise-constant staircase: returns (x, y) with x=value, y=radius."""
    x, y = [], []
    for i, v in enumerate(values):
        x += [v, v]
        y += [boundaries_km[i], boundaries_km[i + 1]]
    return np.array(x), np.array(y)


def compute_live():
    """Build the model and compute every plotted quantity live (no hardcoding)."""
    model = build_mars_model()
    n = model.n_layers
    radii_km = [float(x) for x in np.asarray(model.R0[:n])]
    rho = [float(x) for x in np.asarray(model.rho0[:n])]
    mu = [float(x) for x in np.asarray(model.mu0[:n])]

    mass_achieved, _I = _mass_and_moi(radii_km, rho)
    moi_achieved = mars_moi_factor()

    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
    love, _, _ = get_love(model, forcing, numerics)
    k2_achieved = float(np.real(np.asarray(love.k[0])))

    achieved = {"mass": mass_achieved, "moi_mean": moi_achieved, "k2": k2_achieved}
    return radii_km, rho, mu, achieved


def make_figure():
    radii_km, rho, mu, achieved = compute_live()
    boundaries_km = [0.0] + list(LAYER_RADII_KM)

    apply_style()
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.0], hspace=0.42, wspace=0.32)
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_mu = fig.add_subplot(gs[0, 1])
    ax_res = fig.add_subplot(gs[1, :])

    surface_km = LAYER_RADII_KM[-1]

    # --- rho(r) and mu(r) panels ------------------------------------------------
    for ax, values, xlabel, unit_scale in (
        (ax_rho, rho, r"Density $\rho$ [kg m$^{-3}$]", 1.0),
        (ax_mu, mu, r"Shear modulus $\mu$ [GPa]", 1e-9),
    ):
        style_axes(ax, grid=True)
        vals_scaled = [v * unit_scale for v in values]
        for i in range(4):
            ax.axhspan(boundaries_km[i], boundaries_km[i + 1], color=CATEGORICAL[i], alpha=0.13, zorder=0)
        x, y = _step_arrays(boundaries_km, vals_scaled)
        ax.plot(x, y, color=INK, linewidth=1.3, zorder=3)
        ax.set_ylim(0, surface_km * 1.015)
        ax.set_xlim(left=0)
        ax.set_ylabel("Radius [km]")
        ax.set_xlabel(xlabel)
        # annotate the fitted value at each layer's mid-radius
        for i in range(4):
            r_mid = 0.5 * (boundaries_km[i] + boundaries_km[i + 1])
            v = vals_scaled[i]
            label = f"{v:,.0f}" if unit_scale == 1.0 else f"{v:,.1f}"
            ax.annotate(
                label, xy=(v, r_mid), xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=INK_GRAY, va="center", ha="left",
            )

    # layer names on the density panel only (shared band geometry with mu panel)
    for i, name in enumerate(LAYER_NAMES):
        r_mid = 0.5 * (boundaries_km[i] + boundaries_km[i + 1])
        thin = (boundaries_km[i + 1] - boundaries_km[i]) < 350
        ax_rho.text(
            0.03, r_mid, name, transform=ax_rho.get_yaxis_transform(),
            fontsize=7, color=INK, ha="left", va="center", rotation=90 if thin else 0,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )

    ax_rho.set_title("Density profile", fontsize=8.5)
    ax_mu.set_title("Shear modulus profile", fontsize=8.5)

    # --- constraint residuals panel ---------------------------------------------
    style_axes(ax_res, grid=False)
    name_order = ["mass", "moi_mean", "k2"]
    display_names = {"mass": "Mass $M$", "moi_mean": "Mean MoI\nfactor $I/MR^2$", "k2": "Tidal $k_2$"}
    constraints_by_name = {c.name: c for c in MARS_CONSTRAINTS}

    ax_res.axhspan(-1, 1, color="0.85", alpha=0.6, zorder=0, label=r"$\pm1\sigma$")
    ax_res.axhline(0, color=INK_GRAY, linewidth=0.7, zorder=1)

    xs = np.arange(len(name_order))
    resids = []
    for i, name in enumerate(name_order):
        c = constraints_by_name[name]
        resid = (achieved[name] - c.value) / c.sigma
        resids.append(resid)
        ax_res.plot(i, resid, marker="o", markersize=5.5, color=CATEGORICAL[0], zorder=4)
        ax_res.annotate(
            f"{resid:+.3f}$\\sigma$", xy=(i, resid), xytext=(0, 7 if resid >= 0 else -11),
            textcoords="offset points", ha="center", fontsize=7, color=INK,
        )

    ax_res.set_xlim(-0.6, len(name_order) - 0.4)
    ax_res.set_xticks(xs)
    ax_res.set_xticklabels([display_names[n] for n in name_order], fontsize=7)
    ax_res.set_ylim(-2.2, 2.2)
    ax_res.set_ylabel("Residual\n(achieved - target)/$\\sigma$", fontsize=7)
    ax_res.set_title("Fit constraint residuals (live-computed)", fontsize=8.5)
    ax_res.legend(loc="upper right", fontsize=7, handlelength=1.0)

    fig.suptitle(
        "Mars 1-D interior model (TASK-011 fit): $\\rho(r)$, $\\mu(r)$, and constraint residuals",
        fontsize=9.5, y=0.99,
    )

    return fig


def main():
    fig = make_figure()
    pdf_path, png_path = save_fig(fig, "fig1_mars_interior_model", OUT_DIR)
    plt.close(fig)
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
