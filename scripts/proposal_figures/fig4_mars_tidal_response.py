#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F4 — mars_tidal_response: degree-2 tidal pattern of the fitted Mars model.

Improves on ``scripts/mars_fit_map.py``: two panels on a SHARED diverging
colorbar scale (both panels are the same normalized shape function
P-bar_2^0(sin lat) scaled by a different Love number, so one common
symmetric-about-zero color scale serves both) -- left: radial displacement
h2 * P-bar_2^0(sin lat), right: potential perturbation k2 * P-bar_2^0(sin
lat). Both computed live via ``pylov3d.mars.build_mars_model`` +
``pylov3d.love.get_love``. An under-title strip states the fitted numbers.

m=0 (zonal) pattern only: for a spherically symmetric 1-D reference model,
h2/k2/l2 depend on degree n but not order m, so the *amplitude* scaling
shown is exactly what would apply to the real, longitude-dependent m=2
solar-semidiurnal tide -- only the longitude pattern would differ.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig4_mars_tidal_response.py
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

from common import DIVERGING_CMAP, INK, INK_GRAY, OUT_DIR, apply_style, save_fig, style_axes

from pylov3d.mapping import sh_to_latlon
from pylov3d.mars import MARS_FORCING_TD, build_mars_model
from pylov3d.love import get_love
from pylov3d.types import make_forcing, make_numerics

P2_POLE = math.sqrt(5.0)  # P-bar_2^0(+/-1), 4pi-normalized


def compute_love():
    model = build_mars_model()
    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)
    love, _, _ = get_love(model, forcing, numerics)
    h2 = float(np.real(np.asarray(love.h[0])))
    k2 = float(np.real(np.asarray(love.k[0])))
    l2 = float(np.real(np.asarray(love.l[0])))
    return h2, k2, l2


def make_figure(h2: float, k2: float, l2: float):
    grid_h2 = sh_to_latlon({(2, 0): h2}, lmax=30)
    grid_k2 = sh_to_latlon({(2, 0): k2}, lmax=30)

    vmax = max(abs(h2), abs(k2)) * P2_POLE

    apply_style()
    fig, (ax_h2, ax_k2) = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)

    mesh_h2 = ax_h2.pcolormesh(
        grid_h2.lon, grid_h2.lat, grid_h2.z, cmap=DIVERGING_CMAP,
        vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True,
    )
    mesh_k2 = ax_k2.pcolormesh(
        grid_k2.lon, grid_k2.lat, grid_k2.z, cmap=DIVERGING_CMAP,
        vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True,
    )

    for ax, title, val, sym in (
        (ax_h2, "Radial displacement", h2, "h_2"),
        (ax_k2, "Potential perturbation", k2, "k_2"),
    ):
        style_axes(ax, grid=False)
        ax.set_xlim(grid_h2.lon.min(), grid_h2.lon.max())
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xlabel("Longitude [deg]")
        ax.set_title(f"{title}: ${sym} \\cdot \\bar P_2^0(\\sin\\,{{\\rm lat}})$", fontsize=8)
        pole_val = val * P2_POLE
        ax.annotate(
            f"pole value = {pole_val:+.3f}",
            xy=(0, 88), xytext=(0, 0), textcoords="offset points",
            ha="center", va="top", fontsize=7, color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.85),
        )
    ax_h2.set_ylabel("Latitude [deg]")

    cbar = fig.colorbar(mesh_k2, ax=[ax_h2, ax_k2], pad=0.02, fraction=0.035)
    cbar.set_label("Love number $\\times\\ \\bar P_2^0(\\sin\\,{\\rm lat})$ [dimensionless]", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle("Mars degree-2 ($m$=0) tidal response, TASK-011 fitted model", fontsize=9.5, y=1.1)
    fig.text(
        0.5, 1.005,
        f"$k_2$ = {k2:.3f} (observed 0.169 $\\pm$ 0.006)     "
        f"$h_2$ = {h2:.4f}     $l_2$ = {l2:.4f}",
        fontsize=7.5, color=INK, ha="center", va="bottom", transform=fig.transFigure,
    )
    fig.text(
        0.5, -0.06,
        "$m$=0 (zonal) pattern shown for a single-panel illustration; Love numbers for a\n"
        "spherically symmetric reference model depend on degree $n$ only, not order $m$, so\n"
        "these amplitudes are exactly those of the real (longitude-varying) $m$=2 solar-semidiurnal tide.",
        fontsize=7, color=INK_GRAY, ha="center", va="top", transform=fig.transFigure,
    )

    return fig


def main():
    h2, k2, l2 = compute_love()
    fig = make_figure(h2, k2, l2)
    pdf_path, png_path = save_fig(fig, "fig4_mars_tidal_response", OUT_DIR)
    plt.close(fig)
    print(f"[fit] h2={h2:.6f}, k2={k2:.6f}, l2={l2:.6f}")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
