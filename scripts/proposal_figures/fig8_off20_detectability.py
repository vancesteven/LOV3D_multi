#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F8 -- off20_detectability: TASK-026 off-(2,0) / k2m detectability.

Two panels, same style system as figs 1-7. Left: tier 1, the diagonal
k2m order-splitting (m=0,1,2) predicted by the TASK-016 lateral model
(``pylov3d.mars_detectability_k2m.mars_diagonal_k2m_table``), against the
band of GRAIL's own demonstrated individual-order lunar precision (the
JPL/GL0660B analysis, Konopliv et al. 2013) -- with an explicit
annotation that current Mars Doppler tracking has never even produced a
comparable number (Wörner et al. 2023, "MaQuIs"). Right: tier 2, the
higher-degree off-(2,0) coupled spectrum
(``pylov3d.mars_detectability.mars_off20_detectability_table``), against
a Love-number-space "detection threshold" line built from the
current-orbiter (CO2-seasonal-analogue) benchmark only -- an earlier
version of this figure built the threshold from a GRAIL-class degree-3
benchmark too, on the claim that the two "land within 1% of each other";
that GRAIL comparison was a diagonal/off-diagonal category error
(comparing this tier's off-diagonal |k_(n,m)| against GRAIL's diagonal
sigma(k3)) and has been removed, not just from the threshold line but
from :func:`pylov3d.mars_detectability.mars_off20_detectability_table`
itself. Both panels show the same qualitative result: every point sits
below its respective threshold line -- not currently detectable, at
either tier, by any benchmark used here.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig8_off20_detectability.py
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

from pylov3d.mars import MARS
from pylov3d.mars_detectability import (
    GM_SUN,
    MARS_PERIHELION_M,
    MARS_SIGMA_C30_SEASONAL,
    mars_off20_detectability_table,
    peak_legendre_factor,
    solar_tide_amplitude_parameter,
)
from pylov3d.mars_detectability_k2m import GRAIL_K2M_SIGMA, mars_diagonal_k2m_table

TIER1_COLOR = CATEGORICAL[0]
TIER2_COLOR = CATEGORICAL[1]
N_TIER2_SHOWN = 10


def orbiter_k_equivalent_threshold() -> float:
    """Current-orbiter (CO2-seasonal sigma_C30) precision, converted to an
    equivalent Love-number-space detection threshold at the same
    optimistic (perihelion, sectoral) bound tier 2's table uses. An
    earlier version of this function's docstring claimed this was "within
    1% of GRAIL_SIGMA_K3" and used that near-agreement to justify a
    single combined threshold line; that comparison was a diagonal/
    off-diagonal category error (module docstring,
    ``pylov3d.mars_detectability`` sec. 3) and has been dropped -- this
    threshold now represents the current-orbiter benchmark only."""
    xi = solar_tide_amplitude_parameter(GM_SUN, MARS["GM"], MARS["R"], MARS_PERIHELION_M, n_forcing=2)
    p = peak_legendre_factor(2, 2)
    return MARS_SIGMA_C30_SEASONAL / (xi * p)


def make_figure(k2m_rows, tier2_rows) -> plt.Figure:
    apply_style()
    fig, (ax_k2m, ax_spec) = plt.subplots(
        1, 2, figsize=(7.6, 3.7), gridspec_kw={"wspace": 0.5, "width_ratios": [0.85, 1.15]},
    )

    # ---- left: diagonal k2m order-splitting ---------------------------
    m_vals = [row["m"] for row in k2m_rows]
    delta_vals = np.array([abs(row["delta"]) for row in k2m_rows])
    sigma_vals = np.array([GRAIL_K2M_SIGMA[m] for m in m_vals])

    ax_k2m.fill_between(
        [-0.5, 2.5], sigma_vals.min(), sigma_vals.max(),
        color=INK_GRAY, alpha=0.15, linewidth=0, zorder=0,
        label="GRAIL demonstrated precision\n(lunar k20/k21/k22, Konopliv et al. 2013)",
    )
    x = np.arange(3)
    ax_k2m.vlines(x, delta_vals.min() * 0.4, delta_vals, color=TIER1_COLOR, linewidth=1.4, zorder=2)
    ax_k2m.scatter(x, delta_vals, c=TIER1_COLOR, s=32, zorder=3, edgecolor="white", linewidth=0.5)
    ax_k2m.set_yscale("log")
    ax_k2m.set_xlim(-0.5, 2.5)
    ax_k2m.set_xticks(x)
    ax_k2m.set_xticklabels(["$k_{20}$", "$k_{21}$", "$k_{22}$"])
    ax_k2m.set_ylabel(r"$|\Delta k_{2m}|$ (predicted, Mars)")
    ax_k2m.set_title("Diagonal $k_{2m}$ order-splitting", fontsize=8.5)
    style_axes(ax_k2m, grid=True)
    ax_k2m.legend(loc="upper right", fontsize=5.8, framealpha=0.9)
    ax_k2m.annotate(
        "current Mars Doppler tracking:\nattempted, unsuccessful\n(Wörner et al. 2023)",
        xy=(1.0, delta_vals.min() * 0.55), xytext=(0, 0), textcoords="offset points",
        ha="center", va="top", fontsize=6.0, color=INK,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5, alpha=0.9),
    )

    # ---- right: tier-2 off-(2,0) coupled spectrum ----------------------
    shown = tier2_rows[:N_TIER2_SHOWN]
    xs = np.arange(len(shown))
    k_vals = np.array([row["k_abs"] for row in shown])
    threshold = orbiter_k_equivalent_threshold()

    ax_spec.vlines(xs, k_vals.min() * 0.5, k_vals, color=TIER2_COLOR, linewidth=1.2, zorder=2)
    ax_spec.scatter(xs, k_vals, c=TIER2_COLOR, s=20, zorder=3, edgecolor="white", linewidth=0.4)
    ax_spec.axhline(threshold, color=INK_GRAY, linewidth=0.9, linestyle="--", zorder=1)
    ax_spec.set_yscale("log")
    ax_spec.set_ylim(k_vals.min() * 0.3, threshold * 6.0)
    ax_spec.annotate(
        "detection threshold (current orbiter,\n"
        "Love-number-equivalent, optimistic bound)",
        xy=(0, threshold), xytext=(2, -4), textcoords="offset points",
        ha="left", va="top", fontsize=6.0, color=INK_GRAY,
    )
    ax_spec.set_xticks(xs)
    ax_spec.set_xticklabels([f"({row['n']},{row['m']})" for row in shown], rotation=90, fontsize=6.0)
    ax_spec.set_ylabel(r"$|k_{nm}|$ (off-(2,0) spectrum)")
    ax_spec.set_title(f"Off-(2,0) coupled spectrum (top {len(shown)} of {len(tier2_rows)})", fontsize=8.5)
    style_axes(ax_spec, grid=True)

    fig.suptitle(
        "Mars off-(2,0) tidal detectability (TASK-026): required vs. achieved/demonstrated precision",
        fontsize=9, y=1.04,
    )
    return fig


def main():
    k2m_rows = mars_diagonal_k2m_table()
    tier2_rows = mars_off20_detectability_table()

    fig = make_figure(k2m_rows, tier2_rows)
    pdf_path, png_path = save_fig(fig, "fig8_off20_detectability", OUT_DIR)
    plt.close(fig)

    print("[fit] tier 1 (diagonal k2m splitting vs GRAIL lunar precision, JPL/GSFC):")
    for row in k2m_rows:
        print(f"  m={row['m']}: |Delta k2m|={abs(row['delta']):.3e}, "
              f"GRAIL(JPL) sigma={row['grail_sigma']:.3e}, ratio={row['ratio_grail']:.1f}x, "
              f"GRAIL(GSFC) sigma={row['grail_gsfc_sigma']:.3e}, ratio={row['ratio_grail_gsfc']:.1f}x")
    print("[fit] tier 2 (off-(2,0) spectrum, top 5 vs current-orbiter precision):")
    for row in tier2_rows[:5]:
        print(f"  ({row['n']},{row['m']:+d}): |k|={row['k_abs']:.3e}, "
              f"ratio_orbiter_optimistic={row['ratio_orbiter_optimistic']:.1f}x, "
              f"ratio_orbiter_conservative={row['ratio_orbiter_conservative']:.1f}x")
    print(f"[fit] tier-2 Love-number-equivalent detection threshold: {orbiter_k_equivalent_threshold():.3e}")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
