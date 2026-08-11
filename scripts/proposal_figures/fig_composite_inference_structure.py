# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Composite two-panel figure for the SSS proposal's Figure 3 (fig:ai).

Top: 1-D posterior marginals of the four Mars interior parameters from the
converged pocomc run (chain archived in docs/figures/proposal/
mars_posterior_chain.npz), with the deterministic point fit overlaid.
Bottom: the TASK-021 hydration-front tidal signature — predicted Delta-k2
vs the global hydrated fraction f_h (central serpentinite properties,
literature bracket shaded) against the current observational 1-sigma
(pylov3d.mars_hydration.hydration_forward_sweep; ~150 s of coupled
solves at the module's documented reduced-grid default).

The crust-shell lateral rigidity map that used to sit between these two
panels now lives with the Shadab infiltration map in fig:percolation
(fig_mars_rigidity_percolation.py) -- the inference and the tidal
signature belong together, per the PI.

Sized for a sidecaption figure at ~0.6\\linewidth (portrait-ish stack).
Output: --out (default: the SSS proposal repo's figures/fig3_composite.pdf).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CATEGORICAL as COLORS, apply_style  # noqa: E402  (same dir)

REPO = Path(__file__).resolve().parents[2]
CHAIN = REPO / "docs" / "figures" / "proposal" / "mars_posterior_chain.npz"

LABELS = {
    "rho_core": r"$\rho_{\rm core}$ [kg m$^{-3}$]",
    "rho_lm": r"$\rho_{\rm lm}$ [kg m$^{-3}$]",
    "mu_scale": r"$\mu_{\rm scale}$",
    "R_core": r"$R_{\rm core}$ [km]",
}


def main(out: Path) -> None:
    apply_style()
    d = np.load(CHAIN, allow_pickle=False)
    samples, point = d["samples"], d["point_fit"]
    names = [str(n) for n in d["free_params"]]

    fig = plt.figure(figsize=(4.6, 4.4))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.25], hspace=0.55,
    )

    # --- Top: posterior marginals (1x4) -------------------------------
    gs_top = gs[0].subgridspec(1, 4, wspace=0.35)
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs_top[0, i])
        ax.hist(samples[:, i], bins=28, color=COLORS[0], alpha=0.9)
        ax.axvline(point[i], color=COLORS[1], lw=1.4)
        ax.set_yticks([])
        ax.set_xlabel(LABELS.get(name, name), fontsize=7)
        ax.tick_params(labelsize=6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.text(
        0.5, 0.985,
        f"Bayesian interior inference (pocomc): {samples.shape[0]} samples, "
        f"ESS={float(d['ess']):.0f}",
        ha="center", va="top", fontsize=8,
    )

    # --- Bottom: hydration-front Delta-k2 vs f_h ----------------------
    from pylov3d.mars_hydration import SIGMA_K2, hydration_forward_sweep  # noqa: E402

    rows = hydration_forward_sweep()
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

    ax = fig.add_subplot(gs[1])
    ax.fill_between(
        f_h, d_low, d_high, color=COLORS[0], alpha=0.18, linewidth=0,
        label="serpentinite property bracket",
    )
    ax.plot(f_h, d_central, color=COLORS[0], lw=1.4, marker="o", markersize=2.8,
            label="central properties")
    ax.axhline(SIGMA_K2, color="0.4", lw=0.9, ls="--")
    ax.annotate(
        r"current $1\sigma$ on observed $k_2$",
        xy=(f_h[-1], SIGMA_K2), xytext=(-2, -3), textcoords="offset points",
        ha="right", va="top", fontsize=6, color="0.4",
    )
    ax.set_xlabel(r"global hydrated crust fraction $f_h$", fontsize=7)
    ax.set_ylabel(r"$\Delta k_2$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(
        "Predicted tidal signature of a crustal hydration front",
        fontsize=8,
    )
    ax.legend(loc="center left", fontsize=6, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/svance/SSS_2025_Mars/figures/fig3_composite.pdf"),
    )
    main(p.parse_args().out)
