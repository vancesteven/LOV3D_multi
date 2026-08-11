# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F9 -- the non-Airy crustal-model substitution (TASK-028).

Left: the two first-order zonal channels into the (2,0) forcing mode and
their sum. Parity admits exactly two rheology harmonics at n_lv <= 4 --
(2,0) and (4,0) -- and they enter with opposite signs, so a field
retaining a measured degree-2 term cancels ~91% of the (4,0) contribution
the Airy field reports alone. This is TASK-028's headline: the mechanism
is robust, its net amplitude is convention-dependent.

Right: the (2,0) forcing-mode shift for five InSight-calibrated crustal
models against the Airy baseline, shown both as shipped (degree-2 term
retained, as the models come) and with that term suppressed on both sides
-- the like-for-like pattern comparison, where the five bracket Airy
symmetrically instead of sitting wholly below it.

Reads the committed artifact docs/figures/proposal/mars_crust_models.npz
(produced by the TASK-028 comparison runs) rather than re-solving, in the
same spirit as the TASK-021b and TASK-027 artifacts -- the underlying
comparison is ~15 minutes of coupled solves.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig9_crustal_model_substitution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CATEGORICAL, INK_GRAY, OUT_DIR, apply_style, save_fig, style_axes  # noqa: E402

ARTIFACT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "proposal" / "mars_crust_models.npz"


def make_figure(d) -> plt.Figure:
    apply_style()
    fig, (ax_c, ax_m) = plt.subplots(
        1, 2, figsize=(7.6, 3.4), gridspec_kw={"wspace": 0.42, "width_ratios": [1.0, 1.25]},
    )

    # ---- left: the two first-order channels and their near-cancellation
    labels = [str(x) for x in d["first_order_labels"]]
    vals = d["first_order_shift"] * 1e5
    colors = [CATEGORICAL[3], CATEGORICAL[0], INK_GRAY]
    ax_c.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax_c.axhline(0.0, color=INK_GRAY, lw=0.8)
    for i, v in enumerate(vals):
        ax_c.annotate(
            f"{v:+.2f}", xy=(i, v), xytext=(0, 4 if v > 0 else -11),
            textcoords="offset points", ha="center", fontsize=7,
        )
    ax_c.set_xticks(range(len(labels)))
    ax_c.set_xticklabels(labels, fontsize=7.5)
    ax_c.set_ylabel(r"first-order $\Delta k_2$  [$\times10^{-5}$]")
    ax_c.set_title("Two first-order zonal channels, opposite signs", fontsize=8.5)
    ax_c.set_ylim(min(vals) * 1.28, max(vals) * 1.22)  # headroom for value labels
    ax_c.annotate(
        "91% cancellation", xy=(2, vals[2]), xytext=(-4, -26),
        textcoords="offset points", ha="center", fontsize=7, color=INK_GRAY,
    )
    style_axes(ax_c, grid=True)

    # ---- right: five InSight models vs Airy, shipped and like-for-like
    k2u = complex(d["k2_uniform"])
    airy = abs(complex(d["airy_k20"]) - k2u) * 1e5
    models = [str(m) for m in d["models"]]
    ship = np.abs(d["shipped_k20"] - k2u) * 1e5
    noc = np.abs(d["noc20_k20"] - k2u) * 1e5
    order = np.argsort(noc)
    y = np.arange(len(models))
    ax_m.axvline(airy, color=INK_GRAY, lw=1.1, ls="--", zorder=1)
    ax_m.annotate(
        "Airy", xy=(airy, len(models) - 0.55), xytext=(4, 0),
        textcoords="offset points", fontsize=6.5, color=INK_GRAY, va="center",
    )
    ax_m.scatter(ship[order], y - 0.16, s=26, color=CATEGORICAL[3], zorder=3,
                 label="as shipped (degree-2 term retained)")
    ax_m.scatter(noc[order], y + 0.16, s=26, color=CATEGORICAL[0], zorder=3,
                 label="degree-2 suppressed both sides")
    for i in range(len(models)):
        ax_m.plot([ship[order][i], noc[order][i]], [y[i] - 0.16, y[i] + 0.16],
                  color="0.75", lw=0.7, zorder=2)
    ax_m.set_yticks(y)
    ax_m.set_yticklabels([models[i] for i in order], fontsize=6.5)
    ax_m.set_xlabel(r"$(2,0)$ forcing-mode $\Delta k_2$  [$\times10^{-5}$]")
    ax_m.set_title("InSight-calibrated crustal models vs Airy", fontsize=8.5)
    ax_m.set_ylim(-0.75, len(models) - 0.25)
    ax_m.legend(loc="lower center", fontsize=6.2, frameon=False,
                bbox_to_anchor=(0.5, -0.02), ncol=1)
    style_axes(ax_m, grid=True)

    fig.suptitle(
        "Non-Airy crustal substitution (TASK-028): the Airy pattern holds, "
        "the degree-2 convention does not",
        fontsize=9, y=1.04,
    )
    return fig


def main():
    d = np.load(ARTIFACT, allow_pickle=False)
    fig = make_figure(d)
    pdf_path, png_path = save_fig(fig, "fig9_crustal_model_substitution", OUT_DIR)
    plt.close(fig)

    k2u = complex(d["k2_uniform"])
    airy = abs(complex(d["airy_k20"]) - k2u)
    ship = np.abs(d["shipped_k20"] - k2u)
    noc = np.abs(d["noc20_k20"] - k2u)
    print(f"[fit] Airy {airy:.4e}")
    print(f"[fit] shipped spread x{ship.max()/ship.min():.3f}")
    print(f"[fit] like-for-like spread x{noc.max()/noc.min():.3f} "
          f"({noc.min()/airy - 1:+.1%} to {noc.max()/airy - 1:+.1%} vs Airy)")
    print(f"[output] {pdf_path}\n[output] {png_path}")


if __name__ == "__main__":
    main()
