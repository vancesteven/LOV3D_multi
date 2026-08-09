# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Shared style constants and helpers for the NASA SSW proposal figure set.

Every ``figN_*.py`` script in this directory imports from here so the five
figures read as one consistent publication system. Nothing in here touches
``pylov3d/`` itself — this module only configures matplotlib and provides
small I/O helpers.

Usage
-----
    from common import apply_style, save_fig, CATEGORICAL, FIGSIZE_SINGLE, FIGSIZE_FULL

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    ...
    save_fig(fig, "fig1_mars_interior_model", OUT_DIR)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "docs" / "figures" / "proposal"

# ---------------------------------------------------------------------------
# Colors — Okabe-Ito subset, colorblind-safe, validated with
# dataviz/scripts/validate_palette.js (light mode): ALL CHECKS PASS.
# Fixed order — never cycled, never reordered per-figure.
# ---------------------------------------------------------------------------

CATEGORICAL: list[str] = [
    "#0072B2",  # 1 blue
    "#E69F00",  # 2 orange
    "#009E73",  # 3 green
    "#CC79A7",  # 4 reddish-purple
    "#D55E00",  # 5 vermillion
]

# Sequential (magnitude) ramp: single-hue blues. Use matplotlib's "Blues".
SEQUENTIAL_CMAP = "Blues"

# Diverging (signed-about-zero) ramp. Never rainbow/jet.
DIVERGING_CMAP = "RdBu_r"

# Ink colors — text/axes/grid never use a series color.
INK = "#1a1a1a"        # near-black, primary text/labels
INK_GRAY = "#4d4d4d"   # secondary text (annotations, captions-in-figure)
GRID_GRAY = "#000000"  # grid line color; alpha set separately (recessive)
SPINE_GRAY = "#333333"

# ---------------------------------------------------------------------------
# Sizes (inches). Publication convention: single-column vs full-width.
# ---------------------------------------------------------------------------

FIGSIZE_SINGLE = (3.5, 3.5)   # single-column square-ish default
FIGSIZE_SINGLE_TALL = (3.5, 4.6)
FIGSIZE_FULL = (7.2, 4.2)
FIGSIZE_FULL_WIDE = (7.2, 3.4)
FIGSIZE_FULL_SQUARE = (7.2, 7.2)


def apply_style() -> None:
    """Configure matplotlib rcParams for the whole proposal figure set.

    White background, DejaVu Sans, 8pt base font (7pt floor used locally by
    individual scripts where space is tight), 0.8pt axes linewidth, no
    top/right spines, recessive gray gridlines (alpha 0.15) where used.
    """
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.edgecolor": SPINE_GRAY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID_GRAY,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": SPINE_GRAY,
            "ytick.color": SPINE_GRAY,
            "axes.titlecolor": INK,
            "legend.frameon": False,
            "pdf.fonttype": 42,   # embed as real (editable/searchable) glyphs
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_fig(fig, stem: str, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """Save *fig* as both a vector PDF and a 300-dpi PNG, tight bbox.

    Returns
    -------
    (pdf_path, png_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    return pdf_path, png_path


def style_axes(ax, grid: bool = False) -> None:
    """Apply the no-top/right-spine + optional recessive-grid convention to *ax*."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, alpha=0.15, linewidth=0.6, color=GRID_GRAY)
        ax.set_axisbelow(True)
