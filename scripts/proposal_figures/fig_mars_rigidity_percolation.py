# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Paired figure for the SSS proposal's fig:percolation.

Top: the crust-shell lateral rigidity variation this project derives from
MOLA shape + GMM-3 gravity spherical-harmonic coefficients (Airy
compensation, areoid-referenced; pylov3d.mars_lateral) -- moved here from
the Figure 3 composite so it sits beside the published map it should be
compared against.
Bottom: the published Mars infiltration-time map (Shadab et al. 2025),
read from the proposal repo's figures/surp_image-000.jpg.

Colour convention: the diverging map is drawn with ``RdBu`` rather than
``RdBu_r`` so that, as in the Shadab panel, red marks the low end of the
scale and blue the high end. The two panels therefore read the same way
round even though they show different quantities.

Both panels are plotted on the same longitude range so features line up
column-for-column; the Shadab panel is a raster with its own baked-in
axes, so it is placed as an image rather than re-projected, and its own
axis labels are left visible.

Output: --out (default: the SSS proposal repo's
figures/fig_mars_rigidity_percolation.pdf).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import apply_style  # noqa: E402  (same dir)

SHADAB_IMAGE = Path("/Users/svance/SSS_2025_Mars/figures/surp_image-000.jpg")

# Reversed relative to the old Figure 3 panel, to match the Shadab map's
# red-low / blue-high sense (see module docstring).
DIVERGING = "RdBu"


def main(out: Path) -> None:
    apply_style()

    from pylov3d.mapping import sh_to_latlon  # noqa: E402
    from pylov3d.mars_lateral import dmu_over_mu_real  # noqa: E402

    grid = sh_to_latlon(dmu_over_mu_real(), nlat=91, nlon=181)
    lat, lon, dmu = grid.lat, grid.lon, grid.z

    # The Shadab panel is a raster carrying its own axes, and imshow keeps
    # its aspect, so its rendered width is set by the height it is given.
    # Give it the larger share so the two maps come out comparable in width.
    fig = plt.figure(figsize=(5.2, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.22)

    # --- Top: our derived crustal rigidity variation ------------------
    ax = fig.add_subplot(gs[0])
    vmax = float(np.max(np.abs(dmu))) * 100.0
    im = ax.pcolormesh(
        lon, lat, dmu * 100.0, cmap=DIVERGING, vmin=-vmax, vmax=vmax,
        shading="auto", rasterized=True,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-150, 151, 50))
    ax.set_yticks(np.arange(-50, 51, 50))
    ax.set_xlabel(r"Longitude [$^\circ$]", fontsize=7)
    ax.set_ylabel(r"Latitude [$^\circ$]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(
        "Derived crustal rigidity variation (MOLA + GMM-3, Airy)",
        fontsize=8,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(r"$\delta\mu/\bar\mu$ [%]", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # --- Bottom: the published infiltration-time map ------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(mpimg.imread(SHADAB_IMAGE))
    ax2.set_axis_off()

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/Users/svance/SSS_2025_Mars/figures/fig_mars_rigidity_percolation.pdf"
        ),
    )
    main(p.parse_args().out)
