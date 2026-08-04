#!/usr/bin/env python
"""F3 — mars_topography_pipeline: MOLA topography through the committed pipeline.

``pylov3d.sh_data.load_shape`` -> ``truncate`` to lmax=90 -> subtract C00
(mean radius) and C20 (dominant rotational flattening; an areoid proxy —
see ``pylov3d/tests/test_mapping.py``, ``TestMarsTopoHellasIntegration``) ->
``pylov3d.mapping.sh_to_latlon`` at publication resolution. This exercises
the loader + spherical-harmonic synthesis end to end against the real
planet: sign, phase, and normalization all have to be correct simultaneously
for Hellas to land at the global minimum and Olympus Mons at the global
maximum, at their real locations and heights.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig3_mars_topography_pipeline.py
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

from common import DIVERGING_CMAP, INK, OUT_DIR, apply_style, save_fig, style_axes

from pylov3d.mapping import sh_to_latlon
from pylov3d.sh_data import load_shape, truncate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOPO_PATH = REPO_ROOT / "data" / "mars" / "MarsTopo719.shape.gz"

# Published/measured landmarks (see module docstring + task spec).
HELLAS = {"lat": -33.0, "lon": 61.0, "label": "Hellas basin\n(global min)"}
OLYMPUS = {"lat": 17.0, "lon": -133.0, "label": "Olympus Mons\n(global max, ~21.8 km)"}


def _clm_slm_to_coeffs(clm: np.ndarray, slm: np.ndarray, lmax: int) -> dict:
    """Real-SH ``(clm, slm)`` arrays -> the ``{(n, m): amplitude}`` dict
    :func:`sh_to_latlon` expects (m >= 0 from clm/cosine, m < 0 from
    slm/sine) — the standard real-SH synthesis convention used throughout
    this pipeline (see ``pylov3d/tests/test_mapping.py``)."""
    coeffs = {}
    for n in range(lmax + 1):
        for m in range(n + 1):
            coeffs[(n, m)] = float(clm[n, m])
            if m >= 1:
                coeffs[(n, -m)] = float(slm[n, m])
    return coeffs


def compute_topo_grid(lmax: int = 90, nlat: int = 180, nlon: int = 360):
    shape = load_shape(TOPO_PATH)
    shape_lmax = truncate(shape, lmax)
    clm = shape_lmax["clm"].copy()
    clm[0, 0] = 0.0  # subtract C00 (mean radius)
    clm[2, 0] = 0.0  # subtract C20 (dominant rotational flattening / areoid proxy)
    coeffs = _clm_slm_to_coeffs(clm, shape_lmax["slm"], shape_lmax["lmax"])
    return sh_to_latlon(coeffs, nlat=nlat, nlon=nlon)


def make_figure(grid):
    z_km = grid.z / 1e3

    imin = np.unravel_index(np.argmin(z_km), z_km.shape)
    imax = np.unravel_index(np.argmax(z_km), z_km.shape)
    lat_min, lon_min, z_min = grid.lat[imin[0]], grid.lon[imin[1]], z_km[imin]
    lat_max, lon_max, z_max = grid.lat[imax[0]], grid.lon[imax[1]], z_km[imax]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    vmax = float(np.max(np.abs(z_km)))
    mesh = ax.pcolormesh(
        grid.lon, grid.lat, z_km, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, shading="nearest",
        rasterized=True,
    )
    ax.set_xlim(grid.lon.min(), grid.lon.max())
    ax.set_ylim(grid.lat.min(), grid.lat.max())
    ax.set_xlabel("Longitude [deg E]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    style_axes(ax, grid=False)
    ax.set_title(
        "MOLA topography (relative to areoid), $\\ell_{\\max}=90$, C00+C20 removed",
        fontsize=8.5,
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.032)
    cbar.set_label("Elevation relative to areoid [km]", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    for site, mlat, mlon, mz in (
        (HELLAS, lat_min, lon_min, z_min),
        (OLYMPUS, lat_max, lon_max, z_max),
    ):
        ax.plot(mlon, mlat, marker="o", markersize=4.5, markerfacecolor="none",
                 markeredgecolor=INK, markeredgewidth=1.1, zorder=5)
        dy = 14 if site is HELLAS else -20
        ax.annotate(
            f"{site['label']}\n({mz:+.1f} km, lat {mlat:.0f}, lon {mlon:.0f})",
            xy=(mlon, mlat), xytext=(0, dy), textcoords="offset points",
            ha="center", va="bottom" if dy > 0 else "top", fontsize=7, color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", lw=0.5, alpha=0.85),
        )

    return fig, {
        "lat_min": lat_min, "lon_min": lon_min, "z_min": z_min,
        "lat_max": lat_max, "lon_max": lon_max, "z_max": z_max,
    }


def main():
    grid = compute_topo_grid()
    fig, stats = make_figure(grid)
    pdf_path, png_path = save_fig(fig, "fig3_mars_topography_pipeline", OUT_DIR)
    plt.close(fig)
    print(f"[stats] global min {stats['z_min']:.2f} km at lat={stats['lat_min']:.1f}, lon={stats['lon_min']:.1f}")
    print(f"[stats] global max {stats['z_max']:.2f} km at lat={stats['lat_max']:.1f}, lon={stats['lon_max']:.1f}")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
