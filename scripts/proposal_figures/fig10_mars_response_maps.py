#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F10 -- mars_response_maps: TASK-041 spatial tidal-response maps.

Two stacked panels, same style system as figs 1-9. Top: the lateral-only
gravity anomaly amplitude (uGal). Bottom: the lateral-only radial
displacement amplitude (mm). Both are the amplitude of the periodic
signal, |complex synthesized field|, to a unit (2,0) forcing (scope
caveat: see ``scripts/mars_response_maps.py`` module docstring and
``pylov3d/mars_detectability.py`` module docstring section 2 -- the real
solar tide has m=0/1/2 power at distinct frequencies). Tharsis, the
dichotomy boundary (a representative point on it), Hellas, and the
InSight landing site are annotated on both panels.

Reads the committed artifact
``docs/figures/proposal/mars_response_maps.npz`` (produced by
``scripts/mars_response_maps.py``) rather than re-solving.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig10_mars_response_maps.py
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

from common import INK, INK_GRAY, SEQUENTIAL_CMAP, OUT_DIR, apply_style, save_fig, style_axes  # noqa: E402

ARTIFACT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "proposal" / "mars_response_maps.npz"


def _lon_east_to_grid(lon_east: float) -> float:
    return ((lon_east + 180.0) % 360.0) - 180.0


def _panel(ax, lat, lon, field, cbar_label: str, title: str, sites: list[dict]) -> None:
    mesh = ax.pcolormesh(
        lon, lat, field, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=float(np.max(field)),
        shading="nearest", rasterized=True,
    )
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_ylabel("Latitude [deg]")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    style_axes(ax, grid=False)
    ax.set_title(title, fontsize=8.5)

    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, fraction=0.032)
    cbar.set_label(cbar_label, fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    for site in sites:
        mlon = _lon_east_to_grid(site["lon_east"])
        ax.plot(mlon, site["lat"], marker="o", markersize=4.0, markerfacecolor="none",
                 markeredgecolor=INK, markeredgewidth=1.0, zorder=5)
        ax.annotate(
            site["label"], xy=(mlon, site["lat"]), xytext=(0, site["dy"]),
            textcoords="offset points", ha="center",
            va="bottom" if site["dy"] > 0 else "top", fontsize=6.3, color=INK,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.4, alpha=0.85),
        )


def make_figure(d) -> plt.Figure:
    apply_style()
    fig, (ax_g, ax_h) = plt.subplots(
        2, 1, figsize=(7.2, 6.6), gridspec_kw={"hspace": 0.42},
    )

    site_names = [str(x) for x in d["site_names"]]
    site_lat = d["site_lat"]
    site_lon_e = d["site_lon_east"]
    dy_by_name = {
        "Tharsis": 12, "Dichotomy boundary (representative)": -14,
        "Hellas": -14, "InSight": 12,
    }
    label_by_name = {
        "Tharsis": "Tharsis", "Dichotomy boundary (representative)": "dichotomy\n(repr.)",
        "Hellas": "Hellas", "InSight": "InSight",
    }
    sites = [
        {"label": label_by_name.get(name, name), "lat": float(lat), "lon_east": float(lon_e),
         "dy": dy_by_name.get(name, 12)}
        for name, lat, lon_e in zip(site_names, site_lat, site_lon_e)
    ]

    _panel(
        ax_g, d["lat"], d["lon"], d["gravity_ugal"],
        cbar_label=r"Gravity anomaly amplitude [$\mu$Gal]",
        title="Mars lateral tidal gravity response (unit (2,0) forcing)",
        sites=sites,
    )
    ax_h_field = d["displacement_mm"]
    _panel(
        ax_h, d["lat"], d["lon"], ax_h_field,
        cbar_label="Radial displacement amplitude [mm]",
        title="Mars lateral tidal displacement response (unit (2,0) forcing)",
        sites=sites,
    )
    ax_h.set_xlabel("Longitude [deg E]")

    return fig


def main():
    d = np.load(ARTIFACT, allow_pickle=True)
    fig = make_figure(d)
    pdf_path, png_path = save_fig(fig, "fig10_mars_response_maps", OUT_DIR)
    plt.close(fig)
    print(f"[stats] gravity peak {float(d['gravity_peak_ugal']):.4e} uGal at "
          f"lat={float(d['gravity_peak_lat']):.2f}, lon={float(d['gravity_peak_lon_east']):.2f} E")
    print(f"[stats] displacement peak {float(d['displacement_peak_mm']):.4e} mm at "
          f"lat={float(d['displacement_peak_lat']):.2f}, lon={float(d['displacement_peak_lon_east']):.2f} E")
    print(f"[output] {pdf_path}")
    print(f"[output] {png_path}")


if __name__ == "__main__":
    main()
