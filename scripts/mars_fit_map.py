#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Surface-map demonstration of the TASK-011 Mars point fit (TASK-012).

Renders the degree-2, order-0 (n=2, m=0) tidal surface pattern for the
fitted Mars interior model: a radial-displacement panel scaled by the
computed h2, and a companion gravitational-potential panel scaled by the
computed k2, plus a text panel of the fitted free parameters and each
constraint's target vs. achieved value/residual.

Physical scaling (documented on the colorbars, not just here): this map
uses a *normalized* unit-amplitude degree-2 tidal potential (U2 == 1 in the
body's own units), evaluated with the 4pi-fully-normalized real associated
Legendre function P-bar_2^0 (see ``pylov3d.mapping``, NOT the "geodesy"/
Schmidt/unnormalized P_2 -- P-bar_2^0(x) = sqrt(5)*P_2(x), so its value at
the poles is sqrt(5) ~= 2.236, not 1), so:

    radial displacement panel     =  h2 * P-bar_2^0(sin(lat))   [peaks at +/- h2*sqrt(5)]
    potential perturbation panel  =  k2 * P-bar_2^0(sin(lat))   [peaks at +/- k2*sqrt(5)]

(both are longitude-independent since m=0; see "m=0 vs m=2" below). Recovering
an absolute displacement in metres would require multiplying by Mars's
actual physical degree-2 tidal potential amplitude U2 (from its orbital
eccentricity/solar distance), which is not computed here -- see e.g.
Konopliv, Park & Folkner (2016) for the forcing amplitude if that is wanted
later. The pattern *shape* (h2/k2-scaled P-bar_2^0) and the fitted Love
numbers themselves are exact.

m=0 vs m=2: this map plots the m=0 (zonal, longitude-independent) degree-2
pattern for a simple, single-panel illustration. Mars's actual dominant
tidal forcing (the solar semidiurnal tide) is degree 2, order **m=2**, whose
spatial pattern varies with longitude (unlike what's plotted here) -- but
h2/k2/l2 for a spherically symmetric 1D radial model like this one are the
same regardless of which m is forced (the Love numbers depend only on
degree n, not order m), so the *amplitude* scaling shown is exactly the one
that would apply to the real m=2 tide; only its longitude dependence would
differ from this zonal illustration.

Usage
-----
    venvLOV3Dconv/bin/python scripts/mars_fit_map.py
    venvLOV3Dconv/bin/python scripts/mars_fit_map.py --out scripts/output/mars_fit_map.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np

from pylov3d.forward import build_model, compute_observables
from pylov3d.mapping import sh_to_latlon
from pylov3d.mars_mc import (
    MARS_CONSTRAINTS,
    MARS_FREE_PARAMS,
    MARS_PARAMETERIZATION,
    mars_forcing,
    mars_numerics,
    mars_point_fit_theta,
)

_DEFAULT_OUT = Path(__file__).parent / "output" / "mars_fit_map.png"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--nrbase", type=int, default=100, help="Solver Nrbase (default matches TASK-011 fit).")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    args.out.parent.mkdir(parents=True, exist_ok=True)

    theta = mars_point_fit_theta()
    model = build_model(MARS_PARAMETERIZATION, theta)
    forcing = mars_forcing()
    numerics = mars_numerics(Nrbase=args.nrbase)
    obs = compute_observables(
        model, forcing, numerics,
        which=("mass", "moi_mean", "core_radius_km", "k2", "h2", "l2"),
    )

    h2, k2 = obs["h2"], obs["k2"]
    p2_pole = math.sqrt(5.0)  # P-bar_2^0(+/-1), 4pi-normalized (see module docstring)

    grid_h2 = sh_to_latlon({(2, 0): h2}, lmax=30)
    grid_k2 = sh_to_latlon({(2, 0): k2}, lmax=30)

    fig = plt.figure(figsize=(13, 8.5))
    ax_h2 = fig.add_subplot(2, 2, 1)
    ax_k2 = fig.add_subplot(2, 2, 2)
    ax_text = fig.add_subplot(2, 1, 2)
    ax_text.axis("off")

    vmax_h2 = float(np.max(np.abs(grid_h2.z))) or 1.0
    mesh_h2 = ax_h2.pcolormesh(
        grid_h2.lon, grid_h2.lat, grid_h2.z, cmap="RdBu_r",
        vmin=-vmax_h2, vmax=vmax_h2, shading="nearest",
    )
    ax_h2.set_title("Radial displacement: h2 · P̄_2^0(sin lat)", fontsize=11)
    ax_h2.set_xlabel("Longitude [deg]")
    ax_h2.set_ylabel("Latitude [deg]")
    cb_h2 = fig.colorbar(mesh_h2, ax=ax_h2)
    cb_h2.set_label(
        f"h2 * P̄_2^0(sin lat), 4π-normalized [dimensionless]\n"
        f"(polar value ≈ {h2 * p2_pole:.3f}, NOT h2={h2:.3f})",
        fontsize=8,
    )

    vmax_k2 = float(np.max(np.abs(grid_k2.z))) or 1.0
    mesh_k2 = ax_k2.pcolormesh(
        grid_k2.lon, grid_k2.lat, grid_k2.z, cmap="PuOr_r",
        vmin=-vmax_k2, vmax=vmax_k2, shading="nearest",
    )
    ax_k2.set_title("Potential perturbation: k2 · P̄_2^0(sin lat)", fontsize=11)
    ax_k2.set_xlabel("Longitude [deg]")
    ax_k2.set_ylabel("Latitude [deg]")
    cb_k2 = fig.colorbar(mesh_k2, ax=ax_k2)
    cb_k2.set_label(
        f"k2 * P̄_2^0(sin lat), 4π-normalized [dimensionless]\n"
        f"(polar value ≈ {k2 * p2_pole:.3f}, NOT k2={k2:.3f})",
        fontsize=8,
    )

    param_lines = [f"{name} = {val:.4f}" for name, val in zip(MARS_FREE_PARAMS, theta)]
    param_lines.append(f"h2 = {h2:.6f}, l2 = {obs['l2']:.6f}")

    constraint_lines = []
    for c in MARS_CONSTRAINTS:
        achieved = obs[c.name]
        resid_sigma = (achieved - c.value) / c.sigma
        constraint_lines.append(
            f"{c.name:<10} target={c.value:.6g} +/- {c.sigma:.3g}   "
            f"achieved={achieved:.6g}   residual={resid_sigma:+.3f} sigma"
        )

    text = (
        "TASK-011 Mars point fit (deterministic fit; not a Monte Carlo posterior summary)\n\n"
        "Free parameters:\n  " + "\n  ".join(param_lines) + "\n\n"
        "Constraint residuals:\n  " + "\n  ".join(constraint_lines) + "\n\n"
        "Note: panels show the m=0 (zonal) degree-2 pattern for a single-panel\n"
        "illustration. Mars's actual dominant tide (solar semidiurnal) is m=2,\n"
        "whose longitude pattern differs from this zonal one -- but h2/k2/l2 for a\n"
        "spherically symmetric 1D model are m-independent, so the amplitude scaling\n"
        "shown is the same one that applies to the real m=2 tide."
    )
    ax_text.text(0.02, 0.95, text, va="top", ha="left", fontsize=11, family="monospace", transform=ax_text.transAxes)

    fig.suptitle("Mars degree-2 tidal surface pattern (TASK-012 fit demonstration)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=140)
    plt.close(fig)

    print(f"[output] figure saved to {args.out}")
    print(f"[fit] h2={h2:.6f}, k2={k2:.6f}, l2={obs['l2']:.6f}")
    for c in MARS_CONSTRAINTS:
        achieved = obs[c.name]
        print(f"[constraint] {c.name}: target={c.value:.6g} +/- {c.sigma:.3g}, achieved={achieved:.6g}")

    return args.out


if __name__ == "__main__":
    run(parse_args())
