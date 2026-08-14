# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Moon Airy-crust lateral Love-number pipeline (TASK-031).

This is the Moon analogue of :mod:`pylov3d.mars_lateral`: LOLA shape minus
the low-degree GRAIL equipotential surface gives relief, Airy compensation
maps relief to crustal-thickness variation, and the thickness field is
linearized into lateral rigidity of the Weber model's surface crust layer.

Two Moon-specific convention decisions are explicit:

* Degree 1 is removed from both fields.  MoonTopo719 is in a principal-axis,
  center-of-mass frame, and its large degree-1 shape is the center-of-figure
  offset, a translation rather than a crustal load.
* The default removes C20 from both fields before the areoid subtraction.
  At full amplitude, retaining the residual C20 drives the linearized
  ``|delta_mu/mu_bar|`` to about 1.08, beyond its positive-rigidity domain.
  Removing C20 treats the hydrostatic/tidal zonal figure as reference shape
  and leaves the full non-zonal degree-2 field.  ``include_c20=True`` exists
  only to quantify this first-order convention sensitivity; it is rejected
  by :func:`mu_variable_from_topography` at full amplitude.

The rigidity denominator is the independently adopted 40 km mean crustal
thickness in :data:`pylov3d.moon.MOON`, not the Weber discretization's 34 km
surface shell.  This keeps the Airy perturbation referenced to the observed
mean crust while applying it to the closest available numerical layer.  The
result is close to the linearization boundary (about 0.99 at lmax=4), so this
module reports rather than hides that limitation.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .love import get_love
from .mapping import sh_to_latlon
from .mars_lateral import _real_sh_to_complex_mu_variable
from .moon import (
    LAYER_MU,
    LAYER_RADII_KM,
    LAYER_RHO,
    MOON,
    MOON_FORCING_TD,
    build_moon_model,
)
from .sh_data import load_shadr, load_shape, truncate
from .types import make_forcing, make_numerics


_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data" / "moon"
TOPO_PATH = DATA_DIR / "MoonTopo719.shape.gz"
GRAVITY_PATH = DATA_DIR / "grgm900c_120_sha.tab"

CRUST_LAYER_INDEX = len(LAYER_RADII_KM) - 1
MANTLE_LAYER_INDEX = CRUST_LAYER_INDEX - 1
CRUST_THICKNESS_M = MOON["crust_thickness_adopted"]
WEBER_CRUST_SHELL_THICKNESS_M = (
    LAYER_RADII_KM[CRUST_LAYER_INDEX] - LAYER_RADII_KM[MANTLE_LAYER_INDEX]
) * 1_000.0
AIRY_FACTOR = LAYER_RHO[CRUST_LAYER_INDEX] / (
    LAYER_RHO[MANTLE_LAYER_INDEX] - LAYER_RHO[CRUST_LAYER_INDEX]
)


def _dmu_ddt_coeff() -> float:
    """Return d(mu/mu_bar)/d(dt) [1/m] for the surface crust layer."""
    mu_crust = LAYER_MU[CRUST_LAYER_INDEX]
    mu_mantle = LAYER_MU[MANTLE_LAYER_INDEX]
    return (mu_crust - mu_mantle) / (CRUST_THICKNESS_M * mu_crust)


def _relief_coefficients(
    lmax: int = 4,
    *,
    include_c20: bool = False,
    include_degree1: bool = True,
    topo_path: Path | str = TOPO_PATH,
    gravity_path: Path | str = GRAVITY_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return LOLA relief above the GRAIL equipotential surface [m]."""
    shape = truncate(load_shape(topo_path), lmax)
    gravity = truncate(load_shadr(gravity_path), lmax)
    clm = shape["clm"].copy()
    slm = shape["slm"].copy()
    geoid_clm = gravity["r0_m"] * gravity["clm"].copy()
    geoid_slm = gravity["r0_m"] * gravity["slm"].copy()

    # C00 is the reference radius/GM, not relief.  Degree 1 of the shape is
    # the lunar center-of-figure offset in this principal-axis,
    # center-of-mass frame -- but physically it carries the nearside-farside
    # crustal dichotomy, which the TASK-031 plan specified retaining.  It is
    # retained by default (PI decision, 2026-08-14, resolving the
    # plan/implementation divergence flagged in TASK-035); pass
    # ``include_degree1=False`` to reproduce the pre-2026-08-14 field that
    # the committed MATLAB anchor and TASK-031/034/036/037 artifacts used.
    clm[0, 0] = geoid_clm[0, 0] = 0.0
    if not include_degree1:
        clm[1, :] = slm[1, :] = 0.0
        geoid_clm[1, :] = geoid_slm[1, :] = 0.0
    if not include_c20:
        clm[2, 0] = geoid_clm[2, 0] = 0.0

    return clm - geoid_clm, slm - geoid_slm


def crustal_thickness_variation(
    lmax: int = 4,
    *,
    include_c20: bool = False,
    include_degree1: bool = True,
    topo_path: Path | str = TOPO_PATH,
    gravity_path: Path | str = GRAVITY_PATH,
) -> dict[tuple[int, int], float]:
    """Return Airy crustal-thickness SH coefficients [m].

    Positive relief produces a positive crustal root.  Positive ``m`` keys
    are cosine coefficients and negative ``m`` keys are sine coefficients,
    matching :func:`pylov3d.mapping.sh_to_latlon`.
    """
    clm, slm = _relief_coefficients(
        lmax=lmax,
        include_c20=include_c20,
        include_degree1=include_degree1,
        topo_path=topo_path,
        gravity_path=gravity_path,
    )
    result: dict[tuple[int, int], float] = {}
    for n in range(lmax + 1):
        for m in range(n + 1):
            c = float(clm[n, m]) * AIRY_FACTOR
            if c != 0.0:
                result[(n, m)] = c
            if m:
                s = float(slm[n, m]) * AIRY_FACTOR
                if s != 0.0:
                    result[(n, -m)] = s
    return result


def dmu_over_mu_real(
    lmax: int = 4,
    *,
    include_c20: bool = False,
    include_degree1: bool = True,
) -> dict[tuple[int, int], float]:
    """Return real-SH coefficients of crustal delta_mu/mu_bar."""
    coeff = _dmu_ddt_coeff()
    return {
        nm: coeff * value
        for nm, value in crustal_thickness_variation(
            lmax=lmax, include_c20=include_c20, include_degree1=include_degree1,
        ).items()
    }


def crustal_thickness_diagnostics(
    lmax: int = 4,
    *,
    include_c20: bool = False,
    include_degree1: bool = True,
    nlat: int = 180,
    nlon: int = 360,
) -> dict:
    """Return amplitude, positivity, and per-degree diagnostics."""
    dt = crustal_thickness_variation(
        lmax=lmax, include_c20=include_c20, include_degree1=include_degree1,
    )
    grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon)
    max_abs_dt_m = float(np.max(np.abs(grid.z)))
    max_abs_dmu = abs(_dmu_ddt_coeff()) * max_abs_dt_m
    degree_rms_km = {
        n: math.sqrt(sum(v * v for (nn, _m), v in dt.items() if nn == n)) / 1e3
        for n in range(1, lmax + 1)
    }
    return {
        "max_abs_dt_m": max_abs_dt_m,
        "max_abs_dt_over_reference": max_abs_dt_m / CRUST_THICKNESS_M,
        "max_abs_dmu_over_mubar": max_abs_dmu,
        "degree_rms_km": degree_rms_km,
        "include_c20": include_c20,
        "include_degree1": include_degree1,
    }


def mu_variable_from_topography(
    lmax: int = 4,
    *,
    include_c20: bool = False,
    include_degree1: bool = True,
) -> dict[int, list[tuple[int, int, complex]]]:
    """Return surface-crust ``mu_variable`` entries for the coupled solver.

    Refuse fields outside the linearized positive-rigidity domain instead of
    clipping or silently manufacturing a nonphysical shear modulus.
    """
    diagnostics = crustal_thickness_diagnostics(
        lmax=lmax, include_c20=include_c20, include_degree1=include_degree1,
    )
    if diagnostics["max_abs_dt_over_reference"] >= 1.0:
        raise ValueError("Airy thickness variation exceeds the 40 km reference crust")
    if diagnostics["max_abs_dmu_over_mubar"] >= 1.0:
        raise ValueError("Airy rigidity variation makes the linearized crust non-positive")
    entries = _real_sh_to_complex_mu_variable(
        dmu_over_mu_real(
            lmax=lmax, include_c20=include_c20, include_degree1=include_degree1,
        )
    )
    return {CRUST_LAYER_INDEX: entries}


def moon_lateral_love_spectrum(
    lmax: int = 4,
    forcing: tuple[int, int] = (2, 0),
    perturbation_order: int = 2,
    Nrbase: int = 30,
    method: str = "variable",
    F: complex = 1.0,
    include_degree1: bool = True,
) -> dict:
    """Compute the coupled Moon Love-number spectrum for the default field."""
    n_f, m_f = forcing
    model = build_moon_model()
    forcing_obj = make_forcing(Td=MOON_FORCING_TD, n=n_f, m=m_f, F=F)
    numerics = make_numerics(
        n_layers=model.n_layers,
        method=method,
        Nrbase=Nrbase,
        perturbation_order=perturbation_order,
    )
    mu_variable = mu_variable_from_topography(
        lmax=lmax, include_degree1=include_degree1,
    )
    start = time.perf_counter()
    love, y_rad, model_out = get_love(
        model, forcing_obj, numerics, mu_variable=mu_variable,
    )
    return {
        "love": love,
        "y_rad": y_rad,
        "model": model_out,
        "forcing_obj": forcing_obj,
        "numerics": numerics,
        "mu_variable": mu_variable,
        "wall_s": time.perf_counter() - start,
    }
