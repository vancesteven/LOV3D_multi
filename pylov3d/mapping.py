# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Spectral -> lat/lon surface maps (TASK-012).

Generalizes the ``_delta_unit_map`` machinery originally written (and
validated against MATLAB's ``SPH_LatLon``) in
``pylov3d/tests/test_matlab_validation_ocean.py`` into a reusable real,
4pi-normalized spherical-harmonic-to-grid evaluator. That test module now
calls :func:`sh_to_latlon` instead of carrying its own copy — see
``_delta_unit_map`` there, which is now a thin wrapper.

Conventions
--------------------------------------------------------------------------
- Real, 4pi-fully-normalized associated Legendre functions, **no**
  Condon-Shortley phase, computed by :func:`fully_normalized_legendre` — a
  direct port of the MATLAB reference recursion ``src/SPH_Tools/Legendre.m``
  (Rapp 1982 sectorial/recursion scheme). This is the already-trusted
  convention source for this codebase, and the recursion is stable to high
  degree by construction (no factorial ratios, unlike a
  ``norm(n,m) * scipy.special.lpmv(m, n, t)`` evaluation, which (a) carries
  ``lpmv``'s built-in Condon-Shortley ``(-1)^m`` phase — silently flipping
  the sign of every odd-``m`` map relative to the documented no-CS
  convention — and (b) underflows its ``norm(n, m)`` factor to exactly 0.0
  for ``n + m`` beyond ~170 (`factorial` overflows before the ratio does),
  producing ``0 * inf = NaN`` once ``lpmv`` itself overflows — both bugs are
  invisible to a peak-to-peak check (sign-flips and NaN-free small-degree
  cases don't move ``max - min``), which is why they survived the original
  MATLAB-validated harness; see ``pylov3d/tests/test_mapping.py`` for a
  regression case at real gravity/shape data's degree (120 and 719).
- For ``m >= 0`` the angular factor is ``cos(m*lon)``; for ``m < 0`` (not
  exercised by the original harness, added here only for generality) it is
  ``sin(|m|*lon)`` — the standard real-SH sine complement.
- Grid: half-cell-offset lat/lon centers. The default resolution matches
  MATLAB's ``SPH_LatLon(l_max)``: internal grid degree
  ``lmax_grid = 2*l_max - 1``, spacing ``d = 180 / (2*lmax_grid)`` degrees,
  lat in [-90, 90], lon in [-180, 180], both cell-centered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SHGrid:
    """A lat/lon-gridded scalar field."""

    lat: np.ndarray  # (nlat,) degrees, cell-centered
    lon: np.ndarray  # (nlon,) degrees, cell-centered
    z: np.ndarray  # (nlat, nlon)


def fully_normalized_legendre(lmax: int, t) -> np.ndarray:
    """4pi-fully-normalized associated Legendre functions, no Condon-Shortley phase.

    Direct port of ``src/SPH_Tools/Legendre.m`` (Rapp 1982 recursion): a
    sectorial (diagonal) recursion seeded at ``P_0^0 = 1``, ``P_1^1 =
    sqrt(3)*u``, extended to higher degree along the diagonal, then a
    two-term recursion in degree for each fixed order ``m``. Every
    coefficient in the recursion is a ratio of small integers under a square
    root (never a raw factorial), so this remains finite and accurate at
    degrees far beyond where ``factorial(n +- m)`` would overflow (verified
    to l_max=719, the degree of ``data/mars/MarsTopo719.shape.gz``; see
    ``pylov3d/tests/test_mapping.py``).

    Parameters
    ----------
    lmax : maximum degree.
    t : array_like, evaluation points (``sin(latitude)``, i.e.
        ``cos(colatitude)``), any shape.

    Returns
    -------
    P : ``(lmax+1, lmax+1, *t.shape)`` array; ``P[n, m]`` is
        :math:`\\bar P_n^m(t)` for ``0 <= m <= n``, zero elsewhere.
    """
    t = np.asarray(t, dtype=float)
    orig_shape = t.shape
    t_flat = t.reshape(-1)
    nt = t_flat.size

    P = np.zeros((lmax + 1, lmax + 1, nt), dtype=float)
    u = np.sqrt(1.0 - t_flat**2)

    P[0, 0, :] = 1.0
    if lmax >= 1:
        P[1, 1, :] = math.sqrt(3.0) * u
    for n in range(2, lmax + 1):
        P[n, n, :] = math.sqrt((2 * n + 1) / (2 * n)) * u * P[n - 1, n - 1, :]

    for m in range(0, lmax):
        P[m + 1, m, :] = math.sqrt(2 * m + 3) * t_flat * P[m, m, :]
        for n in range(m + 2, lmax + 1):
            a = math.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
            b = math.sqrt(
                (2 * n + 1) * (n + m - 1) * (n - m - 1) / ((2 * n - 3) * (n + m) * (n - m))
            )
            P[n, m, :] = a * t_flat * P[n - 1, m, :] - b * P[n - 2, m, :]

    return P.reshape((lmax + 1, lmax + 1) + orig_shape)


def latlon_grid(lmax: int = 30, nlat: int | None = None, nlon: int | None = None):
    """Half-cell-offset lat/lon grid centers, in degrees.

    Parameters
    ----------
    lmax : grid-density parameter matching MATLAB's ``SPH_LatLon(l_max)``
        convention (see module docstring). Ignored if both ``nlat`` and
        ``nlon`` are given.
    nlat, nlon : explicit grid size, overriding the ``lmax``-derived default.

    Returns
    -------
    (lat, lon) : 1-D float arrays, degrees.
    """
    if nlat is None or nlon is None:
        lmax_grid = 2 * lmax - 1
        d = 180.0 / (2 * lmax_grid)
        lat = np.arange(-90.0 + d / 2.0, 90.0 - d / 2.0 + 1e-9, d)
        lon = np.arange(-180.0 + d / 2.0, 180.0 - d / 2.0 + 1e-9, d)
    else:
        dlat = 180.0 / nlat
        dlon = 360.0 / nlon
        lat = -90.0 + dlat / 2.0 + dlat * np.arange(nlat)
        lon = -180.0 + dlon / 2.0 + dlon * np.arange(nlon)
    return lat, lon


def sh_to_latlon(
    coeffs: Mapping[tuple[int, int], float],
    lmax: int = 30,
    nlat: int | None = None,
    nlon: int | None = None,
) -> SHGrid:
    """Evaluate a real, 4pi-normalized spherical-harmonic expansion on a grid.

    Parameters
    ----------
    coeffs : ``{(n, m): amplitude}``. ``m >= 0`` contributes
        ``amplitude * P_n^m(sin(lat)) * cos(m*lon)``; ``m < 0`` contributes
        ``amplitude * P_n^|m|(sin(lat)) * sin(|m|*lon)`` (see module
        docstring for the normalization/phase convention).
    lmax, nlat, nlon : grid parameters, see :func:`latlon_grid`.

    Returns
    -------
    SHGrid with ``z.shape == (len(lat), len(lon))``.

    Notes
    -----
    With a single unit coefficient ``{(n, m): 1.0}`` and the default grid,
    ``sh_to_latlon({(n, m): 1.0}, lmax=l_max).z`` reproduces the original
    ``_delta_unit_map(n, m, l_max)`` array's peak-to-peak exactly (same grid
    construction, same 4pi normalization). The raw values of odd-``m`` maps
    changed sign relative to an earlier revision of this function that
    carried ``scipy.special.lpmv``'s Condon-Shortley phase (see module
    docstring) — peak-to-peak is sign-invariant, so this does not affect
    that regression check, but does matter for e.g. the sign/location of a
    tidal bulge on an actual lat/lon map.

    ``lmax`` here is purely the *grid resolution* parameter (see
    :func:`latlon_grid`); the *coefficient* evaluation degree (the ``n`` in
    each ``coeffs`` key) is independent and may exceed it — the underlying
    :func:`fully_normalized_legendre` recursion is built once, up to
    ``max(n for (n, m) in coeffs)``, and is numerically stable at any degree
    this codebase uses (verified to 719; see module docstring).
    """
    lat, lon = latlon_grid(lmax=lmax, nlat=nlat, nlon=nlon)
    t = np.sin(np.radians(lat))
    lon_rad = np.radians(lon)
    z = np.zeros((len(lat), len(lon)))

    if not coeffs:
        return SHGrid(lat=lat, lon=lon, z=z)

    n_max = max(n for (n, _m) in coeffs)
    P = fully_normalized_legendre(n_max, t)  # (n_max+1, n_max+1, nlat)

    for (n, m), amp in coeffs.items():
        mm = abs(m)
        Pnm = P[n, mm, :]
        ang = np.cos(mm * lon_rad) if m >= 0 else np.sin(mm * lon_rad)
        z += amp * (Pnm[:, None] * ang[None, :])

    return SHGrid(lat=lat, lon=lon, z=z)


def plot_map(
    grid: SHGrid,
    title: str = "",
    cbar_label: str = "",
    cmap: str = "RdBu_r",
    ax=None,
    symmetric: bool = True,
):
    """Render an :class:`SHGrid` as a matplotlib pcolormesh map.

    Agg-safe: does not call ``plt.show()``; import ``matplotlib`` and select
    a backend (e.g. ``matplotlib.use("Agg")``) in the caller *before*
    importing this function if running headless. Returns the ``Figure``
    (and, if ``ax`` was not given, the caller owns closing it).

    Parameters
    ----------
    symmetric : if True (default), colour limits are +/- max(|z|) so zero
        maps to the diverging colormap's center — appropriate for signed
        tidal-deformation fields.
    """
    import matplotlib.pyplot as plt

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if symmetric:
        vmax = float(np.max(np.abs(grid.z))) or 1.0
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(grid.z)), float(np.max(grid.z))

    mesh = ax.pcolormesh(grid.lon, grid.lat, grid.z, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xlim(grid.lon.min(), grid.lon.max())
    ax.set_ylim(grid.lat.min(), grid.lat.max())
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    return fig
