# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Bridge 3D Mars alteration/composition fields to gravity harmonics.

This module is deliberately narrow: it converts a bulk density-anomaly grid in
radial shells into the orthonormal potential coefficients used by pylov3d's
finite-shell gravity machinery.  It therefore provides the quantitative link

    alteration/composition field -> density anomaly -> C_lm/S_lm-like signal

without claiming that density anomaly is uniquely caused by hydration.

The surface grids use the LOV3D/MATLAB cell-centred equiangular layout
``(2*lmax, 4*lmax)``. ``grid_to_stokes`` reproduces the literal MATLAB
``LatLon_SPH`` transform, including its deterministic longitude half-cell phase
for m>0. That phase is required for raw-grid MATLAB parity, but it is not a
physical rotation we want to imprint on a new Mars density field. Here we undo
that known phase after analysis. The resulting real Stokes coefficients refer
to the actual cell-centred longitude coordinates of the supplied map.

The Stokes basis has norm 4*pi. The finite-shell gravity formula uses unit-norm
real harmonics, so density coefficients are multiplied by ``sqrt(4*pi)`` before
radial integration.
"""

from __future__ import annotations

import math

import numpy as np

from .mars_gravity_coefficients import (
    MARS_MASS_KG,
    MARS_RADIUS_M,
    layered_density_potential_coefficient,
)
from .matlab_sph import grid_to_stokes

SQRT_4PI = math.sqrt(4.0 * math.pi)


def density_contrast_from_alteration(
    hydrated_fraction: np.ndarray | float,
    reactive_fraction: np.ndarray | float,
    rho_dry_kg_m3: np.ndarray | float,
    rho_hydrated_kg_m3: np.ndarray | float,
) -> np.ndarray:
    """Return bulk density contrast from partial alteration of reactive rock.

    ``hydrated_fraction`` is the fraction of the locally reactive component that
    is altered. ``reactive_fraction`` is the bulk-rock fraction of that reactive
    component. The result is

        delta_rho = f_h * f_reactive * (rho_hydrated - rho_dry).

    This is a material-state map, not an inversion. Porosity, temperature, and
    other density contributions should be added as separate terms when present.
    """
    fh = np.asarray(hydrated_fraction, dtype=float)
    fr = np.asarray(reactive_fraction, dtype=float)
    rd = np.asarray(rho_dry_kg_m3, dtype=float)
    rh = np.asarray(rho_hydrated_kg_m3, dtype=float)

    for name, arr in (("hydrated_fraction", fh), ("reactive_fraction", fr)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must be finite and lie in [0,1]")
    if not np.all(np.isfinite(rd)) or not np.all(np.isfinite(rh)):
        raise ValueError("density endmembers must be finite")
    if np.any(rd <= 0) or np.any(rh <= 0):
        raise ValueError("density endmembers must be positive")

    try:
        return np.asarray(fh * fr * (rh - rd), dtype=float)
    except ValueError as exc:
        raise ValueError("alteration fractions and density endmembers must be broadcastable") from exc


def _remove_matlab_half_cell_phase(
    c_stokes: np.ndarray,
    s_stokes: np.ndarray,
    lmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo the known m-dependent longitude phase in MATLAB ``LatLon_SPH``.

    For synthesized coefficients ``C,S``, the literal MATLAB analysis returns

        C' = C cos(alpha) + S sin(alpha)
        S' = S cos(alpha) - C sin(alpha)

    with ``alpha = m*pi/(4*lmax)``. This function applies the inverse rotation.
    """
    c = np.asarray(c_stokes, dtype=float).copy()
    s = np.asarray(s_stokes, dtype=float).copy()
    for degree in range(lmax + 1):
        for order in range(1, degree + 1):
            alpha = order * np.pi / (4.0 * lmax)
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            cp = c_stokes[degree, order]
            sp = s_stokes[degree, order]
            c[degree, order] = cp * ca - sp * sa
            s[degree, order] = sp * ca + cp * sa
    return c, s


def grid_to_orthonormal_density_coefficients(
    density_grid_kg_m3: np.ndarray,
    lmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform one density grid to unit-norm real cosine/sine coefficients."""
    grid = np.asarray(density_grid_kg_m3, dtype=float)
    if not np.all(np.isfinite(grid)):
        raise ValueError("density grid must be finite")
    c_stokes, s_stokes = grid_to_stokes(grid, lmax)
    c_stokes, s_stokes = _remove_matlab_half_cell_phase(c_stokes, s_stokes, lmax)
    return SQRT_4PI * c_stokes, SQRT_4PI * s_stokes


def layered_density_gravity_coefficients(
    density_anomaly_grids_kg_m3: np.ndarray,
    radius_edges_m: np.ndarray,
    lmax: int,
    *,
    reference_radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-shell orthonormal potential coefficients for 3D density.

    Parameters
    ----------
    density_anomaly_grids_kg_m3
        Array with shape ``(nz, 2*lmax, 4*lmax)``. Each radial cell contains a
        surface map of density anomaly relative to the chosen dry/reference
        composition model.
    radius_edges_m
        ``nz+1`` radial edges increasing outward.
    lmax
        Maximum spherical-harmonic degree.

    Returns
    -------
    q_cos, q_sin
        Arrays of shape ``(lmax+1, lmax+1)``. Degree zero is left at zero because
        the proposal-facing gravity machinery treats mass closure separately.
    """
    if lmax < 1:
        raise ValueError("lmax must be >= 1")
    grids = np.asarray(density_anomaly_grids_kg_m3, dtype=float)
    edges = np.asarray(radius_edges_m, dtype=float)
    expected_grid_shape = (2 * lmax, 4 * lmax)
    if grids.ndim != 3 or grids.shape[1:] != expected_grid_shape:
        raise ValueError(
            f"density grids must have shape (nz, {expected_grid_shape[0]}, {expected_grid_shape[1]})"
        )
    if edges.ndim != 1 or edges.size != grids.shape[0] + 1:
        raise ValueError("radius_edges_m must have length nz+1")
    if not np.all(np.isfinite(grids)) or not np.all(np.isfinite(edges)):
        raise ValueError("density grids and radius edges must be finite")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("radius_edges_m must increase strictly outward")
    if edges[-1] > reference_radius_m:
        raise ValueError("outermost shell cannot exceed reference_radius_m")

    c_layers = np.zeros((grids.shape[0], lmax + 1, lmax + 1), dtype=float)
    s_layers = np.zeros_like(c_layers)
    for iz, grid in enumerate(grids):
        c_layers[iz], s_layers[iz] = grid_to_orthonormal_density_coefficients(grid, lmax)

    q_cos = np.zeros((lmax + 1, lmax + 1), dtype=float)
    q_sin = np.zeros_like(q_cos)
    for degree in range(1, lmax + 1):
        for order in range(degree + 1):
            q_cos[degree, order] = layered_density_potential_coefficient(
                degree,
                edges,
                c_layers[:, degree, order],
                reference_radius_m=reference_radius_m,
                mass_kg=mass_kg,
            )
            if order > 0:
                q_sin[degree, order] = layered_density_potential_coefficient(
                    degree,
                    edges,
                    s_layers[:, degree, order],
                    reference_radius_m=reference_radius_m,
                    mass_kg=mass_kg,
                )
    return q_cos, q_sin
