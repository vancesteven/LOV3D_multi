# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Spherical-harmonic gravity sensitivity for Mars density anomalies.

The original proposal-scale calculation represented a thickness anomaly as a
thin sheet. This module now also supports exact finite radial shells so a 3D
composition/alteration model can pass density-harmonic coefficients as a
function of depth without collapsing them to one equivalent surface sheet.

For unit-norm real spherical harmonics, a thin sheet with thickness coefficient
``H_lm`` has dimensionless external-potential coefficient

    q_lm = 4*pi*delta_rho*R**2 / ((2*l+1)*M) * H_lm.

For a volume-density harmonic ``rho_lm(r)`` the exact generalization is

    q_lm = 4*pi / ((2*l+1)*M*R**l)
           * integral rho_lm(r) r**(l+2) dr.

A constant coefficient between radii ``r_inner`` and ``r_outer`` therefore has
an analytic finite-shell expression. The implementation evaluates it in terms
of radius ratios rather than raw powers of metre-valued radii, avoiding
floating-point overflow through and beyond GMM-3 degree 120.

These are physical orthonormal-harmonic coefficients. Use
``mars_gravity_normalization`` before comparing them with GMM-3 C_lm/S_lm.
"""

from __future__ import annotations

import math

import numpy as np

G = 6.67430e-11
MARS_RADIUS_M = 3389.5e3
MARS_MASS_KG = 6.4171e23


def _validate_gravity_geometry(degree: int, radius_m: float, mass_kg: float) -> None:
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if radius_m <= 0 or mass_kg <= 0:
        raise ValueError("radius_m and mass_kg must be positive")


def thin_sheet_potential_coefficient(
    degree: int,
    thickness_coeff_m: float,
    delta_rho_kg_m3: float,
    *,
    radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
    compensation_fraction: float = 0.0,
    compensation_depth_m: float = 0.0,
) -> float:
    """Return dimensionless orthonormal-harmonic potential coefficient q_lm.

    ``thickness_coeff_m`` is the coefficient of a layer-thickness anomaly, not
    the thickness of a globally uniform layer. Compensation is represented by
    an equal/opposite column-mass harmonic at the specified depth.
    """
    _validate_gravity_geometry(degree, radius_m, mass_kg)
    if not 0.0 <= compensation_fraction <= 1.0:
        raise ValueError("compensation_fraction must lie in [0, 1]")
    if compensation_depth_m < 0 or compensation_depth_m >= radius_m:
        raise ValueError("compensation_depth_m must lie in [0, radius_m)")

    q = (
        4.0
        * math.pi
        * delta_rho_kg_m3
        * radius_m**2
        * thickness_coeff_m
        / ((2 * degree + 1) * mass_kg)
    )
    if compensation_fraction:
        rb = radius_m - compensation_depth_m
        q *= 1.0 - compensation_fraction * (rb / radius_m) ** degree
    return q


def finite_shell_potential_coefficient(
    degree: int,
    density_coeff_kg_m3: float,
    inner_radius_m: float,
    outer_radius_m: float,
    *,
    reference_radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
) -> float:
    """Return exact q_lm for a constant density harmonic in one radial shell.

    ``density_coeff_kg_m3`` is the coefficient multiplying a unit-norm real
    spherical harmonic throughout the shell. The shell must lie at or below
    the external-potential reference radius.
    """
    _validate_gravity_geometry(degree, reference_radius_m, mass_kg)
    ri = float(inner_radius_m)
    ro = float(outer_radius_m)
    R = float(reference_radius_m)
    if ri < 0 or ro <= ri:
        raise ValueError("require 0 <= inner_radius_m < outer_radius_m")
    if ro > R:
        raise ValueError("outer_radius_m cannot exceed reference_radius_m")

    # Algebraically identical to
    # (ro**(l+3)-ri**(l+3))/((l+3)*R**l), but numerically stable at high l.
    radial_factor_m3 = (
        R**3
        / (degree + 3)
        * ((ro / R) ** (degree + 3) - (ri / R) ** (degree + 3))
    )
    return (
        4.0
        * math.pi
        * float(density_coeff_kg_m3)
        * radial_factor_m3
        / ((2 * degree + 1) * mass_kg)
    )


def layered_density_potential_coefficient(
    degree: int,
    radius_edges_m: np.ndarray,
    density_coefficients_kg_m3: np.ndarray,
    *,
    reference_radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
) -> float:
    """Integrate a piecewise-constant radial density-harmonic profile exactly.

    ``radius_edges_m`` must increase outward and have one more entry than
    ``density_coefficients_kg_m3``. Each density value is an l,m harmonic
    coefficient for that radial shell, so positive and negative contributions
    naturally cancel when summed.
    """
    r = np.asarray(radius_edges_m, dtype=float)
    rho = np.asarray(density_coefficients_kg_m3, dtype=float)
    if r.ndim != 1 or rho.ndim != 1 or r.size != rho.size + 1:
        raise ValueError("radius_edges_m must be 1-D with len(edges)=len(density)+1")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(rho)):
        raise ValueError("radius edges and density coefficients must be finite")
    if np.any(np.diff(r) <= 0):
        raise ValueError("radius_edges_m must increase strictly outward")

    terms = [
        finite_shell_potential_coefficient(
            degree,
            rho_i,
            r0,
            r1,
            reference_radius_m=reference_radius_m,
            mass_kg=mass_kg,
        )
        for r0, r1, rho_i in zip(r[:-1], r[1:], rho)
    ]
    return float(math.fsum(terms))


def radial_gravity_from_coefficient(
    degree: int,
    q_lm: float,
    altitude_m: float,
    *,
    radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
) -> float:
    """Return radial-gravity harmonic amplitude in m/s^2 per unit Y_lm."""
    _validate_gravity_geometry(degree, radius_m, mass_kg)
    if altitude_m < 0:
        raise ValueError("altitude_m must be non-negative")
    r = radius_m + altitude_m
    return (
        G
        * mass_kg
        / r**2
        * (degree + 1)
        * q_lm
        * (radius_m / r) ** degree
    )


def gravity_from_thickness_coefficient(
    degree: int,
    thickness_coeff_m: float,
    delta_rho_kg_m3: float,
    altitude_m: float,
    *,
    compensation_fraction: float = 0.0,
    compensation_depth_m: float = 0.0,
    radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
) -> float:
    """Convenience wrapper from H_lm to radial gravity amplitude."""
    q = thin_sheet_potential_coefficient(
        degree,
        thickness_coeff_m,
        delta_rho_kg_m3,
        radius_m=radius_m,
        mass_kg=mass_kg,
        compensation_fraction=compensation_fraction,
        compensation_depth_m=compensation_depth_m,
    )
    return radial_gravity_from_coefficient(
        degree,
        q,
        altitude_m,
        radius_m=radius_m,
        mass_kg=mass_kg,
    )
