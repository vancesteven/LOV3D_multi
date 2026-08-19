# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Finite-shell spherical-harmonic gravity sensitivity for Mars.

The functions here intentionally separate geometry from coefficient normalization.
A layer-thickness anomaly is represented in an orthonormal spherical-harmonic
basis, with coefficient H_lm in metres. For a density contrast ``delta_rho``
and reference radius ``R``, the dimensionless potential coefficient at R is

    q_lm = 4*pi*delta_rho*R**2 / ((2*l+1)*M) * H_lm

for a thin sheet at R. A compensating sheet at radius ``Rb`` can be included
through a compensation fraction c. The convention used here conserves column
mass per solid angle, giving the degree-dependent factor

    1 - c*(Rb/R)**l.

The corresponding radial-gravity harmonic at altitude h is

    g_lm(h) = GM/(R+h)^2 * (l+1) * q_lm * (R/(R+h))**l.

This module is proposal-scale sensitivity machinery, not yet a replacement for
full Mars gravity-field inversion with a published covariance and explicit
normalization convention for C_lm/S_lm products.
"""

from __future__ import annotations

import math

G = 6.67430e-11
MARS_RADIUS_M = 3389.5e3
MARS_MASS_KG = 6.4171e23


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
    """Return dimensionless orthonormal-harmonic potential coefficient q_lm."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if radius_m <= 0 or mass_kg <= 0:
        raise ValueError("radius_m and mass_kg must be positive")
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


def radial_gravity_from_coefficient(
    degree: int,
    q_lm: float,
    altitude_m: float,
    *,
    radius_m: float = MARS_RADIUS_M,
    mass_kg: float = MARS_MASS_KG,
) -> float:
    """Return radial-gravity harmonic amplitude in m/s^2 per unit Y_lm."""
    if degree < 1:
        raise ValueError("degree must be >= 1")
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
