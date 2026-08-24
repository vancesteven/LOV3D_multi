# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-scale spherical-harmonic gravity sensitivity for Mars.

This module deliberately separates two pieces of physics:

1. a thin-sheet surface gravity scale, ``2*pi*G*delta_rho*H``;
2. exact exterior attenuation of a degree-l radial-gravity harmonic,
   ``(R/(R+h))**(l+2)``.

It does not yet impose a specific C_lm/S_lm normalization, compensation model,
or gravity-field covariance. Those belong to the next Mars-specific rung.
"""

from __future__ import annotations

import math

G = 6.67430e-11
MARS_MEAN_RADIUS_M = 3_389_500.0


def degree_from_wavelength(wavelength_m: float, radius_m: float = MARS_MEAN_RADIUS_M) -> float:
    """Return the local spherical-harmonic degree associated with wavelength.

    Uses lambda ~= 2*pi*R/l, appropriate for proposal-scale spectral mapping.
    """
    if wavelength_m <= 0 or radius_m <= 0:
        raise ValueError("wavelength and radius must be positive")
    return 2.0 * math.pi * radius_m / wavelength_m


def wavelength_from_degree(degree: float, radius_m: float = MARS_MEAN_RADIUS_M) -> float:
    """Return approximate surface wavelength for spherical-harmonic degree."""
    if degree <= 0 or radius_m <= 0:
        raise ValueError("degree and radius must be positive")
    return 2.0 * math.pi * radius_m / degree


def radial_gravity_attenuation(
    degree: float,
    altitude_m: float,
    radius_m: float = MARS_MEAN_RADIUS_M,
) -> float:
    """Exterior attenuation of radial gravity from the surface to altitude.

    If a degree-l potential varies as r^{-(l+1)}, its radial acceleration varies
    as r^{-(l+2)}. Therefore a surface radial-gravity harmonic attenuates by
    (R/(R+h))**(l+2).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if altitude_m < 0 or radius_m <= 0:
        raise ValueError("altitude must be non-negative and radius positive")
    return (radius_m / (radius_m + altitude_m)) ** (degree + 2.0)


def thin_sheet_surface_gravity(delta_rho_kg_m3: float, thickness_m: float) -> float:
    """Magnitude of the uncompensated infinite-sheet gravity contrast [m/s^2]."""
    if thickness_m < 0:
        raise ValueError("thickness must be non-negative")
    return 2.0 * math.pi * G * abs(delta_rho_kg_m3) * thickness_m


def harmonic_gravity_bound(
    delta_rho_kg_m3: float,
    thickness_m: float,
    degree: float,
    altitude_m: float,
    radius_m: float = MARS_MEAN_RADIUS_M,
) -> float:
    """Proposal-scale orbital gravity bound [m/s^2].

    The surface amplitude is the uncompensated thin-sheet scale; the orbital
    attenuation is exact for a degree-l exterior radial-gravity harmonic.
    The result is therefore useful for scale discrimination, but is not a
    substitute for a finite-shell C_lm/S_lm forward calculation.
    """
    return thin_sheet_surface_gravity(delta_rho_kg_m3, thickness_m) * radial_gravity_attenuation(
        degree, altitude_m, radius_m
    )
