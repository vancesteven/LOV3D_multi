# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Normalization bridge between pylov3d Mars harmonics and PDS/GMM-3.

The proposal-scale finite-shell gravity machinery represents a physical
potential perturbation using real spherical harmonics normalized to unit norm
on the sphere,

    integral Y_lm^2 dOmega = 1.

The PDS SHADR convention used by GMM-3 stores *normalized* geopotential
coefficients with the Lambeck/geodetic basis documented in the PDS SHADR SIS.
For either the cosine or sine real harmonic that basis has

    integral Ybar_lm^2 dOmega = 4*pi.

Therefore

    Ybar_lm = sqrt(4*pi) Y_lm

and an otherwise identical potential perturbation obeys

    Cbar_lm = q_lm / sqrt(4*pi)

(or Sbar_lm for the sine harmonic).

If the two coefficient sets use different reference radii, equality of the
external potential requires the additional degree-dependent rescaling

    c(R_to) = c(R_from) * (R_from/R_to)^l.

GMM-3 uses R=3396 km.  pylov3d's proposal-scale Mars models currently use
R=3389.5 km.

Primary convention reference: PDS Spherical Harmonics ASCII Data Record
(SHADR) Software Interface Specification, Appendix A.
"""
from __future__ import annotations

import math

SQRT_4PI = math.sqrt(4.0 * math.pi)
GMM3_REFERENCE_RADIUS_M = 3396.0e3
PYLOV3D_MARS_RADIUS_M = 3389.5e3


def orthonormal_to_gmm3_normalized(
    q_lm: float,
    degree: int,
    *,
    source_radius_m: float = PYLOV3D_MARS_RADIUS_M,
    gmm3_radius_m: float = GMM3_REFERENCE_RADIUS_M,
) -> float:
    """Convert a unit-norm real-harmonic potential coefficient to GMM-3.

    ``q_lm`` is the dimensionless coefficient multiplying a real spherical
    harmonic with unit integral norm.  The returned value is the normalized
    C_lm or S_lm coefficient multiplying the corresponding PDS/GMM-3 basis
    function.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if source_radius_m <= 0 or gmm3_radius_m <= 0:
        raise ValueError("reference radii must be positive")
    radius_factor = (source_radius_m / gmm3_radius_m) ** degree
    return float(q_lm) * radius_factor / SQRT_4PI


def gmm3_normalized_to_orthonormal(
    coefficient: float,
    degree: int,
    *,
    source_radius_m: float = PYLOV3D_MARS_RADIUS_M,
    gmm3_radius_m: float = GMM3_REFERENCE_RADIUS_M,
) -> float:
    """Inverse of :func:`orthonormal_to_gmm3_normalized`."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if source_radius_m <= 0 or gmm3_radius_m <= 0:
        raise ValueError("reference radii must be positive")
    radius_factor = (source_radius_m / gmm3_radius_m) ** degree
    return float(coefficient) * SQRT_4PI / radius_factor


def gmm3_conservative_snr(signal_coefficient: float, formal_sigma: float, factor: float = 3.0) -> float:
    """Return coefficient-space signal/noise using a scaled formal sigma."""
    if formal_sigma <= 0 or factor <= 0:
        raise ValueError("formal_sigma and factor must be positive")
    return abs(float(signal_coefficient)) / (factor * float(formal_sigma))
