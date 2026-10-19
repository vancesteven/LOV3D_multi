# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Compare proposal-scale hydration harmonics with the observed GMM-3 field.

Formal coefficient uncertainties are far smaller than the illustrative
hydration signal at low/intermediate degree.  The more relevant next-scale
comparison is therefore with the actual gravity-field amplitude at each degree.

For normalized GMM-3 coefficients this module reports

    degree_norm = sqrt(sum_m(C_lm^2 + S_lm^2))
    coefficient_rms = degree_norm / sqrt(2*l + 1)

where S_l0=0.  These are coefficient-space diagnostics only; they do not
separate topography, crust, mantle, load compensation, or other geological
contributors to the observed gravity field.
"""
from __future__ import annotations

import math
from typing import Iterable

from .mars_gmm3 import GMM3Coefficient, coefficients_at_degree


def degree_coefficient_norm(coeffs: Iterable[GMM3Coefficient], degree: int) -> float:
    """Return sqrt(sum(C_lm^2 + S_lm^2)) at one degree."""
    rows = coefficients_at_degree(coeffs, degree)
    if not rows:
        raise ValueError(f"no coefficients found at degree {degree}")
    return math.sqrt(sum(row.c * row.c + row.s * row.s for row in rows))


def degree_coefficient_rms(coeffs: Iterable[GMM3Coefficient], degree: int) -> float:
    """Return RMS normalized coefficient amplitude across 2*l+1 real modes."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return degree_coefficient_norm(coeffs, degree) / math.sqrt(2 * degree + 1)


def signal_fraction_of_degree_norm(
    signal_coefficient: float,
    coeffs: Iterable[GMM3Coefficient],
    degree: int,
) -> float:
    """Return |signal| divided by total coefficient norm at one degree."""
    norm = degree_coefficient_norm(coeffs, degree)
    if norm == 0.0:
        return math.inf if signal_coefficient else 0.0
    return abs(float(signal_coefficient)) / norm


def signal_over_degree_rms(
    signal_coefficient: float,
    coeffs: Iterable[GMM3Coefficient],
    degree: int,
) -> float:
    """Return |signal| divided by the RMS coefficient amplitude at one degree."""
    rms = degree_coefficient_rms(coeffs, degree)
    if rms == 0.0:
        return math.inf if signal_coefficient else 0.0
    return abs(float(signal_coefficient)) / rms
