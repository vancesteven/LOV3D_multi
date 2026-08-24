# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-scale low-altitude magnetic-resolution bounds for Mars.

For a locally planar potential-field Fourier component of horizontal wavelength
``lambda``, upward continuation attenuates magnetic-field amplitude as

    A(h) / A(0) = exp(-2*pi*h/lambda).

This provides a transparent comparison between helicopter-scale altitudes and
orbital magnetometry.  It is not a full vector crustal-magnetization inversion:
source depth, source thickness, magnetization direction, external fields, and
spacecraft magnetic cleanliness must ultimately be modeled explicitly.
"""
from __future__ import annotations

import math


def upward_continuation_factor(wavelength_m: float, altitude_m: float) -> float:
    """Return potential-field amplitude fraction at altitude ``altitude_m``."""
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive")
    if altitude_m < 0:
        raise ValueError("altitude_m must be non-negative")
    return math.exp(-2.0 * math.pi * altitude_m / wavelength_m)


def relative_low_altitude_gain(
    wavelength_m: float,
    low_altitude_m: float,
    high_altitude_m: float,
) -> float:
    """Return amplitude at low altitude divided by amplitude at high altitude."""
    if high_altitude_m < low_altitude_m:
        raise ValueError("high_altitude_m must be >= low_altitude_m")
    low = upward_continuation_factor(wavelength_m, low_altitude_m)
    high = upward_continuation_factor(wavelength_m, high_altitude_m)
    if high == 0.0:
        return math.inf
    return low / high


def wavelength_for_retained_fraction(altitude_m: float, fraction: float) -> float:
    """Return wavelength whose amplitude retains ``fraction`` at altitude h."""
    if altitude_m < 0:
        raise ValueError("altitude_m must be non-negative")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0,1)")
    if altitude_m == 0.0:
        return 0.0
    return -2.0 * math.pi * altitude_m / math.log(fraction)
