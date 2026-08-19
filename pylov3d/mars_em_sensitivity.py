# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-scale electromagnetic sensitivity utilities for Mars.

This module provides geometry-light quantities useful for planning the Mars
joint-observable experiment: layer conductance and electromagnetic skin depth.
It deliberately does not map conductivity uniquely to liquid water.

For a uniform non-magnetic conductor,

    delta = sqrt(2 / (mu0 * sigma * omega))
          = sqrt(T / (pi * mu0 * sigma)).

The corresponding period for a target skin depth d is

    T = pi * mu0 * sigma * d^2.

Grimm (2002, JGR Planets, doi:10.1029/2001JE001504) showed that low-frequency
EM sounding is highly sensitive to liquid water on Mars because saline water
can be orders of magnitude more conductive than dry rock, while also warning
that conductive clays and ores are potential non-water explanations.  These
utilities are therefore intended for joint seismic/gravity/EM sensitivity
studies, not stand-alone water detection.
"""
from __future__ import annotations

import math

MU0 = 4.0e-7 * math.pi


def layer_conductance_siemens(conductivity_s_m: float, thickness_km: float) -> float:
    """Return sheet conductance ``sigma * H`` in siemens."""
    sigma = float(conductivity_s_m)
    H = float(thickness_km) * 1.0e3
    if sigma < 0 or H < 0:
        raise ValueError("conductivity and thickness must be non-negative")
    return sigma * H


def skin_depth_km(conductivity_s_m: float, period_s: float) -> float:
    """Return electromagnetic skin depth [km] for a uniform conductor."""
    sigma = float(conductivity_s_m)
    T = float(period_s)
    if sigma <= 0 or T <= 0:
        raise ValueError("conductivity and period must be positive")
    return math.sqrt(T / (math.pi * MU0 * sigma)) / 1.0e3


def period_for_skin_depth_s(conductivity_s_m: float, depth_km: float) -> float:
    """Return period [s] whose skin depth equals ``depth_km``."""
    sigma = float(conductivity_s_m)
    d = float(depth_km) * 1.0e3
    if sigma <= 0 or d <= 0:
        raise ValueError("conductivity and depth must be positive")
    return math.pi * MU0 * sigma * d * d


def conductivity_contrast(high_s_m: float, low_s_m: float) -> float:
    """Return dimensionless high/low conductivity contrast."""
    high = float(high_s_m)
    low = float(low_s_m)
    if high <= 0 or low <= 0:
        raise ValueError("conductivities must be positive")
    return high / low
