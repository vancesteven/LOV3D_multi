# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-scale gravity sensitivity for Mars hydration hypotheses.

This module converts a density contrast between seismically similar crustal
models into a first-order gravity signal.  It is intentionally a *sensitivity
bound*, not a replacement for spherical-harmonic crustal inversion.

For a laterally extensive uncompensated layer, the Bouguer/infinite-slab
vertical acceleration is

    dg = 2*pi*G*drho*H.

For a sinusoidal horizontal density anomaly with wavelength lambda observed at
altitude z, the planar thin-sheet approximation attenuates the signal by
exp(-2*pi*z/lambda).  This provides a transparent bridge between a local column
mass contrast and the orbital spatial scales relevant to a future gravity
investigation.

The final proposal analysis should replace these planar bounds with the same
spherical-harmonic gravity/topography machinery used for the Mars crustal
model.  Wieczorek (2022, JGR Planets, doi:10.1029/2022JE007298) demonstrates
why seismic crustal constraints and gravity are naturally complementary on
Mars.
"""
from __future__ import annotations

import math

G_SI = 6.67430e-11
M_S2_PER_MGAL = 1.0e-5


def slab_gravity_mgal(delta_rho_kg_m3: float, thickness_km: float) -> float:
    """Return uncompensated infinite-slab gravity contrast in mGal."""
    drho = float(delta_rho_kg_m3)
    H = float(thickness_km) * 1.0e3
    if H < 0:
        raise ValueError("thickness must be non-negative")
    return 2.0 * math.pi * G_SI * drho * H / M_S2_PER_MGAL


def sinusoidal_sheet_gravity_mgal(
    delta_rho_kg_m3: float,
    thickness_km: float,
    wavelength_km: float,
    altitude_km: float,
) -> float:
    """Planar thin-sheet gravity amplitude at altitude, in mGal.

    The exponential factor is the standard upward continuation of a horizontal
    Fourier mode.  It should be interpreted as a scale estimate only when the
    anomaly thickness is small compared with its lateral wavelength.
    """
    wavelength = float(wavelength_km)
    altitude = float(altitude_km)
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if altitude < 0:
        raise ValueError("altitude must be non-negative")
    surface = slab_gravity_mgal(delta_rho_kg_m3, thickness_km)
    attenuation = math.exp(-2.0 * math.pi * altitude / wavelength)
    return surface * attenuation


def thickness_for_gravity_mgal(delta_rho_kg_m3: float, target_mgal: float) -> float:
    """Return uncompensated layer thickness [km] that yields ``target_mgal``."""
    drho = abs(float(delta_rho_kg_m3))
    target = abs(float(target_mgal))
    if drho <= 0:
        raise ValueError("density contrast magnitude must be positive")
    return target * M_S2_PER_MGAL / (2.0 * math.pi * G_SI * drho) / 1.0e3
