# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-scale electrical-conductivity links for Mars crustal models.

This module provides a generalized Archie-style connected-pore bound,

    sigma_eff = sigma_matrix + sigma_fluid * phi**m * Sw**n,

where phi is connected porosity, Sw is liquid saturation, m is the cementation
exponent, and n is a saturation exponent. The additive matrix term is included
so the expression remains useful as a sensitivity bound outside the ideal
insulating-matrix Archie limit.

This is not a clay/surface-conduction model and should not be interpreted as a
unique mapping from conductivity to water abundance. It is intended as a
bridge from the proposal's poroelastic seismic states to the existing EM
conductance/skin-depth sensitivity machinery until PlanetProfile-derived
mineral/fluid conductivity profiles are available.
"""

from __future__ import annotations


def archie_connected_pore_conductivity(
    porosity: float,
    saturation: float,
    fluid_conductivity_s_m: float,
    *,
    cementation_exponent: float = 2.0,
    saturation_exponent: float = 2.0,
    matrix_conductivity_s_m: float = 0.0,
) -> float:
    """Return an Archie-style effective conductivity in S/m."""
    if not 0.0 <= porosity <= 1.0:
        raise ValueError("porosity must lie in [0, 1]")
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("saturation must lie in [0, 1]")
    if fluid_conductivity_s_m < 0 or matrix_conductivity_s_m < 0:
        raise ValueError("conductivities must be non-negative")
    if cementation_exponent <= 0 or saturation_exponent <= 0:
        raise ValueError("Archie exponents must be positive")

    pore = (
        fluid_conductivity_s_m
        * porosity**cementation_exponent
        * saturation**saturation_exponent
    )
    return matrix_conductivity_s_m + pore


def layered_conductance_s(conductivity_s_m, thickness_m) -> float:
    """Return integrated conductance sum(sigma_i * H_i) in siemens."""
    sigmas = list(conductivity_s_m)
    thicknesses = list(thickness_m)
    if len(sigmas) != len(thicknesses):
        raise ValueError("conductivity and thickness arrays must have equal length")
    if any(s < 0 for s in sigmas) or any(h < 0 for h in thicknesses):
        raise ValueError("conductivities and thicknesses must be non-negative")
    return sum(s * h for s, h in zip(sigmas, thicknesses))
