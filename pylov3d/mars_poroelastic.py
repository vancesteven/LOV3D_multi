# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-scale poroelastic bounds for the InSight mid-crust problem.

This module intentionally implements only the *Gassmann fluid-substitution*
step on a user-supplied drained frame.  It is not a replacement for the full
Berryman self-consistent crack/inclusion model used by Wright, Morzfeld &
Manga (2024).  Keeping the drained-frame moduli explicit lets proposal-facing
experiments separate two questions:

1. how compliant must the fractured/porous frame be?;
2. once that frame exists, how much can pore-fluid saturation change Vp?

For an isotropic, microhomogeneous Gassmann material the saturated bulk
modulus is

    Ksat = Kd + (1-Kd/Ks)^2 /
                 [phi/Kf + (1-phi)/Ks - Kd/Ks^2],

while the saturated shear modulus equals the drained shear modulus.  See
Berryman (2005), JGR Solid Earth, doi:10.1029/2004JB003576, and references
therein.  Heterogeneous/cracked rocks can violate the assumptions behind the
shear-invariance result, which is why this module is labelled a bound rather
than a complete Mars rock-physics model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mars_seismic import isotropic_velocities, seismic_chi2


@dataclass(frozen=True)
class PoroelasticState:
    """Effective saturated elastic state returned by the bounding model."""

    K_pa: float
    mu_pa: float
    rho_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    chi2: float


def gassmann_bulk_modulus(
    K_dry_pa: float,
    K_solid_pa: float,
    K_fluid_pa: float,
    porosity: float,
) -> float:
    """Return the Gassmann saturated bulk modulus [Pa].

    ``porosity=0`` returns ``K_solid`` only when the dry frame itself equals
    the solid modulus; for a prescribed weakened frame the phi->0 Gassmann
    expression tends toward the solid modulus.  We therefore require strictly
    positive porosity for fluid-substitution calculations and provide dry
    states separately in :func:`poroelastic_state`.
    """
    Kd = float(K_dry_pa)
    Ks = float(K_solid_pa)
    Kf = float(K_fluid_pa)
    phi = float(porosity)
    if not (0.0 < phi < 1.0):
        raise ValueError("Gassmann porosity must lie strictly between 0 and 1")
    if Kd <= 0 or Ks <= 0 or Kf <= 0:
        raise ValueError("bulk moduli must be positive")
    if Kd >= Ks:
        raise ValueError("drained-frame bulk modulus must be smaller than solid modulus")

    denom = phi / Kf + (1.0 - phi) / Ks - Kd / (Ks * Ks)
    if denom <= 0:
        raise ValueError("Gassmann denominator is non-positive for this state")
    Ksat = Kd + (1.0 - Kd / Ks) ** 2 / denom
    return float(Ksat)


def saturated_density(
    rho_solid_kg_m3: float,
    rho_fluid_kg_m3: float,
    porosity: float,
) -> float:
    """Simple volume-average density of a fully saturated porous material."""
    rho_s = float(rho_solid_kg_m3)
    rho_f = float(rho_fluid_kg_m3)
    phi = float(porosity)
    if rho_s <= 0 or rho_f < 0:
        raise ValueError("densities must be non-negative and solid density positive")
    if not (0.0 <= phi < 1.0):
        raise ValueError("porosity must lie in [0,1)")
    return float((1.0 - phi) * rho_s + phi * rho_f)


def poroelastic_state(
    *,
    K_solid_pa: float,
    mu_solid_pa: float,
    rho_solid_kg_m3: float,
    K_dry_pa: float,
    mu_dry_pa: float,
    porosity: float,
    saturated: bool,
    K_fluid_pa: float = 2.2e9,
    rho_fluid_kg_m3: float = 1000.0,
) -> PoroelasticState:
    """Evaluate a dry or fully fluid-saturated bounding state.

    The dry-frame moduli are explicit inputs.  This function does *not*
    prescribe how porosity or crack aspect ratio produces those moduli.
    ``saturated=True`` applies Gassmann to ``K_dry`` and leaves ``mu_dry``
    unchanged, as required by the microhomogeneous Gassmann limit.
    """
    Ks = float(K_solid_pa)
    mus = float(mu_solid_pa)
    rhos = float(rho_solid_kg_m3)
    Kd = float(K_dry_pa)
    mud = float(mu_dry_pa)
    phi = float(porosity)
    if mus <= 0 or mud < 0 or mud > mus:
        raise ValueError("require 0 <= mu_dry <= mu_solid with mu_solid > 0")
    if not (0.0 < phi < 1.0):
        raise ValueError("porosity must lie strictly between 0 and 1")

    if saturated:
        K_eff = gassmann_bulk_modulus(Kd, Ks, K_fluid_pa, phi)
        rho_eff = saturated_density(rhos, rho_fluid_kg_m3, phi)
    else:
        K_eff = Kd
        # Dry/gas pore density is neglected in this proposal-scale bound.
        rho_eff = (1.0 - phi) * rhos
    mu_eff = mud
    vp, vs = isotropic_velocities(K_eff, mu_eff, rho_eff)
    return PoroelasticState(
        K_pa=K_eff,
        mu_pa=mu_eff,
        rho_kg_m3=rho_eff,
        vp_m_s=vp,
        vs_m_s=vs,
        chi2=seismic_chi2(vp, vs, rho_eff),
    )
