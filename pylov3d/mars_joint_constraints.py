# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
"""Proposal-facing joint constraints for Mars hydration hypotheses.

This module adds the missing density leg to the preliminary hydrated-solid
seismic experiment. Earlier diagnostics evaluated hydrated elastic moduli at
the *observed* Wright et al. (2024) density to isolate elastic effects. That
was useful, but it prevented density/gravity from acting as an independent
observable.

Here we bracket the grain-scale density of strongly serpentinized ultramafic
rock at 2.5--2.7 g cm^-3, with 2.6 g cm^-3 as a central proposal value. These
are deliberately broad proposal-scale endmembers, consistent with the
serpentinite ranges already used in the SSS proposal and with published
serpentinized-peridotite measurements. They are *not* a replacement for the
planned Perple_X mineralogical calculation.

The mean hydrated crust density is mixed volumetrically with the shipped Mars
reference-crust density (2900 kg m^-3). This allows the same state to predict
rho, Vp, Vs and tidal elastic moduli rather than inserting the seismic density
by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mars import LAYER_MU_CRUST, MARS
from .mars_hydration import K_CRUST, RATIO_SCENARIOS
from .mars_seismic import isotropic_velocities, seismic_chi2


RHO_CRUST_KG_M3 = float(MARS["crust_density"])
RHO_SERPENTINITE_KG_M3 = {
    "low": 2500.0,
    "central": 2600.0,
    "high": 2700.0,
}


@dataclass(frozen=True)
class HydratedSolidState:
    property_scenario: str
    connectivity: str
    rho_scenario: str
    f_h: float
    K_pa: float
    mu_pa: float
    rho_kg_m3: float
    vp_m_s: float
    vs_m_s: float
    chi2_seismic: float


def mixed_moduli(
    f_h: float,
    mu_ratio: float,
    K_ratio: float,
    law: str,
) -> tuple[float, float]:
    """Return effective ``(mu, K)`` [Pa] for a hydrated two-phase mixture.

    Kept inside the package rather than importing the proposal diagnostic
    script, so library modules never depend on ``scripts/``. Voigt is the
    iso-strain upper bound, Reuss the iso-stress lower bound, and Hill their
    arithmetic midpoint.
    """
    f = float(f_h)
    if not 0.0 <= f <= 1.0:
        raise ValueError("f_h must lie in [0,1]")
    if mu_ratio <= 0 or K_ratio <= 0:
        raise ValueError("endmember modulus ratios must be positive")

    mu_dry = float(LAYER_MU_CRUST)
    K_dry = float(K_CRUST)
    mu_wet = float(mu_ratio) * mu_dry
    K_wet = float(K_ratio) * K_dry

    mu_v = (1.0 - f) * mu_dry + f * mu_wet
    K_v = (1.0 - f) * K_dry + f * K_wet
    if law == "voigt":
        return mu_v, K_v

    mu_r = 1.0 / ((1.0 - f) / mu_dry + f / mu_wet)
    K_r = 1.0 / ((1.0 - f) / K_dry + f / K_wet)
    if law == "reuss":
        return mu_r, K_r
    if law == "hill":
        return 0.5 * (mu_v + mu_r), 0.5 * (K_v + K_r)
    raise ValueError(f"unknown mixing law: {law}")


def mixed_density(
    f_h: float,
    rho_serp_kg_m3: float = RHO_SERPENTINITE_KG_M3["central"],
    rho_crust_kg_m3: float = RHO_CRUST_KG_M3,
) -> float:
    """Volume-average density [kg m^-3] for hydrated fraction ``f_h``."""
    f = float(f_h)
    if not 0.0 <= f <= 1.0:
        raise ValueError("f_h must lie in [0,1]")
    if rho_serp_kg_m3 <= 0 or rho_crust_kg_m3 <= 0:
        raise ValueError("densities must be positive")
    return (1.0 - f) * float(rho_crust_kg_m3) + f * float(rho_serp_kg_m3)


def hydrated_solid_state(
    f_h: float,
    property_scenario: str,
    connectivity: str,
    rho_scenario: str = "central",
) -> HydratedSolidState:
    """Return a self-consistent proposal-scale hydrated-solid seismic state."""
    if property_scenario not in RATIO_SCENARIOS:
        raise ValueError(f"unknown property scenario: {property_scenario}")
    if rho_scenario not in RHO_SERPENTINITE_KG_M3:
        raise ValueError(f"unknown density scenario: {rho_scenario}")
    mu_ratio, K_ratio = RATIO_SCENARIOS[property_scenario]
    mu_eff, K_eff = mixed_moduli(float(f_h), mu_ratio, K_ratio, connectivity)
    rho = mixed_density(float(f_h), RHO_SERPENTINITE_KG_M3[rho_scenario])
    vp, vs = isotropic_velocities(K_eff, mu_eff, rho)
    chi2 = seismic_chi2(vp, vs, rho)
    return HydratedSolidState(
        property_scenario=property_scenario,
        connectivity=connectivity,
        rho_scenario=rho_scenario,
        f_h=float(f_h),
        K_pa=float(K_eff),
        mu_pa=float(mu_eff),
        rho_kg_m3=float(rho),
        vp_m_s=float(vp),
        vs_m_s=float(vs),
        chi2_seismic=float(chi2),
    )


def hydrated_solid_grid(f_values=(0.1, 0.25, 0.5, 0.75)) -> list[HydratedSolidState]:
    """Evaluate property/connectivity/density brackets on a small f_h grid."""
    out: list[HydratedSolidState] = []
    for f_h in f_values:
        for prop in ("low", "central", "high"):
            for law in ("voigt", "hill", "reuss"):
                for rho_case in ("low", "central", "high"):
                    out.append(hydrated_solid_state(f_h, prop, law, rho_case))
    return out
