#!/usr/bin/env python3
"""Proposal-facing diagnostic for the InSight mid-crust seismic constraint.

This is intentionally *not* a poroelastic inversion. It asks a narrower first
question: how do the homogeneous elastic moduli currently used by the Mars
reference/hydration models map into the Wright et al. (2024) Vp/Vs/rho
likelihood? A large mismatch demonstrates that hydration cannot be inferred by
simply lowering the solid shear modulus; porosity, pore shape, saturation and/or
metamorphic mineralogy must enter the forward model explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.mars import LAYER_MU_CRUST, MARS
from pylov3d.mars_hydration import K_CRUST, RATIO_SCENARIOS
from pylov3d.mars_seismic import (
    WRIGHT_2024_MIDCRUST,
    isotropic_velocities,
    moduli_from_velocities,
    seismic_chi2,
)
from scripts.mars_serpentinite_connectivity_sensitivity import mixed_moduli


def report(label: str, K: float, mu: float, rho: float) -> None:
    vp, vs = isotropic_velocities(K, mu, rho)
    chi2 = seismic_chi2(vp, vs, rho)
    print(
        f"{label:<28} K={K/1e9:7.2f} GPa mu={mu/1e9:7.2f} GPa "
        f"Vp={vp/1e3:5.2f} km/s Vs={vs/1e3:5.2f} km/s chi2={chi2:8.2f}"
    )


def main() -> int:
    c = WRIGHT_2024_MIDCRUST
    K_obs, mu_obs = moduli_from_velocities(c.vp_m_s, c.vs_m_s, c.rho_kg_m3)

    print("Mars InSight mid-crust seismic diagnostic")
    print(
        f"Wright et al. (2024): Vp={c.vp_m_s/1e3:.2f}+/-0.20 km/s, "
        f"Vs={c.vs_m_s/1e3:.2f}+/-0.30 km/s, rho={c.rho_kg_m3:.0f}+/-157 kg/m^3"
    )
    print(
        f"Equivalent homogeneous isotropic moduli at the observed mean: "
        f"K={K_obs/1e9:.2f} GPa, mu={mu_obs/1e9:.2f} GPa"
    )
    print("\nCurrent solid-modulus models evaluated at their adopted density:")
    report("global reference crust", K_CRUST, LAYER_MU_CRUST, float(MARS["crust_density"]))

    print("\nHydrated-modulus cases at f_h=0.5, using Wright density only as a diagnostic:")
    for scenario in ("low", "central", "high"):
        mu_ratio, K_ratio = RATIO_SCENARIOS[scenario]
        for law in ("voigt", "hill", "reuss"):
            mu, K = mixed_moduli(0.5, mu_ratio, K_ratio, law)
            report(f"{scenario}:{law}", K, mu, c.rho_kg_m3)

    print(
        "\nInterpretation: this diagnostic compares homogeneous elastic solids, not the "
        "Wright et al. poroelastic model. Persistent velocity misfit is therefore a "
        "requirement to add porosity/pore-shape/saturation and metamorphic mineralogy, "
        "not evidence against hydration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
