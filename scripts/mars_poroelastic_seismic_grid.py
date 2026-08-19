#!/usr/bin/env python3
"""Proposal-facing Gassmann bounding grid for the InSight mid-crust.

This is intentionally not a reproduction of Wright et al. (2024)'s full
Berryman self-consistent crack model.  It scans *prescribed* dry-frame bulk
and shear modulus fractions, porosity, and dry-versus-water-saturated state
against the same Vp/Vs/rho likelihood used by pylov3d.mars_seismic.

The output answers a narrower but useful question: once a compliant frame is
allowed, how strongly can saturation move the model through the InSight
observable space, and how degenerate are frame compliance and hydration?
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pylov3d.mars_poroelastic import poroelastic_state  # noqa: E402
from pylov3d.mars_seismic import WRIGHT_2024_MIDCRUST  # noqa: E402


K_SOLID = 70e9
MU_SOLID = 30e9
RHO_SOLID = 2900.0


def run() -> list[dict]:
    rows: list[dict] = []
    for phi in np.linspace(0.05, 0.30, 11):
        for fK in np.linspace(0.15, 0.75, 13):
            for fmu in np.linspace(0.20, 0.80, 13):
                for saturated in (False, True):
                    state = poroelastic_state(
                        K_solid_pa=K_SOLID,
                        mu_solid_pa=MU_SOLID,
                        rho_solid_kg_m3=RHO_SOLID,
                        K_dry_pa=fK * K_SOLID,
                        mu_dry_pa=fmu * MU_SOLID,
                        porosity=float(phi),
                        saturated=saturated,
                    )
                    rows.append({
                        "porosity": float(phi),
                        "Kdry_over_Ksolid": float(fK),
                        "mudry_over_musolid": float(fmu),
                        "saturated": int(saturated),
                        "K_eff_GPa": state.K_pa / 1e9,
                        "mu_eff_GPa": state.mu_pa / 1e9,
                        "rho_kg_m3": state.rho_kg_m3,
                        "Vp_km_s": state.vp_m_s / 1e3,
                        "Vs_km_s": state.vs_m_s / 1e3,
                        "chi2": state.chi2,
                    })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    rows = run()
    rows_sorted = sorted(rows, key=lambda r: r["chi2"])

    obs = WRIGHT_2024_MIDCRUST
    print("Mars proposal-scale poroelastic seismic grid")
    print(
        f"target: Vp={obs.vp_m_s/1e3:.2f} km/s, Vs={obs.vs_m_s/1e3:.2f} km/s, "
        f"rho={obs.rho_kg_m3:.0f} kg/m^3"
    )
    print("matrix bound: K=70 GPa, mu=30 GPa, rho=2900 kg/m^3")
    print("dry-frame moduli are free bounds; saturated rows use Gassmann with water-like Kf=2.2 GPa")
    print("\nbest states:")
    print(" sat  phi   Kd/Ks  mud/mus   Vp    Vs    rho    chi2")
    for r in rows_sorted[: args.top]:
        print(
            f"  {r['saturated']:d}  {r['porosity']:.3f}  {r['Kdry_over_Ksolid']:.3f}   "
            f"{r['mudry_over_musolid']:.3f}   {r['Vp_km_s']:.3f} {r['Vs_km_s']:.3f} "
            f"{r['rho_kg_m3']:.0f}  {r['chi2']:.3f}"
        )

    for sat in (0, 1):
        subset = [r for r in rows_sorted if r["saturated"] == sat]
        n1 = sum(r["chi2"] <= 3.53 for r in subset)  # ~68% joint region for 3 dof
        n2 = sum(r["chi2"] <= 8.02 for r in subset)  # ~95% joint region for 3 dof
        best = subset[0]
        print(
            f"\n{'saturated' if sat else 'dry'}: best chi2={best['chi2']:.3f}; "
            f"grid points within chi2<=3.53: {n1}, <=8.02: {n2}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("\nsaved:", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
