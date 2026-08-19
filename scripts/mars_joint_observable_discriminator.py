#!/usr/bin/env python3
"""Proposal-scale Mars joint-observable discriminator.

Compares two broad families against the same InSight Vp/Vs/rho likelihood:
  1. self-consistent hydrated-solid states with serpentinite density bracket;
  2. fractured-frame dry/saturated states from the proposal-scale poroelastic grid.

This script does not yet compute a full gravity anomaly or remanent field. It
identifies the seismically acceptable states whose density/composition differs,
which are the exact targets for the next gravity, remanence and EM sensitivity
steps.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.mars_joint_constraints import hydrated_solid_grid


def _load_poroelastic(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            rr = {k: v for k, v in r.items()}
            for key in ("phi", "Kd_over_Ks", "mud_over_mus", "vp_km_s", "vs_km_s", "rho_kg_m3", "chi2"):
                rr[key] = float(rr[key])
            rr["saturated"] = int(rr["saturated"])
            rows.append(rr)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--poroelastic-csv",
        type=Path,
        default=Path("data/tests/mars/mars_poroelastic_seismic_grid.csv"),
    )
    p.add_argument("--chi2-max", type=float, default=3.53)
    args = p.parse_args()

    if not args.poroelastic_csv.exists():
        raise SystemExit(
            f"missing {args.poroelastic_csv}; run scripts/mars_poroelastic_seismic_grid.py first"
        )

    solid = sorted(hydrated_solid_grid(), key=lambda s: s.chi2_seismic)
    poro = sorted(_load_poroelastic(args.poroelastic_csv), key=lambda r: r["chi2"])

    print("Mars joint-observable discriminator")
    print(f"seismic acceptance threshold: chi2 <= {args.chi2_max:.2f}\n")

    print("Best self-consistent hydrated-solid states:")
    print(" f_h prop     law    rho_case  Vp    Vs    rho   chi2")
    for s in solid[:12]:
        print(
            f" {s.f_h:3.2f} {s.property_scenario:<8} {s.connectivity:<6} {s.rho_scenario:<8} "
            f"{s.vp_m_s/1e3:5.3f} {s.vs_m_s/1e3:5.3f} {s.rho_kg_m3:5.0f} {s.chi2_seismic:6.3f}"
        )

    solid_good = [s for s in solid if s.chi2_seismic <= args.chi2_max]
    poro_good = [r for r in poro if r["chi2"] <= args.chi2_max]
    print(
        f"\naccepted counts: hydrated-solid={len(solid_good)}, "
        f"poroelastic={len(poro_good)}"
    )

    if solid_good:
        rho_s = [s.rho_kg_m3 for s in solid_good]
        print(
            "hydrated-solid accepted density range: "
            f"{min(rho_s):.0f}..{max(rho_s):.0f} kg/m^3"
        )
    if poro_good:
        rho_p = [r["rho_kg_m3"] for r in poro_good]
        print(
            "poroelastic accepted density range: "
            f"{min(rho_p):.0f}..{max(rho_p):.0f} kg/m^3"
        )

    if solid_good and poro_good:
        best = None
        for s in solid_good:
            for r in poro_good:
                dvp = s.vp_m_s / 1e3 - r["vp_km_s"]
                dvs = s.vs_m_s / 1e3 - r["vs_km_s"]
                if abs(dvp) > 0.10 or abs(dvs) > 0.15:
                    continue
                drho = abs(s.rho_kg_m3 - r["rho_kg_m3"])
                if best is None or drho > best[0]:
                    best = (drho, s, r, dvp, dvs)
        if best is not None:
            drho, s, r, dvp, dvs = best
            print("\nseismically similar cross-family pair with large density contrast:")
            print(
                f" hydrated solid: f_h={s.f_h:.2f} {s.property_scenario}/{s.connectivity}/"
                f"rho-{s.rho_scenario}, Vp={s.vp_m_s/1e3:.3f}, Vs={s.vs_m_s/1e3:.3f}, "
                f"rho={s.rho_kg_m3:.0f}, chi2={s.chi2_seismic:.3f}"
            )
            print(
                f" poroelastic: sat={r['saturated']} phi={r['phi']:.3f} "
                f"Kd/Ks={r['Kd_over_Ks']:.3f} mud/mus={r['mud_over_mus']:.3f}, "
                f"Vp={r['vp_km_s']:.3f}, Vs={r['vs_km_s']:.3f}, "
                f"rho={r['rho_kg_m3']:.0f}, chi2={r['chi2']:.3f}"
            )
            print(
                f" differences: dVp={dvp:+.3f} km/s dVs={dvs:+.3f} km/s "
                f"|drho|={drho:.0f} kg/m^3"
            )
            print(
                " next measurement test: convert this density contrast into a gravity/"
                "crust-thickness sensitivity, then compare remanence and conductivity."
            )
        else:
            print("\nno cross-family pair met the strict velocity-similarity filter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
