#!/usr/bin/env python
"""Compare proposal-scale hydration coefficients with observed GMM-3 degree power."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_gmm3 import read_gmm3_shadr
from pylov3d.mars_gravity_background import (
    degree_coefficient_norm,
    degree_coefficient_rms,
)
from pylov3d.mars_gravity_coefficients import thin_sheet_potential_coefficient
from pylov3d.mars_gravity_normalization import orthonormal_to_gmm3_normalized

DEFAULT_PATH = ROOT / "data" / "external" / "gmm3_120_sha.tab"

SCENARIOS = (
    ("uncomp", 0.0, 0.0),
    ("half@50km", 0.5, 50e3),
    ("full@20km", 1.0, 20e3),
    ("full@100km", 1.0, 100e3),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument("--thickness-km", type=float, default=10.0)
    p.add_argument("--delta-rho", type=float, default=420.0)
    args = p.parse_args()

    if not args.path.exists():
        print(f"missing {args.path}")
        print("run scripts/mars_gmm3_formal_uncertainty.py --download first")
        return 2

    header, coeffs = read_gmm3_shadr(args.path)
    if header.normalization_state != 1:
        raise ValueError("GMM-3 diagnostic requires normalized coefficients")

    print("Mars hydration gravity versus observed GMM-3 degree amplitude")
    print(f"H_lm={args.thickness_km:.1f} km, |delta rho|={args.delta_rho:.0f} kg/m^3")
    print("GMM-3 columns use normalized coefficient-space degree norm/RMS")
    print()
    print("degree scenario       |signal|      degree norm       degree RMS   signal/norm   signal/RMS")

    for degree in (5, 11, 21, 43, 85):
        norm = degree_coefficient_norm(coeffs, degree)
        rms = degree_coefficient_rms(coeffs, degree)
        for name, comp, depth in SCENARIOS:
            q = thin_sheet_potential_coefficient(
                degree,
                args.thickness_km * 1e3,
                args.delta_rho,
                compensation_fraction=comp,
                compensation_depth_m=depth,
            )
            cbar = abs(orthonormal_to_gmm3_normalized(q, degree))
            print(
                f"{degree:6d} {name:12s} {cbar:12.4e} {norm:15.4e} {rms:15.4e} "
                f"{cbar/norm:12.4f} {cbar/rms:12.3f}"
            )
        print()

    print("interpretation guard rail:")
    print("  This compares one hypothetical harmonic with the entire observed gravity")
    print("  amplitude at that degree. It is not a residual or hydration detection.")
    print("  If the signal is a substantial fraction of degree power, geometry and")
    print("  geological attribution dominate over formal coefficient uncertainty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
