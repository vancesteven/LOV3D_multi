#!/usr/bin/env python
"""Infer hydration-harmonic thickness scales from observed GMM-3 degree power."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_gmm3 import read_gmm3_shadr
from pylov3d.mars_gravity_background import degree_coefficient_norm, degree_coefficient_rms
from pylov3d.mars_gravity_coefficients import thin_sheet_potential_coefficient
from pylov3d.mars_gravity_degree_limits import thickness_for_degree_fraction, thickness_for_degree_rms
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
    p.add_argument("--trial-thickness-km", type=float, default=10.0)
    p.add_argument("--delta-rho", type=float, default=420.0)
    args = p.parse_args()

    if not args.path.exists():
        print(f"missing {args.path}")
        print("run scripts/mars_gmm3_formal_uncertainty.py --download first")
        return 2

    _, coeffs = read_gmm3_shadr(args.path)
    trial_h = args.trial_thickness_km * 1e3

    print("Mars hydration harmonic amplitudes relative to observed GMM-3 degree power")
    print(f"trial H_lm={args.trial_thickness_km:.1f} km, |delta rho|={args.delta_rho:.0f} kg/m^3")
    print("limits exploit linearity of the finite-shell gravity response")
    print()
    print("degree scenario       H@10%norm[km] H@100%norm[km] H@1xRMS[km]")

    for degree in (5, 11, 21, 43, 85):
        norm = degree_coefficient_norm(coeffs, degree)
        rms = degree_coefficient_rms(coeffs, degree)
        for name, comp, depth in SCENARIOS:
            q = thin_sheet_potential_coefficient(
                degree,
                trial_h,
                args.delta_rho,
                compensation_fraction=comp,
                compensation_depth_m=depth,
            )
            cbar = orthonormal_to_gmm3_normalized(q, degree)
            h10 = thickness_for_degree_fraction(trial_h, cbar, norm, 0.10)
            h100 = thickness_for_degree_fraction(trial_h, cbar, norm, 1.0)
            hrms = thickness_for_degree_rms(trial_h, cbar, rms)
            print(
                f"{degree:6d} {name:12s} {h10/1e3:14.4f} {h100/1e3:16.4f} {hrms/1e3:13.4f}"
            )
        print()

    print("interpretation guard rail:")
    print("  These are amplitude scales, not statistical upper limits. Geological")
    print("  contributions can add or cancel within a degree. H@10%norm is a useful")
    print("  proposal-scale reference for keeping a single hydration harmonic from")
    print("  dominating the observed Martian gravity field by construction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
