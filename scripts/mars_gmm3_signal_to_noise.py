#!/usr/bin/env python
"""Compare proposal-scale hydration gravity coefficients with GMM-3 errors.

This diagnostic is intentionally diagonal-only.  It converts the finite-shell
unit-norm q_lm coefficient into the normalized PDS/GMM-3 C_lm/S_lm convention,
then compares its magnitude with the formal SHADR coefficient uncertainties.
Final mission requirements must use the SHBDR covariance and a physical spatial
pattern rather than a single arbitrary order at each degree.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_gmm3 import formal_sigmas_at_degree, read_gmm3_shadr
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
        raise ValueError("GMM-3 diagnostic requires normalized SHADR coefficients")

    print("Mars hydration gravity versus GMM-3 formal errors")
    print(f"H_lm={args.thickness_km:.1f} km, |delta rho|={args.delta_rho:.0f} kg/m^3")
    print("signal converted from unit-norm pylov3d q_lm to PDS normalized C_lm/S_lm")
    print("SNR values use 3x formal errors; 'worst' uses the largest sigma at that degree")
    print()
    print("degree scenario       |signal|       3x med sigma    SNR_med    3x max sigma    SNR_worst")

    for degree in (5, 11, 21, 43, 85):
        sigmas = formal_sigmas_at_degree(coeffs, degree)
        med = statistics.median(sigmas)
        mx = max(sigmas)
        for name, comp, depth in SCENARIOS:
            q = thin_sheet_potential_coefficient(
                degree,
                args.thickness_km * 1e3,
                args.delta_rho,
                compensation_fraction=comp,
                compensation_depth_m=depth,
            )
            cbar = orthonormal_to_gmm3_normalized(q, degree)
            signal = abs(cbar)
            snr_med = signal / (3.0 * med)
            snr_worst = signal / (3.0 * mx)
            print(
                f"{degree:6d} {name:12s} {signal:12.4e} {3*med:14.4e} "
                f"{snr_med:10.2f} {3*mx:14.4e} {snr_worst:10.2f}"
            )
        print()

    print("interpretation guard rail:")
    print("  This is a coefficient-space scale test, not a detection claim. A real")
    print("  hydration pattern distributes power over multiple m values and degrees;")
    print("  full GMM-3 covariance, topography/compensation covariance, and a specified")
    print("  spatial model are required before quoting current or MaQuIs requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
