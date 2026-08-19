#!/usr/bin/env python
"""Proposal-scale finite-shell spherical-harmonic gravity diagnostic for Mars."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_gravity_coefficients import gravity_from_thickness_coefficient
from pylov3d.mars_gravity_harmonics import MARS_RADIUS_M, wavelength_to_degree


def main() -> int:
    delta_rho = 420.0
    thickness_coeff = 10e3
    altitude = 300e3
    wavelengths_km = [500, 1000, 2000, 4000]
    compensation_depths_km = [20, 50, 100]
    compensation_fractions = [0.0, 0.5, 1.0]

    print("Mars finite-shell gravity coefficient discriminability")
    print(f"density contrast: |delta rho| = {delta_rho:.0f} kg/m^3")
    print(f"illustrative H_lm amplitude: {thickness_coeff/1e3:.1f} km")
    print(f"orbital altitude: {altitude/1e3:.0f} km")
    print()

    for wavelength in wavelengths_km:
        ell = wavelength_to_degree(wavelength * 1e3, radius_m=MARS_RADIUS_M)
        degree = max(1, int(round(ell)))
        print(f"lambda={wavelength:4d} km -> degree ~{ell:5.2f} (using l={degree})")
        g_un = gravity_from_thickness_coefficient(
            degree, thickness_coeff, delta_rho, altitude
        )
        print(f"  uncompensated: {abs(g_un)*1e5:9.3f} mGal")
        for depth_km in compensation_depths_km:
            vals = []
            for c in compensation_fractions[1:]:
                g = gravity_from_thickness_coefficient(
                    degree,
                    thickness_coeff,
                    delta_rho,
                    altitude,
                    compensation_fraction=c,
                    compensation_depth_m=depth_km * 1e3,
                )
                vals.append(abs(g) * 1e5)
            print(
                f"  comp depth={depth_km:3d} km: "
                f"c=0.5 -> {vals[0]:9.3f} mGal; "
                f"c=1.0 -> {vals[1]:9.3f} mGal"
            )
        print()

    print("interpretation guard rail:")
    print("  H_lm is an orthonormal spherical-harmonic thickness coefficient,")
    print("  not a globally uniform 10-km layer. Compensation is represented as")
    print("  an opposite column-mass harmonic at a chosen depth. This isolates")
    print("  the degree/depth physics before adopting a specific Mars C_lm/S_lm")
    print("  normalization and published gravity covariance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
