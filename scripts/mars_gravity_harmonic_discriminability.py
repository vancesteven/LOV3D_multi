#!/usr/bin/env python3
"""Degree/altitude gravity sensitivity for the Mars seismic-degeneracy pair."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_gravity_harmonics import (
    degree_from_wavelength,
    harmonic_gravity_bound,
    radial_gravity_attenuation,
    thin_sheet_surface_gravity,
)


def main() -> int:
    delta_rho = 420.0
    thickness = 10e3
    wavelengths_km = (250, 500, 1000, 2000, 4000)
    altitudes_km = (100, 300, 500)

    print("Mars spherical-harmonic gravity discriminability bound")
    print()
    print(f"cross-family density contrast: |delta rho| = {delta_rho:.0f} kg/m^3")
    print(f"illustrative anomalous thickness: H = {thickness/1e3:.1f} km")
    print(f"surface thin-sheet scale: {thin_sheet_surface_gravity(delta_rho, thickness)/1e-5:.3f} mGal")
    print()
    print("lambda[km]  degree   attenuation @100/300/500 km    gravity @100/300/500 km [mGal]")
    for wavelength_km in wavelengths_km:
        degree = degree_from_wavelength(wavelength_km * 1e3)
        atten = [radial_gravity_attenuation(degree, h * 1e3) for h in altitudes_km]
        grav = [harmonic_gravity_bound(delta_rho, thickness, degree, h * 1e3) / 1e-5 for h in altitudes_km]
        print(
            f"{wavelength_km:9.0f} {degree:7.2f}   "
            f"{atten[0]:8.4f} {atten[1]:8.4f} {atten[2]:8.4f}      "
            f"{grav[0]:9.3f} {grav[1]:9.3f} {grav[2]:9.3f}"
        )

    print()
    print("interpretation guard rail:")
    print("  radial attenuation is exact for a degree-l exterior harmonic, but the")
    print("  surface amplitude still uses an uncompensated thin-sheet bound. The next")
    print("  rung should generate finite-shell C_lm/S_lm coefficients and compare them")
    print("  with a Mars gravity covariance under explicit compensation assumptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
