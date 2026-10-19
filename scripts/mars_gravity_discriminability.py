#!/usr/bin/env python3
"""Proposal-facing gravity discriminability for the Mars seismic-degeneracy pair.

Uses the 420 kg/m^3 density contrast identified by
``mars_joint_observable_discriminator.py`` and converts it into transparent
Bouguer/slab and upward-continued sinusoidal-sheet gravity amplitudes.

These are first-order sensitivity bounds, not a final Mars spherical-harmonic
forward model.  Their purpose is to determine whether gravity is potentially a
strong or weak degeneracy breaker before investing in a full geometry-specific
calculation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pylov3d.mars_gravity_sensitivity import (  # noqa: E402
    slab_gravity_mgal,
    sinusoidal_sheet_gravity_mgal,
    thickness_for_gravity_mgal,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--delta-rho", type=float, default=420.0, help="kg/m^3")
    p.add_argument("--altitude-km", type=float, default=300.0)
    args = p.parse_args()

    drho = args.delta_rho
    print("Mars gravity discriminability bound")
    print(f"cross-family density contrast: |delta rho| = {abs(drho):.0f} kg/m^3")
    print("uncompensated laterally extensive layer:")
    for H in (0.1, 0.5, 1.0, 5.0, 10.0, 20.0):
        print(f"  H={H:5.1f} km -> |delta g|={abs(slab_gravity_mgal(drho,H)):8.3f} mGal")

    print(f"\nplanar upward continuation to {args.altitude_km:g} km altitude for H=10 km:")
    for wavelength in (250.0, 500.0, 1000.0, 2000.0, 4000.0):
        dg = abs(sinusoidal_sheet_gravity_mgal(drho, 10.0, wavelength, args.altitude_km))
        print(f"  wavelength={wavelength:6.0f} km -> |delta g|={dg:8.3f} mGal")

    print("\nequivalent uncompensated thickness needed for target surface signal:")
    for target in (0.1, 1.0, 10.0):
        H = thickness_for_gravity_mgal(drho, target)
        print(f"  {target:4.1f} mGal -> H={H*1e3:7.1f} m")

    print("\ninterpretation guard rail:")
    print("  these are planar, uncompensated sensitivity bounds; actual Mars gravity")
    print("  depends on anomaly geometry, compensation, wavelength, altitude and covariance")
    print("  with crustal thickness. The next rung should use spherical harmonics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
