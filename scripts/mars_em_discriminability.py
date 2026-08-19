#!/usr/bin/env python3
"""Proposal-facing Mars electromagnetic sensitivity diagnostic.

This script does not assign a unique conductivity to either hydration
hypothesis.  Instead it reports the periods and conductances corresponding to a
broad conductivity sweep for a target mid-crustal depth.  Those quantities can
be compared directly with a future Mars EM instrument/source concept.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pylov3d.mars_em_sensitivity import (  # noqa: E402
    layer_conductance_siemens,
    period_for_skin_depth_s,
    skin_depth_km,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--depth-km", type=float, default=15.0)
    p.add_argument("--thickness-km", type=float, default=10.0)
    p.add_argument(
        "--conductivities",
        default="1e-5,1e-4,1e-3,1e-2,1e-1,1",
        help="comma-separated S/m sweep; generic sensitivity values, not lithology labels",
    )
    args = p.parse_args()
    sigmas = [float(x) for x in args.conductivities.split(",")]

    print("Mars EM discriminability bound")
    print(f"target skin depth: {args.depth_km:g} km")
    print(f"illustrative layer thickness: {args.thickness_km:g} km")
    print("sigma [S/m]   conductance [S]   period for target depth   skin depth at 100 s")
    for sigma in sigmas:
        conductance = layer_conductance_siemens(sigma, args.thickness_km)
        period = period_for_skin_depth_s(sigma, args.depth_km)
        depth100 = skin_depth_km(sigma, 100.0)
        print(
            f"{sigma:10.1e}   {conductance:14.3g}   "
            f"{period:12.3g} s   {depth100:10.3f} km"
        )

    print("\ninterpretation guard rail:")
    print("  conductivity is a present-state observable but not a unique water proxy;")
    print("  saline pore water can be highly conductive, while clays, ores and")
    print("  alteration products can also produce conductive anomalies.")
    print("  The proposal should therefore use EM jointly with seismic, gravity,")
    print("  remanence and tides, and later replace this bound with PlanetProfile")
    print("  conductivity profiles plus a Mars-specific EM/induction observation model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
