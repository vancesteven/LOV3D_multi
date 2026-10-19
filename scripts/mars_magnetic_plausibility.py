#!/usr/bin/env python3
"""Proposal-facing Mars magnetite/remanence plausibility diagnostic.

Prints the Bultel et al. (2025) magnetite requirements for the InSight field
and strongest orbital anomaly, plus simple paleofield scaling.  This is not a
present-hydration inversion; it is a nuisance-aware benchmark for future
coupling to geochemical magnetite production.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pylov3d.mars_magnetic import (  # noqa: E402
    INSIGHT_BENCHMARKS,
    ORBITAL_BENCHMARKS,
    paleofield_scaled_required_magnetite,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paleofields", default="25,50,100", help="comma-separated microtesla values")
    args = p.parse_args()
    paleofields = [float(x) for x in args.paleofields.split(",")]

    print("Mars remanent-magnetization plausibility benchmarks")
    print("Bultel et al. (2025) Table 1; quoted magnetite abundances are minimum values at 50 uT")

    for title, group in (("InSight surface field", INSIGHT_BENCHMARKS),
                         ("Strongest orbital anomaly", ORBITAL_BENCHMARKS)):
        print(f"\n{title}:")
        for b in group:
            scaled = ", ".join(
                f"{B:g}uT->{paleofield_scaled_required_magnetite(b, B):.2f} wt%"
                for B in paleofields
            )
            print(
                f"{b.label:<22} depth={b.depth_top_km:g}-{b.depth_bottom_km:g} km "
                f"M={b.magnetization_A_m:g} A/m  ref={b.magnetite_wt_percent:.2f} wt%  {scaled}"
            )

    print("\nInterpretation guard rail:")
    print("  remanence constrains alteration/mineral production plus paleofield/source geometry;")
    print("  it is not a direct measure of present hydration or pore water.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
