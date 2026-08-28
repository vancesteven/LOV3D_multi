#!/usr/bin/env python3
"""Love-number convergence report for a reduced radial artifact.

Implements the publication gate from docs/RADIAL_PROFILE_REDUCTION_2026-08-21.md:
reduce the same high-resolution profile to a sequence of target layer counts,
compute degree-2 Love numbers for each, and report how k2 moves with reduction
resolution. Pass a PlanetProfile radial artifact, or --synthetic to exercise
the machinery on the built-in Mars-like fixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pylov3d.constants import MAX_LAYERS
from pylov3d.profile_convergence import (
    love_number_convergence,
    successive_k2_differences,
    synthetic_mars_like_shells,
)
from pylov3d.profile_io import load_radial_artifact_shells
from pylov3d.types import make_forcing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--synthetic",
        type=int,
        metavar="N",
        default=None,
        help="use the built-in N-shell Mars-like fixture instead of a file",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12, 14, MAX_LAYERS],
        help="target layer counts to compare",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=44387.62,
        help="tidal forcing period in seconds (default: Mars solar semidiurnal)",
    )
    parser.add_argument("--Nrbase", type=int, default=100)
    args = parser.parse_args()

    if (args.profile is None) == (args.synthetic is None):
        parser.error("give exactly one of: a profile path, or --synthetic N")
    if args.synthetic is not None:
        shells = synthetic_mars_like_shells(args.synthetic)
        source = f"synthetic Mars-like fixture ({args.synthetic} shells)"
    else:
        shells = load_radial_artifact_shells(args.profile, enforce_max_layers=False)
        source = str(args.profile)

    forcing = make_forcing(Td=args.period_s, n=2, m=0, F=1.0)
    entries = love_number_convergence(
        shells, forcing, layer_counts=args.layers, Nrbase=args.Nrbase
    )
    diffs = successive_k2_differences(entries) if len(entries) > 1 else []

    print("pylov3d radial reduction Love-number convergence")
    print(f"profile: {source}")
    print(f"forcing: degree-2 order-0, period {args.period_s} s (elastic)")
    print(f"{'layers':>7} {'k2':>22} {'h2':>22} {'|dk2|/|k2|':>12}")
    for j, e in enumerate(entries):
        diff = f"{diffs[j - 1]:.3e}" if j > 0 else "-"
        print(
            f"{e.layers:>7} {e.k2.real:>22.9f} {e.h2.real:>22.9f} {diff:>12}"
        )
    print(
        "guard rail: choose a layer count only where |dk2|/|k2| is far below "
        "the science error budget; mass closure is exact but C/MR^2 and k2 "
        "are diagnostics, not preserved quantities"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
