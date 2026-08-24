#!/usr/bin/env python3
"""Inspect and explicitly reduce a high-resolution radial profile artifact.

This command reports mass and C/MR^2 changes caused by the reduction. It does
not claim tidal convergence; run Love-number calculations at multiple target
layer counts before selecting a publication model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pylov3d.constants import MAX_LAYERS
from pylov3d.profile_reduction import reduce_radial_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--layers", type=int, default=MAX_LAYERS)
    args = parser.parse_args()

    _, diag = reduce_radial_artifact(args.profile, target_layers=args.layers)
    print("pylov3d radial artifact reduction")
    print(f"profile: {args.profile}")
    print(f"layers: {diag.original_layers} -> {diag.reduced_layers}")
    print(f"mass original: {diag.original.mass_kg:.12e} kg")
    print(f"mass reduced:  {diag.reduced.mass_kg:.12e} kg")
    print(f"relative mass change: {diag.mass_relative_change:+.12e}")
    print(f"C/MR^2 original: {diag.original.cmr2:.12g}")
    print(f"C/MR^2 reduced:  {diag.reduced.cmr2:.12g}")
    print(f"delta C/MR^2:     {diag.cmr2_change:+.12e}")
    if diag.original.mass_relative_error is not None:
        print(f"original mass vs metadata: {diag.original.mass_relative_error:+.12e}")
    if diag.reduced.mass_relative_error is not None:
        print(f"reduced mass vs metadata:  {diag.reduced.mass_relative_error:+.12e}")
    if diag.original.cmr2_error is not None:
        print(f"original C/MR^2 vs metadata: {diag.original.cmr2_error:+.12e}")
    if diag.reduced.cmr2_error is not None:
        print(f"reduced C/MR^2 vs metadata:  {diag.reduced.cmr2_error:+.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
