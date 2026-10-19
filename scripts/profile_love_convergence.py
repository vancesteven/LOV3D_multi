#!/usr/bin/env python3
"""Report Love-number convergence for a radial PlanetProfile artifact."""

from __future__ import annotations

import argparse

from pylov3d.profile_convergence import love_number_reduction_convergence
from pylov3d.profile_io import load_radial_artifact_shells
from pylov3d.types import make_forcing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact")
    p.add_argument(
        "--layers",
        default="4,6,8,10,12,14,16",
        help="Comma-separated target layer counts; highest count is reference.",
    )
    p.add_argument("--period-s", type=float, default=44387.62)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--order", type=int, default=0)
    p.add_argument("--nrbase", type=int, default=100)
    p.add_argument("--body-radius-m", type=float)
    args = p.parse_args()

    counts = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    shells = load_radial_artifact_shells(
        args.artifact,
        body_radius_m=args.body_radius_m,
        enforce_max_layers=False,
    )
    forcing = make_forcing(
        Td=args.period_s,
        n=args.degree,
        m=args.order,
        F=1.0,
    )
    out = love_number_reduction_convergence(
        shells,
        counts,
        forcing,
        Nrbase=args.nrbase,
    )

    print(f"reference layers: {out.reference_layers}")
    print(
        "layers"
        "            Re(k)"
        "            Im(k)"
        "        rel|dk|"
        "        rel|dh|"
        "        rel|dl|"
    )
    for point in out.points:
        print(
            f"{point.actual_layers:6d}"
            f" {point.k.real:16.9e}"
            f" {point.k.imag:16.9e}"
            f" {point.rel_dk:14.6e}"
            f" {point.rel_dh:14.6e}"
            f" {point.rel_dl:14.6e}"
        )


if __name__ == "__main__":
    main()
