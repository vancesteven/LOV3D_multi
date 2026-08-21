#!/usr/bin/env python3
"""End-to-end synthetic Mars alteration-density gravity diagnostic.

The script synthesizes one real spherical-harmonic density anomaly in a finite
radial shell, sends it through the same map -> harmonic -> finite-shell path a
future PlanetThrak/PlanetProfile field will use, converts the result to GMM-3
normalization, and reports radial gravity at a chosen altitude.

It is a convention/physics diagnostic, not a claim about Mars hydration.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pylov3d.mars_alteration_gravity import (
    layered_density_gravity_coefficients,
    orthonormal_gravity_arrays_to_gmm3,
)
from pylov3d.mars_gravity_coefficients import (
    MARS_RADIUS_M,
    radial_gravity_from_coefficient,
)
from pylov3d.matlab_sph import stokes_to_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=11)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument(
        "--density-coeff",
        type=float,
        default=-250.0,
        help="unit-norm real cosine-harmonic density coefficient [kg/m3]",
    )
    parser.add_argument("--top-depth-km", type=float, default=0.0)
    parser.add_argument("--bottom-depth-km", type=float, default=20.0)
    parser.add_argument("--altitude-km", type=float, default=300.0)
    args = parser.parse_args()

    if args.degree < 1 or args.order < 0 or args.order > args.degree:
        raise SystemExit("require degree >= 1 and 0 <= order <= degree")
    if args.top_depth_km < 0 or args.bottom_depth_km <= args.top_depth_km:
        raise SystemExit("require 0 <= top depth < bottom depth")

    # Use enough bandwidth to represent the requested mode. The native grid is
    # 2*lmax by 4*lmax. A few degrees of headroom make leakage diagnostics clear.
    lmax = max(args.degree + 2, 4)
    c = np.zeros((lmax + 1, lmax + 1))
    s = np.zeros_like(c)
    # stokes_to_grid uses the 4pi-normalized real basis. Divide by sqrt(4pi)
    # so the physical unit-norm density coefficient equals the requested value.
    c[args.degree, args.order] = args.density_coeff / math.sqrt(4.0 * math.pi)
    _, _, density_map = stokes_to_grid(c, s, lmax)

    inner = MARS_RADIUS_M - args.bottom_depth_km * 1e3
    outer = MARS_RADIUS_M - args.top_depth_km * 1e3
    q_cos, q_sin = layered_density_gravity_coefficients(
        density_map[None, :, :],
        np.array([inner, outer]),
        lmax,
    )
    gmm_c, gmm_s = orthonormal_gravity_arrays_to_gmm3(q_cos, q_sin)

    q = q_cos[args.degree, args.order]
    g = radial_gravity_from_coefficient(
        args.degree,
        q,
        args.altitude_km * 1e3,
    )

    mask_c = q_cos.copy()
    mask_s = q_sin.copy()
    mask_c[args.degree, args.order] = 0.0
    if args.order > 0:
        mask_s[args.degree, args.order] = 0.0
    leakage = max(float(np.max(np.abs(mask_c))), float(np.max(np.abs(mask_s))))

    print("Mars synthetic 3D alteration-density gravity bridge")
    print(f"input density mode: l={args.degree} m={args.order} C={args.density_coeff:+.9g} kg/m3")
    print(f"shell depth: {args.top_depth_km:g} to {args.bottom_depth_km:g} km")
    print(f"native grid/lmax: {density_map.shape[0]}x{density_map.shape[1]} / {lmax}")
    print(f"orthonormal q_lm: {q:+.12e}")
    print(f"GMM-3 normalized C_lm: {gmm_c[args.degree, args.order]:+.12e}")
    if args.order > 0:
        print(f"GMM-3 normalized S_lm: {gmm_s[args.degree, args.order]:+.12e}")
    print(f"radial gravity at {args.altitude_km:g} km: {g*1e5:+.9g} mGal per unit Y_lm")
    print(f"max off-target q coefficient: {leakage:.3e}")
    print("guard rail: density attribution and covariance must be modeled before detectability claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
