# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Compute and archive the TASK-031 Moon lateral Love-number spectrum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.moon import MOON, WEBER_K2_UNIFORM
from pylov3d.moon_lateral import (
    crustal_thickness_diagnostics,
    moon_lateral_love_spectrum,
)


DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "proposal" / "moon_lateral_spectrum.npz"


def _forcing_index(love) -> int:
    matches = np.where((love.n == love.nf) & (love.m == love.mf))[0]
    if len(matches) != 1:
        raise RuntimeError("forcing mode is missing or duplicated")
    return int(matches[0])


def _save_figure(path: Path, love, forcing_index: int, lmax: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(np.abs(love.k))[::-1][:20]
    labels = [f"({int(love.n[i])},{int(love.m[i]):+d})" for i in order]
    colors = ["#d95f02" if i == forcing_index else "#2c7fb8" for i in order]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(np.arange(len(order)), np.abs(love.k[order]), color=colors)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(order)), labels, rotation=60, ha="right")
    ax.set_ylabel(r"$|k_{nm}|$")
    ax.set_xlabel("response mode (n,m)")
    ax.set_title(f"Moon Airy-crust lateral Love spectrum (lmax={lmax})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--nrbase", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-lmax2",
        action="store_true",
        help="Skip the inexpensive lmax=2 angular-sensitivity comparison.",
    )
    args = parser.parse_args()

    result = moon_lateral_love_spectrum(lmax=args.lmax, Nrbase=args.nrbase)
    love = result["love"]
    forcing_index = _forcing_index(love)
    k_forcing = complex(love.k[forcing_index])
    delta_k = k_forcing - WEBER_K2_UNIFORM
    diagnostics = crustal_thickness_diagnostics(lmax=args.lmax)

    lmax2_shift = np.nan
    lmax2_modes = -1
    lmax2_wall_s = np.nan
    if not args.skip_lmax2:
        low = moon_lateral_love_spectrum(lmax=2, Nrbase=args.nrbase)
        low_love = low["love"]
        low_index = _forcing_index(low_love)
        lmax2_shift = abs(complex(low_love.k[low_index]) - WEBER_K2_UNIFORM)
        lmax2_modes = len(low_love.k)
        lmax2_wall_s = low["wall_s"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    degree_rms = diagnostics["degree_rms_km"]
    np.savez(
        args.output,
        n=np.asarray(love.n, dtype=int),
        m=np.asarray(love.m, dtype=int),
        k=np.asarray(love.k, dtype=complex),
        h=np.asarray(love.h, dtype=complex),
        l=np.asarray(love.l, dtype=complex),
        lmax=args.lmax,
        Nrbase=args.nrbase,
        perturbation_order=2,
        method="variable",
        mode_count=len(love.k),
        wall_s=result["wall_s"],
        k2_uniform=WEBER_K2_UNIFORM,
        k2_forcing=k_forcing,
        delta_k2=delta_k,
        delta_k2_over_sigma=abs(delta_k) / MOON["k2_sigma"],
        max_abs_dt_m=diagnostics["max_abs_dt_m"],
        max_abs_dmu_over_mubar=diagnostics["max_abs_dmu_over_mubar"],
        degree=np.asarray(sorted(degree_rms), dtype=int),
        degree_rms_km=np.asarray([degree_rms[n] for n in sorted(degree_rms)]),
        lmax2_delta_k2=lmax2_shift,
        lmax2_mode_count=lmax2_modes,
        lmax2_wall_s=lmax2_wall_s,
        degree_one_removed=False,  # PI decision 2026-08-14: dichotomy retained
        c20_removed=True,
    )
    _save_figure(args.output.with_suffix(".png"), love, forcing_index, args.lmax)

    off = [
        (abs(complex(k)), int(n), int(m), complex(k))
        for i, (n, m, k) in enumerate(zip(love.n, love.m, love.k))
        if i != forcing_index
    ]
    off.sort(reverse=True)
    print(f"saved {args.output} and {args.output.with_suffix('.png')}")
    print(f"N={len(love.k)}, wall={result['wall_s']:.1f} s")
    print(f"k2_uniform={WEBER_K2_UNIFORM:.15g}")
    print(f"k2_forcing={k_forcing.real:.15g}{k_forcing.imag:+.3e}j")
    print(f"delta_k2={delta_k.real:.6e}{delta_k.imag:+.3e}j")
    print(f"|delta_k2|/sigma_k2={abs(delta_k) / MOON['k2_sigma']:.4%}")
    print("top off-forcing modes:")
    for amplitude, n, m, value in off[:10]:
        print(f"  ({n},{m:+d}) |k|={amplitude:.6e} k={value.real:+.6e}{value.imag:+.6e}j")
    if not args.skip_lmax2:
        ratio = abs(delta_k) / lmax2_shift
        print(f"lmax=2: N={lmax2_modes}, |delta_k2|={lmax2_shift:.6e}; lmax4/lmax2={ratio:.3f}")


if __name__ == "__main__":
    main()
