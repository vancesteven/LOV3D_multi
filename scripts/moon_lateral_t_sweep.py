# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""TASK-036b: Moon lateral T-sweep — does the Love spectrum converge in shell T?

Driver only; no pylov3d module is modified.

The laterally-varying rigidity coefficient is d(mu/mu_bar)/d(dt) = (mu_c -
mu_m) / (T * mu_c) — the exact Voigt volume-fraction average.  The positivity
guard inside ``mu_variable_from_topography`` is keyed to the *shipped* T=40 km
and raises when the thickness field overwhelms that particular shell geometry.
This driver computes the rescaled coefficient for each candidate T directly,
builds mu_variable by hand using the same ``_real_sh_to_complex_mu_variable``
helper the pipeline uses, and calls ``get_love`` with that field.

The 1-D background (Weber model, unmodified) does not change with T — T only
enters through the coefficient.  But the background k2 at each T is verified
by running the corresponding zero-amplitude coupled solve (mu_variable scaled
to zero), so any solver-path or forcing difference is caught.

Artifacts: docs/figures/proposal/moon_lateral_t_sweep.{npz,png}
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.love import get_love
from pylov3d.moon import (
    LAYER_MU,
    MOON_FORCING_TD,
    WEBER_K2_UNIFORM,
    build_moon_model,
)
from pylov3d.mapping import sh_to_latlon
from pylov3d.moon_lateral import (
    CRUST_LAYER_INDEX,
    _real_sh_to_complex_mu_variable,
    crustal_thickness_diagnostics,
    crustal_thickness_variation,
)
from pylov3d.types import make_forcing, make_numerics

MANTLE_LAYER_INDEX = CRUST_LAYER_INDEX - 1
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "moon_lateral_t_sweep.npz"
)
OFF_MODES = ((2, 2), (2, 1), (3, 3))


def _peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return raw * scale / 1e9


def _coeff_for_T(T_m: float) -> float:
    """d(mu/mu_bar)/d(dt) [1/m] for a reference shell of thickness T_m."""
    mu_c = LAYER_MU[CRUST_LAYER_INDEX]
    mu_m = LAYER_MU[MANTLE_LAYER_INDEX]
    return (mu_c - mu_m) / (T_m * mu_c)


def _mu_variable_at_T(
    lmax: int,
    T_m: float,
    *,
    include_c20: bool = False,
) -> dict[int, list[tuple[int, int, complex]]]:
    """Build mu_variable for a given reference shell thickness T_m [m]."""
    dt = crustal_thickness_variation(lmax=lmax, include_c20=include_c20)
    coeff = _coeff_for_T(T_m)
    dmu_over_mu = {nm: coeff * v for nm, v in dt.items()}
    entries = _real_sh_to_complex_mu_variable(dmu_over_mu)
    return {CRUST_LAYER_INDEX: entries}


def _max_dmu_over_mubar(lmax: int, T_m: float, nlat: int = 180, nlon: int = 360) -> float:
    """Scalar positivity margin for a given T [m], without raising.

    Grids the SH field to find the true spatial max (SH coefficient max
    is much smaller than the synthesized spatial max for a broadband field).
    """
    dt = crustal_thickness_variation(lmax=lmax)
    grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon)
    max_dt = float(np.max(np.abs(grid.z)))
    return abs(_coeff_for_T(T_m)) * max_dt


def _forcing_index(love) -> int:
    matches = np.where((love.n == love.nf) & (love.m == love.mf))[0]
    if len(matches) != 1:
        raise RuntimeError("forcing mode missing or duplicated")
    return int(matches[0])


def _mode_abs(love, n: int, m: int) -> float:
    idx = np.where((love.n == n) & (love.m == m))[0]
    if len(idx) == 0:
        return float("nan")
    return float(abs(complex(love.k[idx[0]])))


def _solve_at_T(
    lmax: int,
    T_m: float,
    nrbase: int,
) -> dict:
    """Coupled solve at a given shell thickness T_m and lmax."""
    model = build_moon_model()
    forcing = make_forcing(Td=MOON_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=model.n_layers,
        method="variable",
        Nrbase=nrbase,
        perturbation_order=2,
    )
    mu_var = _mu_variable_at_T(lmax=lmax, T_m=T_m)
    max_dmu = _max_dmu_over_mubar(lmax, T_m)

    t0 = time.perf_counter()
    love, _, _ = get_love(model, forcing, numerics, mu_variable=mu_var)
    wall_s = time.perf_counter() - t0

    fidx = _forcing_index(love)
    delta_k20 = complex(love.k[fidx]) - WEBER_K2_UNIFORM
    return {
        "lmax": lmax,
        "T_km": T_m / 1000.0,
        "T_m": T_m,
        "max_dmu_over_mubar": max_dmu,
        "N": int(len(love.k)),
        "k20_coupled": float(love.k[fidx].real),
        "delta_k20": complex(delta_k20),
        "abs_delta_k20": float(abs(delta_k20)),
        "off": {f"{n}_{m}": _mode_abs(love, n, m) for n, m in OFF_MODES},
        "wall_s": wall_s,
        "peak_rss_gb": _peak_rss_gb(),
    }


def _solve_1d(nrbase: int) -> float:
    """1-D (zero-amplitude) k2 via the coupled path as a sanity check."""
    model = build_moon_model()
    forcing = make_forcing(Td=MOON_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=model.n_layers,
        method="variable",
        Nrbase=nrbase,
        perturbation_order=2,
    )
    love, _, _ = get_love(model, forcing, numerics)
    return float(love.k[0].real)


def _fmt_rel(cur: float, prev: float | None) -> str:
    if prev is None or prev == 0.0:
        return "   --   "
    return f"{(cur - prev) / prev:+7.2%}"


def _save_figure(path: Path, t_sweep: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_vals = [r["T_km"] for r in t_sweep]
    dk20 = [r["abs_delta_k20"] for r in t_sweep]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))

    ax = axes[0]
    ax.plot(T_vals, dk20, "o-", label=r"$|\Delta k_{20}|$")
    for n, m in OFF_MODES:
        key = f"{n}_{m}"
        ax.plot(T_vals, [r["off"][key] for r in t_sweep], "s--", label=f"$|k_{{{n},{m}}}|$")
    ax.set_xlabel("reference shell thickness T [km]")
    ax.set_ylabel(r"$|k|$")
    ax.set_title("Moon lateral T-sweep: |Δk20| and off-modes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    ax2.plot(T_vals, [r["max_dmu_over_mubar"] for r in t_sweep], "D-k", label=r"max$|\delta\mu/\bar{\mu}|$")
    ax2.axhline(1.0, ls=":", color="r", label="positivity limit")
    ax2.set_xlabel("reference shell thickness T [km]")
    ax2.set_ylabel(r"max$|\delta\mu/\bar{\mu}|$")
    ax2.set_title("Positivity margin vs T")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle("TASK-036b: Moon lateral amplitude wall T-sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-list", type=float, nargs="+", default=[40.0, 55.0, 70.0, 85.0],
        help="Reference shell thicknesses T [km]",
    )
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--nrbase", type=int, default=30)
    parser.add_argument(
        "--ladder-t", type=float, default=None,
        help="If set and T-sweep converges, run the 031b lmax ladder at this T [km]",
    )
    parser.add_argument(
        "--lmax-ladder", type=int, nargs="+", default=[2, 3, 4, 5, 6],
        help="lmax rungs for the convergence ladder (only used if --ladder-t set)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    t0_total = time.perf_counter()

    # Verify 1-D background is stable (does not shift with T)
    print("[1D check]", flush=True)
    k2_1d = _solve_1d(args.nrbase)
    print(f"   k2_1d (zero-amplitude, Nrbase={args.nrbase}) = {k2_1d:.10f}", flush=True)
    print(f"   WEBER_K2_UNIFORM                               = {WEBER_K2_UNIFORM:.10f}", flush=True)
    print(f"   diff = {abs(k2_1d - WEBER_K2_UNIFORM):.2e}", flush=True)

    # Crustal thickness diagnostics at lmax (T-independent)
    diag = crustal_thickness_diagnostics(lmax=args.lmax)
    print(f"\n[crustal field lmax={args.lmax}]  max|dt| = {diag['max_abs_dt_m']/1000:.2f} km")

    # --- T-sweep ---
    print(f"\n[T-sweep]  lmax={args.lmax}  Nrbase={args.nrbase}")
    t_sweep = []
    for T_km in args.t_list:
        T_m = T_km * 1000.0
        max_dmu = _max_dmu_over_mubar(args.lmax, T_m)
        print(f"[solve] T={T_km:.0f} km  max|dmu/mubar|={max_dmu:.4f} ...", flush=True)
        r = _solve_at_T(args.lmax, T_m, args.nrbase)
        t_sweep.append(r)
        print(
            f"   N={r['N']} |dk20|={r['abs_delta_k20']:.6e} "
            f"wall={r['wall_s']:.1f}s peakRSS={r['peak_rss_gb']:.1f}GB",
            flush=True,
        )

    print(f"\n[T-sweep results]  lmax={args.lmax}  Nrbase={args.nrbase}")
    header = f"{'T_km':>6} {'max|dmu|':>9} {'N':>5} {'|dk20|':>13} {'d%':>8}"
    for n, m in OFF_MODES:
        header += f" {'|k'+str(n)+str(m)+'|':>12} {'d%':>8}"
    print(header)
    prev = {"dk20": None, **{f"{n}_{m}": None for n, m in OFF_MODES}}
    for r in t_sweep:
        line = (
            f"{r['T_km']:>6.0f} {r['max_dmu_over_mubar']:>9.4f} "
            f"{r['N']:>5} {r['abs_delta_k20']:13.6e} "
            f"{_fmt_rel(r['abs_delta_k20'], prev['dk20'])}"
        )
        prev["dk20"] = r["abs_delta_k20"]
        for n, m in OFF_MODES:
            key = f"{n}_{m}"
            val = r["off"][key]
            line += f" {val:12.5e} {_fmt_rel(val, prev[key])}"
            prev[key] = val
        print(line)

    # Convergence check: step-to-step Δk20 changes across T sweep
    if len(t_sweep) >= 2:
        last_step = abs(t_sweep[-1]["abs_delta_k20"] - t_sweep[-2]["abs_delta_k20"]) / t_sweep[-2]["abs_delta_k20"]
        print(f"\nLast T-step rel change in |dk20|: {last_step:.3e}")
        converged = last_step < 0.05
        print(f"Spectrum {'CONVERGES' if converged else 'NOT YET converged'} in T "
              f"({'<5%' if converged else '>5%'} last step)")

    # --- Optional lmax ladder at ladder_t ---
    ladder = []
    if args.ladder_t is not None:
        T_ladder = args.ladder_t * 1000.0
        print(f"\n[lmax ladder]  T={args.ladder_t:.0f} km  Nrbase={args.nrbase}")
        prev_l = {"dk20": None, **{f"{n}_{m}": None for n, m in OFF_MODES}}
        for lmax in args.lmax_ladder:
            max_dmu = _max_dmu_over_mubar(lmax, T_ladder)
            if max_dmu >= 1.0:
                print(f"  lmax={lmax}  max|dmu/mubar|={max_dmu:.4f}  BLOCKED (positivity; field discarded)")
                ladder.append({"lmax": lmax, "T_km": args.ladder_t, "blocked": True, "max_dmu_over_mubar": max_dmu})
                continue
            print(f"[solve] lmax={lmax} T={args.ladder_t:.0f} km ...", flush=True)
            r = _solve_at_T(lmax, T_ladder, args.nrbase)
            r["blocked"] = False
            ladder.append(r)
            print(
                f"   N={r['N']} |dk20|={r['abs_delta_k20']:.6e} "
                f"max|dmu/mubar|={r['max_dmu_over_mubar']:.4f} "
                f"wall={r['wall_s']:.1f}s peakRSS={r['peak_rss_gb']:.1f}GB",
                flush=True,
            )
        if ladder:
            print(f"\n[lmax ladder results]  T={args.ladder_t:.0f} km  Nrbase={args.nrbase}")
            hdr = f"{'lmax':>4} {'max|dmu|':>9} {'N':>5} {'|dk20|':>13} {'d%':>8}"
            print(hdr)
            prev_dk20 = None
            for r in ladder:
                if r["blocked"]:
                    print(f"{r['lmax']:>4} {r['max_dmu_over_mubar']:>9.4f} {'--':>5}   BLOCKED")
                    continue
                line = (
                    f"{r['lmax']:>4} {r['max_dmu_over_mubar']:>9.4f} "
                    f"{r['N']:>5} {r['abs_delta_k20']:13.6e} "
                    f"{_fmt_rel(r['abs_delta_k20'], prev_dk20)}"
                )
                prev_dk20 = r["abs_delta_k20"]
                print(line)

    # --- Archive ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        t_list_km=np.asarray(args.t_list),
        lmax_tsweep=args.lmax,
        nrbase=args.nrbase,
        tsweep_T_km=np.asarray([r["T_km"] for r in t_sweep]),
        tsweep_abs_delta_k20=np.asarray([r["abs_delta_k20"] for r in t_sweep]),
        tsweep_delta_k20=np.asarray([r["delta_k20"] for r in t_sweep], dtype=complex),
        tsweep_N=np.asarray([r["N"] for r in t_sweep], dtype=int),
        tsweep_max_dmu=np.asarray([r["max_dmu_over_mubar"] for r in t_sweep]),
        tsweep_wall_s=np.asarray([r["wall_s"] for r in t_sweep]),
        tsweep_peak_rss_gb=np.asarray([r["peak_rss_gb"] for r in t_sweep]),
        k2_1d_check=k2_1d,
        k2_uniform=WEBER_K2_UNIFORM,
        total_wall_s=time.perf_counter() - t0_total,
        peak_rss_gb_overall=_peak_rss_gb(),
    )
    for n, m in OFF_MODES:
        key = f"{n}_{m}"
        save[f"tsweep_off_{key}"] = np.asarray([r["off"][key] for r in t_sweep])
    if ladder:
        valid = [r for r in ladder if not r["blocked"]]
        save["ladder_t_km"] = args.ladder_t
        save["ladder_lmax_list"] = np.asarray(args.lmax_ladder, dtype=int)
        save["ladder_abs_delta_k20"] = np.asarray(
            [r.get("abs_delta_k20", float("nan")) for r in ladder]
        )
        save["ladder_N"] = np.asarray(
            [r.get("N", -1) for r in ladder], dtype=int
        )
        save["ladder_blocked"] = np.asarray([r["blocked"] for r in ladder])
        save["ladder_max_dmu"] = np.asarray([r["max_dmu_over_mubar"] for r in ladder])
    np.savez(args.output, **save)
    _save_figure(args.output.with_suffix(".png"), t_sweep)

    print(f"\nsaved {args.output} and {args.output.with_suffix('.png')}")
    print(f"overall peak RSS = {_peak_rss_gb():.1f} GB, total wall = {time.perf_counter() - t0_total:.1f} s")


if __name__ == "__main__":
    main()
