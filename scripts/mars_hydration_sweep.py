#!/usr/bin/env python
"""Independent hydration-amplitude sweep for TASK-021b (cross-check of A's TASK-021).

Re-runs the serpentinization-front tidal-signature sweep from
``pylov3d.mars_hydration`` on independent hardware at *higher radial
resolution* than A's shipped default (``Nrbase=50`` vs the validated 30),
at the validated ``lmax=4`` -- the config that resolves the (4,0)->(2,0)
first-order self-coupling channel (A's ``lmax=2`` default underestimates
the lateral term 11x-73x; see ``docs/MARS_MODEL.md`` section 4). Confirms,
on this machine and at finer radial grid:

  * the central-ratio ``Delta k2(f_h)`` curve (mean / lateral / total), and
  * the ~57-64:1 mean:lateral dominance ratio at ``lmax=4``,

then runs an optional ``lmax=6`` convergence spot check at
``f_h=0.1/0.3/0.5`` (central ratio) to confirm the (4,0)<->(2,0) lateral
channel has converged by ``lmax=4`` (i.e. degrees >=5 add negligibly).

A's reference numbers to reproduce (``docs/MARS_MODEL.md`` section 4,
``lmax=4``, ``Nrbase=30``, central ratio, K_ROW0_FACTOR-corrected;
baseline k2 = 0.169000000000):

    f_h   mean (lmax-indep)   lateral (lmax=4)   mean:lateral
    0.1   6.336e-05           9.89e-07           64:1
    0.3   1.911e-04           3.14e-06           61:1
    0.5   3.203e-04           5.62e-06           57:1

The mean term is lmax-independent (a fresh 1D solve with softened crust
mu0/Ks0), so raising Nrbase 30->50 tests radial-grid convergence of BOTH
the mean 1D solve and the coupled lateral solve. No ``pylov3d`` module is
modified -- this driver only *calls* ``mars_hydration``.

Headless (no plotting dependency beyond matplotlib Agg). Saves the full
row table (.npz) plus a plain-text summary, and prints an A-vs-B
comparison table with relative errors.

Usage
-----
    venvLOV3Dconv/bin/python scripts/mars_hydration_sweep.py            # full: Nrbase=50, lmax=4, +lmax=6 check
    venvLOV3Dconv/bin/python scripts/mars_hydration_sweep.py --quick    # smoke: coarse, seconds
    venvLOV3Dconv/bin/python scripts/mars_hydration_sweep.py --no-lmax6 # skip the lmax=6 convergence check

``--quick`` (lmax=2, Nrbase=15, a 3-point f_h grid, no lmax=6 check) just
exercises the pipeline fast; it is NOT a converged, comparable result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from the repo root; make sure the repo root (parent of pylov3d/) is
# importable regardless of invocation cwd (scripts/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np

from pylov3d.mars_hydration import (
    DEFAULT_F_H_GRID,
    MU_SERP_RATIO_CENTRAL,
    K_SERP_RATIO_CENTRAL,
    RATIO_SCENARIOS,
    SIGMA_K2,
    detectability_summary,
    hydration_forward_sweep,
    hydration_k2,
)

_DEFAULT_OUT_DIR = Path(__file__).parent / "output"

# A's reference (docs/MARS_MODEL.md section 4, lmax=4, Nrbase=30, central):
# f_h -> (mean, lateral, mean:lateral ratio). Used only for the printed
# comparison; the sweep does not depend on these.
_A_REF_LMAX4_CENTRAL = {
    0.1: {"mean": 6.336e-05, "lateral": 9.89e-07, "ratio": 64.0},
    0.3: {"mean": 1.911e-04, "lateral": 3.14e-06, "ratio": 61.0},
    0.5: {"mean": 3.203e-04, "lateral": 5.62e-06, "ratio": 57.0},
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nrbase", type=int, default=None,
                   help="Radial grid Nrbase (default: 50, or 15 with --quick).")
    p.add_argument("--lmax", type=int, default=None,
                   help="Main-sweep lmax (default: 4, or 2 with --quick).")
    p.add_argument("--lmax6", type=int, default=6,
                   help="Convergence-check lmax for the f_h spot points (default: 6).")
    p.add_argument("--no-lmax6", action="store_true",
                   help="Skip the higher-lmax convergence spot check.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke run (lmax=2, Nrbase=15, 3-point grid, no lmax=6 check).")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        nrbase = args.nrbase or 15
        lmax = args.lmax or 2
        f_h_grid = (0.0, 0.1, 0.3)
        do_lmax6 = False
    else:
        nrbase = args.nrbase or 50   # TASK-021b: higher radial resolution than A's 30
        lmax = args.lmax or 4        # validated config resolving (4,0)->(2,0)
        f_h_grid = DEFAULT_F_H_GRID
        do_lmax6 = not args.no_lmax6

    print(f"[config] Nrbase={nrbase}, lmax={lmax}, f_h_grid={f_h_grid}, "
          f"scenarios={list(RATIO_SCENARIOS)}, lmax6_check={do_lmax6}, quick={args.quick}")

    # --- main sweep: f_h x ratio-bracket at (lmax, nrbase) --------------------
    t0 = time.perf_counter()
    rows = hydration_forward_sweep(
        f_h_grid=f_h_grid, ratio_scenarios=RATIO_SCENARIOS, lmax=lmax, Nrbase=nrbase,
    )
    sweep_wall = time.perf_counter() - t0
    n_coupled = sum(1 for r in rows if r["f_h"] > 0.0)
    print(f"[sweep] {len(rows)} points ({n_coupled} coupled solves) in {sweep_wall:.1f}s")

    _print_central_table(rows)
    _print_a_comparison(rows, lmax)

    # detectability (needs f_h=0.0 and 0.1 in the grid)
    det = None
    if 0.0 in f_h_grid and 0.1 in f_h_grid:
        det = detectability_summary(rows, scenario="central")
        cross = det["crossing_f_h"]
        print(f"\n[detectability] sigma_k2={det['sigma_k2']:.4f}; "
              f"crossing f_h={'none over grid' if cross is None else cross}; "
              f"|Delta k2| at f_h=0.1 = {det['precision_to_resolve_f_h_0p1']:.3e} "
              f"(precision that would resolve f_h=0.1 at 1 sigma)")

    # --- lmax convergence spot check ------------------------------------------
    conv = None
    if do_lmax6:
        conv = _lmax_convergence_check(rows, lmax_lo=lmax, lmax_hi=args.lmax6, nrbase=nrbase)

    # --- persist --------------------------------------------------------------
    npz_path = args.out_dir / ("mars_hydration_sweep_quick.npz" if args.quick
                               else "mars_hydration_sweep.npz")
    _save_npz(npz_path, rows, conv, nrbase=nrbase, lmax=lmax)
    print(f"\n[output] sweep table saved to {npz_path}")

    fig_path = args.out_dir / ("mars_hydration_sweep_quick.png" if args.quick
                               else "mars_hydration_sweep.png")
    _plot(rows, fig_path, nrbase=nrbase, lmax=lmax)
    print(f"[output] plot saved to {fig_path}")

    return {"rows": rows, "sweep_wall_s": sweep_wall, "detectability": det, "convergence": conv}


def _central(rows: list[dict]) -> list[dict]:
    return sorted((r for r in rows if r["scenario"] == "central"), key=lambda r: r["f_h"])


def _print_central_table(rows: list[dict]) -> None:
    print(f"\n{'f_h':>5} {'mean':>12} {'lateral':>12} {'total Δk2':>12} "
          f"{'mean:lat':>9} {'total/σ':>9}")
    base = next(r["k2_total"].real for r in _central(rows) if r["f_h"] == 0.0)
    for r in _central(rows):
        mean_d = r["k2_mean"].real - base
        lat = abs(r["k2_lateral"])
        tot = abs(r["k2_total"] - base)
        ratio = (abs(mean_d) / lat) if lat > 0 else float("inf")
        print(f"{r['f_h']:>5.1f} {mean_d:>12.4e} {lat:>12.4e} {tot:>12.4e} "
              f"{ratio:>9.1f} {tot / SIGMA_K2 * 100:>8.2f}%")


def _print_a_comparison(rows: list[dict], lmax: int) -> None:
    """Compare this run's central-ratio mean/lateral against A's lmax=4
    reference (only meaningful at lmax=4; skipped otherwise)."""
    if lmax != 4:
        print(f"\n[A-comparison] skipped (this sweep ran lmax={lmax}; "
              f"A's reference table is lmax=4).")
        return
    base = next(r["k2_total"].real for r in _central(rows) if r["f_h"] == 0.0)
    print(f"\n[A-comparison] this machine (Nrbase from config) vs A "
          f"(Nrbase=30), lmax=4, central ratio:")
    print(f"{'f_h':>5} {'quantity':>9} {'B (this)':>13} {'A (ref)':>13} {'rel err':>10}")
    for f_h, ref in _A_REF_LMAX4_CENTRAL.items():
        r = next((x for x in _central(rows) if abs(x["f_h"] - f_h) < 1e-9), None)
        if r is None:
            continue
        b_mean = r["k2_mean"].real - base
        b_lat = abs(r["k2_lateral"])
        b_ratio = (abs(b_mean) / b_lat) if b_lat > 0 else float("inf")
        for q, b_val, a_val in (
            ("mean", b_mean, ref["mean"]),
            ("lateral", b_lat, ref["lateral"]),
            ("ratio", b_ratio, ref["ratio"]),
        ):
            rel = abs(b_val - a_val) / abs(a_val) if a_val else float("nan")
            print(f"{f_h:>5.1f} {q:>9} {b_val:>13.4e} {a_val:>13.4e} {rel:>9.2%}")


def _lmax_convergence_check(rows, lmax_lo: int, lmax_hi: int, nrbase: int) -> list[dict]:
    """Recompute the central-ratio lateral term at lmax_hi for the
    f_h=0.1/0.3/0.5 spot points and report the lmax_lo -> lmax_hi change,
    testing (4,0)<->(2,0) channel convergence."""
    print(f"\n[lmax-convergence] central ratio, Nrbase={nrbase}: "
          f"lateral term at lmax={lmax_lo} vs lmax={lmax_hi}")
    print(f"{'f_h':>5} {'lat lmax_lo':>14} {'lat lmax_hi':>14} {'Δ rel':>10} "
          f"{'N_lo':>6} {'N_hi':>6} {'wall_hi(s)':>11}")
    out = []
    lo_by_fh = {r["f_h"]: r for r in _central(rows)}
    for f_h in (0.1, 0.3, 0.5):
        if f_h not in lo_by_fh:
            continue
        r_lo = lo_by_fh[f_h]
        lat_lo = abs(r_lo["k2_lateral"])
        t0 = time.perf_counter()
        r_hi = hydration_k2(f_h, mu_ratio=MU_SERP_RATIO_CENTRAL, K_ratio=K_SERP_RATIO_CENTRAL,
                            lmax=lmax_hi, Nrbase=nrbase)
        wall_hi = time.perf_counter() - t0
        lat_hi = abs(r_hi["k2_lateral"])
        rel = abs(lat_hi - lat_lo) / lat_hi if lat_hi > 0 else float("nan")
        print(f"{f_h:>5.1f} {lat_lo:>14.4e} {lat_hi:>14.4e} {rel:>9.2%} "
              f"{r_lo['n_coupled_modes']:>6d} {r_hi['n_coupled_modes']:>6d} {wall_hi:>11.1f}")
        out.append({"f_h": f_h, "lat_lo": lat_lo, "lat_hi": lat_hi, "rel": rel,
                    "n_lo": r_lo["n_coupled_modes"], "n_hi": r_hi["n_coupled_modes"],
                    "lmax_lo": lmax_lo, "lmax_hi": lmax_hi})
    return out


def _save_npz(path: Path, rows, conv, nrbase: int, lmax: int) -> None:
    keys = ("f_h", "scenario", "mu_ratio", "K_ratio", "n_coupled_modes")
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    for k in ("k2_mean", "k2_lateral", "k2_total"):
        arr[k] = np.array([complex(r[k]) for r in rows])
    meta = {"nrbase": nrbase, "lmax": lmax, "sigma_k2": SIGMA_K2}
    if conv:
        arr["conv_f_h"] = np.array([c["f_h"] for c in conv])
        arr["conv_lat_lo"] = np.array([c["lat_lo"] for c in conv])
        arr["conv_lat_hi"] = np.array([c["lat_hi"] for c in conv])
        arr["conv_lmax_lo"] = np.array([c["lmax_lo"] for c in conv])
        arr["conv_lmax_hi"] = np.array([c["lmax_hi"] for c in conv])
    np.savez(path, **arr, **{f"meta_{k}": v for k, v in meta.items()})


def _plot(rows, path: Path, nrbase: int, lmax: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    base = next(r["k2_total"].real for r in _central(rows) if r["f_h"] == 0.0)
    for label in RATIO_SCENARIOS:
        scen = sorted((r for r in rows if r["scenario"] == label), key=lambda r: r["f_h"])
        f_hs = [r["f_h"] for r in scen]
        tot = [abs(r["k2_total"] - base) for r in scen]
        ax1.plot(f_hs, tot, "o-", label=label)
    ax1.axhline(SIGMA_K2, color="crimson", linestyle="--", label=r"$\sigma_{k2}=0.006$")
    ax1.set_xlabel(r"$f_h$"); ax1.set_ylabel(r"$|\Delta k_2|$")
    ax1.set_yscale("log"); ax1.legend(fontsize=8)
    ax1.set_title(f"Total Δk2 vs f_h (Nrbase={nrbase}, lmax={lmax})")

    cen = _central(rows)
    f_hs = [r["f_h"] for r in cen if r["f_h"] > 0]
    mean_d = [r["k2_mean"].real - base for r in cen if r["f_h"] > 0]
    lat = [abs(r["k2_lateral"]) for r in cen if r["f_h"] > 0]
    ax2.plot(f_hs, mean_d, "s-", label="mean (degree 0)")
    ax2.plot(f_hs, lat, "^-", label="lateral (degree>=1)")
    ax2.set_xlabel(r"$f_h$"); ax2.set_ylabel(r"contribution to $\Delta k_2$")
    ax2.set_yscale("log"); ax2.legend(fontsize=8)
    ax2.set_title("Mean vs lateral (central ratio)")
    fig.suptitle("TASK-021b independent hydration sweep (Machine B)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run(parse_args())
