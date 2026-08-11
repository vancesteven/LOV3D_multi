#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Mars lateral spectrum robustness sweep (TASK-027, Machine B).

Two independent robustness questions for the Airy-compensated Mars lateral
Love spectrum (``pylov3d.mars_lateral.mars_lateral_love_spectrum`` /
``docs/MARS_MODEL.md`` section 6), both driver-only -- no ``pylov3d`` module
is modified (TASK-021b precedent, ``scripts/mars_hydration_sweep.py``).

Part 1 -- truncation convergence
--------------------------------
``docs/MARS_MODEL.md`` section 6 reports one spot check: the (2,0) k2 shift
moves 5.517e-5 -> 5.973e-5 from lmax=4 to lmax=5 (an 8.3% step). That is one
step of a sequence nobody continued. This driver runs lmax = 4, 5, 6 (and 7
if ``--lmax7``) at a fixed modest ``Nrbase`` and reports the step-to-step
relative change of BOTH:

  * the (2,0) forcing-mode k2 lateral shift, and
  * the top off-(2,0) response amplitudes -- (3,0), (2,+-2), (3,+-1), the
    modes TASK-026 assesses for detectability -- whose convergence matters
    as much as the forcing mode's.

Truncation (an angular question) is radial-independent to good
approximation (TASK-021b's argument); this driver *verifies* that here by
holding lmax fixed and sweeping ``Nrbase`` = 15/30/50, rather than assuming
it. TASK-021b hit >15 GB at lmax=6/Nrbase=50, so the lmax ladder runs at a
modest Nrbase and the radial ladder runs only at lmax=4.

Part 2 -- Airy calibration sensitivity
--------------------------------------
Section 5 flags Airy as the weakest crustal assumption: the (4,0)-driven
forcing-mode shift "scales ~1:1 (not quadratically) with the Airy
calibration". The lateral rigidity field is *exactly linear* in
``AIRY_FACTOR = rho_crust / (rho_mantle - rho_crust)`` (see
``mars_lateral.crustal_thickness_variation``: ``dt = clm * AIRY_FACTOR``,
and the ``d(mu)/d(dt)`` coefficient is density-independent), and
``AIRY_FACTOR`` is the ONLY place crust/mantle density enters the lateral
field. So sweeping the crust/mantle density bracket is a clean linear
rescale of the baseline ``mu_variable`` followed by an honest coupled
re-solve (perturbation_order=2 retains quadratic terms, so we re-solve
rather than scale outputs). This driver sweeps a defensible published Mars
crust/mantle density bracket and reports the induced spread in the (2,0)
shift and the top off-modes.

The non-Airy (InSight-calibrated) crustal-thickness substitution (Part 2
second pass) needs a data fetch and is deferred to Machine A per the task.

Constraints honored: driver only (no solver module touched); artifacts as
.npz + figure under ``docs/figures/proposal/``.

Usage
-----
    venvLOV3Dconv/bin/python scripts/mars_lateral_robustness.py            # full
    venvLOV3Dconv/bin/python scripts/mars_lateral_robustness.py --quick    # smoke, seconds-ish
    venvLOV3Dconv/bin/python scripts/mars_lateral_robustness.py --lmax7     # add lmax=7 (heavy)
    venvLOV3Dconv/bin/python scripts/mars_lateral_robustness.py --no-airy   # skip Part 2

``--quick`` (lmax ladder [4] only, Nrbase=15, 3-point Airy bracket) just
exercises the pipeline; it is NOT a converged, comparable result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from the repo root; make the repo root (parent of pylov3d/) importable
# regardless of invocation cwd (scripts/ is not a package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np

from pylov3d.love import get_love
from pylov3d.mars import MARS, MARS_FORCING_TD
from pylov3d.mars_lateral import (
    AIRY_FACTOR,
    CRUST_LAYER_INDEX,
    mu_variable_from_topography,
)
from pylov3d.mars import build_mars_model
from pylov3d.types import make_forcing, make_numerics

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures" / "proposal"

# Off-(2,0) response modes to track for convergence + Airy sensitivity.
# These are the detectability-relevant modes TASK-026 assesses (docs/MARS_MODEL.md
# section 6): the (3,0) zonal, the (2,+-2) sectoral splitting pair, and the
# (3,+-1) tesseral pair.
_OFF_MODES = ((3, 0), (2, 2), (2, -2), (3, 1), (3, -1))
_FORCING = (2, 0)

# A's section-6 reference for the forcing-mode shift (lmax=4/5, Nrbase=30):
_A_REF_SHIFT = {4: 5.517e-5, 5: 5.973e-5}

# Defensible published Mars crust/upper-mantle density brackets [kg/m^3].
# Baseline is (2900, 3400) -> AIRY_FACTOR = 5.8 (mars_lateral). Crust grain
# density from InSight-era estimates spans ~2700-3100; upper-mantle ~3400-3500.
# The bracket below is intentionally wide so the reported spread is an upper
# bound on the Airy-calibration sensitivity, not a best-case.
_RHO_CRUST_BRACKET = (2700.0, 2900.0, 3100.0)
_RHO_MANTLE_BRACKET = (3400.0, 3500.0)


def _airy_factor(rho_c: float, rho_m: float) -> float:
    return rho_c / (rho_m - rho_c)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrbase", type=int, default=None,
                   help="Fixed Nrbase for the lmax ladder (default: 30, or 15 with --quick).")
    p.add_argument("--lmax-list", type=int, nargs="+", default=None,
                   help="lmax ladder (default: 4 5 6, or just 4 with --quick).")
    p.add_argument("--lmax7", action="store_true",
                   help="Append lmax=7 to the ladder (heavy: memory + wall time).")
    p.add_argument("--nrbase-list", type=int, nargs="+", default=None,
                   help="Nrbase ladder for the radial-independence check at lmax=4 "
                        "(default: 15 30 50; skipped in --quick).")
    p.add_argument("--no-nrbase-check", action="store_true",
                   help="Skip the Nrbase-independence check.")
    p.add_argument("--no-airy", action="store_true",
                   help="Skip Part 2 (the Airy-calibration sweep).")
    p.add_argument("--airy-lmax", type=int, default=4,
                   help="lmax for the Airy sweep (default: 4).")
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke run (lmax=[4], Nrbase=15, 3-point Airy, no Nrbase check).")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# core solve helpers
# ---------------------------------------------------------------------------

def _extract(love, n: int, m: int) -> complex:
    """Complex k of mode (n, m) from a LoveSpectra, or nan if absent."""
    for i in range(len(love.n)):
        if int(love.n[i]) == n and int(love.m[i]) == m:
            return complex(love.k[i])
    return complex("nan")


def _uniform_k2(nrbase: int) -> float:
    """Real k2 of the uniform (no-lateral) model -- the shift baseline.

    lmax/coupling-independent (no mu_variable => single forcing mode), so one
    solve at a representative Nrbase suffices for all lmax."""
    model = build_mars_model()
    forcing = make_forcing(Td=MARS_FORCING_TD, n=_FORCING[0], m=_FORCING[1], F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=nrbase,
                             perturbation_order=2)
    love, _y, _m = get_love(model, forcing, numerics, mu_variable=None)
    return float(np.real(_extract(love, *_FORCING)))


def _solve(lmax: int, nrbase: int, k2_uniform: float,
           airy_scale: float = 1.0) -> dict:
    """One coupled (2,0)-forced solve; return forcing shift + off-mode amps.

    ``airy_scale`` linearly rescales the baseline lateral ``mu_variable``
    (exact for the Airy-factor dependence -- see module docstring)."""
    mu_variable = mu_variable_from_topography(lmax=lmax)
    if airy_scale != 1.0:
        mu_variable = {
            CRUST_LAYER_INDEX: [(n, m, amp * airy_scale)
                                for (n, m, amp) in mu_variable[CRUST_LAYER_INDEX]]
        }
    model = build_mars_model()
    forcing = make_forcing(Td=MARS_FORCING_TD, n=_FORCING[0], m=_FORCING[1], F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=nrbase,
                             perturbation_order=2)
    t0 = time.perf_counter()
    love, _y, _m = get_love(model, forcing, numerics, mu_variable=mu_variable)
    wall = time.perf_counter() - t0

    k20 = _extract(love, *_FORCING)
    row = {
        "lmax": lmax, "nrbase": nrbase, "airy_scale": airy_scale,
        "n_modes": int(len(love.n)), "wall_s": wall,
        "k20": k20, "shift20": k20 - k2_uniform,
    }
    for (n, m) in _OFF_MODES:
        row[f"k_{n}_{m}"] = _extract(love, n, m)
    return row


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _fmt_off_header() -> str:
    return "".join(f"{'|k(%d,%d)|' % nm:>13}" for nm in _OFF_MODES)


def _print_ladder(rows: list[dict], title: str, key: str) -> None:
    print(f"\n[{title}]  (2,0) shift and off-mode |k| vs {key}")
    print(f"{key:>6} {'N':>5} {'Re shift20':>13} {'|shift20|':>13}"
          f"{_fmt_off_header()} {'wall(s)':>9}")
    prev = None
    for r in rows:
        line = (f"{r[key]:>6} {r['n_modes']:>5} "
                f"{r['shift20'].real:>13.4e} {abs(r['shift20']):>13.4e}")
        for nm in _OFF_MODES:
            line += f"{abs(r[f'k_{nm[0]}_{nm[1]}']):>13.4e}"
        line += f" {r['wall_s']:>9.1f}"
        print(line)
        if prev is not None:
            drow = f"{'  Δrel':>6} {'':>5} "
            drow += f"{_relstr(r['shift20'].real, prev['shift20'].real):>13} "
            drow += f"{_relstr(abs(r['shift20']), abs(prev['shift20'])):>13}"
            for nm in _OFF_MODES:
                k = f"k_{nm[0]}_{nm[1]}"
                drow += f"{_relstr(abs(r[k]), abs(prev[k])):>13}"
            print(drow)
        prev = r


def _relstr(cur: float, ref: float) -> str:
    if ref == 0 or not np.isfinite(cur) or not np.isfinite(ref):
        return "  n/a"
    return f"{(cur - ref) / ref * 100:+.2f}%"


def _print_a_comparison(ladder: list[dict]) -> None:
    print("\n[A-comparison] forcing-mode shift vs docs/MARS_MODEL.md section 6")
    print(f"{'lmax':>6} {'B Re shift20':>15} {'A ref':>12} {'rel err':>10}")
    for r in ladder:
        ref = _A_REF_SHIFT.get(r["lmax"])
        if ref is None:
            continue
        rel = abs(r["shift20"].real - ref) / abs(ref)
        print(f"{r['lmax']:>6} {r['shift20'].real:>15.4e} {ref:>12.4e} {rel:>9.2%}")


def _airy_spread(rows: list[dict]) -> dict:
    """max/min spread over the Airy bracket for shift20 (real) and each off-mode |k|."""
    out = {}
    sh = [r["shift20"].real for r in rows]
    out["shift20_re"] = {"min": min(sh), "max": max(sh),
                         "spread_frac": (max(sh) - min(sh)) / abs(np.mean(sh))}
    for nm in _OFF_MODES:
        vals = [abs(r[f"k_{nm[0]}_{nm[1]}"]) for r in rows]
        mean = float(np.mean(vals))
        out[f"k_{nm[0]}_{nm[1]}"] = {"min": min(vals), "max": max(vals),
                                     "spread_frac": (max(vals) - min(vals)) / mean if mean else float("nan")}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        nrbase = args.nrbase or 15
        lmax_list = args.lmax_list or [4]
        nrbase_list = None if args.no_nrbase_check else (args.nrbase_list or [15, 30])
        do_airy = not args.no_airy
        airy_lmax = args.airy_lmax
        rho_crusts = (2700.0, 2900.0, 3100.0)
        rho_mantles = (3400.0,)
    else:
        nrbase = args.nrbase or 30
        lmax_list = args.lmax_list or [4, 5, 6]
        if args.lmax7 and 7 not in lmax_list:
            lmax_list = list(lmax_list) + [7]
        nrbase_list = None if args.no_nrbase_check else (args.nrbase_list or [15, 30, 50])
        do_airy = not args.no_airy
        airy_lmax = args.airy_lmax
        rho_crusts = _RHO_CRUST_BRACKET
        rho_mantles = _RHO_MANTLE_BRACKET

    print(f"[config] lmax_list={lmax_list}, nrbase={nrbase}, "
          f"nrbase_list={nrbase_list}, airy={do_airy} (lmax={airy_lmax}), "
          f"baseline AIRY_FACTOR={AIRY_FACTOR:.4f}, quick={args.quick}")

    # Uniform baseline (one solve; lmax/coupling-independent).
    t0 = time.perf_counter()
    k2_uniform = _uniform_k2(nrbase)
    print(f"[uniform] k2_uniform = {k2_uniform:.12f} "
          f"(MARS['k2']={MARS['k2']}, {time.perf_counter() - t0:.1f}s)")

    # --- Part 1a: lmax ladder at fixed Nrbase --------------------------------
    lmax_ladder = []
    for lmax in lmax_list:
        r = _solve(lmax, nrbase, k2_uniform)
        lmax_ladder.append(r)
        print(f"  [lmax={lmax}] N={r['n_modes']} shift20={r['shift20'].real:.4e} "
              f"({r['wall_s']:.1f}s)")
    _print_ladder(lmax_ladder, "Part 1a: truncation ladder", "lmax")
    _print_a_comparison(lmax_ladder)

    # --- Part 1b: Nrbase-independence check at lmax=4 ------------------------
    nrbase_ladder = []
    if nrbase_list:
        for nb in nrbase_list:
            r = _solve(4, nb, k2_uniform)
            nrbase_ladder.append(r)
            print(f"  [Nrbase={nb}] N={r['n_modes']} shift20={r['shift20'].real:.4e} "
                  f"({r['wall_s']:.1f}s)")
        _print_ladder(nrbase_ladder, "Part 1b: Nrbase-independence (lmax=4)", "nrbase")

    # --- Part 2: Airy-calibration sweep --------------------------------------
    airy_rows = []
    airy_spread = None
    if do_airy:
        print(f"\n[Part 2] Airy sweep at lmax={airy_lmax}, Nrbase={nrbase}")
        for rho_m in rho_mantles:
            for rho_c in rho_crusts:
                af = _airy_factor(rho_c, rho_m)
                scale = af / AIRY_FACTOR
                r = _solve(airy_lmax, nrbase, k2_uniform, airy_scale=scale)
                r["rho_crust"] = rho_c
                r["rho_mantle"] = rho_m
                r["airy_factor"] = af
                airy_rows.append(r)
                print(f"  [rho_c={rho_c:.0f} rho_m={rho_m:.0f}] "
                      f"AF={af:.3f} (x{scale:.3f}) shift20={r['shift20'].real:.4e}")
        airy_spread = _airy_spread(airy_rows)
        _print_airy(airy_rows, airy_spread)

    # --- persist -------------------------------------------------------------
    tag = "_quick" if args.quick else ""
    npz_path = args.out_dir / f"mars_lateral_robustness{tag}.npz"
    _save_npz(npz_path, lmax_ladder, nrbase_ladder, airy_rows,
              k2_uniform=k2_uniform, nrbase=nrbase)
    print(f"\n[output] table saved to {npz_path}")

    fig_path = args.out_dir / f"mars_lateral_robustness{tag}.png"
    _plot(lmax_ladder, nrbase_ladder, airy_rows, fig_path, nrbase=nrbase)
    print(f"[output] figure saved to {fig_path}")

    return {"lmax_ladder": lmax_ladder, "nrbase_ladder": nrbase_ladder,
            "airy_rows": airy_rows, "airy_spread": airy_spread,
            "k2_uniform": k2_uniform}


def _print_airy(rows: list[dict], spread: dict) -> None:
    print(f"\n[Part 2 spread] induced range over the crust/mantle-density bracket")
    print(f"  (2,0) Re shift20 : {spread['shift20_re']['min']:.4e} .. "
          f"{spread['shift20_re']['max']:.4e}  "
          f"(spread {spread['shift20_re']['spread_frac'] * 100:.1f}% of mean)")
    for nm in _OFF_MODES:
        s = spread[f"k_{nm[0]}_{nm[1]}"]
        print(f"  |k{nm}|         : {s['min']:.4e} .. {s['max']:.4e}  "
              f"(spread {s['spread_frac'] * 100:.1f}% of mean)")


def _save_npz(path, lmax_ladder, nrbase_ladder, airy_rows, k2_uniform, nrbase):
    arr = {}

    def pack(prefix, rows):
        if not rows:
            return
        arr[f"{prefix}_lmax"] = np.array([r["lmax"] for r in rows])
        arr[f"{prefix}_nrbase"] = np.array([r["nrbase"] for r in rows])
        arr[f"{prefix}_n_modes"] = np.array([r["n_modes"] for r in rows])
        arr[f"{prefix}_wall_s"] = np.array([r["wall_s"] for r in rows])
        arr[f"{prefix}_shift20"] = np.array([complex(r["shift20"]) for r in rows])
        arr[f"{prefix}_k20"] = np.array([complex(r["k20"]) for r in rows])
        for nm in _OFF_MODES:
            k = f"k_{nm[0]}_{nm[1]}"
            arr[f"{prefix}_{k}"] = np.array([complex(r[k]) for r in rows])

    pack("lmax", lmax_ladder)
    pack("nrbase", nrbase_ladder)
    pack("airy", airy_rows)
    if airy_rows:
        arr["airy_rho_crust"] = np.array([r["rho_crust"] for r in airy_rows])
        arr["airy_rho_mantle"] = np.array([r["rho_mantle"] for r in airy_rows])
        arr["airy_factor"] = np.array([r["airy_factor"] for r in airy_rows])
    np.savez(path, k2_uniform=k2_uniform, base_nrbase=nrbase,
             base_airy_factor=AIRY_FACTOR, **arr)


def _plot(lmax_ladder, nrbase_ladder, airy_rows, path, nrbase):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    lm = [r["lmax"] for r in lmax_ladder]
    ax.plot(lm, [r["shift20"].real for r in lmax_ladder], "o-", label="(2,0) Re shift")
    for nm in _OFF_MODES:
        k = f"k_{nm[0]}_{nm[1]}"
        ax.plot(lm, [abs(r[k]) for r in lmax_ladder], "^--", label=f"|k{nm}|")
    ax.set_xlabel("lmax"); ax.set_ylabel("k amplitude"); ax.set_yscale("log")
    ax.set_title(f"Part 1a: truncation ladder (Nrbase={nrbase})")
    ax.legend(fontsize=7)

    ax = axes[1]
    if nrbase_ladder:
        nb = [r["nrbase"] for r in nrbase_ladder]
        ax.plot(nb, [r["shift20"].real for r in nrbase_ladder], "s-", label="(2,0) Re shift")
        for nm in _OFF_MODES:
            k = f"k_{nm[0]}_{nm[1]}"
            ax.plot(nb, [abs(r[k]) for r in nrbase_ladder], "^--", label=f"|k{nm}|")
        ax.set_xlabel("Nrbase"); ax.set_ylabel("k amplitude"); ax.set_yscale("log")
        ax.set_title("Part 1b: radial independence (lmax=4)")
        ax.legend(fontsize=7)
    else:
        ax.set_visible(False)

    ax = axes[2]
    if airy_rows:
        af = [r["airy_factor"] for r in airy_rows]
        order = np.argsort(af)
        af = np.array(af)[order]
        ax.plot(af, np.array([r["shift20"].real for r in airy_rows])[order],
                "o-", label="(2,0) Re shift")
        for nm in _OFF_MODES:
            k = f"k_{nm[0]}_{nm[1]}"
            ax.plot(af, np.array([abs(r[k]) for r in airy_rows])[order],
                    "^--", label=f"|k{nm}|")
        ax.axvline(AIRY_FACTOR, color="crimson", ls=":", label=f"baseline AF={AIRY_FACTOR:.2f}")
        ax.set_xlabel("Airy factor"); ax.set_ylabel("k amplitude"); ax.set_yscale("log")
        ax.set_title("Part 2: Airy-calibration sensitivity")
        ax.legend(fontsize=7)
    else:
        ax.set_visible(False)

    fig.suptitle("TASK-027 Mars lateral robustness (Machine B)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run(parse_args())
