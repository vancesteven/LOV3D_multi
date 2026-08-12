#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Mars diagonal k2m order-splitting: is the m-ordering stable? (TASK-030, B).

The proposal reports the diagonal-splitting ordering among m (which of
k20/k21/k22 shifts most from the uniform k2) but flags it as "should not be
relied upon", because only m=0 was carried past lmax=4: TASK-027 ran the m=0
truncation ladder to lmax=6, while the m=1 and m=2 diagonal Love numbers
(``pylov3d.mars_detectability_k2m.MARS_K21_FORCING`` / ``MARS_K22_FORCING``,
feeding TASK-026's 8.2x/12.0x/8.2x GRAIL-precision ratios) exist only at
lmax=4. This driver runs the m=1 and m=2 diagonal forcing solves at lmax=5
and 6 as well, at the identical TASK-027 part-1 config, and reports whether
the ordering of the splitting |dk2m| = |k2m(lmax) - k2_uniform| holds as the
harmonic cutoff climbs.

Method (identical to TASK-027 part 1 / mars_lateral_robustness.py, and to
the config MARS_K21/22_FORCING were computed at -- lmax=4, Nrbase=30,
method='combination', perturbation_order=2):

  * for each m in {0, 1, 2} and each lmax in {4, 5, 6}, solve the coupled
    diagonal forcing (2, m) via the VALIDATED ``mars_lateral_love_spectrum``
    path (the same call ``recompute_k2m_diagonal_shift`` uses), extract the
    forced mode's own real k, and report dk2m = k2m - k2_uniform;
  * the uniform (no-lateral) k2 is degree-only for a spherically symmetric
    body -- m-INDEPENDENT -- so one uniform solve is the baseline for all
    three m (mirrors mars_lateral_robustness._uniform_k2);
  * the ordering question is invariant to the constant baseline (the same
    k2_uniform is subtracted from all three m at a given lmax), so a sign or
    magnitude wobble that reorders the m at higher lmax is a real result, and
    stability retires the proposal caveat.

At lmax=4 the documented ordering is m=0 (5.52e-5) > m=2 (3.40e-5) >
m=1 (2.09e-5). The deliverable is the same table at lmax=5 and 6 and a plain
statement of whether that ordering is preserved.

Driver only -- no ``pylov3d`` module is modified (TASK-021b/TASK-027
precedent). Artifacts: ``.npz`` + figure under ``docs/figures/proposal/``.

Usage
-----
    venvLOV3Dconv/bin/python scripts/mars_m_ordering.py            # full (lmax 4,5,6)
    venvLOV3Dconv/bin/python scripts/mars_m_ordering.py --quick    # smoke, lmax=4 only
    venvLOV3Dconv/bin/python scripts/mars_m_ordering.py --lmax-list 4 5 6 7   # add lmax=7 (heavy)

``--quick`` (lmax=[4], Nrbase=15) only exercises the pipeline; it is NOT a
converged, comparable result.
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
from pylov3d.mars import MARS, MARS_FORCING_TD, build_mars_model
from pylov3d.mars_lateral import mars_lateral_love_spectrum
from pylov3d.types import make_forcing, make_numerics

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures" / "proposal"

_M_LIST = (0, 1, 2)

# Documented lmax=4 diagonal shifts (pylov3d.mars_detectability_k2m; m=0 is
# the TASK-016/TASK-027 forcing-mode shift). Used only for an at-a-glance
# regression print, not as the source of truth (this driver recomputes them).
_REF_LMAX4_SHIFT = {0: 5.517e-5, 1: 2.0913e-5, 2: 3.3995e-5}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--nrbase", type=int, default=None,
                   help="Fixed Nrbase (default: 30, or 15 with --quick).")
    p.add_argument("--lmax-list", type=int, nargs="+", default=None,
                   help="lmax ladder (default: 4 5 6, or just 4 with --quick).")
    p.add_argument("--m-list", type=int, nargs="+", default=None,
                   help="Forcing orders m to run (default: 0 1 2).")
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke run (lmax=[4], Nrbase=15).")
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

    Degree-only for a spherically symmetric body, so m-independent: one
    solve at forcing (2,0) is the baseline for m=0/1/2 alike."""
    model = build_mars_model()
    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=nrbase,
                             perturbation_order=2)
    love, _y, _m = get_love(model, forcing, numerics, mu_variable=None)
    return float(np.real(_extract(love, 2, 0)))


def _solve_diag(m: int, lmax: int, nrbase: int, k2_uniform: float) -> dict:
    """One coupled diagonal (2,m)-forced solve; return the forced-mode shift.

    Uses the validated ``mars_lateral_love_spectrum`` path (the same call
    ``mars_detectability_k2m.recompute_k2m_diagonal_shift`` uses), so this
    reproduces MARS_K21/22_FORCING exactly at lmax=4/Nrbase=30."""
    t0 = time.perf_counter()
    result = mars_lateral_love_spectrum(
        lmax=lmax, forcing=(2, m), Nrbase=nrbase,
        method="combination", perturbation_order=2,
    )
    wall = time.perf_counter() - t0
    love = result["love"]
    k2m = _extract(love, 2, m)
    return {
        "m": m, "lmax": lmax, "nrbase": nrbase,
        "n_modes": int(len(love.n)), "wall_s": wall,
        "k2m": k2m, "shift": k2m - k2_uniform,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _ordering(rows_at_lmax: list[dict]) -> list[int]:
    """m values sorted by |shift| descending -- the splitting ordering."""
    return [r["m"] for r in sorted(rows_at_lmax, key=lambda r: -abs(r["shift"].real))]


def _ord_str(order: list[int]) -> str:
    return " > ".join(f"m={m}" for m in order)


def _print_table(rows: list[dict], lmax_list, m_list) -> dict:
    print("\n[diagonal k2m shift table]  dk2m = Re[k2m(lmax)] - k2_uniform")
    header = f"{'lmax':>6} " + "".join(f"{'dk2(2,%d)' % m:>15}" for m in m_list) \
             + f"{'N(m=%d)' % m_list[-1]:>9} {'wall(s)':>9}   ordering (|dk| desc)"
    print(header)
    by_lmax = {}
    orderings = {}
    for lmax in lmax_list:
        rows_l = [r for r in rows if r["lmax"] == lmax]
        by_lmax[lmax] = {r["m"]: r for r in rows_l}
        order = _ordering(rows_l)
        orderings[lmax] = order
        line = f"{lmax:>6} "
        for m in m_list:
            r = by_lmax[lmax].get(m)
            line += f"{r['shift'].real:>15.4e}" if r else f"{'--':>15}"
        n_last = by_lmax[lmax].get(m_list[-1], {}).get("n_modes", 0)
        wall_max = max((r["wall_s"] for r in rows_l), default=0.0)
        line += f"{n_last:>9} {wall_max:>9.1f}   {_ord_str(order)}"
        print(line)
    return orderings


def _print_lmax4_regression(rows: list[dict]) -> None:
    print("\n[lmax=4 regression vs documented shifts]")
    print(f"{'m':>4} {'this run dk':>15} {'documented':>15} {'rel err':>10}")
    for m in _M_LIST:
        r = next((r for r in rows if r["lmax"] == 4 and r["m"] == m), None)
        ref = _REF_LMAX4_SHIFT.get(m)
        if r is None or ref is None:
            continue
        rel = abs(r["shift"].real - ref) / abs(ref)
        print(f"{m:>4} {r['shift'].real:>15.4e} {ref:>15.4e} {rel:>9.2%}")


def _print_verdict(orderings: dict, lmax_list) -> tuple[bool, str]:
    lmax_max = max(lmax_list)
    ref_order = orderings[min(lmax_list)]
    stable = all(orderings[lm] == ref_order for lm in lmax_list)
    print("\n[verdict] m-ordering of the diagonal splitting across the lmax ladder")
    for lm in lmax_list:
        flag = "" if orderings[lm] == ref_order else "   <-- REORDERED"
        print(f"  lmax={lm}: {_ord_str(orderings[lm])}{flag}")
    if stable:
        msg = (f"STABLE: the ordering {_ord_str(ref_order)} holds at every lmax "
               f"in {list(lmax_list)}. The proposal's m-ordering caveat can retire "
               f"(converged to lmax={lmax_max}).")
    else:
        msg = (f"NOT STABLE: the ordering changes across the lmax ladder "
               f"(lmax=4 {_ord_str(orderings[min(lmax_list)])} vs "
               f"lmax={lmax_max} {_ord_str(orderings[lmax_max])}). The caveat is "
               f"warranted -- report the reordering as the result.")
    print(f"\n  ==> {msg}")
    return stable, msg


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def _save_npz(path, rows, k2_uniform, nrbase, orderings, stable):
    arr = {
        "m": np.array([r["m"] for r in rows]),
        "lmax": np.array([r["lmax"] for r in rows]),
        "nrbase": np.array([r["nrbase"] for r in rows]),
        "n_modes": np.array([r["n_modes"] for r in rows]),
        "wall_s": np.array([r["wall_s"] for r in rows]),
        "k2m": np.array([complex(r["k2m"]) for r in rows]),
        "shift": np.array([complex(r["shift"]) for r in rows]),
    }
    # ordering per lmax, as a 2D int array (rows = lmax ascending)
    lmax_sorted = sorted(orderings)
    arr["ordering_lmax"] = np.array(lmax_sorted)
    arr["ordering"] = np.array([orderings[lm] for lm in lmax_sorted])
    np.savez(path, k2_uniform=k2_uniform, base_nrbase=nrbase,
             mars_k2=MARS["k2"], ordering_stable=bool(stable), **arr)


def _plot(rows, path, nrbase, m_list, lmax_list):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m in m_list:
        xs = [lm for lm in lmax_list]
        ys = [abs(next(r for r in rows if r["lmax"] == lm and r["m"] == m)["shift"].real)
              for lm in lmax_list]
        ax.plot(xs, ys, "o-", label=f"|dk2(2,{m})|")
    ax.set_xlabel("lmax")
    ax.set_ylabel("|diagonal shift| = |Re k2m - k2_uniform|")
    ax.set_yscale("log")
    ax.set_xticks(list(lmax_list))
    ax.set_title(f"TASK-030 Mars diagonal k2m ordering (Nrbase={nrbase})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        nrbase = args.nrbase or 15
        lmax_list = tuple(args.lmax_list or [4])
    else:
        nrbase = args.nrbase or 30
        lmax_list = tuple(args.lmax_list or [4, 5, 6])
    m_list = tuple(args.m_list or _M_LIST)

    print(f"[config] m_list={list(m_list)}, lmax_list={list(lmax_list)}, "
          f"nrbase={nrbase}, method='combination', perturbation_order=2, "
          f"quick={args.quick}")

    t0 = time.perf_counter()
    k2_uniform = _uniform_k2(nrbase)
    print(f"[uniform] k2_uniform = {k2_uniform:.12f} "
          f"(MARS['k2']={MARS['k2']}, {time.perf_counter() - t0:.1f}s)")

    rows = []
    for lmax in lmax_list:
        for m in m_list:
            r = _solve_diag(m, lmax, nrbase, k2_uniform)
            rows.append(r)
            print(f"  [lmax={lmax} m={m}] N={r['n_modes']} "
                  f"k2m={r['k2m'].real:.12f} dk2m={r['shift'].real:.4e} "
                  f"({r['wall_s']:.1f}s)")

    orderings = _print_table(rows, lmax_list, m_list)
    if 4 in lmax_list:
        _print_lmax4_regression(rows)
    stable, _msg = _print_verdict(orderings, lmax_list)

    tag = "_quick" if args.quick else ""
    npz_path = args.out_dir / f"mars_m_ordering{tag}.npz"
    _save_npz(npz_path, rows, k2_uniform, nrbase, orderings, stable)
    print(f"\n[output] table saved to {npz_path}")

    fig_path = args.out_dir / f"mars_m_ordering{tag}.png"
    _plot(rows, fig_path, nrbase, m_list, lmax_list)
    print(f"[output] figure saved to {fig_path}")

    return {"rows": rows, "k2_uniform": k2_uniform,
            "orderings": orderings, "stable": stable}


if __name__ == "__main__":
    run(parse_args())
