# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""TASK-031b: Moon lateral truncation-convergence study (driver only).

The Moon analogue of TASK-027 part 1.  Runs the validated
``moon_lateral.moon_lateral_love_spectrum`` path over an lmax ladder at fixed
Nrbase, tracking the (2,0) forcing-mode ``Delta k20`` and the three named
off-forcing pairs ((2,+/-2), (2,+/-1), (3,+/-3)); separately verifies (does
not import from Mars) the Nrbase-independence of the angular result; watches
peak resident memory throughout; and -- as a cheap extra -- fits the scaling
exponent of the isolated (4,0) rheology channel to test whether TASK-028's
Mars first-order cancellation generalises (the Moon field has no (2,0)
channel, C20 being removed by default).

No pylov3d module is modified; this is a pure consumer of the committed
pipeline.  Artifacts -> docs/figures/proposal/moon_lateral_convergence.{npz,png}.
"""

from __future__ import annotations

import argparse
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.love import get_love
from pylov3d.moon import MOON, MOON_FORCING_TD, WEBER_K2_UNIFORM, build_moon_model
from pylov3d.moon_lateral import (
    CRUST_LAYER_INDEX,
    crustal_thickness_diagnostics,
    moon_lateral_love_spectrum,
    mu_variable_from_topography,
)
from pylov3d.types import make_forcing, make_numerics

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "proposal" / "moon_lateral_convergence.npz"

# Off-forcing pairs the TASK-031 table names (positive-m member tracked; the
# pair members are equal in magnitude by construction).
OFF_MODES = ((2, 2), (2, 1), (3, 3))


def _peak_rss_gb() -> float:
    """Peak resident set size so far, in GB (macOS reports bytes, Linux KB)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return raw * scale / 1e9


def _forcing_index(love) -> int:
    matches = np.where((love.n == love.nf) & (love.m == love.mf))[0]
    if len(matches) != 1:
        raise RuntimeError("forcing mode is missing or duplicated")
    return int(matches[0])


def _mode_abs(love, n: int, m: int) -> float:
    idx = np.where((love.n == n) & (love.m == m))[0]
    if len(idx) == 0:
        return float("nan")
    return float(abs(complex(love.k[idx[0]])))


def _solve(lmax: int, nrbase: int) -> dict:
    """One full-field solve; returns Delta k20, off-mode |k|, N, wall, peak RSS.

    Every rung reports its positivity margin (max|dmu/mu_bar|) from the cheap
    diagnostics, which never raise.  If that margin has reached the linearized
    positive-rigidity bound the coupled solve is *not* attempted -- the field
    is non-physical -- and the rung is recorded as ``blocked`` with the reason.
    This is the expected behaviour above the cutoff where the Airy
    linearization breaks down, not an error to route around.
    """
    diag = crustal_thickness_diagnostics(lmax=lmax)
    row = {
        "lmax": lmax,
        "nrbase": nrbase,
        "max_abs_dmu_over_mubar": float(diag["max_abs_dmu_over_mubar"]),
        "max_abs_dt_over_reference": float(diag["max_abs_dt_over_reference"]),
        "blocked": False,
        "block_reason": "",
        "N": -1,
        "delta_k20": complex(float("nan"), float("nan")),
        "abs_delta_k20": float("nan"),
        "off": {f"{n}_{m}": float("nan") for n, m in OFF_MODES},
        "wall_s": float("nan"),
        "peak_rss_gb": _peak_rss_gb(),
    }
    try:
        result = moon_lateral_love_spectrum(lmax=lmax, Nrbase=nrbase)
    except ValueError as exc:
        row["blocked"] = True
        row["block_reason"] = str(exc)
        return row
    love = result["love"]
    fidx = _forcing_index(love)
    delta_k20 = complex(love.k[fidx]) - WEBER_K2_UNIFORM
    row.update(
        N=int(len(love.k)),
        delta_k20=complex(delta_k20),
        abs_delta_k20=float(abs(delta_k20)),
        off={f"{n}_{m}": _mode_abs(love, n, m) for n, m in OFF_MODES},
        wall_s=float(result["wall_s"]),
        peak_rss_gb=_peak_rss_gb(),
    )
    return row


def _fmt_rel(cur: float, prev: float | None) -> str:
    if prev is None or prev == 0.0:
        return "   --   "
    return f"{(cur - prev) / prev:+7.2%}"


def _c20_channel_exponent(nrbase: int) -> dict:
    """Fit the (4,0)-only k20-shift scaling exponent (first vs second order).

    Isolates the (4,0) rheology harmonic from the default lmax=4 field, scales
    it by eps in {1e-3, 1e-2}, and fits log-log slope of |Delta k20|.  A slope
    ~1 means first order (self-coupling of the (2,0) tide), ~2 means second.
    """
    base = mu_variable_from_topography(lmax=4)[CRUST_LAYER_INDEX]
    e40 = [e for e in base if e[0] == 4 and e[1] == 0]
    if not e40:
        raise RuntimeError("no (4,0) harmonic in the default Moon field")

    model = build_moon_model()
    forcing = make_forcing(Td=MOON_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=model.n_layers, method="variable",
        Nrbase=nrbase, perturbation_order=2,
    )
    love_1d, _, _ = get_love(model, forcing, numerics)
    k2_1d = complex(love_1d.k[0])

    shifts = {}
    for eps in (1e-3, 1e-2):
        scaled = [(n, m, eps * amp) for n, m, amp in e40]
        love, _, _ = get_love(
            model, forcing, numerics,
            mu_variable={CRUST_LAYER_INDEX: scaled},
        )
        fidx = _forcing_index(love)
        shifts[eps] = abs(complex(love.k[fidx]) - k2_1d)
    exponent = math.log(shifts[1e-2] / shifts[1e-3]) / math.log(10.0)
    return {
        "shift_1e-3": shifts[1e-3],
        "shift_1e-2": shifts[1e-2],
        "exponent": exponent,
    }


def _save_figure(path: Path, ladder: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lmax = [r["lmax"] for r in ladder]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(lmax, [r["abs_delta_k20"] for r in ladder], "o-", label=r"$|\Delta k_{20}|$ (forcing)")
    for n, m in OFF_MODES:
        key = f"{n}_{m}"
        ax.plot(lmax, [r["off"][key] for r in ladder], "s--", label=f"$|k_{{{n},{m}}}|$")
    ax.set_yscale("log")
    ax.set_xticks(lmax)
    ax.set_xlabel("angular truncation $l_{max}$")
    ax.set_ylabel(r"$|k|$")
    ax.set_title("Moon lateral spectrum: angular convergence (TASK-031b)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax-list", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--ladder-nrbase", type=int, default=30)
    parser.add_argument("--nrbase-list", type=int, nargs="+", default=[15, 30, 50])
    parser.add_argument("--nrbase-lmax", type=int, default=4)
    parser.add_argument("--skip-c20-channel", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache: dict[tuple[int, int], dict] = {}

    def solve_cached(lmax: int, nrbase: int) -> dict:
        key = (lmax, nrbase)
        if key not in cache:
            print(f"[solve] lmax={lmax} Nrbase={nrbase} ...", flush=True)
            cache[key] = _solve(lmax, nrbase)
            r = cache[key]
            if r["blocked"]:
                print(
                    f"   BLOCKED (max|dmu/mubar|={r['max_abs_dmu_over_mubar']:.4f} "
                    f">= 1): {r['block_reason']}",
                    flush=True,
                )
            else:
                print(
                    f"   N={r['N']} |dk20|={r['abs_delta_k20']:.6e} "
                    f"max|dmu/mubar|={r['max_abs_dmu_over_mubar']:.4f} "
                    f"wall={r['wall_s']:.1f}s peakRSS={r['peak_rss_gb']:.1f}GB",
                    flush=True,
                )
        return cache[key]

    # --- 1. Angular ladder at fixed Nrbase ---
    ladder = [solve_cached(lmax, args.ladder_nrbase) for lmax in args.lmax_list]

    print("\n[angular ladder]  Nrbase=%d" % args.ladder_nrbase)
    header = f"{'lmax':>4} {'N':>5} {'max|dmu/mb|':>11} {'|dk20|':>13} {'d%':>8}"
    for n, m in OFF_MODES:
        header += f" {'|k'+str(n)+str(m)+'|':>12} {'d%':>8}"
    print(header)
    prev = {"dk20": None, **{f"{n}_{m}": None for n, m in OFF_MODES}}
    for r in ladder:
        if r["blocked"]:
            print(
                f"{r['lmax']:>4} {'--':>5} {r['max_abs_dmu_over_mubar']:11.4f}"
                f"   BLOCKED (linearization non-positive; field discarded)"
            )
            continue
        line = (
            f"{r['lmax']:>4} {r['N']:>5} {r['max_abs_dmu_over_mubar']:11.4f} "
            f"{r['abs_delta_k20']:13.6e} {_fmt_rel(r['abs_delta_k20'], prev['dk20'])}"
        )
        prev["dk20"] = r["abs_delta_k20"]
        for n, m in OFF_MODES:
            key = f"{n}_{m}"
            val = r["off"][key]
            line += f" {val:12.5e} {_fmt_rel(val, prev[key])}"
            prev[key] = val
        print(line)

    # --- 2. Nrbase-independence (verified, not imported) ---
    nrbase_runs = [solve_cached(args.nrbase_lmax, nb) for nb in args.nrbase_list]
    print(f"\n[Nrbase independence]  lmax={args.nrbase_lmax}")
    print(f"{'Nrbase':>6} {'N':>5} {'|dk20|':>13} {'rel to max Nrbase':>18}")
    valid = [r for r in nrbase_runs if not r["blocked"]]
    ref = valid[-1]["abs_delta_k20"] if valid else 0.0
    for r in nrbase_runs:
        if r["blocked"]:
            print(f"{r['nrbase']:>6} {'--':>5} {'BLOCKED':>13} {'--':>18}")
            continue
        rel = abs(r["abs_delta_k20"] - ref) / ref if ref else float("nan")
        print(f"{r['nrbase']:>6} {r['N']:>5} {r['abs_delta_k20']:13.6e} {rel:18.2e}")

    # --- 3. (4,0) first-order channel check ---
    channel = None
    if not args.skip_c20_channel:
        print("\n[(4,0) channel exponent]")
        channel = _c20_channel_exponent(args.ladder_nrbase)
        print(
            f"   shift(1e-3)={channel['shift_1e-3']:.6e} "
            f"shift(1e-2)={channel['shift_1e-2']:.6e} "
            f"exponent={channel['exponent']:.4f} "
            f"({'first' if channel['exponent'] < 1.5 else 'second'} order)"
        )

    # --- archive ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        lmax_list=np.asarray(args.lmax_list, dtype=int),
        ladder_nrbase=args.ladder_nrbase,
        ladder_N=np.asarray([r["N"] for r in ladder], dtype=int),
        ladder_abs_delta_k20=np.asarray([r["abs_delta_k20"] for r in ladder]),
        ladder_delta_k20=np.asarray([r["delta_k20"] for r in ladder], dtype=complex),
        ladder_wall_s=np.asarray([r["wall_s"] for r in ladder]),
        ladder_peak_rss_gb=np.asarray([r["peak_rss_gb"] for r in ladder]),
        nrbase_list=np.asarray(args.nrbase_list, dtype=int),
        nrbase_lmax=args.nrbase_lmax,
        nrbase_abs_delta_k20=np.asarray([r["abs_delta_k20"] for r in nrbase_runs]),
        nrbase_N=np.asarray([r["N"] for r in nrbase_runs], dtype=int),
        k2_uniform=WEBER_K2_UNIFORM,
        k2_sigma=MOON["k2_sigma"],
        peak_rss_gb_overall=_peak_rss_gb(),
        total_wall_s=time.perf_counter() - t0,
    )
    for n, m in OFF_MODES:
        key = f"{n}_{m}"
        save[f"ladder_off_{key}"] = np.asarray([r["off"][key] for r in ladder])
    if channel is not None:
        save["c40_exponent"] = channel["exponent"]
        save["c40_shift_1e-3"] = channel["shift_1e-3"]
        save["c40_shift_1e-2"] = channel["shift_1e-2"]
    np.savez(args.output, **save)
    _save_figure(args.output.with_suffix(".png"), ladder)

    print(f"\nsaved {args.output} and {args.output.with_suffix('.png')}")
    print(f"overall peak RSS = {_peak_rss_gb():.1f} GB, total wall = {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
