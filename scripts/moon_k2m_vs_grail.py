# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""TASK-034: Predicted Moon k2m splitting vs GRAIL's measured k20/k21/k22.

Driver only; no pylov3d module is modified.

Runs three diagonal coupled solves — forcing (2,0), (2,1), (2,2) — on the
Weber model with the shipped Airy lateral field at lmax=4 (the highest
cutoff the linearization admits: the dichotomy-retaining default's margin is
max|dmu/mu_bar|=0.9898 at lmax=4 and crosses unity at lmax=5; the superseded
degree-1-removed field's margin was 0.9902).  Reports each predicted k2m
against:
  - the uniform Weber background k2 = WEBER_K2_UNIFORM (0.02315914223)
  - GRAIL's measured values: k20=0.02408±0.00045, k21=0.02414±0.00025,
    k22=0.02394±0.00028 (Konopliv et al. 2013 Table 4, already committed in
    pylov3d/mars_detectability_k2m.py as GRAIL_K2M / GRAIL_K2M_SIGMA).

The expected outcome is a null: the predicted splitting is far below both the
individual GRAIL uncertainties and the observed order-to-order scatter.
The observed GRAIL spread (~2.0e-4) is itself smaller than the individual
uncertainties, so the measured order-to-order variation is likely not
statistically significant — the prediction landing two orders of magnitude
below all of these is the publishable result.

Artifacts: docs/figures/proposal/moon_k2m_vs_grail.{npz,png}
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

from pylov3d.mars_detectability_k2m import GRAIL_K2M, GRAIL_K2M_SIGMA
from pylov3d.moon import WEBER_K2_UNIFORM
from pylov3d.moon_lateral import moon_lateral_love_spectrum

DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "figures" / "proposal" / "moon_k2m_vs_grail.npz"
)
FORCING_ORDERS = (0, 1, 2)


def _peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return raw * scale / 1e9


def _forcing_index(love, n: int, m: int) -> int:
    matches = np.where((love.n == love.nf) & (love.m == love.mf))[0]
    if len(matches) != 1:
        raise RuntimeError(f"forcing mode ({n},{m}) missing or duplicated")
    return int(matches[0])


def _solve_k2m(m: int, lmax: int, nrbase: int) -> dict:
    """Coupled solve for forcing (2, m)."""
    t0 = time.perf_counter()
    result = moon_lateral_love_spectrum(lmax=lmax, forcing=(2, m), Nrbase=nrbase)
    wall_s = time.perf_counter() - t0
    love = result["love"]
    fidx = _forcing_index(love, 2, m)
    k2m = complex(love.k[fidx])
    delta_k2m = k2m.real - WEBER_K2_UNIFORM
    return {
        "m": m,
        "k2m": k2m.real,
        "delta_k2m": delta_k2m,
        "abs_delta_k2m": abs(delta_k2m),
        "N": int(len(love.k)),
        "wall_s": wall_s,
        "peak_rss_gb": _peak_rss_gb(),
    }


def _save_figure(path: Path, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = [r["m"] for r in results]
    pred = [abs(r["delta_k2m"]) for r in results]
    grail_vals = [GRAIL_K2M[m] for m in ms]
    grail_sig = [GRAIL_K2M_SIGMA[m] for m in ms]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: predicted k2m vs GRAIL
    ax = axes[0]
    x = np.arange(len(ms))
    ax.errorbar(x, grail_vals, yerr=grail_sig, fmt="o", capsize=5,
                color="#2c7fb8", label="GRAIL k2m ± 1σ")
    ax.axhline(WEBER_K2_UNIFORM, ls="--", color="gray", lw=0.8,
               label=f"Weber uniform k2 = {WEBER_K2_UNIFORM:.5f}")
    ax.set_xticks(x, [f"k2{m}" for m in ms])
    ax.set_ylabel("k2m")
    ax.set_title("Predicted vs GRAIL (lmax=4)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Right: predicted |Δk2m| vs GRAIL uncertainty and observed spread
    ax2 = axes[1]
    grail_spread = max(grail_vals) - min(grail_vals)
    ax2.bar(x, pred, color="#d95f02", label=r"predicted $|\Delta k_{2m}|$", zorder=3)
    ax2.axhline(min(grail_sig), ls=":", color="#2c7fb8", lw=1.2,
                label=f"min GRAIL σ = {min(grail_sig):.2e}")
    ax2.axhline(grail_spread, ls="-.", color="#756bb1", lw=1.2,
                label=f"GRAIL obs. spread = {grail_spread:.2e}")
    ax2.set_yscale("log")
    ax2.set_xticks(x, [f"|Δk2{m}|" for m in ms])
    ax2.set_ylabel(r"$|\Delta k_{2m}|$")
    ax2.set_title("Splitting vs precision and scatter")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle("TASK-034: Moon k2m splitting vs GRAIL", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--nrbase", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    t0_total = time.perf_counter()

    print(f"[TASK-034] Moon k2m vs GRAIL  lmax={args.lmax}  Nrbase={args.nrbase}")
    print(f"  Weber uniform k2  = {WEBER_K2_UNIFORM:.10f}")
    print(f"  GRAIL k20={GRAIL_K2M[0]:.5f}±{GRAIL_K2M_SIGMA[0]:.5f}  "
          f"k21={GRAIL_K2M[1]:.5f}±{GRAIL_K2M_SIGMA[1]:.5f}  "
          f"k22={GRAIL_K2M[2]:.5f}±{GRAIL_K2M_SIGMA[2]:.5f}")
    grail_spread = max(GRAIL_K2M.values()) - min(GRAIL_K2M.values())
    print(f"  GRAIL obs. spread (max−min) = {grail_spread:.2e}  "
          f"(cf. min σ = {min(GRAIL_K2M_SIGMA.values()):.2e}  → "
          f"{'NOT significant' if grail_spread < min(GRAIL_K2M_SIGMA.values()) else 'significant'})")

    results = []
    for m in FORCING_ORDERS:
        print(f"\n[solve] forcing (2,{m}) ...", flush=True)
        r = _solve_k2m(m, args.lmax, args.nrbase)
        results.append(r)
        ratio = r["abs_delta_k2m"] / GRAIL_K2M_SIGMA[m]
        print(f"  k2{m} = {r['k2m']:.10f}  Δk2{m} = {r['delta_k2m']:+.4e}  "
              f"|Δk2{m}|/σ_GRAIL = {ratio:.3e}  "
              f"N={r['N']}  wall={r['wall_s']:.1f}s  RSS={r['peak_rss_gb']:.1f}GB",
              flush=True)

    # --- Summary table ---
    print("\n[results]")
    print(f"{'m':>2} {'k2m':>14} {'Δk2m':>13} {'|Δk2m|':>12} "
          f"{'σ_GRAIL':>9} {'|Δ|/σ':>9} {'GRAIL_k2m':>11}")
    pred_vals = []
    for r in results:
        m = r["m"]
        ratio = r["abs_delta_k2m"] / GRAIL_K2M_SIGMA[m]
        pred_vals.append(r["k2m"])
        print(f"{m:>2} {r['k2m']:>14.10f} {r['delta_k2m']:>+13.4e} "
              f"{r['abs_delta_k2m']:>12.4e} "
              f"{GRAIL_K2M_SIGMA[m]:>9.5f} {ratio:>9.3e} "
              f"{GRAIL_K2M[m]:>11.5f}")

    pred_spread = max(pred_vals) - min(pred_vals)
    print(f"\nPredicted k2m spread (max−min) = {pred_spread:.4e}")
    print(f"GRAIL obs. spread              = {grail_spread:.4e}")
    print(f"Ratio pred/GRAIL-spread        = {pred_spread/grail_spread:.3e}")
    print(f"Min |Δk2m|/σ_GRAIL             = "
          f"{min(r['abs_delta_k2m']/GRAIL_K2M_SIGMA[r['m']] for r in results):.3e}")
    print(f"Max |Δk2m|/σ_GRAIL             = "
          f"{max(r['abs_delta_k2m']/GRAIL_K2M_SIGMA[r['m']] for r in results):.3e}")

    # Elastic/anelastic gap vs lateral splitting
    grail_mean_k2 = sum(GRAIL_K2M.values()) / 3
    elastic_anelastic_gap = grail_mean_k2 - WEBER_K2_UNIFORM
    mean_pred_split = sum(r["abs_delta_k2m"] for r in results) / len(results)
    print(f"\n[elastic/anelastic gap analysis]")
    print(f"  Mean GRAIL k2           = {grail_mean_k2:.5f}")
    print(f"  Weber elastic k2        = {WEBER_K2_UNIFORM:.5f}")
    print(f"  Elastic/anelastic gap   = {elastic_anelastic_gap:.4e}  "
          f"({elastic_anelastic_gap/grail_mean_k2*100:.1f}%)")
    print(f"  Mean |Δk2m| (lateral)   = {mean_pred_split:.4e}")
    print(f"  Gap / lateral splitting = {elastic_anelastic_gap/mean_pred_split:.1f}×")

    # --- Archive ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        lmax=args.lmax,
        nrbase=args.nrbase,
        forcing_orders=np.array(list(FORCING_ORDERS), dtype=int),
        k2m=np.array([r["k2m"] for r in results]),
        delta_k2m=np.array([r["delta_k2m"] for r in results]),
        abs_delta_k2m=np.array([r["abs_delta_k2m"] for r in results]),
        N=np.array([r["N"] for r in results], dtype=int),
        grail_k2m=np.array([GRAIL_K2M[m] for m in FORCING_ORDERS]),
        grail_k2m_sigma=np.array([GRAIL_K2M_SIGMA[m] for m in FORCING_ORDERS]),
        grail_obs_spread=grail_spread,
        pred_spread=pred_spread,
        weber_k2_uniform=WEBER_K2_UNIFORM,
        elastic_anelastic_gap=elastic_anelastic_gap,
        wall_s=np.array([r["wall_s"] for r in results]),
        total_wall_s=time.perf_counter() - t0_total,
        peak_rss_gb_overall=_peak_rss_gb(),
    )
    _save_figure(args.output.with_suffix(".png"), results)
    print(f"\nsaved {args.output} and {args.output.with_suffix('.png')}")
    print(f"total wall = {time.perf_counter() - t0_total:.1f} s  "
          f"peak RSS = {_peak_rss_gb():.1f} GB")


if __name__ == "__main__":
    main()
