#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Monte Carlo sampler driver for the TASK-018 Moon interior model.

Runs ``pocomc`` (Preconditioned Monte Carlo) over the 4 free parameters of
``pylov3d.moon_mc.MOON_PARAMETERIZATION`` (core_rho_scale, mu_scale,
R_fluid_core, mantle_rho_scale), constrained by mass / mean MoI / k2 /
core_radius_km (see ``pylov3d.moon_mc``, "Identifiability" -- the 4th
constraint is required for a well-posed 4-parameter fit). This is the Moon
analogue of ``scripts/mars_pocomc.py`` and mirrors its structure exactly.

Headless (matplotlib Agg backend, set before any pyplot import). Saves the
chain (samples + weights, .npz) and a hand-rolled pairplot (no ``corner``
dependency), and prints posterior medians +/- 1 sigma (3 significant
figures) plus the effective sample size (ESS), alongside the as-built Weber
profile theta (``moon_point_estimate_theta``) for comparison.

TASK-019 context
----------------
``pylov3d.moon_mc``'s docstring quotes a reference R_fluid_core =
363.7 +/- 33.2 km run that was made *without* the shipped 0.88 core-density
FLOOR (an intermediate, unphysical [0.75, 1.25] core_rho_scale range). This
driver's non-quick default reproduces the run *properly-resolved* settings
(``n_active >= 64``, ``n_effective >= 128``) under the bounds this module
actually ships -- including the 0.88 floor -- so the posterior reflects the
model as released, not the pre-floor draft.

Usage
-----
    venvLOV3Dconv/bin/python scripts/moon_pocomc.py --quick
    venvLOV3Dconv/bin/python scripts/moon_pocomc.py --n-particles 64

``--quick`` is a tiny smoke run (n_particles=16, a coarse solver grid,
dynamic=False, a small fixed n_total) meant to finish in well under a
minute, *not* to produce a converged, publication-grade posterior -- for
that, use the (much slower) default settings or larger ``--n-particles``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from the repo root with `venvLOV3Dconv/bin/python scripts/moon_pocomc.py`;
# `scripts/` is not an installed package, so make sure the repo root (parent
# of `pylov3d/`) is importable regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pocomc
from scipy.stats import uniform

from pylov3d.forward import build_model, compute_observables
from pylov3d.moon_mc import (
    MOON_BOUNDS,
    MOON_FREE_PARAMS,
    MOON_PARAMETERIZATION,
    moon_forcing,
    moon_log_posterior,
    moon_numerics,
    moon_point_estimate_theta,
)

_DEFAULT_OUT_DIR = Path(__file__).parent / "output"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-particles", type=int, default=None,
        help="pocomc n_active (default: 64, or 16 with --quick).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help=(
            "Tiny smoke run: n_particles=16 (unless overridden), coarse solver "
            "grid (Nrbase=15), non-dynamic termination, small n_total."
        ),
    )
    p.add_argument("--nrbase", type=int, default=None, help="Override solver Nrbase.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    if args.quick:
        # Small n_active dominates wall time far more than n_total does: the
        # SMC annealing schedule (beta: 0 -> 1) always runs to completion
        # regardless of n_total, so keeping n_active small (rather than
        # relying on n_total as a cutoff) is what keeps --quick fast.
        n_particles = args.n_particles or 16
        n_effective = n_particles
        Nrbase = args.nrbase or 15
        n_total = 32
        dynamic = False
    else:
        # TASK-019 target: n_active >= 64, n_effective >= 128 -- the
        # "properly-resolved" settings quoted in pylov3d.moon_mc's docstring.
        n_particles = args.n_particles or 64
        n_effective = 2 * n_particles
        Nrbase = args.nrbase or 50  # Moon default (MOON_NUMERICS_NRBASE)
        n_total = None  # let pocomc's dynamic termination decide
        dynamic = True

    forcing = moon_forcing()
    numerics = moon_numerics(Nrbase=Nrbase)
    log_post = moon_log_posterior(Nrbase=Nrbase)

    # Per-eval timing (warm, post-JAX-compile) -- informs full-run cost estimates.
    theta0 = np.array(moon_point_estimate_theta())
    log_post(theta0)  # warm-up / JIT compile
    n_timing = 10
    t0 = time.perf_counter()
    for _ in range(n_timing):
        log_post(theta0)
    per_eval_s = (time.perf_counter() - t0) / n_timing
    print(f"[timing] per forward-model evaluation (Nrbase={Nrbase}, warm): {per_eval_s * 1000:.1f} ms")

    dists = [
        uniform(loc=MOON_BOUNDS[name][0], scale=MOON_BOUNDS[name][1] - MOON_BOUNDS[name][0])
        for name in MOON_FREE_PARAMS
    ]
    prior = pocomc.Prior(dists)

    sampler = pocomc.Sampler(
        prior, log_post, n_dim=len(MOON_FREE_PARAMS),
        n_active=n_particles, n_effective=n_effective,
        dynamic=dynamic, random_state=args.seed,
    )

    t0 = time.perf_counter()
    run_kwargs = {"progress": True}
    if n_total is not None:
        run_kwargs["n_total"] = n_total
        run_kwargs["n_evidence"] = 0
    sampler.run(**run_kwargs)
    wall_s = time.perf_counter() - t0

    samples, weights, logl, logp = sampler.posterior()
    ess = _effective_sample_size(weights)
    medians = np.array(
        [_weighted_quantile(samples[:, i], weights, 0.5) for i in range(samples.shape[1])]
    )
    lo1 = np.array(
        [_weighted_quantile(samples[:, i], weights, 0.1587) for i in range(samples.shape[1])]
    )
    hi1 = np.array(
        [_weighted_quantile(samples[:, i], weights, 0.8413) for i in range(samples.shape[1])]
    )

    point_estimate = np.array(moon_point_estimate_theta())

    print(f"\n[run] n_particles={n_particles}, Nrbase={Nrbase}, quick={args.quick}, "
          f"n_samples={samples.shape[0]}, ESS={ess:.1f}, wall={wall_s:.1f}s")
    if args.quick:
        print(
            "\n*** NOT CONVERGED -- demo only ***\n"
            "--quick uses a small n_active and a coarse solver grid purely to\n"
            "exercise the pipeline fast; its marginal widths are NOT a converged\n"
            "posterior. Use the (much slower) default settings for a result\n"
            "meant to be interpreted quantitatively."
        )
    print(f"\n{'param':<16} {'median':>14} {'-1sigma':>10} {'+1sigma':>10} {'as-built theta':>16}")
    for i, name in enumerate(MOON_FREE_PARAMS):
        med_r = _round_sig(medians[i])
        lo_r = _round_sig(medians[i] - lo1[i])
        hi_r = _round_sig(hi1[i] - medians[i])
        print(f"{name:<16} {med_r:>14.6g} {lo_r:>10.6g} {hi_r:>10.6g} {point_estimate[i]:>16.6g}")

    model = build_model(MOON_PARAMETERIZATION, medians)
    obs = compute_observables(
        model, forcing, numerics,
        which=("mass", "moi_mean", "core_radius_km", "k2"),
        core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
    )
    print(
        f"\n[median-model observables] mass={obs['mass']:.6e} kg, "
        f"moi_mean={obs['moi_mean']:.6f}, core_radius_km={obs['core_radius_km']:.3f}, "
        f"k2={obs['k2']:.6f}"
    )

    npz_path = args.out_dir / ("moon_chain_quick.npz" if args.quick else "moon_chain.npz")
    np.savez(
        npz_path, samples=samples, weights=weights, logl=logl, logp=logp,
        free_params=np.array(MOON_FREE_PARAMS), point_estimate=point_estimate,
    )
    print(f"\n[output] chain saved to {npz_path}")

    fig_path = args.out_dir / ("moon_pairplot_quick.png" if args.quick else "moon_pairplot.png")
    _pairplot(samples, weights, MOON_FREE_PARAMS, point_estimate, fig_path)
    print(f"[output] pairplot saved to {fig_path}")

    return {
        "medians": medians, "lo1": lo1, "hi1": hi1, "point_estimate": point_estimate,
        "per_eval_s": per_eval_s, "wall_s": wall_s, "n_samples": samples.shape[0],
        "ess": ess,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2), on posterior()'s
    (possibly unnormalized) importance weights -- invariant to an overall
    scale factor on the weights, ranges from 1 (all mass on one sample) to
    len(weights) (uniform weights)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w**2))


def _round_sig(x: float, sig: int = 3) -> float:
    """Round ``x`` to ``sig`` significant figures (0.0 and non-finite pass through)."""
    if x == 0.0 or not np.isfinite(x):
        return float(x)
    return float(round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1)))


def _pairplot(samples: np.ndarray, weights: np.ndarray, names, truth, out_path: Path) -> None:
    """Hand-rolled scatter-matrix (no ``corner`` dependency): diagonal =
    weighted 1-D histogram, off-diagonal = weight-alpha scatter, the as-built
    theta marked with red lines."""
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.4 * d, 2.4 * d))
    w_norm = weights / weights.max()

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], weights=weights, bins=25, color="steelblue")
                ax.axvline(truth[i], color="crimson", linestyle="--", linewidth=1)
            elif i > j:
                ax.scatter(
                    samples[:, j], samples[:, i], s=6, c="steelblue",
                    alpha=np.clip(w_norm, 0.05, 1.0).mean() * 0.6 + 0.1,
                    edgecolors="none",
                )
                ax.axvline(truth[j], color="crimson", linestyle="--", linewidth=0.7)
                ax.axhline(truth[i], color="crimson", linestyle="--", linewidth=0.7)
            else:
                ax.axis("off")
            if i == d - 1:
                ax.set_xlabel(names[j], fontsize=8)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Moon posterior (pocomc) vs. as-built Weber theta (dashed red)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run(parse_args())
