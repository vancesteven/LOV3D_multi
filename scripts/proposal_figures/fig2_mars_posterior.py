#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""F2 — mars_posterior: 4x4 corner plot from a REAL pocomc run.

Runs ``pocomc`` (Preconditioned Monte Carlo) over the 4 free parameters of
``pylov3d.mars_mc.MARS_PARAMETERIZATION`` (rho_core, rho_lm, mu_scale,
R_core), using the committed ``pylov3d.mars_mc`` machinery (mass, mean MoI,
k2, core_radius_km constraints). Production settings are n_active=256,
Nrbase=100, and a fixed seed, matching the TASK-011 deterministic fit grid.
Saves the chain to
``docs/figures/proposal/mars_posterior_chain.npz`` for reproducibility.

Diagonal panels are 1-D weighted marginal histograms; the R_core diagonal
additionally overlays the Stahler et al. (2021) 1830 +/- 40 km seismology
prior as a dashed Gaussian. Off-diagonal panels are 2-D density (hexbin,
single-hue Blues) built from an equal-weight resample of the posterior.
The TASK-011 deterministic point fit is overlaid as a cross in every panel.

Usage
-----
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig2_mars_posterior.py
    venvLOV3Dconv/bin/python scripts/proposal_figures/fig2_mars_posterior.py --resample
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from common import (
    CATEGORICAL,
    FIGSIZE_FULL_SQUARE,
    INK,
    INK_GRAY,
    OUT_DIR,
    SEQUENTIAL_CMAP,
    apply_style,
    save_fig,
    style_axes,
)

from pylov3d.mars import MARS
from pylov3d.mars_mc import (
    MARS_BOUNDS,
    MARS_FREE_PARAMS,
    mars_log_posterior,
    mars_point_fit_theta,
)

PARAM_LABELS = {
    "rho_core": r"$\rho_{\rm core}$ [kg m$^{-3}$]",
    "rho_lm": r"$\rho_{\rm lm}$ [kg m$^{-3}$]",
    "mu_scale": r"$\mu_{\rm scale}$",
    "R_core": r"$R_{\rm core}$ [km]",
}

POINT_FIT_COLOR = CATEGORICAL[1]  # "categorical color 2" (1-based index 2)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-active", type=int, default=256)
    p.add_argument("--nrbase", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--chain-path", type=Path, default=None,
        help="Defaults to <out-dir>/mars_posterior_chain.npz",
    )
    p.add_argument(
        "--resample", action="store_true",
        help="Force a fresh pocomc run even if a chain file already exists.",
    )
    p.add_argument(
        "--ess-min", type=float, default=400.0,
        help="Minimum acceptable ESS; reruns once at n_active=512 if not met.",
    )
    return p.parse_args(argv)


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS: (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w**2))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def _run_pocomc(n_active: int, Nrbase: int, seed: int) -> dict:
    import pocomc
    from scipy.stats import uniform

    np.random.seed(seed)
    log_post = mars_log_posterior(Nrbase=Nrbase)

    dists = [
        uniform(loc=MARS_BOUNDS[name][0], scale=MARS_BOUNDS[name][1] - MARS_BOUNDS[name][0])
        for name in MARS_FREE_PARAMS
    ]
    prior = pocomc.Prior(dists)

    sampler = pocomc.Sampler(
        prior, log_post, n_dim=len(MARS_FREE_PARAMS),
        n_active=n_active, n_effective=2 * n_active,
        dynamic=True, random_state=seed,
    )
    t0 = time.perf_counter()
    sampler.run(progress=True)
    wall_s = time.perf_counter() - t0

    samples, weights, logl, logp = sampler.posterior()
    ess = _effective_sample_size(weights)
    print(f"[run] n_active={n_active}, Nrbase={Nrbase}, seed={seed}, "
          f"n_samples={samples.shape[0]}, ESS={ess:.1f}, wall={wall_s:.1f}s")
    return {
        "samples": samples, "weights": weights, "logl": logl, "logp": logp,
        "ess": ess, "wall_s": wall_s, "n_active": n_active, "Nrbase": Nrbase,
    }


def get_chain(args: argparse.Namespace) -> dict:
    chain_path = args.chain_path or (args.out_dir / "mars_posterior_chain.npz")

    if chain_path.exists() and not args.resample:
        d = np.load(chain_path, allow_pickle=True)
        print(f"[chain] loaded existing chain from {chain_path}")
        return {
            "samples": d["samples"], "weights": d["weights"],
            "logl": d["logl"], "logp": d["logp"],
            "ess": float(d["ess"]), "wall_s": float(d["wall_s"]),
            "n_active": int(d["n_active"]), "Nrbase": int(d["Nrbase"]),
            "chain_path": chain_path,
        }

    result = _run_pocomc(args.n_active, args.nrbase, args.seed)
    if result["ess"] <= args.ess_min:
        print(f"[run] ESS={result['ess']:.1f} <= {args.ess_min}; rerunning at n_active=512")
        result = _run_pocomc(512, args.nrbase, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    point_fit = np.array(mars_point_fit_theta())
    np.savez(
        chain_path,
        samples=result["samples"], weights=result["weights"],
        logl=result["logl"], logp=result["logp"],
        free_params=np.array(MARS_FREE_PARAMS), point_fit=point_fit,
        ess=result["ess"], wall_s=result["wall_s"],
        n_active=result["n_active"], Nrbase=result["Nrbase"], seed=args.seed,
    )
    print(f"[output] chain saved to {chain_path}")
    result["chain_path"] = chain_path
    return result


def make_figure(chain: dict, args: argparse.Namespace):
    samples = np.asarray(chain["samples"])
    weights = np.asarray(chain["weights"], dtype=float)
    ess = chain["ess"]
    names = list(MARS_FREE_PARAMS)
    d = len(names)
    point_fit = np.array(mars_point_fit_theta())

    # Equal-weight resample for smooth off-diagonal density plots.
    rng = np.random.default_rng(0)
    p = weights / weights.sum()
    n_resample = min(20000, max(4000, int(10 * ess)))
    idx = rng.choice(len(samples), size=n_resample, replace=True, p=p)
    rs = samples[idx]

    medians = np.array([_weighted_quantile(samples[:, i], weights, 0.5) for i in range(d)])
    lo1 = np.array([_weighted_quantile(samples[:, i], weights, 0.1587) for i in range(d)])
    hi1 = np.array([_weighted_quantile(samples[:, i], weights, 0.8413) for i in range(d)])

    apply_style()
    fig, axes = plt.subplots(d, d, figsize=FIGSIZE_FULL_SQUARE)

    r_core_km = MARS["core_radius"] / 1e3
    r_core_sigma_km = MARS["core_radius_sigma"] / 1e3

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            style_axes(ax)
            ax.tick_params(labelsize=7, length=2.5)

            if i == j:
                counts, bin_edges, _ = ax.hist(
                    samples[:, i], weights=weights, bins=32, density=True,
                    color=CATEGORICAL[0], alpha=0.85, edgecolor="none",
                )
                ax.axvline(point_fit[i], color=POINT_FIT_COLOR, linewidth=1.1, zorder=5)
                if names[i] == "R_core":
                    xs = np.linspace(*ax.get_xlim(), 300)
                    ax.plot(
                        xs, norm.pdf(xs, loc=r_core_km, scale=r_core_sigma_km),
                        color=INK_GRAY, linestyle="--", linewidth=1.1,
                        label="Stahler+2021 prior\n(1830 +/- 40 km)",
                    )
                    ax.set_ylim(top=ax.get_ylim()[1] * 1.4)  # headroom for the legend
                    ax.legend(loc="upper left", fontsize=7, handlelength=1.4,
                              borderaxespad=0.2, labelspacing=0.3)
                ax.set_yticks([])
            elif i > j:
                hb = ax.hexbin(
                    rs[:, j], rs[:, i], gridsize=28, cmap=SEQUENTIAL_CMAP,
                    mincnt=1, linewidths=0.0, rasterized=True,
                )
                ax.scatter(
                    [point_fit[j]], [point_fit[i]], marker="x", s=32,
                    linewidths=1.4, color=POINT_FIT_COLOR, zorder=6,
                )

            if i == d - 1:
                ax.set_xlabel(PARAM_LABELS[names[j]], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(PARAM_LABELS[names[i]], fontsize=7)
            elif j == 0 and i == 0:
                ax.set_ylabel("")

    fig.tight_layout(rect=(0.0, 0.025, 1.0, 0.95))

    # Single shared legend entry for the point-fit cross, placed inside the
    # blank upper-right corner of the grid (avoids colliding with the
    # suptitle and avoids repeating per-panel legends).
    axes[0, d - 1].text(
        0.5, 0.5, "x  TASK-011\ndeterministic\npoint fit",
        color=POINT_FIT_COLOR, fontsize=7.5, ha="center", va="center",
        transform=axes[0, d - 1].transAxes,
    )

    n_active = chain["n_active"]
    Nrbase = chain["Nrbase"]
    n_samples = samples.shape[0]
    fig.suptitle(
        "Mars interior posterior (pocomc, real run): "
        f"n_active={n_active}, Nrbase={Nrbase}, n_samples={n_samples}, ESS={ess:.0f}",
        fontsize=8.5, y=0.985,
    )
    fig.text(
        0.01, 0.012,
        "rho_core, rho_lm, mu_scale constrained by mass + mean MoI + k2; "
        "R_core is prior/seismology-driven (see core_radius_km constraint).",
        fontsize=7, color=INK_GRAY, ha="left", va="bottom", transform=fig.transFigure,
    )

    return fig, {"medians": medians, "lo1": lo1, "hi1": hi1, "names": names, "point_fit": point_fit}


def main(argv=None):
    args = parse_args(argv)
    chain = get_chain(args)
    fig, summary = make_figure(chain, args)
    pdf_path, png_path = save_fig(fig, "fig2_mars_posterior", args.out_dir)
    plt.close(fig)

    print(f"\n[output] {pdf_path}")
    print(f"[output] {png_path}")
    print(f"\n{'param':<10} {'median':>14} {'-1sigma':>10} {'+1sigma':>10} {'point fit':>14}")
    for i, name in enumerate(summary["names"]):
        print(f"{name:<10} {summary['medians'][i]:>14.6g} "
              f"{summary['medians'][i] - summary['lo1'][i]:>10.6g} "
              f"{summary['hi1'][i] - summary['medians'][i]:>10.6g} "
              f"{summary['point_fit'][i]:>14.6g}")
    print(f"\n[summary] ESS={chain['ess']:.1f}, n_samples={chain['samples'].shape[0]}, "
          f"wall_s={chain['wall_s']:.1f}")


if __name__ == "__main__":
    main()
