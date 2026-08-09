#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Joint anelastic Monte Carlo sampler for the Moon (TASK-025b).

The anelastic successor to ``scripts/moon_pocomc.py`` (the TASK-019 elastic
posterior). It samples rheology jointly with structure: the four structural
free parameters of ``pylov3d.moon_mc.MOON_PARAMETERIZATION``
(core_rho_scale, mu_scale, R_fluid_core, mantle_rho_scale) PLUS a mantle
viscosity ``log10_eta`` and -- for the Andrade path -- an Andrade
``alpha``, against FIVE observables: the four elastic-MC constraints
(mass, mean MoI, k2, core_radius_km) plus **Q = 38 +/- 4 at the draconic
month** (Williams & Boggs 2015; :data:`pylov3d.anelastic_moon.
MOON_MONTHLY_Q`), the fifth observable that lets the fit try to break the
rigidity/anelasticity degeneracy the elastic TASK-019 posterior cannot.

Two rheology paths (``--rheology``), per the TASK-025b spec (option c):

* ``maxwell`` (default, fast, fully validated): the complex k2 comes from
  pylov3d's OWN solver on the **real 10-layer Weber body** (ocean-aware,
  no PyALMA3) via :func:`pylov3d.anelastic._with_eta` + the ordinary
  ``compute_observables`` path -- so mass/MoI/core_radius/k2/Q all describe
  ONE theta-scaled body. Expected to rail the Q prior at its edge (Maxwell
  at the gap-closing viscosity implies Q ~ 0.79, 9.3 sigma from 38 +/- 4 --
  see docs/MOON_MODEL.md "Anelasticity (TASK-025a)"); that is a validated
  negative and a pipeline check, not the science answer.

* ``andrade`` (the science run, slow): the complex k2 comes from PyALMA3
  (pylov3d has no Andrade path -- see :mod:`pylov3d.anelastic` docstring)
  on a **theta-aware simplified body**. PyALMA3 cannot represent the Weber
  profile's internal ocean, so the three innermost layers are merged into a
  single fluid core (exactly as :func:`pylov3d.anelastic_moon.
  moon_simplified_body`, but rebuilt HERE from the theta-scaled model so
  ``mu_scale``/``mantle_rho_scale``/``R_fluid_core`` flow into the k2/Q body
  -- if the ALMA body ignored theta, mu_scale could not move k2/Q and the
  mu_scale-alpha correlation this run exists to measure would be forced to
  zero). Structural observables (mass/MoI/core_radius) still come from the
  REAL 10-layer body; only k2/Q use the simplified one. The residual
  real-vs-simplified structure penalty is small (~0.19% on Q at fixed
  parameters, TASK-025a Maxwell control) -- reported, not hidden.

``--no-q`` drops the Q constraint (keeping eta/alpha free): the controlled
"without Q" comparison for the degeneracy question -- run it alongside the
default (with Q) to read off how much the Q observable moves the mu_scale
marginal and the mu_scale-eta / mu_scale-alpha correlations.

Headless (Agg). Saves the chain (.npz, rheology + q/noq suffix) and a
hand-rolled pairplot, and prints posterior medians +/- 1 sigma, the Kish
ESS, weighted mu_scale-eta / mu_scale-alpha correlations, and the
median-model observables (including the implied Q).

Usage
-----
    venvLOV3Dconv/bin/python scripts/moon_anelastic_pocomc.py --quick
    venvLOV3Dconv/bin/python scripts/moon_anelastic_pocomc.py --rheology maxwell
    venvLOV3Dconv/bin/python scripts/moon_anelastic_pocomc.py --rheology andrade --n-particles 48
    venvLOV3Dconv/bin/python scripts/moon_anelastic_pocomc.py --rheology andrade --no-q

``--quick`` is a tiny smoke run (n_particles=16, coarse solver grid,
dynamic=False, small n_total; low ALMA precision for --rheology andrade)
meant to finish fast and exercise the pipeline, NOT to produce a converged
posterior. The Andrade path is arbitrary-precision (PyALMA3, ``--ndigits``)
and can be seconds per evaluation: the driver TIMES one warm evaluation and
prints it before the run so the wall-clock cost is visible up front.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Run from the repo root with `venvLOV3Dconv/bin/python scripts/...`; `scripts/`
# is not an installed package, so make the repo root (parent of `pylov3d/`)
# importable regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: must precede any pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pocomc
from scipy.stats import uniform

from pylov3d.anelastic import (
    ALMA_T0_S,
    ANDRADE_ALPHA_RANGE,
    MARS_MANTLE_ETA_ANDRADE_RANGE,
    _import_alma,
    _with_eta,
    implied_Q,
)
from pylov3d.anelastic_moon import (
    MOON_DRACONIC_MONTH_TD,
    MOON_MANTLE_LAYERS,
    MOON_MONTHLY_Q,
    MOON_MONTHLY_Q_SIGMA,
)
from pylov3d.forward import build_model, compute_observables, log_prior
from pylov3d.moon import MOON
from pylov3d.moon_mc import (
    MOON_BOUNDS,
    MOON_CONSTRAINTS,
    MOON_FREE_PARAMS,
    MOON_PARAMETERIZATION,
    moon_numerics,
    moon_point_estimate_theta,
)
from pylov3d.types import make_forcing

_DEFAULT_OUT_DIR = Path(__file__).parent / "output"

# Structural Gaussian constraints, keyed by name, pulled from moon_mc (not
# re-typed): mass, moi_mean, k2, core_radius_km. The Q constraint is added
# separately (MOON_MONTHLY_Q +/- MOON_MONTHLY_Q_SIGMA).
_STRUCT_C = {c.name: c for c in MOON_CONSTRAINTS}

#: eta prior [Pa s], log-uniform. Borrowed from the Mars Andrade mantle
#: range (Bagheri et al. 2019; :data:`pylov3d.anelastic.
#: MARS_MANTLE_ETA_ANDRADE_RANGE`) as a silicate-mantle ANALOGUE -- 025a's
#: literature table records no Moon-specific cited mantle-viscosity range
#: (only Goossens 2024's qualitative low-viscosity lower mantle), so this
#: is deliberately a Mars-derived analogue prior, not a Moon citation.
_ETA_RANGE = MARS_MANTLE_ETA_ANDRADE_RANGE


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rheology", choices=("maxwell", "andrade"), default="maxwell",
        help="maxwell = native real-body (fast, validated); andrade = PyALMA3 "
             "theta-aware simplified body (slow science run).",
    )
    p.add_argument(
        "--no-q", action="store_true",
        help="Drop the Q=38+/-4 constraint (controlled 'without Q' comparison "
             "for the degeneracy analysis); eta/alpha stay free.",
    )
    p.add_argument("--n-particles", type=int, default=None, help="pocomc n_active (default 64, or 16 with --quick).")
    p.add_argument(
        "--quick", action="store_true",
        help="Tiny smoke run: n_particles=16, coarse grid (Nrbase=15), "
             "non-dynamic, small n_total, low ALMA precision.",
    )
    p.add_argument("--nrbase", type=int, default=None, help="Override solver Nrbase (Maxwell path).")
    p.add_argument("--ndigits", type=int, default=None, help="PyALMA3 mantissa digits (Andrade path; default 48, or 24 with --quick).")
    p.add_argument("--order", type=int, default=8, help="PyALMA3 Gaver-Stehfest order (Andrade path).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# theta-aware anelastic complex k2
# ---------------------------------------------------------------------------

def _maxwell_k2_real_body(model, eta: float, forcing, numerics) -> complex:
    """Complex k2 of the REAL theta-scaled 10-layer body with a Maxwell
    mantle (all layers in :data:`MOON_MANTLE_LAYERS` share ``eta``), via
    pylov3d's own ocean-aware solver. ``compute_observables`` returns a
    complex k2 because the model is viscoelastic (mantle eta finite)."""
    model_ve = _with_eta(model, {i: eta for i in MOON_MANTLE_LAYERS})
    obs = compute_observables(model_ve, forcing, numerics, which=("k2",))
    return complex(obs["k2"])


def _simplified_body_from_model(model):
    """Rebuild :func:`pylov3d.anelastic_moon.moon_simplified_body`'s merge
    on the theta-SCALED model: merge the three innermost layers (0,1,2 =
    artificial core, inner core, fluid outer core) into one fluid layer at
    radius R0[2], density = combined mass / combined volume; keep layers
    3-9 with the model's (theta-scaled) rho/mu. Returns ALMA-ready
    (radii_km, rho, mu, mantle_layer_indices)."""
    n = model.n_layers
    radii = np.asarray(model.R0[:n], dtype=float)  # km
    rho = np.asarray(model.rho0[:n], dtype=float)
    mu = np.asarray(model.mu0[:n], dtype=float)

    bnd = np.concatenate([[0.0], radii])  # inner->outer boundaries, km
    vol = bnd[1:4] ** 3 - bnd[0:3] ** 3  # proportional to shell volume (layers 0,1,2)
    core_rho = float(np.sum(rho[:3] * vol) / np.sum(vol))

    simp_radii = [float(radii[2])] + [float(x) for x in radii[3:]]
    simp_rho = [core_rho] + [float(x) for x in rho[3:]]
    simp_mu = [0.0] + [float(x) for x in mu[3:]]
    mantle_layers = list(range(1, len(simp_radii) - 1))  # exclude core (0) and crust (last)
    return simp_radii, simp_rho, simp_mu, mantle_layers


def _andrade_k2_theta_aware(model, eta: float, alpha: float, Td: float, ndigits: int, order: int) -> complex:
    """Complex k2 via PyALMA3 on the theta-aware simplified body -- the
    Andrade analogue of :func:`_maxwell_k2_real_body`. Composes the same
    ALMA call ``pylov3d.anelastic_moon.moon_alma_k2`` makes, but on the
    body rebuilt from the theta-scaled ``model`` (so mu_scale/rho/radius
    flow into k2/Q) rather than the fixed as-built simplified body."""
    alma = _import_alma()
    radii, rho, mu, mantle_layers = _simplified_body_from_model(model)
    n = len(radii)
    r_in = [x * 1e3 for x in radii]  # m

    eta_in = [0.0] * n
    rheol = ["fluid"]
    params = np.zeros((n, 2))
    for i in range(1, n):
        if i in mantle_layers:
            eta_in[i] = eta
            rheol.append("andrade")
            params[i, 0] = alpha
        else:
            rheol.append("elastic")

    alma_model = alma.build_model(
        r_in, rho, mu, eta_in, rheol, params,
        ndigits=ndigits, verbose=False, parallel=False,
    )
    _h, _l, k = alma.love_numbers(
        [2], [Td / ALMA_T0_S], "tidal", "step", 0,
        alma_model, "complex", order=order, verbose=False, parallel=False,
    )
    return complex(k[0, 0])


# ---------------------------------------------------------------------------
# Log-posterior (custom: Q constrains Im(k2), which forward.log_likelihood
# cannot -- it compares Re only)
# ---------------------------------------------------------------------------

def _gaussian_ll(val: float, value: float, sigma: float) -> float:
    resid = (val - value) / sigma
    return -0.5 * resid * resid - math.log(sigma * math.sqrt(2.0 * math.pi))


def make_anelastic_log_posterior(rheology: str, *, use_q: bool, forcing, numerics,
                                 ndigits: int, order: int):
    """Build ``callable(theta) -> float``. theta = the 4 structural params +
    log10(eta) [+ alpha for andrade]. Structural observables come from the
    real theta-scaled body; complex k2 (hence Re k2 and Q) from the
    rheology-appropriate anelastic forward. Any solver/ALMA failure on an
    unphysical theta degrades to -inf (never crashes the sampler)."""
    is_andrade = rheology == "andrade"
    warn = {"n": 0}

    def log_post(theta) -> float:
        theta = np.asarray(theta, dtype=float)
        theta_struct = theta[:4]
        log10_eta = float(theta[4])
        alpha = float(theta[5]) if is_andrade else None

        # box gate: structural (via forward.log_prior) + eta/alpha bounds
        if not np.isfinite(log_prior(theta_struct, MOON_PARAMETERIZATION)):
            return -np.inf
        if not (math.log10(_ETA_RANGE[0]) <= log10_eta <= math.log10(_ETA_RANGE[1])):
            return -np.inf
        if is_andrade and not (ANDRADE_ALPHA_RANGE[0] <= alpha <= ANDRADE_ALPHA_RANGE[1]):
            return -np.inf

        eta = 10.0 ** log10_eta
        try:
            model = build_model(MOON_PARAMETERIZATION, theta_struct)
            # structural observables: analytic, rheology-independent, REAL body
            struct = compute_observables(
                model, forcing, numerics,
                which=("mass", "moi_mean", "core_radius_km"),
                core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
            )
            if is_andrade:
                k2 = _andrade_k2_theta_aware(model, eta, alpha, forcing.Td, ndigits, order)
            else:
                k2 = _maxwell_k2_real_body(model, eta, forcing, numerics)
        except Exception as exc:  # noqa: BLE001 - unphysical theta -> -inf, don't crash sampler
            warn["n"] += 1
            return -np.inf

        ll = 0.0
        ll += _gaussian_ll(struct["mass"], _STRUCT_C["mass"].value, _STRUCT_C["mass"].sigma)
        ll += _gaussian_ll(struct["moi_mean"], _STRUCT_C["moi_mean"].value, _STRUCT_C["moi_mean"].sigma)
        ll += _gaussian_ll(struct["core_radius_km"], _STRUCT_C["core_radius_km"].value, _STRUCT_C["core_radius_km"].sigma)
        ll += _gaussian_ll(k2.real, _STRUCT_C["k2"].value, _STRUCT_C["k2"].sigma)
        if use_q:
            ll += _gaussian_ll(implied_Q(k2), MOON_MONTHLY_Q, MOON_MONTHLY_Q_SIGMA)
        if not math.isfinite(ll):
            return -np.inf
        return ll

    log_post.warn = warn
    return log_post


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    is_andrade = args.rheology == "andrade"
    use_q = not args.no_q

    if args.quick:
        n_particles = args.n_particles or 16
        n_effective = n_particles
        Nrbase = args.nrbase or 15
        ndigits = args.ndigits or 24
        n_total = 32
        dynamic = False
    else:
        # TASK-019 standard: n_active >= 64, n_effective >= 128, dynamic.
        n_particles = args.n_particles or 64
        n_effective = 2 * n_particles
        Nrbase = args.nrbase or 50
        ndigits = args.ndigits or 48
        n_total = None
        dynamic = True

    free_params = list(MOON_FREE_PARAMS) + ["log10_eta"] + (["alpha"] if is_andrade else [])
    n_dim = len(free_params)

    forcing = make_forcing(Td=MOON_DRACONIC_MONTH_TD, n=2, m=0, F=1.0)
    numerics = moon_numerics(Nrbase=Nrbase)
    log_post = make_anelastic_log_posterior(
        args.rheology, use_q=use_q, forcing=forcing, numerics=numerics,
        ndigits=ndigits, order=args.order,
    )

    # priors: structural (uniform over MOON_BOUNDS) + log10_eta (log-uniform
    # => uniform on log10) + alpha (uniform). pocomc.Prior is the normalized
    # prior; log_post is the (unnormalized-box) likelihood.
    dists = [
        uniform(loc=MOON_BOUNDS[name][0], scale=MOON_BOUNDS[name][1] - MOON_BOUNDS[name][0])
        for name in MOON_FREE_PARAMS
    ]
    dists.append(uniform(loc=math.log10(_ETA_RANGE[0]),
                         scale=math.log10(_ETA_RANGE[1]) - math.log10(_ETA_RANGE[0])))
    if is_andrade:
        dists.append(uniform(loc=ANDRADE_ALPHA_RANGE[0],
                             scale=ANDRADE_ALPHA_RANGE[1] - ANDRADE_ALPHA_RANGE[0]))
    prior = pocomc.Prior(dists)

    # per-eval timing at a plausible interior point (warm, post-compile) --
    # matters most for the Andrade/PyALMA3 path (arbitrary precision).
    theta0 = list(moon_point_estimate_theta()) + [math.log10(math.sqrt(_ETA_RANGE[0] * _ETA_RANGE[1]))]
    if is_andrade:
        theta0.append(0.3)
    theta0 = np.array(theta0)
    log_post(theta0)  # warm-up / JIT compile
    n_timing = 3 if is_andrade else 10
    t0 = time.perf_counter()
    for _ in range(n_timing):
        log_post(theta0)
    per_eval_s = (time.perf_counter() - t0) / n_timing
    print(f"[timing] per log-posterior eval (rheology={args.rheology}, "
          f"Nrbase={Nrbase}, ndigits={ndigits if is_andrade else 'n/a'}, warm): "
          f"{per_eval_s * 1000:.1f} ms")
    print(f"[config] free_params={free_params}, use_q={use_q}, "
          f"n_active={n_particles}, n_effective={n_effective}, dynamic={dynamic}")

    sampler = pocomc.Sampler(
        prior, log_post, n_dim=n_dim,
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
    medians = np.array([_weighted_quantile(samples[:, i], weights, 0.5) for i in range(n_dim)])
    lo1 = np.array([_weighted_quantile(samples[:, i], weights, 0.1587) for i in range(n_dim)])
    hi1 = np.array([_weighted_quantile(samples[:, i], weights, 0.8413) for i in range(n_dim)])

    print(f"\n[run] rheology={args.rheology}, use_q={use_q}, n_dim={n_dim}, "
          f"n_samples={samples.shape[0]}, ESS={ess:.1f}, wall={wall_s:.1f}s, "
          f"solver_failures={log_post.warn['n']}")
    if args.quick:
        print("\n*** NOT CONVERGED -- demo only (--quick) ***")
    print(f"\n{'param':<16} {'median':>14} {'-1sigma':>12} {'+1sigma':>12}")
    for i, name in enumerate(free_params):
        print(f"{name:<16} {_round_sig(medians[i]):>14.6g} "
              f"{_round_sig(medians[i] - lo1[i]):>12.6g} {_round_sig(hi1[i] - medians[i]):>12.6g}")

    # degeneracy diagnostics: weighted correlations mu_scale vs eta / alpha.
    mu_i = free_params.index("mu_scale")
    eta_i = free_params.index("log10_eta")
    corr_mu_eta = _weighted_corr(samples[:, mu_i], samples[:, eta_i], weights)
    print(f"\n[degeneracy] mu_scale median = {medians[mu_i]:.4f} "
          f"(elastic TASK-019 reference: 0.965)")
    print(f"[degeneracy] weighted corr(mu_scale, log10_eta) = {corr_mu_eta:+.3f}")
    if is_andrade:
        a_i = free_params.index("alpha")
        corr_mu_a = _weighted_corr(samples[:, mu_i], samples[:, a_i], weights)
        print(f"[degeneracy] weighted corr(mu_scale, alpha)    = {corr_mu_a:+.3f}")
        print(f"[degeneracy] alpha median = {medians[a_i]:.3f} "
              f"(Efroimsky common range 0.2-0.3)")

    # median-model observables, including implied Q
    model = build_model(MOON_PARAMETERIZATION, medians[:4])
    struct = compute_observables(
        model, forcing, numerics, which=("mass", "moi_mean", "core_radius_km"),
        core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
    )
    eta_med = 10.0 ** medians[eta_i]
    if is_andrade:
        k2_med = _andrade_k2_theta_aware(model, eta_med, medians[free_params.index("alpha")],
                                         forcing.Td, ndigits, args.order)
    else:
        k2_med = _maxwell_k2_real_body(model, eta_med, forcing, numerics)
    print(f"\n[median-model observables] mass={struct['mass']:.6e} kg, "
          f"moi_mean={struct['moi_mean']:.6f}, core_radius_km={struct['core_radius_km']:.3f}")
    print(f"[median-model observables] k2={k2_med.real:.6f} (obs {MOON['k2']:.5f}), "
          f"Q={implied_Q(k2_med):.2f} (obs {MOON_MONTHLY_Q:.0f}+/-{MOON_MONTHLY_Q_SIGMA:.0f})")

    tag = f"{args.rheology}{'' if use_q else '_noq'}{'_quick' if args.quick else ''}"
    npz_path = args.out_dir / f"moon_anelastic_chain_{tag}.npz"
    np.savez(
        npz_path, samples=samples, weights=weights, logl=logl, logp=logp,
        free_params=np.array(free_params), rheology=args.rheology, use_q=use_q,
        eta_range=np.array(_ETA_RANGE), alpha_range=np.array(ANDRADE_ALPHA_RANGE),
    )
    print(f"\n[output] chain saved to {npz_path}")
    fig_path = args.out_dir / f"moon_anelastic_pairplot_{tag}.png"
    _pairplot(samples, weights, free_params, fig_path,
              title=f"Moon anelastic posterior ({args.rheology}, "
                    f"{'with' if use_q else 'without'} Q)")
    print(f"[output] pairplot saved to {fig_path}")

    return {
        "medians": medians, "lo1": lo1, "hi1": hi1, "free_params": free_params,
        "per_eval_s": per_eval_s, "wall_s": wall_s, "ess": ess,
        "corr_mu_eta": corr_mu_eta, "k2_med": k2_med, "Q_med": implied_Q(k2_med),
    }


# ---------------------------------------------------------------------------
# Weighted statistics + plotting (mirrors scripts/moon_pocomc.py)
# ---------------------------------------------------------------------------

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def _effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w**2))


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation coefficient."""
    w = np.asarray(w, dtype=float)
    wsum = np.sum(w)
    mx = np.sum(w * x) / wsum
    my = np.sum(w * y) / wsum
    cov = np.sum(w * (x - mx) * (y - my)) / wsum
    vx = np.sum(w * (x - mx) ** 2) / wsum
    vy = np.sum(w * (y - my) ** 2) / wsum
    denom = math.sqrt(vx * vy)
    return float(cov / denom) if denom > 0 else float("nan")


def _round_sig(x: float, sig: int = 3) -> float:
    if x == 0.0 or not np.isfinite(x):
        return float(x)
    return float(round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1)))


def _pairplot(samples: np.ndarray, weights: np.ndarray, names, out_path: Path, title: str) -> None:
    """Hand-rolled scatter-matrix (no ``corner`` dependency): diagonal =
    weighted 1-D histogram, off-diagonal = weight-alpha scatter."""
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.4 * d, 2.4 * d))
    w_norm = weights / weights.max()
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], weights=weights, bins=25, color="steelblue")
            elif i > j:
                ax.scatter(
                    samples[:, j], samples[:, i], s=6, c="steelblue",
                    alpha=np.clip(w_norm, 0.05, 1.0).mean() * 0.6 + 0.1, edgecolors="none",
                )
            else:
                ax.axis("off")
            if i == d - 1:
                ax.set_xlabel(names[j], fontsize=8)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run(parse_args())
