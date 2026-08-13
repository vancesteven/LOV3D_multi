#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""TASK-037: excursion-determined shell -- does the ladder converge in lmax?

Driver only; no ``pylov3d`` module is modified.

TASK-036b's T-sweep (``scripts/moon_lateral_t_sweep.py``) showed the lateral
Love spectrum does not converge in a *fixed* shell thickness T -- widening T
chases the amplitude toward zero as 1/T, so T cannot be picked freely and
then swept away. TASK-037's proposal is that T is not free: it should be
tied to the crustal-thickness excursion itself, so it varies with the SH
truncation ``lmax`` rather than being chosen by hand.

Two ways to define "the excursion" are implemented, and they are NOT
equivalent for the Moon's Airy-crust field, which is strongly asymmetric
(the far-side highlands push |dt| positive; South Pole-Aitken pushes it
sharply negative):

  --rule twice_max :  T = 2 * max|dt|                (spatial max of |dt|)
  --rule range     :  T = dt_max - dt_min             (true spatial excursion)

Under "twice_max" the perturbation margin ``max|delta_mu/mu_bar|`` is
*identically* K/2 at every truncation (K = |mu_c - mu_m| / mu_c is the
material contrast alone) -- that is the rule's defining property, and this
driver asserts it. Under "range" the margin is not pinned and is expected to
drift with lmax.

Both spatial extrema (``dt_max``, ``dt_min``) are computed by GRIDDING the
SH field (:func:`pylov3d.mapping.sh_to_latlon`, nlat=360/nlon=720), not read
off the SH coefficients -- the largest SH coefficient underestimates the
spatial extremum by a large factor for a broadband field. This driver
builds the T-rescaled ``mu_variable`` by hand (the same pattern as
``scripts/moon_lateral_t_sweep.py``): scale the dt SH coefficients by
``(mu_c - mu_m) / (T * mu_c)`` and convert with
``_real_sh_to_complex_mu_variable`` directly, bypassing
``mu_variable_from_topography`` (its positivity guard is keyed to the
shipped default T and would raise for these rescaled shells).

The 1-D background is re-checked at every T via a zero-amplitude coupled
solve, because widening the shell also changes the *reference* model that
the lateral perturbation for the background 1-D case (Weber Moon / Mars
uniform) is being compared against -- TASK-036b's caveat, carried forward
here rather than assumed to still hold.

Artifacts: docs/figures/proposal/{body}_excursion_shell_ladder.{npz,png}
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
from pylov3d.mapping import sh_to_latlon
from pylov3d.types import make_forcing, make_numerics

from pylov3d.moon import (
    LAYER_MU as MOON_LAYER_MU,
    MOON_FORCING_TD,
    WEBER_K2_UNIFORM,
    build_moon_model,
)
from pylov3d.moon_lateral import (
    CRUST_LAYER_INDEX as MOON_CRUST_LAYER_INDEX,
    _real_sh_to_complex_mu_variable,
    crustal_thickness_variation as moon_crustal_thickness_variation,
)

from pylov3d.mars import (
    LAYER_MU_CRUST as MARS_LAYER_MU_CRUST,
    MARS_FORCING_TD,
    MARS_MU_SCALE,
    _MU_UM_BASE as MARS_MU_UM_BASE,
    build_mars_model,
)
from pylov3d.mars_lateral import (
    CRUST_LAYER_INDEX as MARS_CRUST_LAYER_INDEX,
    crustal_thickness_variation as mars_crustal_thickness_variation,
)

MOON_MANTLE_LAYER_INDEX = MOON_CRUST_LAYER_INDEX - 1

OFF_MODES = ((2, 2), (2, 1), (3, 3))
RULES = ("twice_max", "range")
POSITIVITY_TOL = 1e-3

DEFAULT_LMAX_LIST = [4, 5, 6]

# TASK-037 reference numbers (team lead, computed ahead of this driver), for
# the diagnostics self-check. Moon only -- Mars was not pre-computed by the
# team lead and is reported here instead of asserted against.
_MOON_REF = {
    4: {"dt_max_km": 20.062, "dt_min_km": -32.630, "T_twice_max_km": 65.260, "T_range_km": 52.692},
    5: {"dt_max_km": 22.095, "dt_min_km": -38.000, "T_twice_max_km": 75.999, "T_range_km": 60.095},
    6: {"dt_max_km": 20.615, "dt_min_km": -42.506, "T_twice_max_km": 85.012, "T_range_km": 63.121},
}
_MOON_REF_TOL_KM = 0.05  # generous vs. the 3-decimal reference printout


def _peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return raw * scale / 1e9


def _fmt_rel(cur: float, prev: float | None) -> str:
    if prev is None or prev == 0.0:
        return "   --   "
    return f"{(cur - prev) / prev:+7.2%}"


# ---------------------------------------------------------------------------
# Body-specific adapters: everything downstream is body-agnostic.
# ---------------------------------------------------------------------------

class _MoonBody:
    name = "moon"
    K = abs(MOON_LAYER_MU[MOON_CRUST_LAYER_INDEX] - MOON_LAYER_MU[MOON_MANTLE_LAYER_INDEX]) / \
        MOON_LAYER_MU[MOON_CRUST_LAYER_INDEX]
    crust_layer_index = MOON_CRUST_LAYER_INDEX
    forcing_Td = MOON_FORCING_TD
    k2_uniform = WEBER_K2_UNIFORM
    numerics_method = "variable"

    @staticmethod
    def crustal_thickness_variation(lmax: int) -> dict:
        return moon_crustal_thickness_variation(lmax=lmax)

    @staticmethod
    def coeff_for_T(T_m: float) -> float:
        mu_c = MOON_LAYER_MU[MOON_CRUST_LAYER_INDEX]
        mu_m = MOON_LAYER_MU[MOON_MANTLE_LAYER_INDEX]
        return (mu_c - mu_m) / (T_m * mu_c)

    @staticmethod
    def build_model():
        return build_moon_model()


class _MarsBody:
    name = "mars"
    K = abs(MARS_LAYER_MU_CRUST - MARS_MU_UM_BASE * MARS_MU_SCALE) / MARS_LAYER_MU_CRUST
    crust_layer_index = MARS_CRUST_LAYER_INDEX
    forcing_Td = MARS_FORCING_TD
    k2_uniform = None  # not a shipped constant; measured via a zero-amplitude solve (see main())
    numerics_method = "combination"

    @staticmethod
    def crustal_thickness_variation(lmax: int) -> dict:
        return mars_crustal_thickness_variation(lmax=lmax)

    @staticmethod
    def coeff_for_T(T_m: float) -> float:
        mu_c = MARS_LAYER_MU_CRUST
        mu_o = MARS_MU_UM_BASE * MARS_MU_SCALE
        return (mu_c - mu_o) / (T_m * mu_c)

    @staticmethod
    def build_model():
        return build_mars_model()


BODIES = {"moon": _MoonBody, "mars": _MarsBody}


# ---------------------------------------------------------------------------
# T / margin computation (diagnostics; no solves)
# ---------------------------------------------------------------------------

def _dt_extrema(body, lmax: int, nlat: int = 360, nlon: int = 720) -> tuple[float, float]:
    """(dt_max, dt_min) [m] on a fine lat/lon grid -- NOT the SH coefficient max."""
    dt = body.crustal_thickness_variation(lmax=lmax)
    grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon)
    return float(np.max(grid.z)), float(np.min(grid.z))


def _T_for_rule(rule: str, dt_max: float, dt_min: float) -> float:
    if rule == "twice_max":
        return 2.0 * max(abs(dt_max), abs(dt_min))
    if rule == "range":
        return dt_max - dt_min
    raise ValueError(f"unknown rule {rule!r}")


def _margin_for_T(body, T_m: float, dt_max: float, dt_min: float) -> float:
    """max|delta_mu/mu_bar| for this T, using the spatial |dt| extremum."""
    return abs(body.coeff_for_T(T_m)) * max(abs(dt_max), abs(dt_min))


def _diagnostics_row(body, lmax: int) -> dict:
    dt_max, dt_min = _dt_extrema(body, lmax)
    row = {"lmax": lmax, "dt_max_m": dt_max, "dt_min_m": dt_min}
    for rule in RULES:
        T_m = _T_for_rule(rule, dt_max, dt_min)
        margin = _margin_for_T(body, T_m, dt_max, dt_min)
        row[f"T_{rule}_m"] = T_m
        row[f"margin_{rule}"] = margin
    return row


def _print_diagnostics(body, lmax_list: list[int]) -> list[dict]:
    print(f"\n[diagnostics] body={body.name}  K=|mu_c-mu_m|/mu_c={body.K:.4f}  K/2={body.K/2:.4f}")
    header = (
        f"{'lmax':>4} {'dt_max_km':>10} {'dt_min_km':>10} "
        f"{'T_twice_max_km':>14} {'margin_tm':>10} "
        f"{'T_range_km':>11} {'margin_rng':>10}"
    )
    print(header)
    rows = []
    for lmax in lmax_list:
        row = _diagnostics_row(body, lmax)
        rows.append(row)
        print(
            f"{lmax:>4} {row['dt_max_m']/1000:>10.3f} {row['dt_min_m']/1000:>10.3f} "
            f"{row['T_twice_max_m']/1000:>14.3f} {row['margin_twice_max']:>10.4f} "
            f"{row['T_range_m']/1000:>11.3f} {row['margin_range']:>10.4f}"
        )

    # Reference self-check (Moon only; Mars has no pre-computed reference).
    if body.name == "moon":
        print("\n[reference self-check] (team-lead precomputed Moon numbers)")
        all_ok = True
        for row in rows:
            lmax = row["lmax"]
            ref = _MOON_REF.get(lmax)
            if ref is None:
                continue
            checks = [
                ("dt_max_km", row["dt_max_m"] / 1000, ref["dt_max_km"]),
                ("dt_min_km", row["dt_min_m"] / 1000, ref["dt_min_km"]),
                ("T_twice_max_km", row["T_twice_max_m"] / 1000, ref["T_twice_max_km"]),
                ("T_range_km", row["T_range_m"] / 1000, ref["T_range_km"]),
            ]
            for label, computed, reference in checks:
                diff = abs(computed - reference)
                ok = diff < _MOON_REF_TOL_KM
                all_ok &= ok
                status = "OK" if ok else "MISMATCH"
                print(
                    f"  lmax={lmax} {label}: computed={computed:.3f} "
                    f"reference={reference:.3f} diff={diff:.4f} [{status}]"
                )
        print(f"  reference self-check: {'ALL PASSED' if all_ok else 'FAILURES ABOVE'}")

    # Positivity-margin assertion: under "twice_max" the margin must equal
    # K/2 identically at every lmax (the rule's defining algebraic property).
    print("\n[twice_max margin assertion]  margin must equal K/2 at every lmax")
    K_half = body.K / 2.0
    assertion_ok = True
    for row in rows:
        diff = abs(row["margin_twice_max"] - K_half)
        ok = diff < POSITIVITY_TOL
        assertion_ok &= ok
        status = "OK" if ok else "FAILED"
        print(
            f"  lmax={row['lmax']}: margin_twice_max={row['margin_twice_max']:.6f} "
            f"K/2={K_half:.6f} diff={diff:.2e} [{status}]"
        )
    if not assertion_ok:
        raise AssertionError(
            f"{body.name}: twice_max margin assertion FAILED (tol={POSITIVITY_TOL}) -- "
            "see per-lmax diffs above."
        )
    print("  twice_max margin assertion: PASSED at every lmax")

    return rows


# ---------------------------------------------------------------------------
# Solve path
# ---------------------------------------------------------------------------

def _mu_variable_at_T(body, lmax: int, T_m: float) -> dict[int, list[tuple[int, int, complex]]]:
    """Build mu_variable by hand for a given reference shell thickness T_m [m].

    Bypasses the shipped ``mu_variable_from_topography`` positivity guard
    (keyed to the default T), matching ``scripts/moon_lateral_t_sweep.py``.
    """
    dt = body.crustal_thickness_variation(lmax=lmax)
    coeff = body.coeff_for_T(T_m)
    dmu_over_mu = {nm: coeff * v for nm, v in dt.items()}
    entries = _real_sh_to_complex_mu_variable(dmu_over_mu)
    return {body.crust_layer_index: entries}


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


def _make_numerics(body, nrbase: int):
    model = body.build_model()
    return make_numerics(
        n_layers=model.n_layers,
        method=body.numerics_method,
        Nrbase=nrbase,
        perturbation_order=2,
    )


def _solve_1d(body, nrbase: int) -> float:
    """Zero-amplitude coupled solve -- the background 1-D k2, re-checked at
    every T rather than assumed T-independent (TASK-036b caveat)."""
    model = body.build_model()
    forcing = make_forcing(Td=body.forcing_Td, n=2, m=0, F=1.0)
    numerics = _make_numerics(body, nrbase)
    love, _, _ = get_love(model, forcing, numerics)
    return float(love.k[0].real)


def _solve_at_T(body, lmax: int, T_m: float, nrbase: int, k2_uniform: float) -> dict:
    """Coupled solve at a given reference-shell thickness T_m and lmax."""
    model = body.build_model()
    forcing = make_forcing(Td=body.forcing_Td, n=2, m=0, F=1.0)
    numerics = _make_numerics(body, nrbase)
    mu_var = _mu_variable_at_T(body, lmax, T_m)

    dt_max, dt_min = _dt_extrema(body, lmax)
    margin = _margin_for_T(body, T_m, dt_max, dt_min)

    t0 = time.perf_counter()
    love, _, _ = get_love(model, forcing, numerics, mu_variable=mu_var)
    wall_s = time.perf_counter() - t0

    fidx = _forcing_index(love)
    delta_k20 = complex(love.k[fidx]) - k2_uniform
    return {
        "lmax": lmax,
        "T_km": T_m / 1000.0,
        "T_m": T_m,
        "margin": margin,
        "N": int(len(love.k)),
        "k20_coupled": float(love.k[fidx].real),
        "delta_k20": complex(delta_k20),
        "abs_delta_k20": float(abs(delta_k20)),
        "off": {f"{n}_{m}": _mode_abs(love, n, m) for n, m in OFF_MODES},
        "wall_s": wall_s,
        "peak_rss_gb": _peak_rss_gb(),
    }


def _run_ladder(body, rule: str, lmax_list: list[int], nrbase: int, k2_uniform: float) -> list[dict]:
    print(f"\n[ladder]  body={body.name}  rule={rule}  Nrbase={nrbase}")
    header = f"{'lmax':>4} {'T_km':>8} {'margin':>8} {'N':>5} {'|dk20|':>13} {'d%':>8}"
    for n, m in OFF_MODES:
        header += f" {'|k'+str(n)+str(m)+'|':>12} {'d%':>8}"
    print(header)

    rows = []
    prev = {"dk20": None, **{f"{n}_{m}": None for n, m in OFF_MODES}}
    for lmax in lmax_list:
        dt_max, dt_min = _dt_extrema(body, lmax)
        T_m = _T_for_rule(rule, dt_max, dt_min)
        print(f"[solve] lmax={lmax} T={T_m/1000:.2f} km ...", flush=True)
        r = _solve_at_T(body, lmax, T_m, nrbase, k2_uniform)
        rows.append(r)
        line = (
            f"{r['lmax']:>4} {r['T_km']:>8.2f} {r['margin']:>8.4f} "
            f"{r['N']:>5} {r['abs_delta_k20']:13.6e} "
            f"{_fmt_rel(r['abs_delta_k20'], prev['dk20'])}"
        )
        prev["dk20"] = r["abs_delta_k20"]
        for n, m in OFF_MODES:
            key = f"{n}_{m}"
            val = r["off"][key]
            line += f" {val:12.5e} {_fmt_rel(val, prev[key])}"
            prev[key] = val
        print(line, flush=True)
    return rows


def _save_figure(path: Path, body_name: str, K: float, ladders: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    styles = {"twice_max": ("o-", "s--"), "range": ("^-", "D--")}
    for rule, rows in ladders.items():
        if not rows:
            continue
        lmax_vals = [r["lmax"] for r in rows]
        dk20 = [r["abs_delta_k20"] for r in rows]
        main_style, off_style = styles.get(rule, ("o-", "s--"))
        ax.plot(lmax_vals, dk20, main_style, label=rf"$|\Delta k_{{20}}|$ ({rule})")
        for n, m in OFF_MODES:
            key = f"{n}_{m}"
            ax.plot(
                lmax_vals, [r["off"][key] for r in rows], off_style,
                alpha=0.6, label=f"$|k_{{{n},{m}}}|$ ({rule})",
            )
    ax.set_xlabel("lmax")
    ax.set_ylabel(r"$|k|$")
    ax.set_title(f"{body_name}: excursion-shell ladder, $|\\Delta k_{{20}}|$ and off-modes")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    for rule, rows in ladders.items():
        if not rows:
            continue
        lmax_vals = [r["lmax"] for r in rows]
        margins = [r["margin"] for r in rows]
        main_style, _ = styles.get(rule, ("o-", "s--"))
        ax2.plot(lmax_vals, margins, main_style, label=rule)
    ax2.axhline(K / 2.0, ls=":", color="g", label="K/2")
    ax2.axhline(1.0, ls=":", color="r", label="positivity limit")
    ax2.set_xlabel("lmax")
    ax2.set_ylabel(r"max$|\delta\mu/\bar{\mu}|$")
    ax2.set_title("Positivity margin vs lmax")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(f"TASK-037: {body_name} excursion-determined shell ladder", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--body", choices=("moon", "mars"), default="moon")
    parser.add_argument("--rule", choices=("twice_max", "range", "both"), default="both")
    parser.add_argument("--lmax-list", type=int, nargs="+", default=DEFAULT_LMAX_LIST)
    parser.add_argument("--nrbase", type=int, default=30)
    parser.add_argument(
        "--diagnostics-only", action="store_true",
        help="Print the T/margin table for every lmax and rule; run NO coupled solves.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    body = BODIES[args.body]
    output = args.output or (
        REPO_ROOT / "docs" / "figures" / "proposal" / f"{args.body}_excursion_shell_ladder.npz"
    )

    t0_total = time.perf_counter()

    diag_rows = _print_diagnostics(body, args.lmax_list)

    if args.diagnostics_only:
        print(f"\n--diagnostics-only: exiting without any coupled solve.")
        return

    rules = [args.rule] if args.rule != "both" else list(RULES)

    # Uniform (background) k2 -- for Mars this is not a shipped constant, so
    # measure it explicitly rather than assume it matches MARS["k2"]=0.169
    # (the fitted MARS_MU_SCALE target, not necessarily what get_love with
    # method="combination" reproduces bit-for-bit at this Nrbase).
    if body.k2_uniform is not None:
        k2_uniform = body.k2_uniform
        print(f"\n[k2_uniform]  body={body.name}  shipped constant = {k2_uniform:.10f}")
    else:
        k2_uniform = _solve_1d(body, args.nrbase)
        print(f"\n[k2_uniform]  body={body.name}  measured (zero-amplitude solve) = {k2_uniform:.10f}")

    # Re-check the 1-D background is stable across the ladder's T values
    # (TASK-036b caveat: widening the shell changes the background model;
    # do not assume it is T-independent).
    print(f"\n[1D background re-check]  Nrbase={args.nrbase}")
    k2_1d = _solve_1d(body, args.nrbase)
    diff = abs(k2_1d - k2_uniform)
    print(f"   k2_1d (zero-amplitude) = {k2_1d:.10f}")
    print(f"   k2_uniform (baseline)  = {k2_uniform:.10f}")
    print(f"   diff                   = {diff:.2e}")
    print(f"   background is {'T-independent as expected' if diff < 1e-8 else 'NOT confirmed T-independent -- see diff above'}")

    ladders: dict[str, list[dict]] = {}
    for rule in rules:
        ladders[rule] = _run_ladder(body, rule, args.lmax_list, args.nrbase, k2_uniform)

    # --- Archive ---
    output.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        body=args.body,
        rules=np.asarray(rules),
        lmax_list=np.asarray(args.lmax_list, dtype=int),
        nrbase=args.nrbase,
        K=body.K,
        k2_uniform=k2_uniform,
        k2_1d_check=k2_1d,
        diag_lmax=np.asarray([r["lmax"] for r in diag_rows], dtype=int),
        diag_dt_max_m=np.asarray([r["dt_max_m"] for r in diag_rows]),
        diag_dt_min_m=np.asarray([r["dt_min_m"] for r in diag_rows]),
        diag_T_twice_max_m=np.asarray([r["T_twice_max_m"] for r in diag_rows]),
        diag_margin_twice_max=np.asarray([r["margin_twice_max"] for r in diag_rows]),
        diag_T_range_m=np.asarray([r["T_range_m"] for r in diag_rows]),
        diag_margin_range=np.asarray([r["margin_range"] for r in diag_rows]),
        total_wall_s=time.perf_counter() - t0_total,
        peak_rss_gb_overall=_peak_rss_gb(),
    )
    for rule, rows in ladders.items():
        if not rows:
            continue
        save[f"ladder_{rule}_lmax"] = np.asarray([r["lmax"] for r in rows], dtype=int)
        save[f"ladder_{rule}_T_km"] = np.asarray([r["T_km"] for r in rows])
        save[f"ladder_{rule}_margin"] = np.asarray([r["margin"] for r in rows])
        save[f"ladder_{rule}_N"] = np.asarray([r["N"] for r in rows], dtype=int)
        save[f"ladder_{rule}_abs_delta_k20"] = np.asarray([r["abs_delta_k20"] for r in rows])
        save[f"ladder_{rule}_delta_k20"] = np.asarray([r["delta_k20"] for r in rows], dtype=complex)
        save[f"ladder_{rule}_wall_s"] = np.asarray([r["wall_s"] for r in rows])
        save[f"ladder_{rule}_peak_rss_gb"] = np.asarray([r["peak_rss_gb"] for r in rows])
        for n, m in OFF_MODES:
            key = f"{n}_{m}"
            save[f"ladder_{rule}_off_{key}"] = np.asarray([r["off"][key] for r in rows])

    np.savez(output, **save)
    if any(ladders.values()):
        _save_figure(output.with_suffix(".png"), args.body, body.K, ladders)
        print(f"\nsaved {output} and {output.with_suffix('.png')}")
    else:
        print(f"\nsaved {output} (no solves run; figure skipped)")

    print(f"overall peak RSS = {_peak_rss_gb():.1f} GB, total wall = {time.perf_counter() - t0_total:.1f} s")


if __name__ == "__main__":
    main()
