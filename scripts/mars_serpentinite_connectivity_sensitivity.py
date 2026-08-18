#!/usr/bin/env python3
"""Mars serpentinite effective-medium sensitivity experiment.

Tests how the hydration-front headline k2 signal changes when the hydrated
crust is not forced to obey the Voigt (iso-strain, stiff upper-bound) mixing
law used by TASK-021.  The experiment keeps the existing Mars reference body,
serpentinite endmember properties, forcing period, and hydration fraction,
and changes only the effective-medium law for the *mean* hydrated crust.

Why mean-only first?
--------------------
TASK-021 found the degree-0 mean-softening contribution dominates the lateral
front contribution by roughly 60:1 at the validated angular truncation.  A
mean-only experiment therefore tests the proposal's headline detectability
claim while cleanly isolating effective-medium physics from lateral-SH
bookkeeping.  A later nonlinear-grid extension can apply the same laws
pointwise and re-expand the lateral residual.

Mixing laws
-----------
voigt : arithmetic average of K and mu (iso-strain; stiff bound)
reuss : harmonic average of K and mu (iso-stress / connected weak-layer limit)
hill  : arithmetic mean of Voigt and Reuss (VRH midpoint)

The Reuss model is intentionally a limiting sensitivity case, not a claim that
Martian serpentinite has that topology.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from pylov3d.love import get_love
from pylov3d.mars import LAYER_MU_CRUST, MARS, MARS_FORCING_TD, build_mars_model
from pylov3d.mars_hydration import K_CRUST, RATIO_SCENARIOS
from pylov3d.mars_lateral import CRUST_LAYER_INDEX
from pylov3d.types import make_forcing, make_numerics


def _mix_voigt(f: float, dry: float, wet: float) -> float:
    return (1.0 - f) * dry + f * wet


def _mix_reuss(f: float, dry: float, wet: float) -> float:
    if dry <= 0 or wet <= 0:
        raise ValueError("Reuss mixing requires positive moduli")
    return 1.0 / ((1.0 - f) / dry + f / wet)


def mixed_moduli(f: float, mu_ratio: float, K_ratio: float, law: str) -> tuple[float, float]:
    """Return effective (mu, K) [Pa] for hydrated fraction f."""
    mu_dry = float(LAYER_MU_CRUST)
    K_dry = float(K_CRUST)
    mu_wet = mu_ratio * mu_dry
    K_wet = K_ratio * K_dry

    mu_v = _mix_voigt(f, mu_dry, mu_wet)
    K_v = _mix_voigt(f, K_dry, K_wet)
    if law == "voigt":
        return mu_v, K_v

    mu_r = _mix_reuss(f, mu_dry, mu_wet)
    K_r = _mix_reuss(f, K_dry, K_wet)
    if law == "reuss":
        return mu_r, K_r
    if law == "hill":
        return 0.5 * (mu_v + mu_r), 0.5 * (K_v + K_r)
    raise ValueError(f"unknown mixing law: {law}")


def solve_k2(mu_crust: float, K_crust: float, nrbase: int) -> float:
    model = build_mars_model()
    model = model._replace(
        mu0=model.mu0.at[CRUST_LAYER_INDEX].set(mu_crust),
        Ks0=model.Ks0.at[CRUST_LAYER_INDEX].set(K_crust),
    )
    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=model.n_layers,
        method="combination",
        Nrbase=nrbase,
        perturbation_order=2,
    )
    love, _, _ = get_love(model, forcing, numerics)
    return float(complex(love.k[0]).real)


def run(f_grid: list[float], nrbase: int) -> list[dict]:
    baseline = solve_k2(float(LAYER_MU_CRUST), float(K_CRUST), nrbase)
    rows: list[dict] = []
    for scenario, (mu_ratio, K_ratio) in RATIO_SCENARIOS.items():
        for law in ("voigt", "hill", "reuss"):
            for f in f_grid:
                mu_eff, K_eff = mixed_moduli(f, mu_ratio, K_ratio, law)
                k2 = baseline if f == 0 else solve_k2(mu_eff, K_eff, nrbase)
                dk = k2 - baseline
                rows.append({
                    "scenario": scenario,
                    "law": law,
                    "f_h": f,
                    "mu_eff_gpa": mu_eff / 1e9,
                    "K_eff_gpa": K_eff / 1e9,
                    "k2": k2,
                    "delta_k2": dk,
                    "delta_k2_over_sigma": dk / float(MARS["k2_sigma"]),
                    "sigma_required_1sigma": abs(dk),
                })
    return rows


def _print_summary(rows: list[dict]) -> None:
    print("Mars serpentinite connectivity sensitivity")
    print(f"current sigma_k2 = {float(MARS['k2_sigma']):.6g}")
    print("\nAt f_h=0.5:")
    print("scenario  law      mu_eff[GPa]  K_eff[GPa]    delta_k2    % current sigma")
    for r in rows:
        if math.isclose(r["f_h"], 0.5):
            print(
                f"{r['scenario']:<9} {r['law']:<8} "
                f"{r['mu_eff_gpa']:11.4f} {r['K_eff_gpa']:11.4f} "
                f"{r['delta_k2']:+.6e} {100*r['delta_k2_over_sigma']:+12.3f}%"
            )

    print("\nAt f_h=0.1 (future 1-sigma precision requirement):")
    print("scenario  law         |delta_k2|   improvement vs current")
    for r in rows:
        if math.isclose(r["f_h"], 0.1):
            req = abs(r["delta_k2"])
            improvement = float(MARS["k2_sigma"]) / req if req > 0 else math.inf
            print(f"{r['scenario']:<9} {r['law']:<8} {req:.6e} {improvement:12.2f}x")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nrbase", type=int, default=30)
    p.add_argument(
        "--f-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        help="comma-separated hydration fractions",
    )
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    f_grid = [float(x) for x in args.f_grid.split(",")]
    if any(f < 0 or f > 1 for f in f_grid):
        raise SystemExit("all f_h values must lie in [0,1]")

    rows = run(f_grid, args.nrbase)
    _print_summary(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
