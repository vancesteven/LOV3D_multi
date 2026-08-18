#!/usr/bin/env python3
"""TASK-046 Gate B/C cross-check using each forcing's native coupling basis.

This is the quantitative follow-up to the MATLAB anchor committed in
``data/tests/io/io_energy_cross_check.{log,mat}``.  The old Python driver
incorrectly reused the (2,0) coupling closure for the (2,-2) and (2,+2)
forcings.  Here each forcing is solved on the closure generated from its own
(n,m), and the resulting stress/strain fields are combined only at the energy
contraction stage with ``get_energy_coupled_multibasis``.

The archived MATLAB reference at Nrbase=50 is:

    k_uni(2,m) = 0.7337217069 - 0.0151236751 i  (all m)
    k_lat(2,0) = 0.7325399703 - 0.0153355564 i
    k_lat(2,±2)= 0.7381214321 - 0.0198692819 i
    N_modes lateral = 125 per forcing
    E_direct uniform/lateral = 2.1668778416 / 2.8404609804
    E_Love   uniform/lateral = 2.2144024348 / 2.9026033327
    direct-vs-Love mismatch  = ~2.19% for both

Run a cheap structural check first with ``--nrbase 10``.  Use ``--nrbase 50``
for the MATLAB quantitative anchor; coupling construction can be expensive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.couplings import get_couplings
from pylov3d.energy import get_energy
from pylov3d.energy_multibasis import get_energy_coupled_multibasis
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_variable,
)
from pylov3d.love import extract_love_numbers
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.solver import get_solution

MATLAB = {
    "k_uni": 0.7337217069 - 0.0151236751j,
    "k_lat_m0": 0.7325399703 - 0.0153355564j,
    "k_lat_m2": 0.7381214321 - 0.0198692819j,
    "e_direct_uni": 2.1668778416,
    "e_direct_lat": 2.8404609804,
    "e_love_uni": 2.2144024348,
    "e_love_lat": 2.9026033327,
    "n_modes": 125,
}


def love_energy_estimate(love_list, forcings) -> float:
    """Match the MATLAB cross-forcing double sum for E_k."""
    out = 0.0
    for fi in forcings:
        for j, fj in enumerate(forcings):
            love = love_list[j]
            idx = np.where((np.asarray(love.n) == fi.n) & (np.asarray(love.m) == fi.m))[0]
            if len(idx):
                out -= float(fi.F) * float(fj.F) * complex(love.k[idx[0]]).imag
    return out


def forcing_mode_k(love, forcing) -> complex:
    idx = np.where((np.asarray(love.n) == forcing.n) & (np.asarray(love.m) == forcing.m))[0]
    if not len(idx):
        raise RuntimeError(f"forcing mode ({forcing.n},{forcing.m}) missing from Love spectrum")
    return complex(love.k[idx[0]])


def solve_uniform(raw_model, forcings, numerics):
    numerics, model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(model, forcings)
    results = []
    for forcing in forcings:
        y, r, _Y, aprop = get_solution(model, forcing, numerics)
        love = extract_love_numbers(y, model, forcing)
        results.append((y, r, aprop, love))
    return model, numerics, results


def solve_lateral_native(raw_model, forcings, numerics, mu_variable, eta_variable):
    numerics, model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(model, forcings)
    model, lateral = process_lateral_variations(
        model,
        forcings,
        mu_variable=mu_variable,
        eta_variable=eta_variable,
        rheology_cutoff=numerics.rheology_cutoff,
    )

    results = []
    couplings_list = []
    for forcing in forcings:
        couplings = get_couplings(
            lateral.variations,
            forcing.n,
            forcing.m,
            perturbation_order=numerics.perturbation_order,
        )
        y, r, _Y, aprop = get_solution(
            model, forcing, numerics, couplings=couplings, lateral=lateral,
        )
        love = extract_love_numbers(y, model, forcing, couplings=couplings)
        results.append((y, r, aprop, love))
        couplings_list.append(couplings)
    return model, numerics, lateral, couplings_list, results


def relerr(a, b) -> float:
    return abs(a - b) / max(abs(b), np.finfo(float).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrbase", type=int, default=10)
    parser.add_argument(
        "--assert-matlab",
        action="store_true",
        help="at Nrbase=50, fail if forcing-mode k or energy closure is inconsistent with the archived MATLAB anchor",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    raw_model = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(args.nrbase)
    mu_variable, eta_variable, _diagnostics = io_mu_eta_variable()

    model_u, num_u, uniform = solve_uniform(raw_model, forcings, numerics)
    model_l, num_l, _lat, couplings, lateral = solve_lateral_native(
        raw_model, forcings, numerics, mu_variable, eta_variable,
    )

    k_u = [forcing_mode_k(r[3], f) for r, f in zip(uniform, forcings)]
    k_l = [forcing_mode_k(r[3], f) for r, f in zip(lateral, forcings)]
    mode_counts = [len(c.n_s) for c in couplings]

    e_direct_u_raw = 0.0
    for forcing, (y, r, aprop, _love) in zip(forcings, uniform):
        e = get_energy(y, r, aprop, model_u, forcing, num_u)
        e_direct_u_raw += float(forcing.F) ** 2 * float(e.energy_integral[0])
    e_direct_u = -e_direct_u_raw

    y_solutions = [(r[0], r[1], r[2]) for r in lateral]
    e_lat = get_energy_coupled_multibasis(
        y_solutions,
        forcings,
        model_l,
        num_l,
        [c.n_s for c in couplings],
        [c.m_s for c in couplings],
        Nenergy=num_l.Nenergy,
    )
    zero = np.where((np.asarray(e_lat.n) == 0) & (np.asarray(e_lat.m) == 0))[0]
    if not len(zero):
        raise RuntimeError("monopole energy term (0,0) missing")
    e_direct_l = -float(e_lat.energy_integral[zero[0]])

    love_u = [r[3] for r in uniform]
    love_l = [r[3] for r in lateral]
    ek_u = love_energy_estimate(love_u, forcings)
    ek_l = love_energy_estimate(love_l, forcings)

    matlab_prefactor = 5.0 / float(model_u.Gg)
    e_love_u = matlab_prefactor * ek_u
    e_love_l = matlab_prefactor * ek_l
    mismatch_u = abs(e_direct_u - e_love_u) / abs(e_love_u)
    mismatch_l = abs(e_direct_l - e_love_l) / abs(e_love_l)

    print(f"TASK-046 Gate B/C, Nrbase={args.nrbase}")
    print(f"native lateral mode counts: {mode_counts}")
    for f, ku, kl in zip(forcings, k_u, k_l):
        print(f"forcing (n={f.n},m={f.m:+d})  k_uni={ku.real:+.10f}{ku.imag:+.10f}i  k_lat={kl.real:+.10f}{kl.imag:+.10f}i")
    print(f"direct energy uniform/lateral: {e_direct_u:.10e}  {e_direct_l:.10e}")
    print(f"Love energy   uniform/lateral: {e_love_u:.10e}  {e_love_l:.10e}")
    print(f"direct/Love mismatch: uniform={100*mismatch_u:.4f}%  lateral={100*mismatch_l:.4f}%")
    print(f"wall time: {time.perf_counter()-t0:.1f} s")

    if args.nrbase == 50:
        print("\nMATLAB Gate C anchor comparison:")
        for f, ku, kl in zip(forcings, k_u, k_l):
            k_lat_ref = MATLAB["k_lat_m0"] if f.m == 0 else MATLAB["k_lat_m2"]
            print(
                f"  m={f.m:+d}: relerr k_uni={relerr(ku, MATLAB['k_uni']):.3e}, "
                f"k_lat={relerr(kl, k_lat_ref):.3e}"
            )
        print(
            "  relerr E_direct uni/lat="
            f"{relerr(e_direct_u, MATLAB['e_direct_uni']):.3e}/"
            f"{relerr(e_direct_l, MATLAB['e_direct_lat']):.3e}"
        )
        print(
            "  relerr E_love uni/lat="
            f"{relerr(e_love_u, MATLAB['e_love_uni']):.3e}/"
            f"{relerr(e_love_l, MATLAB['e_love_lat']):.3e}"
        )

        if args.assert_matlab:
            assert all(relerr(x, MATLAB["k_uni"]) < 1e-7 for x in k_u)
            for f, x in zip(forcings, k_l):
                ref = MATLAB["k_lat_m0"] if f.m == 0 else MATLAB["k_lat_m2"]
                assert relerr(x, ref) < 1e-6
            assert mode_counts == [MATLAB["n_modes"]] * len(forcings)
            assert relerr(e_direct_u, MATLAB["e_direct_uni"]) < 5e-3
            assert relerr(e_direct_l, MATLAB["e_direct_lat"]) < 5e-3
            assert relerr(e_love_u, MATLAB["e_love_uni"]) < 5e-3
            assert relerr(e_love_l, MATLAB["e_love_lat"]) < 5e-3
            assert mismatch_u < 0.03
            assert mismatch_l < 0.03
            print("MATLAB Gate C assertions: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
