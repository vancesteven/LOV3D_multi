#!/usr/bin/env python3
"""TASK-046 strict Gate C using the MATLAB-faithful raw-grid rheology path.

This is the decisive end-to-end test after isolating the historical
``SPH_LatLon`` / ``LatLon_SPH`` transform convention.  The nonlinear Io
fractional mu/eta maps are passed directly through the MATLAB-faithful grid
processor and then through the unchanged pylov3d coupled solver.
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
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_grids,
)
from pylov3d.love import extract_love_numbers
from pylov3d.rheology import get_rheology
from pylov3d.rheology_grid import process_lateral_fractional_grids
from pylov3d.solver import get_solution
from scripts.io_energy_gate_bc_multibasis import (
    MATLAB,
    TARGET_MODE_COUNTS,
    forcing_mode_k,
    love_energy_estimate,
    matlab_lateral_k_ref,
    monopole_direct_energy,
    relerr,
    solve_uniform,
)


def solve_lateral_raw_grid(raw_model, forcings, numerics):
    numerics, model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(model, forcings)
    _lat, _lon, dmu, deta = io_mu_eta_grids(l_max=100)
    ilayer = IO_ASTHENOSPHERE_LAYER_INDEX
    model, lateral = process_lateral_fractional_grids(
        model,
        dmu_grids={ilayer: dmu},
        deta_grids={ilayer: deta},
        rheology_cutoff=numerics.rheology_cutoff,
        minimum_rheology_value=-13.0,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrbase", type=int, default=50)
    parser.add_argument("--assert-matlab", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    raw_model = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(args.nrbase)

    model_u, num_u, uniform = solve_uniform(raw_model, forcings, numerics)
    model_l, num_l, lateral_rheology, couplings, lateral = solve_lateral_raw_grid(
        raw_model, forcings, numerics,
    )

    k_u = [forcing_mode_k(r[3], f) for r, f in zip(uniform, forcings)]
    k_l = [forcing_mode_k(r[3], f) for r, f in zip(lateral, forcings)]
    mode_counts = [len(c.n_s) for c in couplings]

    uniform_n_s = [np.asarray([f.n], dtype=int) for f in forcings]
    uniform_m_s = [np.asarray([f.m], dtype=int) for f in forcings]
    e_direct_u = monopole_direct_energy(
        uniform, forcings, model_u, num_u, uniform_n_s, uniform_m_s,
    )
    e_direct_l = monopole_direct_energy(
        lateral,
        forcings,
        model_l,
        num_l,
        [c.n_s for c in couplings],
        [c.m_s for c in couplings],
        couplings_list=couplings,
        lateral=lateral_rheology,
    )

    love_u = [r[3] for r in uniform]
    love_l = [r[3] for r in lateral]
    prefactor = 5.0 / float(model_u.Gg)
    e_love_u = prefactor * love_energy_estimate(love_u, forcings)
    e_love_l = prefactor * love_energy_estimate(love_l, forcings)

    print(f"TASK-046 strict raw-grid Gate C, Nrbase={args.nrbase}")
    print(f"native lateral mode counts: {mode_counts} (MATLAB target {TARGET_MODE_COUNTS})")
    for f, ku, kl in zip(forcings, k_u, k_l):
        print(
            f"forcing (n={f.n},m={f.m:+d})  "
            f"k_uni={ku.real:+.10f}{ku.imag:+.10f}i  "
            f"k_lat={kl.real:+.10f}{kl.imag:+.10f}i"
        )
    print(f"direct energy uniform/lateral: {e_direct_u:.12e}  {e_direct_l:.12e}")
    print(f"Love energy   uniform/lateral: {e_love_u:.12e}  {e_love_l:.12e}")
    print(f"wall time: {time.perf_counter()-t0:.1f} s")
    print("\nMATLAB comparison:")
    worst_k = 0.0
    for f, ku, kl in zip(forcings, k_u, k_l):
        eu = relerr(ku, MATLAB["k_uni"])
        el = relerr(kl, matlab_lateral_k_ref(f.m))
        worst_k = max(worst_k, el)
        print(f"  m={f.m:+d}: relerr k_uni={eu:.3e}, k_lat={el:.3e}")
    ed_u = relerr(e_direct_u, MATLAB["e_direct_uni"])
    ed_l = relerr(e_direct_l, MATLAB["e_direct_lat"])
    el_u = relerr(e_love_u, MATLAB["e_love_uni"])
    el_l = relerr(e_love_l, MATLAB["e_love_lat"])
    print(f"  relerr E_direct uni/lat={ed_u:.3e}/{ed_l:.3e}")
    print(f"  relerr E_love   uni/lat={el_u:.3e}/{el_l:.3e}")
    print(f"  worst lateral k relerr={worst_k:.3e}")

    if args.assert_matlab:
        assert mode_counts == TARGET_MODE_COUNTS
        assert all(relerr(x, MATLAB["k_uni"]) < 1e-7 for x in k_u)
        for f, x in zip(forcings, k_l):
            assert relerr(x, matlab_lateral_k_ref(f.m)) < 1e-7
        assert ed_u < 1e-7
        assert ed_l < 1e-7
        assert el_u < 1e-7
        assert el_l < 1e-7
        print("strict raw-grid MATLAB Gate C parity: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
