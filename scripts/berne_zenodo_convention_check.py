#!/usr/bin/env python3
"""Full five-forcing pylov3d comparison against the Berne 2026 Zenodo anchors.

Runs both real->complex sine conventions ('script' = run_forward_shear.m,
'canonical' = get_rheology.m) over forcings (2,m'), m' = -2..2, and reports
the worst relative difference to the shipped native-MATLAB k2_responses.txt.
Result on 2026-09-28: script 1.0e-12, canonical 7.1e-2 (365 modes compared).
Expect roughly two hours of CPU at the archive's Nr=200.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pylov3d.berne2026 import (
    build_zenodo_example_model,
    read_zenodo_shear_coefficients,
    zenodo_mu_variable,
)
from pylov3d.love import get_love
from pylov3d.types import make_forcing, make_numerics

DATA = Path(__file__).resolve().parents[1] / "data" / "tests" / "berne_zenodo"


def main() -> int:
    rows = read_zenodo_shear_coefficients(DATA / "shear_modulus_coefficients.txt")
    ship: dict[int, dict[tuple[int, int], complex]] = {}
    for line in open(DATA / "k2_responses.txt"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        mp = int(re.match(r"\((-?\d+),(-?\d+)\)", t[0]).group(2))
        ship.setdefault(mp, {})[(int(t[1]), int(t[2]))] = float(t[3]) + 1j * float(t[4])

    model = build_zenodo_example_model()
    numerics = make_numerics(n_layers=2, method="variable", Nrbase=200,
                             perturbation_order=2, rheology_cutoff=2.0)
    for conv in ("script", "canonical"):
        mu_variable = zenodo_mu_variable(rows, convention=conv)
        worst, where, n_cmp = 0.0, None, 0
        for mp in sorted(ship):
            love, _, _ = get_love(model, make_forcing(686.98 * 86400.0, 2, mp, 1.0),
                                  numerics, mu_variable=mu_variable)
            got = {(int(n), int(m)): complex(k)
                   for n, m, k in zip(love.n, love.m, love.k)}
            kf = abs(ship[mp][(2, mp)]) or 1.0
            for mode in set(ship[mp]) & set(got):
                d = abs(got[mode] - ship[mp][mode]) / kf
                n_cmp += 1
                if d > worst:
                    worst, where = d, (mp, mode)
            missing, extra = set(ship[mp]) - set(got), set(got) - set(ship[mp])
            if missing or extra:
                print(f"conv={conv} mp={mp} mode-set mismatch "
                      f"missing={sorted(missing)} extra={sorted(extra)}", flush=True)
        print(f"convention={conv:10s} compared={n_cmp} "
              f"worst rel diff={worst:.3e} at {where}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
