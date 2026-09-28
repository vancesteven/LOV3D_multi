#!/usr/bin/env python3
"""PRELIMINARY higher-degree tidal signal of pointwise-mixed crustal hydration.

Extends the proposal's Figure-3 study (mean-only Voigt/Hill/Reuss k2
sensitivity) to the lateral problem: for each (scenario, law, f_h) the
mixing law is applied pointwise to the crustal-thickness hydration field
(pylov3d.mars_hydration_connectivity), and the full degree-2-forced
coupled response is computed with the extended Love tensor. Reported per
combination:

* delta_k2_mean = k2(mixed-mean 1D model) - k2(dry)   [the Fig-3 quantity,
  now with the pointwise mean];
* k_2m splitting |k_2m - k2_mean_1d| for m' = 0, 1, 2 (diagonals);
* the strongest degree-3 and degree-4 responses to (2,0) forcing, and the
  overall max |K| off-diagonal per degree over all five forcings.

PRELIMINARY (2026-09-29): supports scoping only; Berne will provide
updated calculations, which supersede these numbers.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.extended_love import extended_love_tensor
from pylov3d.love import get_love
from pylov3d.mars import MARS_FORCING_TD, build_mars_model
from pylov3d.mars_hydration import RATIO_SCENARIOS
from pylov3d.mars_hydration_connectivity import pointwise_hydrated_crust
from pylov3d.types import make_forcing, make_numerics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", default=["central", "low"],
                   choices=sorted(RATIO_SCENARIOS))
    p.add_argument("--laws", nargs="+", default=["voigt", "hill", "reuss"])
    p.add_argument("--f-h", nargs="+", type=float, default=[0.1, 0.5])
    p.add_argument("--lmax-field", type=int, default=4)
    p.add_argument("--lmax-out", type=int, default=2)
    p.add_argument("--nrbase", type=int, default=30)
    p.add_argument("--csv", type=Path,
                   default=REPO_ROOT / "data/tests/mars/connectivity_higher_degree_preliminary.csv")
    args = p.parse_args()

    numerics = make_numerics(n_layers=4, method="combination",
                             Nrbase=args.nrbase, perturbation_order=2)
    dry, _, _ = get_love(build_mars_model(),
                         make_forcing(MARS_FORCING_TD, 2, 0, 1.0), numerics)
    k2_dry = complex(dry.k[0]).real

    rows = []
    for scenario in args.scenarios:
        mu_r, K_r = RATIO_SCENARIOS[scenario]
        for law in args.laws:
            for f_h in args.f_h:
                t0 = time.time()
                pc = pointwise_hydrated_crust(
                    f_h, mu_r, K_r, law,
                    lmax_field=args.lmax_field, lmax_out=args.lmax_out)
                mean_love, _, _ = get_love(
                    pc.model, make_forcing(MARS_FORCING_TD, 2, 0, 1.0), numerics)
                k2_mean = complex(mean_love.k[0]).real
                T = extended_love_tensor(
                    pc.model, numerics, MARS_FORCING_TD,
                    mu_variable=pc.mu_variable, K_variable=pc.K_variable)
                row = {
                    "scenario": scenario, "law": law, "f_h": f_h,
                    "mu_bar_GPa": pc.mu_bar / 1e9, "Ks_bar_GPa": pc.Ks_bar / 1e9,
                    "clip_fraction": pc.clip_fraction,
                    "delta_k2_mean": k2_mean - k2_dry,
                }
                for m in (0, 1, 2):
                    row[f"split_k2{m}"] = abs(T.entry(2, m, 2, m) - k2_mean)
                row["max_l3_from_20"] = max(
                    abs(T.entry(3, m, 2, 0)) for m in range(-3, 4))
                row["max_l4_from_20"] = max(
                    abs(T.entry(4, m, 2, 0)) for m in range(-4, 5))
                for l in (3, 4):
                    row[f"max_l{l}_any"] = max(
                        (abs(T.K[i, j])
                         for i, mode in enumerate(T.response_modes)
                         for j in range(len(T.forcing_modes))
                         if mode[0] == l), default=0.0)
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in row.items()), flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        fh.write("# PRELIMINARY (2026-09-29): pointwise-connectivity higher-degree "
                 "tidal sensitivity; supports scoping only, superseded by Berne's "
                 "updated calculations when available.\n")
        fh.write(f"# k2_dry={k2_dry!r} lmax_field={args.lmax_field} "
                 f"lmax_out={args.lmax_out} Nrbase={args.nrbase} "
                 f"perturbation_order=2 Td={MARS_FORCING_TD}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved: {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
