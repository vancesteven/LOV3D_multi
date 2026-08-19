#!/usr/bin/env python
"""Connected-pore conductivity sensitivity for the seismic look-alike pair."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_em_profiles import archie_connected_pore_conductivity
from pylov3d.mars_em_sensitivity import (
    layer_conductance_siemens,
    period_for_skin_depth_s,
    skin_depth_km,
)


def main() -> int:
    phi = 0.30
    sw = 1.0
    H_km = 10.0
    matrix_sigma = 1e-4
    fluid_sigmas = [0.1, 1.0, 10.0]
    cementation = [1.5, 2.0, 2.5]
    target_depth_km = 15.0

    print("Mars Archie-style EM discriminability bound")
    print("seismic survivor: saturated fractured frame, phi=0.30")
    print(f"matrix conductivity bound: {matrix_sigma:.1e} S/m")
    print("fully saturated connected pores; saturation exponent n=2")
    print()
    print("sigma_f  m    sigma_eff[S/m]  conductance[S]  T(delta=15km)[s]  delta@100s[km]")
    for sigma_f in fluid_sigmas:
        for m in cementation:
            sigma_eff = archie_connected_pore_conductivity(
                phi,
                sw,
                sigma_f,
                cementation_exponent=m,
                saturation_exponent=2.0,
                matrix_conductivity_s_m=matrix_sigma,
            )
            conductance = layer_conductance_siemens(sigma_eff, H_km)
            period = period_for_skin_depth_s(sigma_eff, target_depth_km)
            delta100 = skin_depth_km(sigma_eff, 100.0)
            print(
                f"{sigma_f:7.2g} {m:3.1f} {sigma_eff:15.5g} "
                f"{conductance:14.4g} {period:17.4g} {delta100:15.3f}"
            )

    print()
    print("interpretation guard rail:")
    print("  this is an Archie-style connected-pore sensitivity bound, not a")
    print("  Mars mineral-physics conductivity model. Matrix conduction is kept")
    print("  explicit; clay/surface conduction and alteration-mineral conduction")
    print("  are not represented. The next rung should replace these inputs with")
    print("  PlanetProfile-derived mineral/fluid conductivity profiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
