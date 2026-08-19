#!/usr/bin/env python
"""Compare helicopter and orbital magnetic anomaly attenuation on Mars."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_magnetic_aerial import upward_continuation_factor


def main() -> int:
    wavelengths_km = (1, 2, 5, 10, 20, 50, 100, 200)
    altitudes_m = (50.0, 100.0, 500.0, 10_000.0, 100_000.0, 150_000.0)

    print("Mars helicopter magnetic anomaly resolution bound")
    print("potential-field upward continuation: A(h)/A0 = exp(-2*pi*h/lambda)")
    print()
    print("lambda[km]   50m       100m      500m      10km      100km     150km")
    for lam_km in wavelengths_km:
        vals = [upward_continuation_factor(lam_km * 1e3, h) for h in altitudes_m]
        print(
            f"{lam_km:9.0f} "
            + " ".join(f"{v:10.3e}" for v in vals)
        )

    print()
    print("interpretation:")
    print("  helicopter altitudes preserve km-to-tens-of-km crustal magnetic structure")
    print("  that is exponentially filtered at orbital altitude. Multi-altitude vector")
    print("  traverses and landed measurements can therefore constrain source geometry")
    print("  and reduce the source-depth/thickness nuisance parameters in remanence models.")
    print("  This is a scale bound only; vehicle magnetic cleanliness, external fields,")
    print("  source depth, magnetization direction, and full vector inversion remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
