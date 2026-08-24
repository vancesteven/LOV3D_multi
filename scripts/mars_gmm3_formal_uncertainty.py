#!/usr/bin/env python
"""Inspect formal GMM-3 coefficient uncertainties at proposal-relevant degrees."""

from __future__ import annotations

import argparse
import statistics
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylov3d.mars_gmm3 import formal_sigmas_at_degree, read_gmm3_shadr

PDS_URL = (
    "https://pds-geosciences.wustl.edu/mro/mro-m-rss-5-sdp-v1/"
    "mrors_1xxx/data/shadr/gmm3_120_sha.tab"
)
DEFAULT_PATH = ROOT / "data" / "external" / "gmm3_120_sha.tab"


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading official PDS GMM-3 SHADR to {path}")
    urllib.request.urlretrieve(PDS_URL, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    p.add_argument(
        "--download",
        action="store_true",
        help="Download the official PDS ASCII GMM-3 product if --path is absent.",
    )
    args = p.parse_args()

    if not args.path.exists():
        if not args.download:
            print(f"missing {args.path}")
            print("rerun with --download, or provide --path to gmm3_120_sha.tab")
            return 2
        _download(args.path)

    header, coeffs = read_gmm3_shadr(args.path)
    print("GMM-3 formal coefficient uncertainty diagnostic")
    print(f"reference radius: {header.reference_radius_km:.1f} km")
    print(f"maximum degree/order: {header.max_degree}/{header.max_order}")
    print(f"normalization state: {header.normalization_state} (PDS: 1=normalized)")
    print("PDS label recommends 3x formal uncertainties for conservative errors")
    print()
    print("degree   Nsig    min sigma       median sigma    max sigma       3x median")
    for degree in (5, 11, 21, 43, 85):
        vals = formal_sigmas_at_degree(coeffs, degree)
        if not vals:
            print(f"{degree:6d}      0")
            continue
        med = statistics.median(vals)
        print(
            f"{degree:6d} {len(vals):6d} "
            f"{min(vals):13.5e} {med:13.5e} {max(vals):13.5e} {3*med:13.5e}"
        )

    print()
    print("interpretation guard rail:")
    print("  These are coefficient-space formal errors from the public GMM-3 SHADR")
    print("  product. Do not compare pylov3d q_lm directly with them until the")
    print("  orthonormal-harmonic to GMM-3 normalized C_lm/S_lm convention bridge")
    print("  is explicitly validated. Full covariance is available in the SHBDR")
    print("  binary product and should replace diagonal-only errors for final work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
