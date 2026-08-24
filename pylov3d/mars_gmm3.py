# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Read the public GMM-3 Mars gravity SHADR product.

The PDS ASCII SHADR product ``gmm3_120_sha.tab`` contains a one-line header
followed by normalized C_lm/S_lm coefficients and their formal uncertainties.
The associated PDS label recommends multiplying the formal uncertainties by
three for conservative error estimates.

This module intentionally stops short of converting pylov3d's orthonormal
thickness-harmonic coefficients to the GMM-3 normalization.  It provides the
observational uncertainty side of that bridge first, so normalization can be
validated explicitly rather than assumed.

Official PDS product:
https://pds-geosciences.wustl.edu/mro/mro-m-rss-5-sdp-v1/mrors_1xxx/data/shadr/gmm3_120_sha.tab
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GMM3Header:
    reference_radius_km: float
    gm_km3_s2: float
    gm_sigma_km3_s2: float
    max_degree: int
    max_order: int
    normalization_state: int


@dataclass(frozen=True)
class GMM3Coefficient:
    degree: int
    order: int
    c: float
    s: float
    sigma_c: float
    sigma_s: float


def _fields(line: str) -> list[str]:
    return [x.strip() for x in line.strip().split(",")]


def parse_header(line: str) -> GMM3Header:
    fields = _fields(line)
    if len(fields) < 6:
        raise ValueError("GMM-3 header must contain at least six comma-separated fields")
    return GMM3Header(
        reference_radius_km=float(fields[0]),
        gm_km3_s2=float(fields[1]),
        gm_sigma_km3_s2=float(fields[2]),
        max_degree=int(fields[3]),
        max_order=int(fields[4]),
        normalization_state=int(fields[5]),
    )


def parse_coefficient(line: str) -> GMM3Coefficient:
    fields = _fields(line)
    if len(fields) < 6:
        raise ValueError("GMM-3 coefficient row must contain six comma-separated fields")
    return GMM3Coefficient(
        degree=int(fields[0]),
        order=int(fields[1]),
        c=float(fields[2]),
        s=float(fields[3]),
        sigma_c=float(fields[4]),
        sigma_s=float(fields[5]),
    )


def read_gmm3_shadr(path: str | Path) -> tuple[GMM3Header, list[GMM3Coefficient]]:
    """Read a GMM-3 SHADR ASCII file."""
    path = Path(path)
    with path.open("r", encoding="ascii") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise ValueError("empty GMM-3 SHADR file")
    header = parse_header(lines[0])
    coeffs = [parse_coefficient(line) for line in lines[1:]]
    return header, coeffs


def coefficients_at_degree(
    coeffs: Iterable[GMM3Coefficient], degree: int
) -> list[GMM3Coefficient]:
    """Return all coefficient rows at one harmonic degree."""
    return [row for row in coeffs if row.degree == degree]


def formal_sigmas_at_degree(
    coeffs: Iterable[GMM3Coefficient], degree: int, *, include_s_zero: bool = False
) -> list[float]:
    """Return positive formal C/S uncertainties at one degree.

    ``S_l0`` is identically zero and its stored uncertainty is therefore zero;
    by default zeros are omitted from the returned sample.
    """
    out: list[float] = []
    for row in coefficients_at_degree(coeffs, degree):
        for sigma in (row.sigma_c, row.sigma_s):
            if sigma > 0.0 or include_s_zero:
                out.append(sigma)
    return out
