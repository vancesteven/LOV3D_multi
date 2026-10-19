# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Love-number convergence diagnostics for controlled radial reduction.

This module answers a publication-facing question: how sensitive are the
computed Love numbers to the number of retained radial shells after a
high-resolution PlanetProfile artifact is reduced to pylov3d's static layer
limit?

The highest requested layer count is used as the internal reference. This is a
numerical convergence diagnostic, not proof that the high-resolution physical
profile itself is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MAX_LAYERS
from .love import get_love
from .profile_io import RadialArtifactShells
from .profile_reduction import reduce_radial_shells, reduced_shells_to_interior_model
from .types import Forcing, make_numerics


@dataclass(frozen=True)
class LoveReductionPoint:
    target_layers: int
    actual_layers: int
    k: complex
    h: complex
    l: complex
    abs_dk: float
    abs_dh: float
    abs_dl: float
    rel_dk: float
    rel_dh: float
    rel_dl: float


@dataclass(frozen=True)
class LoveReductionConvergence:
    reference_layers: int
    points: tuple[LoveReductionPoint, ...]


def _relative(delta: float, reference: complex) -> float:
    denom = abs(reference)
    if denom == 0.0:
        return 0.0 if delta == 0.0 else np.inf
    return float(delta / denom)


def love_number_reduction_convergence(
    shells: RadialArtifactShells,
    target_layers,
    forcing: Forcing,
    *,
    Nrbase: int = 100,
    method: str = "combination",
    fluid_mu_tol_Pa: float = 1.0,
) -> LoveReductionConvergence:
    """Compute 1-D Love-number convergence across reduced layer counts.

    The highest requested count is only an internal convergence reference.
    If the result has not plateaued by that count, the static MAX_LAYERS limit
    is itself part of the numerical uncertainty and should be reported.
    """
    counts = sorted({int(n) for n in target_layers})
    if not counts:
        raise ValueError("target_layers must contain at least one layer count")
    if counts[0] < 2:
        raise ValueError("target layer counts must be >= 2")
    if counts[-1] > MAX_LAYERS:
        raise ValueError(
            f"target layer counts cannot exceed pylov3d MAX_LAYERS={MAX_LAYERS}"
        )

    solved = []
    for target in counts:
        reduced, _ = reduce_radial_shells(
            shells,
            target_layers=target,
            fluid_mu_tol_Pa=fluid_mu_tol_Pa,
        )
        model = reduced_shells_to_interior_model(
            reduced,
            fluid_mu_tol_Pa=fluid_mu_tol_Pa,
        )
        numerics = make_numerics(
            n_layers=model.n_layers,
            method=method,
            Nrbase=Nrbase,
        )
        love, _, _ = get_love(model, forcing, numerics)
        solved.append(
            (
                target,
                int(model.n_layers),
                complex(love.k[0]),
                complex(love.h[0]),
                complex(love.l[0]),
            )
        )

    _, reference_layers, k_ref, h_ref, l_ref = solved[-1]
    points = []
    for target, actual, k, h, l_love in solved:
        dk = abs(k - k_ref)
        dh = abs(h - h_ref)
        dl = abs(l_love - l_ref)
        points.append(
            LoveReductionPoint(
                target_layers=target,
                actual_layers=actual,
                k=k,
                h=h,
                l=l_love,
                abs_dk=float(dk),
                abs_dh=float(dh),
                abs_dl=float(dl),
                rel_dk=_relative(float(dk), k_ref),
                rel_dh=_relative(float(dh), h_ref),
                rel_dl=_relative(float(dl), l_ref),
            )
        )
    return LoveReductionConvergence(
        reference_layers=reference_layers,
        points=tuple(points),
    )
