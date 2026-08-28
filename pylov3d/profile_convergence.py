# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Love-number convergence diagnostic for reduced radial artifacts.

``profile_reduction`` guarantees mass closure but explicitly does not claim
tidal convergence. This module implements the publication gate documented in
``docs/RADIAL_PROFILE_REDUCTION_2026-08-21.md``: reduce the same
high-resolution radial artifact to a sequence of target layer counts, compute
Love numbers for each reduced model, and report how the tidal response changes
with resolution. Selecting a reduced profile for science requires the reported
successive differences to be small at the chosen layer count; if they are not,
raise the static layer limit or improve the reduction instead of relaxing the
requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MAX_LAYERS
from .love import get_love
from .profile_io import RadialArtifactShells
from .profile_reduction import (
    ReductionDiagnostics,
    reduce_radial_shells,
    reduced_shells_to_interior_model,
)
from .types import Forcing, make_numerics


@dataclass(frozen=True)
class LoveConvergenceEntry:
    """Love numbers for one reduced-resolution version of the same artifact."""

    target_layers: int
    layers: int
    k2: complex
    h2: complex
    l2: complex
    reduction: ReductionDiagnostics


def love_number_convergence(
    shells: RadialArtifactShells,
    forcing: Forcing,
    *,
    layer_counts: list[int],
    Nrbase: int = 100,
    method: str = "combination",
    fluid_mu_tol_Pa: float = 1.0,
) -> list[LoveConvergenceEntry]:
    """Compute degree-2 Love numbers over a sequence of target layer counts.

    Each entry reduces the *original* high-resolution shells independently, so
    the sequence probes reduction resolution rather than compounding merges.
    """
    counts = sorted(set(int(c) for c in layer_counts))
    if not counts:
        raise ValueError("layer_counts must be non-empty")
    if counts[0] < 2:
        raise ValueError("layer counts must be >= 2")
    if counts[-1] > MAX_LAYERS:
        raise ValueError(
            f"layer counts above MAX_LAYERS={MAX_LAYERS} cannot be solved; "
            "raise the static limit instead"
        )

    entries: list[LoveConvergenceEntry] = []
    for target in counts:
        reduced, diag = reduce_radial_shells(
            shells, target_layers=target, fluid_mu_tol_Pa=fluid_mu_tol_Pa
        )
        model = reduced_shells_to_interior_model(
            reduced, fluid_mu_tol_Pa=fluid_mu_tol_Pa
        )
        numerics = make_numerics(
            n_layers=model.n_layers, method=method, Nrbase=Nrbase
        )
        love, _, _ = get_love(model, forcing, numerics)
        entries.append(
            LoveConvergenceEntry(
                target_layers=target,
                layers=model.n_layers,
                k2=complex(love.k[0]),
                h2=complex(love.h[0]),
                l2=complex(love.l[0]),
                reduction=diag,
            )
        )
    return entries


def synthetic_mars_like_shells(n: int = 64) -> RadialArtifactShells:
    """High-resolution Mars-like radial artifact for diagnostics and tests.

    Fully liquid core to 1830 km, then a solid mantle and crust with smooth
    modulus/density gradients. This is a numerical fixture for exercising the
    reduction/convergence machinery, not a fitted Mars interior model.
    """
    if n < 8:
        raise ValueError("need at least 8 shells for a meaningful fixture")
    r = np.linspace(300e3, 3389.5e3, n)
    rho = np.where(
        r < 1830e3,
        6100.0,
        np.interp(r, [1830e3, 3280e3, 3389.5e3], [4100.0, 3400.0, 2900.0]),
    )
    K = np.where(
        r < 1830e3, 180e9, np.interp(r, [1830e3, 3389.5e3], [160e9, 60e9])
    )
    mu = np.where(
        r < 1830e3, 0.0, np.interp(r, [1830e3, 3389.5e3], [110e9, 30e9])
    )
    return RadialArtifactShells(r, rho, K, mu, {"body": "synthetic-mars-like"})


def successive_k2_differences(entries: list[LoveConvergenceEntry]) -> np.ndarray:
    """Relative |k2(n_i) - k2(n_{i-1})| / |k2(n_i)| between successive counts.

    This is the convergence diagnostic to report: the value at the layer count
    chosen for science quantifies how much the tidal response is still moving
    with reduction resolution.
    """
    if len(entries) < 2:
        raise ValueError("need at least two layer counts to assess convergence")
    k2 = np.array([e.k2 for e in entries])
    return np.abs(np.diff(k2)) / np.abs(k2[1:])
