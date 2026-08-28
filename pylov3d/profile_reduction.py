# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Scientifically explicit reduction of high-resolution radial artifacts.

pylov3d uses a static JAX layer limit. High-resolution PlanetProfile profiles
therefore require a controlled reduction rather than silent decimation. This
module greedily merges adjacent shells while:

* preserving total mass exactly at every merge,
* never merging across a fluid/solid shear-modulus boundary,
* choosing the merge with the smallest axial-moment perturbation,
* volume-averaging K and mu within the merged shell.

Mass preservation is exact by construction. Moment of inertia and, especially,
Love-number convergence remain diagnostics, not assumptions. Publication runs
must report both the bulk diagnostics and a target-layer convergence test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MAX_LAYERS
from .profile_io import (
    RadialArtifactShells,
    RadialBulkDiagnostics,
    load_radial_artifact_shells,
    radial_shell_bulk_diagnostics,
)
from .types import InteriorModel, make_interior_model


@dataclass(frozen=True)
class ReductionDiagnostics:
    original_layers: int
    reduced_layers: int
    mass_relative_change: float
    cmr2_change: float
    original: RadialBulkDiagnostics
    reduced: RadialBulkDiagnostics


def _shell_geometry(r_out: np.ndarray):
    r_in = np.concatenate(([0.0], r_out[:-1]))
    volume = (4.0 * np.pi / 3.0) * (r_out**3 - r_in**3)
    moi_factor = (8.0 * np.pi / 15.0) * (r_out**5 - r_in**5)
    return r_in, volume, moi_factor


def _merge_pair(shells: RadialArtifactShells, i: int) -> RadialArtifactShells:
    r = np.asarray(shells.outer_radius_m, dtype=float)
    rho = np.asarray(shells.rho_kgm3, dtype=float)
    K = np.asarray(shells.K_Pa, dtype=float)
    mu = np.asarray(shells.mu_Pa, dtype=float)
    if not 0 <= i < r.size - 1:
        raise IndexError("merge index must select an adjacent shell pair")

    r_in, volume, _ = _shell_geometry(r)
    merged_volume = volume[i] + volume[i + 1]
    merged_mass = rho[i] * volume[i] + rho[i + 1] * volume[i + 1]
    merged_rho = merged_mass / merged_volume
    merged_K = (K[i] * volume[i] + K[i + 1] * volume[i + 1]) / merged_volume
    merged_mu = (mu[i] * volume[i] + mu[i + 1] * volume[i + 1]) / merged_volume

    # Pair i spans from its original inner boundary to shell i+1's outer edge.
    r_new = np.delete(r, i)
    rho_new = np.delete(rho, i)
    K_new = np.delete(K, i)
    mu_new = np.delete(mu, i)
    rho_new[i] = merged_rho
    K_new[i] = merged_K
    mu_new[i] = merged_mu
    return RadialArtifactShells(
        outer_radius_m=r_new,
        rho_kgm3=rho_new,
        K_Pa=K_new,
        mu_Pa=mu_new,
        metadata=dict(shells.metadata),
    )


def reduce_radial_shells(
    shells: RadialArtifactShells,
    *,
    target_layers: int = MAX_LAYERS,
    fluid_mu_tol_Pa: float = 1.0,
) -> tuple[RadialArtifactShells, ReductionDiagnostics]:
    """Reduce a radial model with exact mass closure and explicit MoI error."""
    if target_layers < 2:
        raise ValueError("target_layers must be >= 2")
    if fluid_mu_tol_Pa < 0:
        raise ValueError("fluid_mu_tol_Pa must be non-negative")
    n0 = shells.outer_radius_m.size
    if target_layers >= n0:
        d = radial_shell_bulk_diagnostics(shells)
        return shells, ReductionDiagnostics(n0, n0, 0.0, 0.0, d, d)

    current = shells
    original_diag = radial_shell_bulk_diagnostics(shells)
    while current.outer_radius_m.size > target_layers:
        fluid = np.asarray(current.mu_Pa) <= fluid_mu_tol_Pa
        candidates = [i for i in range(fluid.size - 1) if fluid[i] == fluid[i + 1]]
        if not candidates:
            raise ValueError(
                "cannot reach target_layers without crossing a fluid/solid boundary"
            )

        base_C = radial_shell_bulk_diagnostics(current).axial_moi_kgm2
        best_i = None
        best_score = None
        for i in candidates:
            trial = _merge_pair(current, i)
            trial_C = radial_shell_bulk_diagnostics(trial).axial_moi_kgm2
            score = abs(trial_C - base_C) / abs(base_C)
            if best_score is None or score < best_score:
                best_score = score
                best_i = i
        current = _merge_pair(current, int(best_i))

    reduced_diag = radial_shell_bulk_diagnostics(current)
    mass_change = (reduced_diag.mass_kg - original_diag.mass_kg) / original_diag.mass_kg
    cmr2_change = reduced_diag.cmr2 - original_diag.cmr2
    return current, ReductionDiagnostics(
        original_layers=n0,
        reduced_layers=current.outer_radius_m.size,
        mass_relative_change=mass_change,
        cmr2_change=cmr2_change,
        original=original_diag,
        reduced=reduced_diag,
    )


def reduce_radial_artifact(
    path,
    *,
    target_layers: int = MAX_LAYERS,
    body_radius_m: float | None = None,
    fluid_mu_tol_Pa: float = 1.0,
) -> tuple[RadialArtifactShells, ReductionDiagnostics]:
    """Load a high-resolution artifact and reduce it explicitly."""
    shells = load_radial_artifact_shells(
        path,
        body_radius_m=body_radius_m,
        enforce_max_layers=False,
    )
    return reduce_radial_shells(
        shells,
        target_layers=target_layers,
        fluid_mu_tol_Pa=fluid_mu_tol_Pa,
    )


def reduced_shells_to_interior_model(
    shells: RadialArtifactShells,
    *,
    fluid_mu_tol_Pa: float = 1.0,
) -> InteriorModel:
    """Convert an already-controlled reduced shell set to ``InteriorModel``."""
    n = shells.outer_radius_m.size
    if n > MAX_LAYERS:
        raise ValueError(f"reduced model still has {n} shells > MAX_LAYERS={MAX_LAYERS}")
    fluid = np.asarray(shells.mu_Pa) <= fluid_mu_tol_Pa
    ocean = fluid.astype(int)
    # The innermost contiguous fluid run is a liquid core, not an ocean: the
    # solver (like upstream MATLAB LOV3D) rejects ocean flags at layer
    # index < 2, and build_mars_model represents the liquid core as mu=0
    # with the ocean flag unset.
    i = 0
    while i < n and fluid[i]:
        ocean[i] = 0
        i += 1
    mu = np.asarray(shells.mu_Pa, dtype=float).copy()
    mu[fluid] = 0.0
    return make_interior_model(
        R0_km=(np.asarray(shells.outer_radius_m) / 1e3).tolist(),
        rho0=np.asarray(shells.rho_kgm3).tolist(),
        mu0=mu.tolist(),
        Ks0=np.asarray(shells.K_Pa).tolist(),
        eta0=[None] * n,
        ocean=ocean.tolist(),
    )
