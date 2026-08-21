# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Forward inputs for gravity and tides from one Mars alteration state.

This module enforces a key Task-1 modeling rule: gravity and tidal rigidity
should not receive unrelated perturbations. A single ``f_h * f_reactive`` field
is converted to density, shear modulus and bulk modulus, then passed to the
radially resolved gravity bridge and the effective crustal LOV3D layer bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars_alteration_gravity import layered_density_gravity_coefficients
from .mars_alteration_state import (
    AlterationEndmembers,
    ElasticLayerState,
    alteration_material_fields,
    elastic_layer_state_from_3d,
)


@dataclass(frozen=True)
class CommonAlterationForwardInputs:
    density_anomaly_kg_m3: np.ndarray
    mu_Pa: np.ndarray
    K_Pa: np.ndarray
    gravity_q_cos: np.ndarray
    gravity_q_sin: np.ndarray
    tidal_layer: ElasticLayerState


def build_common_alteration_forward_inputs(
    hydrated_fraction: np.ndarray,
    reactive_fraction: np.ndarray | float,
    radius_edges_m: np.ndarray,
    lmax: int,
    endmembers: AlterationEndmembers,
    *,
    mixing_law: str = "hill",
    tidal_layer_index: int = 3,
) -> CommonAlterationForwardInputs:
    """Map one 3D alteration state into gravity and tidal material inputs.

    ``hydrated_fraction`` must have shape ``(2*lmax, 4*lmax, nz)`` on the
    physical cell-centred equiangular map used by the Mars forward bridges.
    ``reactive_fraction`` may be scalar or broadcastable to that shape.

    Gravity retains every radial shell. Tides currently reduce the same mu/K
    fields to one effective radial crustal layer because the proposal Mars
    reference model has a single crust layer. That approximation is therefore
    explicit at one boundary rather than hidden in separate sensitivity models.
    """
    fh = np.asarray(hydrated_fraction, dtype=float)
    expected_surface = (2 * lmax, 4 * lmax)
    if fh.ndim != 3 or fh.shape[:2] != expected_surface:
        raise ValueError(
            f"hydrated_fraction must have shape ({expected_surface[0]}, {expected_surface[1]}, nz)"
        )
    if not np.all(np.isfinite(fh)) or np.any(fh < 0) or np.any(fh > 1):
        raise ValueError("hydrated_fraction must be finite and lie in [0,1]")

    rho, mu, K = alteration_material_fields(
        fh,
        reactive_fraction,
        endmembers,
        mixing_law=mixing_law,
    )
    if rho.shape != fh.shape:
        rho = np.broadcast_to(rho, fh.shape).copy()
        mu = np.broadcast_to(mu, fh.shape).copy()
        K = np.broadcast_to(K, fh.shape).copy()

    density_anomaly = rho - endmembers.rho_dry_kg_m3
    # Gravity bridge orders radial shells first; physical state orders radial last.
    q_cos, q_sin = layered_density_gravity_coefficients(
        np.moveaxis(density_anomaly, -1, 0),
        radius_edges_m,
        lmax,
    )
    tidal = elastic_layer_state_from_3d(
        mu,
        K,
        radius_edges_m,
        lmax,
        layer_index=tidal_layer_index,
    )
    return CommonAlterationForwardInputs(
        density_anomaly_kg_m3=np.asarray(density_anomaly),
        mu_Pa=np.asarray(mu),
        K_Pa=np.asarray(K),
        gravity_q_cos=q_cos,
        gravity_q_sin=q_sin,
        tidal_layer=tidal,
    )
