# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Radial grid setup — translated from set_boundary_indices.m.

Four discretization methods are available:

- **combination**: ``Nrbase`` points plus a fraction proportional to layer
  thickness.  Default and recommended.
- **variable**: ``Nrbase`` points per layer (step-size varies).
- **fixed**: Total of ``Nrbase`` points with uniform step-size.  *Adjusts
  boundary radii* to land on grid points.
- **manual**: User-provided ``Nrlayer`` array.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from .constants import MAX_LAYERS
from .types import NumericsConfig, InteriorModel


def set_boundary_indices(
    numerics: NumericsConfig,
    model: InteriorModel,
    Nrlayer_manual: list[int] | None = None,
) -> tuple[NumericsConfig, InteriorModel]:
    """Compute radial grid indices per layer.

    Returns updated ``(NumericsConfig, InteriorModel)`` (the latter is only
    modified by the *fixed* method which adjusts boundary radii).

    Parameters
    ----------
    numerics : NumericsConfig
    model : InteriorModel
    Nrlayer_manual : list[int] | None
        Required when ``numerics.method == 'manual'``.  Format:
        ``[0, n_pts_layer1, n_pts_layer2, ...]`` (core entry is 0).
    """
    n_layers = numerics.n_layers
    Nrbase = numerics.Nrbase
    method = numerics.method

    # Extract plain Python floats for R0
    R0 = [float(model.R0[i]) for i in range(n_layers)]

    if n_layers <= 2:
        # Single active layer
        Nrlayer = [0] * MAX_LAYERS
        if n_layers == 2:
            Nrlayer[1] = Nrbase
        Nr = Nrbase
        BCindices = [0] * MAX_LAYERS
        return (
            numerics._replace(
                Nr=Nr,
                Nrlayer=jnp.array(Nrlayer, dtype=jnp.int32),
                BCindices=jnp.array(BCindices, dtype=jnp.int32),
            ),
            model,
        )

    Nrlayer = [0] * MAX_LAYERS
    BCindices = [0] * MAX_LAYERS

    if method == "combination":
        # Fractional thickness of each layer
        total_thickness = R0[-1] - R0[0]
        frac_r = []
        for i in range(1, n_layers):
            frac_r.append((R0[i] - R0[i - 1]) / total_thickness)
        for i in range(1, n_layers):
            Nrlayer[i] = int(math.floor(frac_r[i - 1] * Nrbase)) + Nrbase
        Nr = sum(Nrlayer)
        # BCindices: 0-based grid-node index of each internal layer boundary
        # (node k = last point of layer k's cumulative count). MATLAB stores
        # these 1-based for combination/variable/fixed but 0-based for manual
        # (set_boundary_indices.m:131-140, a latent MATLAB inconsistency);
        # here they are normalized to 0-based for every method.
        for i in range(1, n_layers):
            BCindices[i - 1] = sum(Nrlayer[: i + 1])

    elif method == "variable":
        for i in range(1, n_layers):
            Nrlayer[i] = Nrbase
        Nr = sum(Nrlayer)
        for i in range(1, n_layers):
            BCindices[i - 1] = sum(Nrlayer[: i + 1])

    elif method == "fixed":
        delta_r = (R0[-1] - R0[0]) / Nrbase
        R0_new = list(R0)
        for i in range(1, n_layers):
            if i == 1:
                Nrlayer[i] = int(math.floor((R0[i] - R0[0]) / delta_r))
                BCindices[0] = Nrlayer[i]
                R0_new[i] = R0[0] + Nrlayer[i] * delta_r
            elif i == n_layers - 1:
                Nrlayer[i] = Nrbase - sum(Nrlayer[:i])
                BCindices[i - 1] = sum(Nrlayer[: i + 1])
                R0_new[i] = R0_new[i - 1] + Nrlayer[i] * delta_r
            else:
                Nrlayer[i] = int(math.floor((R0[i] - R0_new[i - 1]) / delta_r))
                BCindices[i - 1] = sum(Nrlayer[: i + 1])
                R0_new[i] = R0_new[i - 1] + Nrlayer[i] * delta_r
        Nr = sum(Nrlayer)
        assert Nr == Nrbase, f"Nr ({Nr}) != Nrbase ({Nrbase})"
        # Update R0 in model
        R0_arr = jnp.array(R0_new + [0.0] * (MAX_LAYERS - n_layers), dtype=jnp.float64)
        model = model._replace(R0=R0_arr)

    elif method == "manual":
        if Nrlayer_manual is None:
            raise ValueError("manual method requires Nrlayer_manual argument")
        for i in range(min(len(Nrlayer_manual), MAX_LAYERS)):
            Nrlayer[i] = Nrlayer_manual[i]
        Nr = sum(Nrlayer)
        for i in range(1, n_layers - 1):
            BCindices[i - 1] = sum(Nrlayer[: i + 1])

    else:
        raise ValueError(f"Unknown method: {method!r}")

    numerics = numerics._replace(
        Nr=Nr,
        Nrlayer=jnp.array(Nrlayer, dtype=jnp.int32),
        BCindices=jnp.array(BCindices, dtype=jnp.int32),
    )
    return numerics, model
