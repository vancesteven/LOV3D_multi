# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Auxiliary propagator rows for the coupled JAX scan solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .jax_coupled import (
    _build_aprop_coupled_jax,
    _get_cached_scan,
    _grid_and_props,
)
from .propagator import compute_gravity


_AUX_CACHE: dict = {}


def _ocean_layer_bounds(model, numerics):
    """Ocean-node lookup with the full guard set; (0, 0, 0) if no ocean."""
    from .jax_ocean_scan import _detect_ocean
    return _detect_ocean(model, numerics)


def _get_cached_aux_builder(n_s, Coup, Gg):
    """Return a memoized jitted-vmapped coupled propagator builder."""
    key = (np.asarray(n_s).tobytes(), np.asarray(Coup).tobytes(), float(Gg))
    builder = _AUX_CACHE.get(key)
    if builder is None:
        static, _ = _get_cached_scan(n_s, Coup, Gg)

        def build_one(r, g, dg, muC, lam, rho, muC_amp, K_amp):
            return _build_aprop_coupled_jax(
                r, g, dg, muC, lam, rho, muC_amp, K_amp, static,
            )

        builder = jax.jit(jax.vmap(build_one))
        _AUX_CACHE[key] = builder
    return builder


def compute_aprop_aux_coupled(model, forcing, numerics, couplings, lateral):
    """Return first 3N propagator rows at every coupled radial grid point."""
    del forcing  # Signature parity with the coupled solution functions.
    r_grid, _, M_at_boundary, arrays = _grid_and_props(
        model, numerics, couplings, lateral,
    )

    def prepend(value, name):
        return np.concatenate((np.asarray([value]), arrays[name]))

    r = prepend(r_grid[0], "r_curr")
    muC = prepend(complex(model.muC[1]), "muC")
    lam = prepend(complex(model.lam[1]), "lam")
    rho = prepend(float(model.rho[1]), "rho")
    M_inner = prepend(M_at_boundary[1], "M_inner")
    R_inner = prepend(float(model.R[0]), "R_inner")
    muC_amp = np.vstack((lateral.muC_amp[1], arrays["muC_amp"]))
    K_amp = np.vstack((lateral.K_amp[1], arrays["K_amp"]))

    g = np.zeros_like(r)
    dg = np.zeros_like(r)
    for k in range(len(r)):
        g[k], dg[k] = compute_gravity(
            r[k], rho[k], M_inner[k], R_inner[k], model.Gg,
        )

    builder = _get_cached_aux_builder(
        couplings.n_s, couplings.Coup, model.Gg,
    )
    Aprop = builder(*(
        jnp.array(value)
        for value in (r, g, dg, muC, lam, rho, muC_amp, K_amp)
    ))
    jax.block_until_ready(Aprop)
    N3 = 3 * len(couplings.n_s)
    out = np.array(Aprop[:, :N3, :])

    # In-ocean nodes: displacement/stress recovery is undefined there
    # (matches NumPy's ``Aprop_aux[k_idx] = 0.0`` for layer_map==ocean_layer).
    ocean_layer, ocean_start, ocean_end = _ocean_layer_bounds(model, numerics)
    if ocean_layer:
        out[ocean_start + 1: ocean_end + 1] = 0.0
    return out
