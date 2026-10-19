# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
"""MATLAB-faithful raw-grid lateral rheology processing.

This module provides the map-input counterpart to
:func:`pylov3d.rheology.process_lateral_variations`.  It is intentionally
separate while TASK-046 closes MATLAB/Python transform parity.

Inputs are fractional deviations from the scalar layer means, i.e.
``mu_rel = 1 + dmu`` and ``eta_rel = 1 + deta``.  For viscoelastic layers the
nonlinear Maxwell transform is evaluated on the supplied equiangular map and
analysed with the literal MATLAB ``LatLon_SPH`` convention in
:mod:`pylov3d.matlab_sph`.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .matlab_sph import filter_rheology_modes, maxwell_rheology_from_fractional_grid
from .types import InteriorModel, LateralRheology


def _infer_lmax(grid: np.ndarray) -> int:
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("fractional rheology grid must be 2-D")
    nlat, nlon = grid.shape
    if nlat % 2 or nlon != 2 * nlat:
        raise ValueError(
            "MATLAB equiangular rheology grid must have shape (2*lmax, 4*lmax)"
        )
    return nlat // 2


def process_lateral_fractional_grids(
    model: InteriorModel,
    *,
    dmu_grids: dict[int, np.ndarray] | None = None,
    deta_grids: dict[int, np.ndarray] | None = None,
    rheology_cutoff: float = 2.0,
    minimum_rheology_value: float = -14.0,
) -> tuple[InteriorModel, LateralRheology]:
    """Process lateral viscoelastic rheology supplied as fractional maps.

    Parameters
    ----------
    model
        Already-normalized model (normally returned by ``get_rheology``).
    dmu_grids, deta_grids
        Mapping from 0-based layer index to fractional-deviation maps.  Maps
        must use the native LOV3D/MATLAB equiangular shape
        ``(2*lmax, 4*lmax)``.  Missing ``dmu`` or ``deta`` for a layer is
        treated as zero variation on the other field's grid.
    rheology_cutoff
        Retain modes within this many decades of the strongest real or
        imaginary nonzero coefficient, matching ``get_rheology.m``.
    minimum_rheology_value
        MATLAB noise guard in log10 relative amplitude.

    Notes
    -----
    This first production path is intentionally limited to viscoelastic
    layers because that is the nonlinear transform whose MATLAB parity is
    load-bearing for TASK-046.  Elastic map inputs should continue to use the
    established coefficient path until a separate map regression is added.
    """
    dmu_grids = dmu_grids or {}
    deta_grids = deta_grids or {}
    layers = sorted(set(dmu_grids) | set(deta_grids))

    n_layers = model.n_layers
    uniform = np.ones(n_layers, dtype=bool)
    muC_new = np.asarray(model.muC, dtype=complex).copy()
    lam_new = np.asarray(model.lam, dtype=complex).copy()
    layer_muC: dict[int, dict[tuple[int, int], complex]] = {}

    for ilayer in range(1, n_layers):
        if ilayer not in layers:
            layer_muC[ilayer] = {}
            continue
        if bool(model.elastic[ilayer]):
            raise NotImplementedError(
                "fractional-grid processing is currently validated only for viscoelastic layers"
            )

        template = dmu_grids.get(ilayer)
        if template is None:
            template = deta_grids[ilayer]
        template = np.asarray(template, dtype=float)
        lmax = _infer_lmax(template)

        dmu = np.asarray(dmu_grids.get(ilayer, np.zeros_like(template)), dtype=float)
        deta = np.asarray(deta_grids.get(ilayer, np.zeros_like(template)), dtype=float)
        if dmu.shape != template.shape or deta.shape != template.shape:
            raise ValueError("dmu and deta grids for a layer must have identical shapes")

        mu00, modes = maxwell_rheology_from_fractional_grid(
            dmu,
            deta,
            mu_mean=float(model.mu[ilayer]),
            maxwell_mean=float(model.MaxTime[ilayer]),
            lmax=lmax,
        )
        kept = filter_rheology_modes(
            modes,
            cutoff=rheology_cutoff,
            minimum_log_value=minimum_rheology_value,
        )
        significant = {(n, m): amp for n, m, amp, _lr, _li in kept}

        muC_new[ilayer] = mu00
        lam_new[ilayer] = complex(model.Ks[ilayer]) - (2.0 / 3.0) * mu00
        layer_muC[ilayer] = significant
        uniform[ilayer] = not bool(significant)

    all_nm: set[tuple[int, int]] = set()
    for layer in layer_muC.values():
        all_nm.update(layer)

    if all_nm:
        sorted_nm = sorted(all_nm)
        variations = np.asarray(sorted_nm, dtype=int)
        muC_amp = np.zeros((n_layers, len(sorted_nm)), dtype=complex)
        K_amp = np.zeros_like(muC_amp)
        for ilayer in range(1, n_layers):
            for j, nm in enumerate(sorted_nm):
                muC_amp[ilayer, j] = layer_muC.get(ilayer, {}).get(nm, 0j)
    else:
        variations = np.zeros((1, 2), dtype=int)
        muC_amp = np.zeros((n_layers, 1), dtype=complex)
        K_amp = np.zeros_like(muC_amp)

    model = model._replace(
        muC=jnp.asarray(muC_new, dtype=jnp.complex128),
        lam=jnp.asarray(lam_new, dtype=jnp.complex128),
    )
    return model, LateralRheology(
        variations=variations,
        muC_amp=muC_amp,
        K_amp=K_amp,
        uniform=uniform,
    )
