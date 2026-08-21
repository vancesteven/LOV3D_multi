# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Lateral bulk-modulus support for the coupled LOV3D constitutive equations.

The upstream MATLAB interfaces advertise ``K_variable``, and ``get_solution.m``
contains a bulk-coupling slot in ``rheology_variable(:,3)``. However,
``get_rheology.m`` never populates that slot and even filters active modes using
only the complex-shear column. Therefore bulk-modulus heterogeneity is an
unfinished feature in the reference code, not a MATLAB-validated capability.

The coupled constitutive equation itself is unambiguous. Its scalar-stress row
contains

    3*lambda + 2*mu = 3*K.

Consequently a physical bulk-modulus harmonic ``delta K_lm`` must enter the
coupled row as ``K_amp = 3*delta K_lm`` in the same normalized modulus units as
``muC_amp``. The low-level algebra is regression-tested directly against the
uniform constitutive matrix before this helper is used for Mars science.

This module leaves the MATLAB-parity path untouched and provides an explicit,
separately tested extension for physical K variations.
"""

from __future__ import annotations

import numpy as np

from .rheology import _ensure_conjugate_pairs, process_lateral_variations
from .types import Forcing, InteriorModel, LateralRheology


def bulk_constitutive_amplitude(mean_K: float, fractional_amplitude: complex) -> complex:
    """Convert ``delta K / K`` to the scalar constitutive coupling amplitude.

    ``mean_K`` must already be in the normalized modulus units used by the
    solver. The factor of three follows exactly from ``3 lambda + 2 mu = 3 K``.
    """
    if not np.isfinite(mean_K) or mean_K <= 0:
        raise ValueError("mean_K must be finite and positive")
    amp = complex(fractional_amplitude)
    if not np.isfinite(amp.real) or not np.isfinite(amp.imag):
        raise ValueError("fractional_amplitude must be finite")
    return 3.0 * float(mean_K) * amp


def inject_bulk_modulus_variations(
    model: InteriorModel,
    lateral: LateralRheology,
    K_variable: dict[int, list[tuple[int, int, complex]]] | None,
) -> LateralRheology:
    """Add physical bulk-modulus harmonics to processed lateral rheology.

    ``K_variable[layer]`` uses the same public convention as the MATLAB input:
    tuples ``(degree, order, deltaK/Kmean)``. Missing conjugate partners are
    added so the physical K field is real. Existing shear-rheology modes are
    preserved and the mode union is returned in deterministic sorted order.
    """
    K_variable = K_variable or {}
    mode_maps: dict[int, dict[tuple[int, int], complex]] = {}
    all_modes = {tuple(map(int, nm)) for nm in np.asarray(lateral.variations)}
    # A no-variation placeholder is represented as [[0,0]] by the old path.
    all_modes.discard((0, 0))

    for ilayer in range(1, model.n_layers):
        modes = K_variable.get(ilayer) or []
        if not modes:
            mode_maps[ilayer] = {}
            continue
        modes = _ensure_conjugate_pairs(modes)
        combined: dict[tuple[int, int], complex] = {}
        for degree, order, amp in modes:
            if degree <= 0:
                continue
            key = (int(degree), int(order))
            combined[key] = combined.get(key, 0.0j) + complex(amp)
        combined = {key: amp for key, amp in combined.items() if abs(amp) > 0}
        mode_maps[ilayer] = combined
        all_modes.update(combined)

    if not all_modes:
        return lateral

    sorted_modes = sorted(all_modes)
    mode_index = {nm: i for i, nm in enumerate(sorted_modes)}
    n_modes = len(sorted_modes)
    mu_new = np.zeros((model.n_layers, n_modes), dtype=complex)
    K_new = np.zeros_like(mu_new)

    old_modes = [tuple(map(int, nm)) for nm in np.asarray(lateral.variations)]
    for old_j, nm in enumerate(old_modes):
        if nm == (0, 0) or nm not in mode_index:
            continue
        new_j = mode_index[nm]
        mu_new[:, new_j] = lateral.muC_amp[:, old_j]
        K_new[:, new_j] = lateral.K_amp[:, old_j]

    uniform = np.asarray(lateral.uniform, dtype=bool).copy()
    for ilayer, mapping in mode_maps.items():
        if not mapping:
            continue
        mean_K = float(model.Ks[ilayer])
        for nm, frac_amp in mapping.items():
            K_new[ilayer, mode_index[nm]] += bulk_constitutive_amplitude(mean_K, frac_amp)
        if np.any(np.abs(K_new[ilayer]) > 0):
            uniform[ilayer] = False

    return LateralRheology(
        variations=np.asarray(sorted_modes, dtype=int),
        muC_amp=mu_new,
        K_amp=K_new,
        uniform=uniform,
    )


def process_lateral_variations_with_bulk(
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
    *,
    mu_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    eta_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    K_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    rheology_cutoff: float = 2.0,
) -> tuple[InteriorModel, LateralRheology]:
    """Process shear/viscosity through the parity path, then add physical K.

    Keeping the two stages explicit is intentional: it protects the strict
    MATLAB raw-grid shear-rheology validation while enabling a separately
    tested bulk-modulus extension for composition-derived Mars models.
    """
    model_out, lateral = process_lateral_variations(
        model,
        forcing,
        mu_variable=mu_variable,
        eta_variable=eta_variable,
        K_variable=None,
        rheology_cutoff=rheology_cutoff,
    )
    return model_out, inject_bulk_modulus_variations(model_out, lateral, K_variable)
