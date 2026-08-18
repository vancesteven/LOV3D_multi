# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Coupled tidal-energy contraction for forcings with different mode bases.

TASK-046 showed that each forcing component in a laterally heterogeneous
problem must retain its own perturbation-order closure.  Solving all forcings
on the closure generated from a single forcing can truncate legitimate modes
and corrupt the quadratic energy contraction.

This module preserves each forcing's native coupled solution basis, computes
stress/strain on that native basis, maps the resulting *fields* into the union
of all (n,m) modes, and only then performs the quadratic energy contraction.
That is equivalent to summing the physical fields before evaluating the
energy, while avoiding an oversized solve basis that could introduce coupling
paths beyond the requested perturbation order.
"""

from __future__ import annotations

import math

import numpy as np

from .energy import compute_stress_strain_coupled
from .energy_couplings import get_energy_couplings
from .types import EnergySpectra, Forcing, InteriorModel, NumericsConfig


def get_energy_coupled_multibasis(
    y_solutions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    forcings: list[Forcing],
    model: InteriorModel,
    numerics: NumericsConfig,
    n_s_list: list[np.ndarray],
    m_s_list: list[np.ndarray],
    Nenergy: int = 8,
) -> EnergySpectra:
    """Compute coupled dissipation from forcing solutions on distinct bases.

    Parameters are the same as :func:`pylov3d.energy.get_energy_coupled`,
    except ``n_s_list`` and ``m_s_list`` provide one native mode basis per
    forcing.  All solutions must share the same radial grid.
    """
    n_forcing = len(forcings)
    if not (
        len(y_solutions) == len(n_s_list) == len(m_s_list) == n_forcing
    ):
        raise ValueError("one solution and one mode basis are required per forcing")
    if n_forcing == 0:
        raise ValueError("at least one forcing is required")

    r_grid = np.asarray(y_solutions[0][1])
    nr = numerics.Nr
    for y_sol, r_j, _ in y_solutions:
        if y_sol.shape[0] != nr + 1:
            raise ValueError("solution radial dimension is inconsistent with numerics")
        if not np.array_equal(np.asarray(r_j), r_grid):
            raise ValueError("all forcing solutions must use the same radial grid")

    # Union is used only for field bookkeeping and the energy contraction,
    # never as a solve basis.
    union_modes = sorted(
        {
            (int(n), int(m))
            for ns, ms in zip(n_s_list, m_s_list)
            for n, m in zip(np.asarray(ns), np.asarray(ms))
        }
    )
    n_union = np.asarray([p[0] for p in union_modes], dtype=int)
    m_union = np.asarray([p[1] for p in union_modes], dtype=int)
    mode_index = {mode: i for i, mode in enumerate(union_modes)}
    n_mode = len(union_modes)

    stress_union = np.zeros((nr + 1, 6, n_mode), dtype=np.complex128)
    strain_union = np.zeros((nr + 1, 6, n_mode), dtype=np.complex128)

    for j, forcing in enumerate(forcings):
        y_sol, r_j, aprop_j = y_solutions[j]
        ns = np.asarray(n_s_list[j], dtype=int)
        ms = np.asarray(m_s_list[j], dtype=int)
        if y_sol.shape[1] != 8 * len(ns):
            raise ValueError("solution width does not match its native mode basis")

        _, stress_flat, strain_flat = compute_stress_strain_coupled(
            y_sol, np.asarray(r_j), aprop_j, model, ns, numerics,
        )
        stress_native = stress_flat.reshape(nr + 1, len(ns), 6).transpose(0, 2, 1)
        strain_native = strain_flat.reshape(nr + 1, len(ns), 6).transpose(0, 2, 1)

        for k, mode in enumerate(zip(ns, ms)):
            u = mode_index[(int(mode[0]), int(mode[1]))]
            stress_union[:, :, u] += float(forcing.F) * stress_native[:, :, k]
            strain_union[:, :, u] += float(forcing.F) * strain_native[:, :, k]

    # Match get_energy_coupled's MATLAB component ordering.
    reorder = [1, 2, 3, 4, 5, 0]
    stress_p = stress_union[:, reorder, :]
    strain_p = strain_union[:, reorder, :]

    stress_n = np.zeros_like(stress_p)
    strain_n = np.zeros_like(strain_p)
    for i, (n_i, m_i) in enumerate(union_modes):
        neg = mode_index.get((n_i, -m_i))
        if neg is not None:
            stress_n[:, :, i] = np.conj(stress_p[:, :, neg])
            strain_n[:, :, i] = np.conj(strain_p[:, :, neg])
        elif m_i == 0:
            stress_n[:, :, i] = np.conj(stress_p[:, :, i])
            strain_n[:, :, i] = np.conj(strain_p[:, :, i])

    ec = get_energy_couplings(n_union, m_union, Nenergy=Nenergy)
    EC = ec.EC
    n_en = ec.n_en
    m_en = ec.m_en
    n_en_mode = len(n_en)

    nz_i1, nz_i2, nz_k, nz_i3, nz_i4 = np.nonzero(EC)
    n2_offset = np.asarray([-2, -1, 0, 1, 2, 0])
    energy = np.zeros((nr + 1, n_en_mode), dtype=float)

    for idx in range(len(nz_i1)):
        i1 = nz_i1[idx]
        i2 = nz_i2[idx]
        k = nz_k[idx]
        i3 = nz_i3[idx]
        i4 = nz_i4[idx]
        n2a = int(n_union[i1]) + int(n2_offset[i3])
        n2b = int(n_union[i2]) + int(n2_offset[i4])
        ec_val = EC[i1, i2, k, i3, i4]
        phase1 = (-1) ** (n2a + int(n_union[i1]) - int(m_union[i1]))
        phase2 = (-1) ** (n2b + int(n_union[i2]) - int(m_union[i2]))
        term = (
            1j * 2 * math.pi * phase1 * ec_val
            * stress_n[:, i3, i1] * strain_p[:, i4, i2]
            - 1j * 2 * math.pi * phase2 * ec_val
            * stress_p[:, i3, i1] * strain_n[:, i4, i2]
        )
        energy[:, k] += term.real

    r_mid = (r_grid[:-1] + r_grid[1:]) / 2
    dr = r_grid[1:] - r_grid[:-1]
    weights = r_mid ** 2 * dr
    energy_integral = np.zeros(n_en_mode)
    for k in range(n_en_mode):
        energy_integral[k] = np.sum(
            weights * (energy[:-1, k] + energy[1:, k]) / 2
        )

    return EnergySpectra(
        n=n_en,
        m=m_en,
        energy_integral=energy_integral,
        energy_profile=energy,
    )
