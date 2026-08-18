# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Stress/strain recovery for coupled solutions using solver conventions.

The coupled solver stores state by field blocks,
``[U,V,W]_(all modes), [R,S,T]_(all modes), [Phi,dPhi]_(all modes)``.
For laterally heterogeneous layers, stress recovery must also use the same
coupled A1/A2 constitutive matrices used by the propagator, including the
off-diagonal rheology terms.
"""
from __future__ import annotations

import numpy as np

from .energy import build_A14_A15
from .propagator import build_A1_A2, build_A1_A2_coupled, build_A3
from .types import InteriorModel, NumericsConfig


def recover_coupled_fields(
    y_sol: np.ndarray,
    r_grid: np.ndarray,
    Aprop_aux: np.ndarray,
    model: InteriorModel,
    n_s: np.ndarray,
    numerics: NumericsConfig,
    *,
    couplings=None,
    lateral=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover GSH displacement, stress and strain from a coupled solution.

    If ``couplings`` and ``lateral`` are supplied, per-layer constitutive
    matrices include all lateral-rheology coupling terms via the same
    :func:`build_A1_A2_coupled` routine used by the forward propagator.
    """
    ns = np.asarray(n_s, dtype=int)
    N = len(ns)
    Nr = numerics.Nr
    N3 = 3 * N

    if y_sol.shape != (Nr + 1, 8 * N):
        raise ValueError("coupled solution shape is inconsistent with mode basis")
    if Aprop_aux.shape != (Nr + 1, N3, 8 * N):
        raise ValueError("Aprop_aux shape is inconsistent with mode basis")

    # Solver state ordering is grouped by fields, so displacement is the
    # leading 3N block.  Do not index as 8 values per mode here.
    U_all = np.asarray(y_sol[:, :N3])

    A3_inv = np.zeros((N3, N3), dtype=np.complex128)
    A14 = np.zeros((6 * N, N3), dtype=np.complex128)
    A15 = np.zeros((6 * N, N3), dtype=np.complex128)
    for i, n in enumerate(ns):
        A3_inv[3*i:3*i+3, 3*i:3*i+3] = np.linalg.inv(build_A3(int(n)))
        a14, a15 = build_A14_A15(int(n))
        A14[6*i:6*i+6, 3*i:3*i+3] = a14
        A15[6*i:6*i+6, 3*i:3*i+3] = a15

    u = (A3_inv @ U_all.T).T

    # Reconstruct the same layer map used by the solver.
    layer_map = np.zeros(Nr + 1, dtype=int)
    layer_map[0] = 0
    k = 1
    for ilayer in range(1, model.n_layers):
        for _ in range(int(numerics.Nrlayer[ilayer])):
            layer_map[k] = ilayer
            k += 1

    A1_cache: dict[int, np.ndarray] = {}
    A2_cache: dict[int, np.ndarray] = {}
    for ilayer in range(1, model.n_layers):
        if int(model.ocean[ilayer]) == 1:
            continue
        mu = complex(model.muC[ilayer])
        lam = complex(model.lam[ilayer])
        if couplings is not None and lateral is not None and N > 1:
            a1, a2 = build_A1_A2_coupled(
                ns,
                mu,
                lam,
                couplings.Coup,
                np.asarray(lateral.muC_amp[ilayer, :]),
                np.asarray(lateral.K_amp[ilayer, :]),
            )
        else:
            a1 = np.zeros((6*N, N3), dtype=np.complex128)
            a2 = np.zeros_like(a1)
            for i, n in enumerate(ns):
                x1, x2 = build_A1_A2(int(n), mu, lam)
                a1[6*i:6*i+6, 3*i:3*i+3] = x1
                a2[6*i:6*i+6, 3*i:3*i+3] = x2
        A1_cache[ilayer] = a1
        A2_cache[ilayer] = a2

    u_dot = np.zeros((Nr + 1, N3), dtype=np.complex128)
    stress = np.zeros((Nr + 1, 6*N), dtype=np.complex128)
    strain = np.zeros_like(stress)

    for kr in range(Nr + 1):
        ilayer = int(layer_map[kr])
        if ilayer == 0 or int(model.ocean[ilayer]) == 1:
            continue
        if ilayer not in A1_cache:
            continue
        r = float(r_grid[kr])
        x_dot = Aprop_aux[kr] @ y_sol[kr]
        u_dot[kr] = A3_inv @ x_dot

        # Literal MATLAB get_solution.m auxiliary-field convention.
        if r > 0:
            stress[kr] = A1_cache[ilayer] @ u_dot[kr] + A2_cache[ilayer] @ u[kr] / r
            strain[kr] = A14 @ u_dot[kr] + A15 @ u[kr] / r
        else:
            stress[kr] = A1_cache[ilayer] @ u_dot[kr]
            strain[kr] = A14 @ u_dot[kr]

    return u, stress, strain
