# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Boundary-condition assembly for coupled models with a subsurface ocean."""

from __future__ import annotations

import math

import numpy as np


def assemble_bc_ocean_coupled(
    Y_cmb,
    Y_surf,
    Y_ocean_start,
    Y_ocean_end,
    n_s,
    m_s,
    gc,
    Rc,
    rho2,
    rhoK_surface,
    Gg,
    rho1,
    gO,
    gI,
    rhoO,
    rho_below_ocean,
    rho_above_ocean,
    forcing,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 24N x 24N coupled-ocean boundary system.

    Coefficient columns are ordered as sub-ocean, ocean, then shell segments.
    Each mode owns 24 consecutive rows in BC1--BC24 order.
    """
    N = len(n_s)
    N3, N6, N8 = 3 * N, 6 * N, 8 * N
    B = np.zeros((24 * N, 24 * N), dtype=np.complex128)
    B2 = np.zeros(24 * N, dtype=np.complex128)
    identity = np.eye(N8, dtype=np.complex128)
    ocean_cols = slice(N8, 2 * N8)
    shell_cols = slice(2 * N8, 3 * N8)

    for k in range(N):
        n = int(n_s[k])
        m = int(m_s[k])
        row = 24 * k
        U, V, W = 3 * k, 3 * k + 1, 3 * k + 2
        R, S, T = N3 + 3 * k, N3 + 3 * k + 1, N3 + 3 * k + 2
        Phi, dPhi = N6 + 2 * k, N6 + 2 * k + 1

        # BC1--4: core-mantle boundary, sub-ocean coefficients.
        B[row, :N8] = (
            Y_cmb[U] - Y_cmb[R] / (gc * rho2) + Y_cmb[Phi] / gc
        )
        B[row + 1, :N8] = Y_cmb[S]
        B[row + 2, :N8] = Y_cmb[W] if n == 1 else Y_cmb[T]
        if n == 0:
            B[row + 3, :N8] = Y_ocean_start[dPhi]
        else:
            fac = 4.0 * math.pi * Gg / (gc * rho2) * (rho1 - rho2)
            B[row + 3, :N8] = (
                -fac * Y_cmb[R]
                + (n / Rc + fac * rho2) * Y_cmb[Phi]
                - Y_cmb[dPhi]
            )

        # BC5--8: surface boundary, shell coefficients.
        B[row + 4, shell_cols] = Y_surf[R]
        B[row + 5, shell_cols] = Y_surf[V] if n == 0 else Y_surf[S]
        B[row + 6, shell_cols] = Y_surf[W] if n == 0 else Y_surf[T]
        if n < 2:
            B[row + 7, shell_cols] = Y_surf[Phi]
        else:
            B[row + 7, shell_cols] = (
                4.0 * math.pi * Gg * rhoK_surface * Y_surf[U]
                + (n + 1) * Y_surf[Phi] + Y_surf[dPhi]
            )

        # BC9--13: sub-ocean to ocean-floor matching.
        B[row + 8, :N8] = (
            Y_ocean_start[U]
            - Y_ocean_start[R] / (gO * rhoO)
            + Y_ocean_start[Phi] / gO
        )
        B[row + 9, :N8] = Y_ocean_start[V] if n == 0 else Y_ocean_start[S]
        B[row + 10, :N8] = Y_ocean_start[W] if n == 0 else Y_ocean_start[T]
        B[row + 11, :N8] = Y_ocean_start[Phi]
        B[row + 11, ocean_cols] = -identity[Phi]
        if n < 0:
            B[row + 12, :N8] = Y_ocean_start[Phi]
        else:
            delta_rho_floor = rho_below_ocean - rhoO
            B[row + 12, :N8] = (
                Y_ocean_start[dPhi]
                + 4.0 * math.pi * Gg * delta_rho_floor * Y_ocean_start[U]
            )
            B[row + 12, ocean_cols] = -identity[dPhi]

        # BC14--19: displacement and stress variables are undefined in ocean.
        for offset, component in enumerate((U, V, W, R, S, T)):
            B[row + 13 + offset, ocean_cols] = identity[component]

        # BC20--24: ocean-ceiling to shell matching.
        B[row + 19, shell_cols] = (
            identity[U] - identity[R] / (gI * rhoO) + identity[Phi] / gI
        )
        B[row + 20, shell_cols] = identity[S]
        B[row + 21, shell_cols] = identity[W] if n == 1 else identity[T]
        B[row + 22, shell_cols] = identity[Phi]
        B[row + 22, ocean_cols] = -Y_ocean_end[Phi]
        if n < 0:
            B[row + 23, shell_cols] = identity[dPhi]
        else:
            delta_rho_ceiling = rho_above_ocean - rhoO
            B[row + 23, shell_cols] = (
                identity[dPhi]
                + 4.0 * math.pi * Gg * delta_rho_ceiling * identity[U]
            )
            B[row + 23, ocean_cols] = -Y_ocean_end[dPhi]

        forcings = forcing if isinstance(forcing, list) else [forcing]
        if any(f.n == n and f.m == m for f in forcings):
            B2[row + 7] = 2 * n + 1

    return B, B2
