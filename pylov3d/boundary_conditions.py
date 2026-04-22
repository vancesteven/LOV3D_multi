"""Boundary condition assembly — translated from get_solution.m lines 604-850.

Assembles the linear system ``B · C = B2`` whose solution gives the
integration constants.  The physical solution is then ``y = Y · C`` at each
radial point.

Two cases:
- **No ocean** (``assemble_bc_no_ocean``): 8 BCs, 8 unknowns per mode.
- **Ocean** (``assemble_bc_ocean``): 24 BCs, 24 unknowns per mode
  (sub-ocean, ocean, shell).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# No-ocean case
# ---------------------------------------------------------------------------

def assemble_bc_no_ocean(
    Y_cmb: np.ndarray,
    Y_surf: np.ndarray,
    n: int,
    m: int,
    gc: float,
    Rc: float,
    rho2: float,
    rhoK: float,
    Gg: float,
    rho_layer2: float,
    forcing,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble 8×8 boundary condition matrix for the no-ocean case.

    Parameters
    ----------
    Y_cmb : (8, 8) — fundamental matrix at CMB (k=0)
    Y_surf : (8, 8) — fundamental matrix at surface (k=Nr)
    n : int — harmonic degree of mode
    m : int — harmonic order of mode
    gc : float — gravity at CMB
    Rc : float — normalized core radius
    rho2 : float — effective core density (Delta_rho[0] + rho[1])
    rhoK : float — surface layer density
    Gg : float — normalized gravitational constant
    rho_layer2 : float — density of first solid layer
    forcing : Forcing or list[Forcing] — tidal forcing

    Returns
    -------
    B : (8, 8) complex
    B2 : (8,) complex
    """
    B = np.zeros((8, 8), dtype=np.complex128)
    B2 = np.zeros(8, dtype=np.complex128)

    # State indices: U=0, V=1, W=2, R=3, S=4, T=5, Φ=6, dΦ/dr=7

    # ---- Core-mantle boundary (CMB), evaluated at Y_cmb ----

    # BC1: inviscid core condition
    if n == -3:
        # Special: R = 0
        B[0, :] = Y_cmb[3, :]
    else:
        # U - R/(gc*rho2) + Φ/gc = 0
        B[0, :] = Y_cmb[0, :] - Y_cmb[3, :] / (gc * rho2) + Y_cmb[6, :] / gc

    # BC2: no tangential stress S = 0
    B[1, :] = Y_cmb[4, :]

    # BC3: toroidal
    if n == 1:
        B[2, :] = Y_cmb[2, :]   # W = 0
    else:
        B[2, :] = Y_cmb[5, :]   # T = 0

    # BC4: potential continuity at CMB
    if n == 0:
        # dΦ/dr at surface = 0 (special degree-0 case)
        B[3, :] = Y_surf[7, :]
    else:
        fac = 4.0 * math.pi * Gg / (gc * rho2) * (rho_layer2 - rho2)
        B[3, :] = (-fac * Y_cmb[3, :]
                   + (n / Rc + fac * rho2) * Y_cmb[6, :]
                   - Y_cmb[7, :])

    # ---- Surface boundary ----

    # BC5: no radial stress R = 0
    B[4, :] = Y_surf[3, :]

    # BC6: no tangential stress
    if n == 0:
        B[5, :] = Y_surf[1, :]   # V = 0
    else:
        B[5, :] = Y_surf[4, :]   # S = 0

    # BC7: toroidal
    if n == 0:
        B[6, :] = Y_surf[2, :]   # W = 0
    else:
        B[6, :] = Y_surf[5, :]   # T = 0

    # BC8: surface potential condition
    if n < 2:
        B[7, :] = Y_surf[6, :]   # Φ = 0
    else:
        # 4π·Gg·ρK·U + (n+1)·Φ + dΦ/dr = (2n+1)·F
        B[7, :] = (4.0 * math.pi * Gg * rhoK * Y_surf[0, :]
                   + (n + 1) * Y_surf[6, :]
                   + Y_surf[7, :])

    # ---- Forcing RHS ----
    if isinstance(forcing, list):
        for f in forcing:
            if f.n == n and f.m == m:
                B2[7] = (2 * n + 1)
                break
    else:
        if forcing.n == n and forcing.m == m:
            B2[7] = (2 * n + 1)

    return B, B2


# ---------------------------------------------------------------------------
# No-ocean case — coupled (multi-mode)
# ---------------------------------------------------------------------------

def assemble_bc_no_ocean_coupled(
    Y_cmb: np.ndarray,
    Y_surf: np.ndarray,
    n_s: np.ndarray,
    m_s: np.ndarray,
    gc: float,
    Rc: float,
    rho2: float,
    rhoK: float,
    Gg: float,
    rho_layer2: float,
    forcing,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble 8N×8N boundary condition matrix for the coupled no-ocean case.

    Parameters
    ----------
    Y_cmb : (8N, 8N) — fundamental matrix at CMB
    Y_surf : (8N, 8N) — fundamental matrix at surface
    n_s : (N,) int — mode degrees
    m_s : (N,) int — mode orders
    gc, Rc, rho2, rhoK, Gg, rho_layer2 : same as scalar version
    forcing : Forcing or list[Forcing]

    Returns
    -------
    B : (8N, 8N) complex
    B2 : (8N,) complex
    """
    N = len(n_s)
    N3 = 3 * N
    N6 = 6 * N
    N8 = 8 * N

    B = np.zeros((N8, N8), dtype=np.complex128)
    B2 = np.zeros(N8, dtype=np.complex128)

    for k in range(N):
        n = int(n_s[k])
        mk = int(m_s[k])

        # State indices for mode k in coupled ordering
        Uk = 3 * k
        Vk = 3 * k + 1
        Wk = 3 * k + 2
        Rk = N3 + 3 * k
        Sk = N3 + 3 * k + 1
        Tk = N3 + 3 * k + 2
        Phik = N6 + 2 * k
        dPhik = N6 + 2 * k + 1

        # BC row indices
        cmb_base = 4 * k
        surf_base = 4 * N + 4 * k

        # ---- CMB boundary conditions ----

        # BC1: inviscid core
        if n == -3:
            B[cmb_base, :] = Y_cmb[Rk, :]
        else:
            B[cmb_base, :] = (Y_cmb[Uk, :]
                              - Y_cmb[Rk, :] / (gc * rho2)
                              + Y_cmb[Phik, :] / gc)

        # BC2: S = 0
        B[cmb_base + 1, :] = Y_cmb[Sk, :]

        # BC3: toroidal
        if n == 1:
            B[cmb_base + 2, :] = Y_cmb[Wk, :]
        else:
            B[cmb_base + 2, :] = Y_cmb[Tk, :]

        # BC4: potential continuity at CMB
        if n == 0:
            B[cmb_base + 3, :] = Y_surf[dPhik, :]
        else:
            fac = 4.0 * math.pi * Gg / (gc * rho2) * (rho_layer2 - rho2)
            B[cmb_base + 3, :] = (
                -fac * Y_cmb[Rk, :]
                + (n / Rc + fac * rho2) * Y_cmb[Phik, :]
                - Y_cmb[dPhik, :]
            )

        # ---- Surface boundary conditions ----

        # BC5: R = 0
        B[surf_base, :] = Y_surf[Rk, :]

        # BC6
        if n == 0:
            B[surf_base + 1, :] = Y_surf[Vk, :]
        else:
            B[surf_base + 1, :] = Y_surf[Sk, :]

        # BC7
        if n == 0:
            B[surf_base + 2, :] = Y_surf[Wk, :]
        else:
            B[surf_base + 2, :] = Y_surf[Tk, :]

        # BC8: surface potential condition
        if n < 2:
            B[surf_base + 3, :] = Y_surf[Phik, :]
        else:
            B[surf_base + 3, :] = (
                4.0 * math.pi * Gg * rhoK * Y_surf[Uk, :]
                + (n + 1) * Y_surf[Phik, :]
                + Y_surf[dPhik, :]
            )

        # Forcing RHS
        is_forcing = False
        if isinstance(forcing, list):
            for f in forcing:
                if f.n == n and f.m == mk:
                    is_forcing = True
                    break
        else:
            if forcing.n == n and forcing.m == mk:
                is_forcing = True

        if is_forcing:
            B2[surf_base + 3] = (2 * n + 1)

    return B, B2


# ---------------------------------------------------------------------------
# Ocean case
# ---------------------------------------------------------------------------

def assemble_bc_ocean(
    Y_cmb: np.ndarray,
    Y_surf: np.ndarray,
    Y_ocean_start: np.ndarray,
    Y_ocean_end: np.ndarray,
    n: int,
    m: int,
    gc: float,
    Rc: float,
    rho2: float,
    rhoK: float,
    Gg: float,
    rho_layer2: float,
    gO: float,
    gI: float,
    rhoO: float,
    rho_below_ocean: float,
    rho_above_ocean: float,
    forcing,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble 24×24 boundary condition matrix for the ocean case.

    Parameters
    ----------
    Y_cmb : (8, 8) — fundamental matrix at CMB
    Y_surf : (8, 8) — fundamental matrix at surface
    Y_ocean_start : (8, 8) — fundamental matrix at bottom of ocean
    Y_ocean_end : (8, 8) — fundamental matrix at top of ocean
    n, m : harmonic degree, order
    gc, Rc, rho2 : CMB quantities
    rhoK : surface layer density
    Gg : normalized gravitational constant
    rho_layer2 : density of first solid layer
    gO : gravity at bottom of ocean
    gI : gravity at top of ocean
    rhoO : ocean density
    rho_below_ocean : density of layer below ocean
    rho_above_ocean : density of layer above ocean
    forcing : Forcing or list[Forcing]

    Returns
    -------
    B : (24, 24) complex
    B2 : (24,) complex
    """
    B = np.zeros((24, 24), dtype=np.complex128)
    B2 = np.zeros(24, dtype=np.complex128)
    I8 = np.eye(8, dtype=np.complex128)

    # Column blocks:  0:8 = sub-ocean,  8:16 = ocean,  16:24 = shell

    # ---- BC1-4: CMB conditions (same as no-ocean) ----
    # BC1: inviscid core
    B[0, :8] = Y_cmb[0, :] - Y_cmb[3, :] / (gc * rho2) + Y_cmb[6, :] / gc

    # BC2: S = 0
    B[1, :8] = Y_cmb[4, :]

    # BC3: toroidal
    if n == 1:
        B[2, :8] = Y_cmb[2, :]
    else:
        B[2, :8] = Y_cmb[5, :]

    # BC4: potential at CMB
    if n == 0:
        B[3, :8] = Y_ocean_start[7, :]
    else:
        fac = 4.0 * math.pi * Gg / (gc * rho2) * (rho_layer2 - rho2)
        B[3, :8] = (-fac * Y_cmb[3, :]
                    + (n / Rc + fac * rho2) * Y_cmb[6, :]
                    - Y_cmb[7, :])

    # ---- BC5-8: Surface conditions (shell constants) ----
    # BC5: R = 0
    B[4, 16:24] = Y_surf[3, :]

    # BC6
    if n == 0:
        B[5, 16:24] = Y_surf[1, :]
    else:
        B[5, 16:24] = Y_surf[4, :]

    # BC7
    if n == 0:
        B[6, 16:24] = Y_surf[2, :]
    else:
        B[6, 16:24] = Y_surf[5, :]

    # BC8
    if n < 2:
        B[7, 16:24] = Y_surf[6, :]
    else:
        B[7, 16:24] = (4.0 * math.pi * Gg * rhoK * Y_surf[0, :]
                       + (n + 1) * Y_surf[6, :]
                       + Y_surf[7, :])

    # ---- BC9-11: Mantle-ocean boundary (sub-ocean constants) ----
    # BC9: inviscid boundary at ocean floor
    B[8, :8] = (Y_ocean_start[0, :]
                - Y_ocean_start[3, :] / (gO * rhoO)
                + Y_ocean_start[6, :] / gO)

    # BC10: S=0 or V=0
    if n == 0:
        B[9, :8] = Y_ocean_start[1, :]
    else:
        B[9, :8] = Y_ocean_start[4, :]

    # BC11: T=0 or W=0
    if n == 0:
        B[10, :8] = Y_ocean_start[2, :]
    else:
        B[10, :8] = Y_ocean_start[5, :]

    # ---- BC12-13: Potential continuity at ocean floor ----
    # BC12: Φ below = Φ ocean
    B[11, :8] = Y_ocean_start[6, :]
    B[11, 8:16] = -I8[6, :]

    # BC13: dΦ/dr continuity with density jump
    if n < 0:
        B[12, :8] = Y_ocean_start[6, :]
    else:
        Delta_rho_ocean_bottom = rho_below_ocean - rhoO
        B[12, :8] = (Y_ocean_start[7, :]
                     + 4.0 * math.pi * Gg * Delta_rho_ocean_bottom
                     * Y_ocean_start[0, :])
        B[12, 8:16] = -I8[7, :]

    # ---- BC14-19: Ocean variables undefined (U,V,W,R,S,T = 0 in ocean) ----
    for offset, row_idx in enumerate(range(13, 19)):
        B[row_idx, 8:16] = I8[offset, :]

    # ---- BC20-22: Ocean-shell boundary (shell constants) ----
    # BC20: inviscid boundary at ocean ceiling
    B[19, 16:24] = I8[0, :] - I8[3, :] / (gI * rhoO) + I8[6, :] / gI

    # BC21: S = 0
    B[20, 16:24] = I8[4, :]

    # BC22: toroidal
    if n == 1:
        B[21, 16:24] = I8[2, :]
    else:
        B[21, 16:24] = I8[5, :]

    # ---- BC23-24: Potential continuity at ocean ceiling ----
    # BC23: Φ shell = Φ ocean
    B[22, 16:24] = I8[6, :]
    B[22, 8:16] = -Y_ocean_end[6, :]

    # BC24: dΦ/dr continuity with density jump
    if n < 0:
        B[23, 16:24] = I8[7, :]
    else:
        Delta_rho_ocean_top = rho_above_ocean - rhoO
        B[23, 16:24] = I8[7, :] + 4.0 * math.pi * Gg * Delta_rho_ocean_top * I8[0, :]
        B[23, 8:16] = -Y_ocean_end[7, :]

    # ---- Forcing RHS ----
    if isinstance(forcing, list):
        for f in forcing:
            if f.n == n and f.m == m:
                B2[7] = (2 * n + 1)
                break
    else:
        if forcing.n == n and forcing.m == m:
            B2[7] = (2 * n + 1)

    return B, B2
