"""Radial integration via Cash-Karp RK5 — translated from get_solution.m.

Integrates the 8×8 fundamental matrix ODE ``dY/dr = Aprop(r) · Y`` from the
core-mantle boundary (CMB) outward to the surface using the propagator
matrix method.  Boundary conditions are applied after integration.

For Milestone 1 (1D, single mode, no lateral variations) the system is 8×8
with state vector ``[U, V, W, R, S, T, Φ, dΦ/dr]``.
"""

from __future__ import annotations

import math

import numpy as np

from .propagator import build_aprop, build_aprop_coupled, compute_gravity, CK_A, CK_B, CK_C
from .boundary_conditions import (
    assemble_bc_no_ocean,
    assemble_bc_ocean,
    assemble_bc_no_ocean_coupled,
)


# ---------------------------------------------------------------------------
# Cash-Karp RK5 increment
# ---------------------------------------------------------------------------

def cash_karp_increment(
    r_start: float,
    dr: float,
    n: int,
    muC: complex,
    lam: complex,
    rho: float,
    Gg: float,
    M_inner: float,
    R_inner: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one Cash-Karp RK5 matrix increment.

    Returns
    -------
    inc : (8, 8) complex
        Increment matrix such that ``Y(r+dr) ≈ (I + inc) @ Y(r)``.
    Aprop_at_r : (8, 8) complex
        Propagator matrix evaluated at *r_start* (for auxiliary recovery).
    """
    I8 = np.eye(8, dtype=np.complex128)

    def _aprop_at(r):
        g, dg = compute_gravity(r, rho, M_inner, R_inner, Gg)
        return build_aprop(r, g, dg, n, muC, lam, rho, Gg)

    # Stage 1
    A1 = _aprop_at(r_start + CK_A[0] * dr)  # CK_A[0] = 0
    K1 = dr * A1

    # Stage 2
    A2 = _aprop_at(r_start + CK_A[1] * dr)
    K2 = dr * A2 @ (I8 + CK_B[1][0] * K1)

    # Stage 3
    A3 = _aprop_at(r_start + CK_A[2] * dr)
    K3 = dr * A3 @ (I8 + CK_B[2][0] * K1 + CK_B[2][1] * K2)

    # Stage 4
    A4 = _aprop_at(r_start + CK_A[3] * dr)
    K4 = dr * A4 @ (I8 + CK_B[3][0] * K1 + CK_B[3][1] * K2 + CK_B[3][2] * K3)

    # Stage 5
    A5 = _aprop_at(r_start + CK_A[4] * dr)
    K5 = dr * A5 @ (I8
                     + CK_B[4][0] * K1 + CK_B[4][1] * K2
                     + CK_B[4][2] * K3 + CK_B[4][3] * K4)

    # Stage 6
    A6 = _aprop_at(r_start + CK_A[5] * dr)
    K6 = dr * A6 @ (I8
                     + CK_B[5][0] * K1 + CK_B[5][1] * K2
                     + CK_B[5][2] * K3 + CK_B[5][3] * K4
                     + CK_B[5][4] * K5)

    # 5th-order combination (CK_C[1] = CK_C[4] = 0)
    inc = CK_C[0] * K1 + CK_C[2] * K3 + CK_C[3] * K4 + CK_C[5] * K6

    return inc, A1


def cash_karp_increment_coupled(
    r_start: float,
    dr: float,
    n_s: np.ndarray,
    muC: complex,
    lam: complex,
    rho: float,
    Gg: float,
    M_inner: float,
    R_inner: float,
    Coup: np.ndarray,
    muC_amp: np.ndarray,
    K_amp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cash-Karp RK5 increment for the coupled 8N×8N system.

    Same numerical scheme as ``cash_karp_increment`` but uses the coupled
    propagator ``build_aprop_coupled`` at each stage.

    Returns
    -------
    inc : (8N, 8N) complex — increment matrix
    Aprop_at_r : (8N, 8N) complex — propagator at *r_start*
    """
    N = len(n_s)
    N8 = 8 * N
    IN = np.eye(N8, dtype=np.complex128)

    def _aprop_at(r):
        g, dg = compute_gravity(r, rho, M_inner, R_inner, Gg)
        return build_aprop_coupled(r, g, dg, n_s, muC, lam, rho, Gg,
                                   Coup, muC_amp, K_amp)

    # Stage 1
    Ap1 = _aprop_at(r_start + CK_A[0] * dr)
    K1 = dr * Ap1

    # Stage 2
    Ap2 = _aprop_at(r_start + CK_A[1] * dr)
    K2 = dr * Ap2 @ (IN + CK_B[1][0] * K1)

    # Stage 3
    Ap3 = _aprop_at(r_start + CK_A[2] * dr)
    K3 = dr * Ap3 @ (IN + CK_B[2][0] * K1 + CK_B[2][1] * K2)

    # Stage 4
    Ap4 = _aprop_at(r_start + CK_A[3] * dr)
    K4 = dr * Ap4 @ (IN + CK_B[3][0] * K1 + CK_B[3][1] * K2
                      + CK_B[3][2] * K3)

    # Stage 5
    Ap5 = _aprop_at(r_start + CK_A[4] * dr)
    K5 = dr * Ap5 @ (IN + CK_B[4][0] * K1 + CK_B[4][1] * K2
                      + CK_B[4][2] * K3 + CK_B[4][3] * K4)

    # Stage 6
    Ap6 = _aprop_at(r_start + CK_A[5] * dr)
    K6 = dr * Ap6 @ (IN + CK_B[5][0] * K1 + CK_B[5][1] * K2
                      + CK_B[5][2] * K3 + CK_B[5][3] * K4
                      + CK_B[5][4] * K5)

    # 5th-order combination
    inc = CK_C[0] * K1 + CK_C[2] * K3 + CK_C[3] * K4 + CK_C[5] * K6

    return inc, Ap1


# ---------------------------------------------------------------------------
# Full solver
# ---------------------------------------------------------------------------

def get_solution(model, forcing, numerics, couplings=None, lateral=None):
    """Integrate from CMB to surface and apply boundary conditions.

    Parameters
    ----------
    model : InteriorModel
        Normalized model (output of ``get_rheology``).
    forcing : Forcing or list[Forcing]
        Tidal forcing.  For single-mode M1 we use only the first component.
    numerics : NumericsConfig
        Grid configuration (output of ``set_boundary_indices``).
    couplings : Couplings, optional
        Mode coupling coefficients for the laterally heterogeneous problem.
        When provided with N > 1 modes, uses the coupled 8N×8N solver.
    lateral : LateralRheology, optional
        Per-layer lateral variation amplitudes (required when couplings has N > 1).

    Returns
    -------
    y_sol : (Nr+1, 8) or (Nr+1, 8N) complex
        Physical solution at each radial point.
    r_grid : (Nr+1,) float
        Radial coordinate at each point.
    Y_all : (Nr+1, 8, 8) or (Nr+1, 8N, 8N) complex
        Fundamental matrix at each point (for energy calculations).
    Aprop_aux : (Nr+1, 3, 8) or (Nr+1, 3N, 8N) complex
        First 3(N) rows of Aprop at each point (for stress/strain recovery).
    """
    # Dispatch to coupled solver when multi-mode couplings are provided
    if couplings is not None and len(couplings.n_s) > 1:
        if lateral is None:
            raise ValueError(
                "lateral must be provided for coupled solver (N > 1 modes)"
            )
        return _get_solution_coupled(
            model, forcing, numerics, couplings, lateral,
        )


    n_layers = model.n_layers
    Nr = numerics.Nr
    Gg = model.Gg

    # Extract forcing parameters
    if isinstance(forcing, list):
        f0 = forcing[0]
    else:
        f0 = forcing
    n_deg = f0.n
    m_ord = f0.m

    # Detect ocean layer (0 = no ocean)
    ocean_layer = 0
    ocean_start = 0
    ocean_end = 0
    for i in range(n_layers):
        if int(model.ocean[i]) == 1:
            ocean_layer = i  # 0-based layer index
            ocean_start_bc = i - 1  # BCindices index for bottom of ocean
            ocean_end_bc = i        # BCindices index for top of ocean
            ocean_start = int(numerics.BCindices[ocean_start_bc - 1]) if ocean_start_bc > 0 else 0
            ocean_end = int(numerics.BCindices[ocean_end_bc - 1]) if ocean_end_bc > 0 else 0
            break

    # ----- Build radial grid -----
    r_grid = np.zeros(Nr + 1)
    layer_map = np.zeros(Nr + 1, dtype=int)  # layer index for each grid point

    Rc = float(model.R[0])  # core radius
    r_grid[0] = Rc
    layer_map[0] = 0  # CMB belongs to core

    k = 1
    for i_layer in range(1, n_layers):
        R_inner = float(model.R[i_layer - 1])
        R_outer = float(model.R[i_layer])
        npts = int(numerics.Nrlayer[i_layer])
        if npts > 0:
            dr_layer = (R_outer - R_inner) / npts
            for j in range(npts):
                r_grid[k] = R_inner + (j + 1) * dr_layer
                layer_map[k] = i_layer
                k += 1

    # ----- Precompute enclosed mass at inner boundary of each layer -----
    M_at_boundary = np.zeros(n_layers)
    M_at_boundary[0] = (4.0 / 3.0) * math.pi * float(model.rho[0]) * Rc ** 3
    for i in range(1, n_layers):
        R_prev = float(model.R[i - 1])
        R_before = float(model.R[i - 2]) if i >= 2 else 0.0
        if i == 1:
            M_at_boundary[i] = M_at_boundary[0]
        else:
            M_at_boundary[i] = M_at_boundary[i - 1] + \
                (4.0 / 3.0) * math.pi * float(model.rho[i - 1]) * \
                (R_prev ** 3 - R_before ** 3)

    # ----- Initialize fundamental matrix -----
    Y = np.zeros((Nr + 1, 8, 8), dtype=np.complex128)
    Y[0] = np.eye(8, dtype=np.complex128)

    # Auxiliary Aprop (first 3 rows, for stress/strain recovery)
    Aprop_aux = np.zeros((Nr + 1, 3, 8), dtype=np.complex128)

    # Compute Aprop at CMB for auxiliary storage
    g0, dg0 = compute_gravity(Rc, float(model.rho[1]), M_at_boundary[1],
                              float(model.R[0]), Gg)
    Ap0 = build_aprop(Rc, g0, dg0, n_deg,
                      complex(model.muC[1]), complex(model.lam[1]),
                      float(model.rho[1]), Gg)
    Aprop_aux[0] = Ap0[:3, :]

    I8 = np.eye(8, dtype=np.complex128)

    # ----- Forward propagation -----
    prev_layer = 1  # first active layer

    for k_idx in range(1, Nr + 1):
        i_layer = layer_map[k_idx]
        r_curr = r_grid[k_idx]
        r_prev = r_grid[k_idx - 1]
        dr = r_curr - r_prev

        # Material properties for this layer
        muC_k = complex(model.muC[i_layer])
        lam_k = complex(model.lam[i_layer])
        rho_k = float(model.rho[i_layer])
        R_inner_k = float(model.R[i_layer - 1])
        M_inner_k = M_at_boundary[i_layer]

        # Compute CK-RK5 increment
        inc, Ap_at_r = cash_karp_increment(
            r_prev, dr, n_deg, muC_k, lam_k, rho_k, Gg,
            M_inner_k, R_inner_k,
        )

        # Auxiliary Aprop at current point (evaluate at r_curr)
        g_aux, dg_aux = compute_gravity(r_curr, rho_k, M_inner_k, R_inner_k, Gg)
        Ap_aux = build_aprop(r_curr, g_aux, dg_aux, n_deg,
                             muC_k, lam_k, rho_k, Gg)
        Aprop_aux[k_idx] = Ap_aux[:3, :]

        # Get Y_old with possible density discontinuity correction
        Y_old = Y[k_idx - 1].copy()

        if i_layer != prev_layer:
            # Layer boundary crossed
            if ocean_layer > 0 and i_layer == ocean_layer:
                # Entering ocean: reset to identity
                Y_old = I8.copy()
            elif ocean_layer > 0 and i_layer == ocean_layer + 1:
                # Exiting ocean: reset to identity for shell
                Y_old = I8.copy()
            else:
                # Density discontinuity correction
                Delta_rho_k = float(model.Delta_rho[i_layer])
                # dphi row += 4*pi*Gg * Delta_rho * U row
                Y_old[7, :] += 4.0 * math.pi * Gg * Delta_rho_k * Y_old[0, :]

        # Propagate: Y(k) = (I + inc) @ Y_old
        Y[k_idx] = (I8 + inc) @ Y_old
        prev_layer = i_layer

    # ----- Boundary conditions -----
    # Core quantities
    rhoC = float(model.rho[0])
    Mc = (4.0 / 3.0) * math.pi * rhoC * Rc ** 3
    gc = Gg * Mc / Rc ** 2 if Rc > 0 else 0.0

    rho2 = float(model.Delta_rho[0]) + float(model.rho[1])
    rhoK_surface = float(model.rho[n_layers - 1])

    Y_cmb = Y[0]
    Y_surf = Y[Nr]

    if ocean_layer == 0:
        B, B2 = assemble_bc_no_ocean(
            Y_cmb, Y_surf, n_deg, m_ord,
            gc, Rc, rho2, rhoK_surface, Gg,
            float(model.rho[1]), forcing,
        )
        C = np.linalg.solve(B, B2)

        # Assemble physical solution
        y_sol = np.zeros((Nr + 1, 8), dtype=np.complex128)
        for k_idx in range(Nr + 1):
            y_sol[k_idx] = Y[k_idx] @ C
    else:
        # Ocean case: need Y at ocean boundaries
        Y_ocean_start = Y[ocean_start]
        Y_ocean_end = Y[ocean_end]

        gO = float(model.gs[ocean_layer - 1])
        gI = float(model.gs[ocean_layer])
        rhoO = float(model.rho[ocean_layer])
        rho_below_ocean = float(model.rho[ocean_layer - 1])
        rho_above_ocean = float(model.rho[ocean_layer + 1])

        B, B2 = assemble_bc_ocean(
            Y_cmb, Y_surf, Y_ocean_start, Y_ocean_end,
            n_deg, m_ord,
            gc, Rc, rho2, rhoK_surface, Gg,
            float(model.rho[1]),
            gO, gI, rhoO, rho_below_ocean, rho_above_ocean,
            forcing,
        )
        C = np.linalg.solve(B, B2)

        # Assemble physical solution from three segments
        y_sol = np.zeros((Nr + 1, 8), dtype=np.complex128)
        C_below = C[:8]
        C_ocean = C[8:16]
        C_shell = C[16:24]

        for k_idx in range(Nr + 1):
            il = layer_map[k_idx]
            if ocean_layer > 0 and il >= ocean_layer + 1:
                y_sol[k_idx] = Y[k_idx] @ C_shell
            elif ocean_layer > 0 and il == ocean_layer:
                y_sol[k_idx] = Y[k_idx] @ C_ocean
            else:
                y_sol[k_idx] = Y[k_idx] @ C_below

    return y_sol, r_grid, Y, Aprop_aux


# ---------------------------------------------------------------------------
# Coupled (multi-mode) solver
# ---------------------------------------------------------------------------

def _get_solution_coupled(model, forcing, numerics, couplings, lateral):
    """Integrate the coupled 8N×8N system from CMB to surface.

    Private implementation called by ``get_solution`` when ``couplings``
    has N > 1 modes.  Same Cash-Karp RK5 scheme as the 1D solver but with
    8N×8N fundamental matrices and coupled propagator.
    """
    n_layers = model.n_layers
    Nr = numerics.Nr
    Gg = model.Gg
    N = len(couplings.n_s)
    N3 = 3 * N
    N6 = 6 * N
    N8 = 8 * N

    # Ocean not supported yet in coupled solver
    for i in range(n_layers):
        if int(model.ocean[i]) == 1:
            raise NotImplementedError(
                "Ocean layers not yet supported in coupled solver"
            )

    # ----- Build radial grid -----
    r_grid = np.zeros(Nr + 1)
    layer_map = np.zeros(Nr + 1, dtype=int)

    Rc = float(model.R[0])
    r_grid[0] = Rc
    layer_map[0] = 0

    k = 1
    for i_layer in range(1, n_layers):
        R_inner = float(model.R[i_layer - 1])
        R_outer = float(model.R[i_layer])
        npts = int(numerics.Nrlayer[i_layer])
        if npts > 0:
            dr_layer = (R_outer - R_inner) / npts
            for j in range(npts):
                r_grid[k] = R_inner + (j + 1) * dr_layer
                layer_map[k] = i_layer
                k += 1

    # ----- Precompute enclosed mass at inner boundary of each layer -----
    M_at_boundary = np.zeros(n_layers)
    M_at_boundary[0] = (4.0 / 3.0) * math.pi * float(model.rho[0]) * Rc ** 3
    for i in range(1, n_layers):
        R_prev = float(model.R[i - 1])
        R_before = float(model.R[i - 2]) if i >= 2 else 0.0
        if i == 1:
            M_at_boundary[i] = M_at_boundary[0]
        else:
            M_at_boundary[i] = M_at_boundary[i - 1] + \
                (4.0 / 3.0) * math.pi * float(model.rho[i - 1]) * \
                (R_prev ** 3 - R_before ** 3)

    # ----- Initialize fundamental matrix (8N × 8N) -----
    Y = np.zeros((Nr + 1, N8, N8), dtype=np.complex128)
    Y[0] = np.eye(N8, dtype=np.complex128)

    # Auxiliary Aprop (first 3N rows, for stress/strain recovery)
    Aprop_aux = np.zeros((Nr + 1, N3, N8), dtype=np.complex128)

    # Aprop at CMB (using layer 1 properties)
    muC_amp_1 = lateral.muC_amp[1, :]
    K_amp_1 = lateral.K_amp[1, :]
    g0, dg0 = compute_gravity(
        Rc, float(model.rho[1]), M_at_boundary[1], float(model.R[0]), Gg,
    )
    Ap0 = build_aprop_coupled(
        Rc, g0, dg0, couplings.n_s,
        complex(model.muC[1]), complex(model.lam[1]),
        float(model.rho[1]), Gg,
        couplings.Coup, muC_amp_1, K_amp_1,
    )
    Aprop_aux[0] = Ap0[:N3, :]

    IN8 = np.eye(N8, dtype=np.complex128)

    # ----- Forward propagation -----
    prev_layer = 1

    for k_idx in range(1, Nr + 1):
        i_layer = layer_map[k_idx]
        r_curr = r_grid[k_idx]
        r_prev = r_grid[k_idx - 1]
        dr = r_curr - r_prev

        # Material properties for this layer
        muC_k = complex(model.muC[i_layer])
        lam_k = complex(model.lam[i_layer])
        rho_k = float(model.rho[i_layer])
        R_inner_k = float(model.R[i_layer - 1])
        M_inner_k = M_at_boundary[i_layer]

        # Per-layer lateral variation amplitudes
        muC_amp_k = lateral.muC_amp[i_layer, :]
        K_amp_k = lateral.K_amp[i_layer, :]

        # Coupled CK-RK5 increment
        inc, _ = cash_karp_increment_coupled(
            r_prev, dr, couplings.n_s, muC_k, lam_k, rho_k, Gg,
            M_inner_k, R_inner_k,
            couplings.Coup, muC_amp_k, K_amp_k,
        )

        # Auxiliary Aprop at current point
        g_aux, dg_aux = compute_gravity(
            r_curr, rho_k, M_inner_k, R_inner_k, Gg,
        )
        Ap_aux = build_aprop_coupled(
            r_curr, g_aux, dg_aux, couplings.n_s,
            muC_k, lam_k, rho_k, Gg,
            couplings.Coup, muC_amp_k, K_amp_k,
        )
        Aprop_aux[k_idx] = Ap_aux[:N3, :]

        # Density discontinuity correction at layer boundaries
        Y_old = Y[k_idx - 1].copy()
        if i_layer != prev_layer:
            Delta_rho_k = float(model.Delta_rho[i_layer])
            for m_idx in range(N):
                U_row = 3 * m_idx
                dPhi_row = N6 + 2 * m_idx + 1
                Y_old[dPhi_row, :] += (
                    4.0 * math.pi * Gg * Delta_rho_k * Y_old[U_row, :]
                )

        # Propagate: Y(k) = (I + inc) @ Y_old
        Y[k_idx] = (IN8 + inc) @ Y_old
        prev_layer = i_layer

    # ----- Boundary conditions -----
    rhoC = float(model.rho[0])
    Mc = (4.0 / 3.0) * math.pi * rhoC * Rc ** 3
    gc = Gg * Mc / Rc ** 2 if Rc > 0 else 0.0

    rho2 = float(model.Delta_rho[0]) + float(model.rho[1])
    rhoK_surface = float(model.rho[n_layers - 1])

    B, B2 = assemble_bc_no_ocean_coupled(
        Y[0], Y[Nr], couplings.n_s, couplings.m_s,
        gc, Rc, rho2, rhoK_surface, Gg,
        float(model.rho[1]), forcing,
    )
    C = np.linalg.solve(B, B2)

    # Assemble physical solution
    y_sol = np.zeros((Nr + 1, N8), dtype=np.complex128)
    for k_idx in range(Nr + 1):
        y_sol[k_idx] = Y[k_idx] @ C

    return y_sol, r_grid, Y, Aprop_aux
