"""JAX lax.scan radial propagation — JIT-compiled version.

Extends jax_propagator.py with a ``jax.lax.scan`` loop so the full radial
integration is XLA-fused into a single kernel rather than driven by a Python
``for`` loop.

Only ``n=2`` forcing degree is supported (same restriction as jax_propagator).

Public API
----------
propagate_1d_jax_scan(model, forcing, numerics)
    Drop-in replacement for ``propagate_1d_jax`` using lax.scan.
jax_get_love_k2_scan(model, forcing, numerics)
    Love-number wrapper using the scan propagation.
"""

from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .rheology import get_rheology
from .grid import set_boundary_indices
from .boundary_conditions import assemble_bc_no_ocean
from .propagator import (
    CK_A, CK_B, CK_C,
    build_A3, build_A4, build_A5,
)
from .jax_propagator import _build_aprop_pure_jax


# ---------------------------------------------------------------------------
# Precompute rho-independent sub-matrices for n=2
# ---------------------------------------------------------------------------

def _precompute_n2_rho_independent(n: int) -> dict:
    """Return static JAX matrices that depend only on ``n`` (not on rho).

    Used by the scan propagator to keep rho-dependent entries out of the
    static closure and instead handle them as traced scalars inside the loop.
    """
    A3 = build_A3(n)
    A3_inv = np.linalg.inv(A3)
    A4 = build_A4(n)
    A5 = build_A5(n)
    A13 = np.eye(3, dtype=np.complex128)

    nn = n * (n + 1)
    A9 = np.eye(2, dtype=np.complex128)
    A100 = np.zeros((2, 2), dtype=np.complex128)
    A101 = np.zeros((2, 2), dtype=np.complex128)
    A102 = np.zeros((2, 2), dtype=np.complex128)
    A100[0, 1] = 1.0
    A101[1, 1] = -2.0
    A102[1, 0] = float(nn)

    return {
        "A3_inv": jnp.array(A3_inv, dtype=jnp.complex128),
        "A4":     jnp.array(A4,     dtype=jnp.complex128),
        "A5":     jnp.array(A5,     dtype=jnp.complex128),
        "A13":    jnp.array(A13,    dtype=jnp.complex128),
        "A9":     jnp.array(A9,     dtype=jnp.complex128),
        "A100":   jnp.array(A100,   dtype=jnp.complex128),
        "A101":   jnp.array(A101,   dtype=jnp.complex128),
        "A102":   jnp.array(A102,   dtype=jnp.complex128),
    }


# ---------------------------------------------------------------------------
# scan-compatible aprop that accepts rho as a traced scalar
# ---------------------------------------------------------------------------

def _build_aprop_rho_jax(r, g, dg, muC, lam, rho, Gg, static: dict, n: int):
    """Build 8×8 propagator with ``rho`` as a traced JAX scalar.

    The rho-dependent sub-matrices (A71, A72, A81, A82, A11, A12) are
    assembled inline so that the function can be called inside lax.scan
    where ``rho`` varies per layer.  Rho-independent matrices are taken from
    the ``static`` closure.
    """
    nn = n * (n + 1)
    pi4Gg = 4.0 * math.pi * Gg

    # Rho-dependent matrices — linear in rho, built from traced scalar
    A71 = jnp.zeros((3, 3), dtype=jnp.complex128)
    A71 = A71.at[0, 0].set(-2.0 * rho)
    A71 = A71.at[0, 1].set(rho * nn)
    A71 = A71.at[1, 0].set(rho)

    A72 = jnp.zeros((3, 3), dtype=jnp.complex128)
    A72 = A72.at[0, 0].set(rho)

    A81 = jnp.zeros((3, 2), dtype=jnp.complex128)
    A81 = A81.at[0, 1].set(rho)

    A82 = jnp.zeros((3, 2), dtype=jnp.complex128)
    A82 = A82.at[1, 0].set(rho)

    A11 = jnp.zeros((2, 3), dtype=jnp.complex128)
    A11 = A11.at[1, 0].set(-2.0 * pi4Gg * rho)
    A11 = A11.at[1, 1].set(pi4Gg * rho * nn)

    A12 = jnp.zeros((2, 3), dtype=jnp.complex128)
    A12 = A12.at[1, 0].set(-pi4Gg * rho)

    return _build_aprop_pure_jax(
        r, g, dg, muC, lam,
        static["A4"], static["A5"], static["A13"],
        A71, A72, A81, A82,
        static["A9"], static["A100"], static["A101"], static["A102"],
        A11, A12, static["A3_inv"], n,
    )


# ---------------------------------------------------------------------------
# Build JIT-compiled scan function
# ---------------------------------------------------------------------------

def _make_jit_scan(static: dict, n: int, Gg: float):
    """Return a JIT-compiled function that runs the full radial scan.

    Parameters
    ----------
    static : dict  — rho-independent matrices from _precompute_n2_rho_independent
    n : int        — forcing degree (must be 2)
    Gg : float     — normalized gravitational constant

    Returns
    -------
    run_scan : callable(Y_init, xs_tuple) -> (Y_final, Y_all)
      Y_all has shape (Nr, 8, 8) — intermediate solutions, one per step.
    """
    ck_a = CK_A
    ck_b = CK_B
    ck_c = CK_C
    I8 = jnp.eye(8, dtype=jnp.complex128)

    def _ap(r, muC, lam, rho, M_inner, R_inner):
        """Build propagator at radius r given layer material and geometry."""
        M_r = M_inner + (4.0 / 3.0) * math.pi * rho * (r**3 - R_inner**3)
        g  = Gg * M_r / r**2
        dg = Gg * (
            2.0 * (4.0 / 3.0 * math.pi * rho * R_inner**3 - M_inner) / r**3
            + 4.0 / 3.0 * math.pi * rho
        )
        return _build_aprop_rho_jax(r, g, dg, muC, lam, rho, Gg, static, n)

    def _cash_karp(r_start, dr, muC, lam, rho, M_inner, R_inner):
        """One Cash-Karp RK5 step returning the 8×8 increment matrix."""
        K1 = dr * _ap(r_start, muC, lam, rho, M_inner, R_inner)
        K2 = dr * _ap(r_start + ck_a[1]*dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[1][0]*K1)
        K3 = dr * _ap(r_start + ck_a[2]*dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[2][0]*K1 + ck_b[2][1]*K2)
        K4 = dr * _ap(r_start + ck_a[3]*dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[3][0]*K1 + ck_b[3][1]*K2 + ck_b[3][2]*K3)
        K5 = dr * _ap(r_start + ck_a[4]*dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[4][0]*K1 + ck_b[4][1]*K2
                   + ck_b[4][2]*K3 + ck_b[4][3]*K4)
        K6 = dr * _ap(r_start + ck_a[5]*dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[5][0]*K1 + ck_b[5][1]*K2
                   + ck_b[5][2]*K3 + ck_b[5][3]*K4 + ck_b[5][4]*K5)
        return ck_c[0]*K1 + ck_c[2]*K3 + ck_c[3]*K4 + ck_c[5]*K6

    def step(Y_prev, xs):
        """Single scan step: density correction + RK5 propagation."""
        r_prev, r_curr, muC_k, lam_k, rho_k, M_inner_k, R_inner_k, drho_k = xs
        dr = r_curr - r_prev

        # Density-discontinuity correction (nonzero only at layer boundaries)
        Y_corr = Y_prev.at[7, :].add(
            4.0 * jnp.pi * Gg * drho_k * Y_prev[0, :]
        )

        inc = _cash_karp(r_prev, dr, muC_k, lam_k, rho_k, M_inner_k, R_inner_k)
        Y_new = (I8 + inc) @ Y_corr
        return Y_new, Y_new

    @jax.jit
    def run_scan(Y_init, xs_tuple):
        """JIT-compiled scan over all Nr radial steps."""
        Y_final, Y_all = jax.lax.scan(step, Y_init, xs_tuple)
        return Y_final, Y_all

    return run_scan


# ---------------------------------------------------------------------------
# Public API: scan-based propagation
# ---------------------------------------------------------------------------

def propagate_1d_jax_scan(model, forcing, numerics):
    """JIT-compiled radial integration using jax.lax.scan.

    Identical semantics to ``propagate_1d_jax`` (same math, same boundary
    corrections), but the radial loop is expressed as a single lax.scan so
    XLA fuses it into one compiled kernel.

    Parameters
    ----------
    model : InteriorModel (normalized, output of get_rheology)
    forcing : Forcing or list[Forcing]
    numerics : NumericsConfig (output of set_boundary_indices)

    Returns
    -------
    Y_all : (Nr+1, 8, 8) complex128 numpy array
    r_grid : (Nr+1,) float64 numpy array
    """
    n_layers = model.n_layers
    Nr = numerics.Nr
    Gg = model.Gg

    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_deg = f0.n
    if n_deg != 2:
        raise NotImplementedError(
            f"jax_scan: only n=2 is supported; got n={n_deg}."
        )
    if any(int(model.ocean[i]) == 1 for i in range(n_layers)):
        raise NotImplementedError(
            "jax_scan: ocean layers are not supported; use the NumPy "
            "solver (get_solution) for ocean-bearing models."
        )

    # ----- Radial grid -----
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

    # ----- Enclosed mass at inner boundary of each layer -----
    M_at_boundary = np.zeros(n_layers)
    M_at_boundary[0] = (4.0 / 3.0) * math.pi * float(model.rho[0]) * Rc ** 3
    for i in range(1, n_layers):
        R_prev = float(model.R[i - 1])
        if i == 1:
            M_at_boundary[i] = M_at_boundary[0]
        else:
            R_before = float(model.R[i - 2])
            M_at_boundary[i] = M_at_boundary[i - 1] + \
                (4.0 / 3.0) * math.pi * float(model.rho[i - 1]) * \
                (R_prev ** 3 - R_before ** 3)

    # ----- Per-point input arrays (shape (Nr,)) -----
    r_prev_arr    = np.zeros(Nr, dtype=np.float64)
    r_curr_arr    = np.zeros(Nr, dtype=np.float64)
    muC_arr       = np.zeros(Nr, dtype=np.complex128)
    lam_arr       = np.zeros(Nr, dtype=np.complex128)
    rho_arr       = np.zeros(Nr, dtype=np.float64)
    M_inner_arr   = np.zeros(Nr, dtype=np.float64)
    R_inner_arr   = np.zeros(Nr, dtype=np.float64)
    delta_rho_arr = np.zeros(Nr, dtype=np.float64)

    # delta_rho_arr: nonzero only at the first grid point of each new layer.
    # The Python loop initializes prev_layer=1, so no correction fires at k=1
    # (the first mantle point) even though layer_map[0]=0 != layer_map[1]=1.
    # We replicate this exactly: at xs index 0 (k_idx=1) use prev_layer=1;
    # at xs index k>0 (k_idx=k+1) use layer_map[k_idx-1].
    for k_idx in range(1, Nr + 1):
        xs_k = k_idx - 1  # 0-indexed xs slot
        i_layer = layer_map[k_idx]
        prev_layer_k = 1 if k_idx == 1 else layer_map[k_idx - 1]

        r_prev_arr[xs_k]  = r_grid[k_idx - 1]
        r_curr_arr[xs_k]  = r_grid[k_idx]
        muC_arr[xs_k]     = complex(model.muC[i_layer])
        lam_arr[xs_k]     = complex(model.lam[i_layer])
        rho_arr[xs_k]     = float(model.rho[i_layer])
        M_inner_arr[xs_k] = M_at_boundary[i_layer]
        R_inner_arr[xs_k] = float(model.R[i_layer - 1])

        if i_layer != prev_layer_k:
            delta_rho_arr[xs_k] = float(model.Delta_rho[i_layer])

    # ----- Build JIT-compiled scan -----
    static = _precompute_n2_rho_independent(n_deg)
    run_scan = _make_jit_scan(static, n_deg, Gg)

    Y_init = jnp.eye(8, dtype=jnp.complex128)
    xs_tuple = (
        jnp.array(r_prev_arr),
        jnp.array(r_curr_arr),
        jnp.array(muC_arr),
        jnp.array(lam_arr),
        jnp.array(rho_arr),
        jnp.array(M_inner_arr),
        jnp.array(R_inner_arr),
        jnp.array(delta_rho_arr),
    )

    _, Y_steps = run_scan(Y_init, xs_tuple)
    jax.block_until_ready(Y_steps)

    # Prepend the identity initial matrix to get shape (Nr+1, 8, 8)
    Y_all_jax = jnp.concatenate(
        [jnp.eye(8, dtype=jnp.complex128)[None, :, :], Y_steps], axis=0
    )
    return np.array(Y_all_jax), r_grid


def jax_get_love_k2_scan(model, forcing, numerics):
    """Compute k2 for a 1D model using the scan-based JAX propagator.

    Identical to ``jax_get_love_k2`` but uses the lax.scan integration path.

    Returns
    -------
    k2 : complex — gravity Love number k₂ = Φ_surf − 1
    """
    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_deg = f0.n
    m_ord = f0.m

    Y_all, r_grid = propagate_1d_jax_scan(model, forcing, numerics)

    Nr = len(r_grid) - 1
    Y_cmb  = Y_all[0]
    Y_surf = Y_all[Nr]

    n_layers = model.n_layers
    Gg = model.Gg
    Rc = float(model.R[0])
    rhoC = float(model.rho[0])
    Mc = (4.0 / 3.0) * math.pi * rhoC * Rc ** 3
    gc = Gg * Mc / Rc ** 2 if Rc > 0 else 0.0

    rho2 = float(model.Delta_rho[0]) + float(model.rho[1])
    rhoK_surface = float(model.rho[n_layers - 1])

    B, B2 = assemble_bc_no_ocean(
        Y_cmb, Y_surf, n_deg, m_ord,
        gc, Rc, rho2, rhoK_surface, Gg,
        float(model.rho[1]), forcing,
    )
    C = np.linalg.solve(B, B2)

    y_surf = Y_surf @ C
    k2 = complex(y_surf[6]) - 1.0
    return k2
