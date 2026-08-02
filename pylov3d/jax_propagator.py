"""JAX 1D propagator — JIT-compiled forward propagation for the spherically
symmetric (uncoupled) Love-number problem.

This module re-implements the hot path from ``propagator.build_aprop`` and
``solver.get_solution`` (1D path only) using JAX / XLA so that the entire
radial integration loop can be JIT-compiled, and later vmap-ped over batches
of interior models or tidal frequencies.

Current limitation: ``n=2`` is hard-wired (the standard tidal forcing degree).
This avoids the ``if n <= 0`` branches in the A-matrix builders.  Generalising
to arbitrary ``n`` via ``jax.lax.cond`` is left for a later increment.

The boundary-condition assembly (``assemble_bc_no_ocean``) is intentionally
left in NumPy since it runs once per evaluation and its contribution to total
runtime is negligible.

Usage example::

    from pylov3d.types import make_interior_model, make_forcing, make_numerics
    from pylov3d.jax_propagator import jax_get_love_k2

    model  = make_interior_model(...)
    forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=2, method='variable', Nrbase=500)
    k2 = jax_get_love_k2(model, forcing, numerics)
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
    build_A3, build_A4, build_A5, build_others,
)


# ---------------------------------------------------------------------------
# Static n=2 sub-matrix building
# ---------------------------------------------------------------------------

def _precompute_static_n2(n: int, rho: float, Gg: float):
    """Precompute all static sub-matrices for a given (n, rho, Gg).

    These sub-matrices depend only on ``n`` (integer) and the mean material
    properties ``rho`` and ``Gg``, not on radius or moduli.  We compute them
    once and embed them as constant JAX arrays inside the JIT-compiled function.

    Returns a dict of JAX arrays.
    """
    A3 = build_A3(n)
    A3_inv = np.linalg.inv(A3)
    A4 = build_A4(n)
    A5 = build_A5(n)
    A13, A6, A71, A72, A81, A82, A9, A100, A101, A102, A11, A12 = \
        build_others(n, rho, Gg)

    return {
        "A3_inv": jnp.array(A3_inv, dtype=jnp.complex128),
        "A4": jnp.array(A4, dtype=jnp.complex128),
        "A5": jnp.array(A5, dtype=jnp.complex128),
        "A13": jnp.array(A13, dtype=jnp.complex128),
        "A71": jnp.array(A71, dtype=jnp.complex128),
        "A72": jnp.array(A72, dtype=jnp.complex128),
        "A81": jnp.array(A81, dtype=jnp.complex128),
        "A82": jnp.array(A82, dtype=jnp.complex128),
        "A9": jnp.array(A9, dtype=jnp.complex128),
        "A100": jnp.array(A100, dtype=jnp.complex128),
        "A101": jnp.array(A101, dtype=jnp.complex128),
        "A102": jnp.array(A102, dtype=jnp.complex128),
        "A11": jnp.array(A11, dtype=jnp.complex128),
        "A12": jnp.array(A12, dtype=jnp.complex128),
    }


# ---------------------------------------------------------------------------
# Pure-JAX build_aprop for use inside lax.scan
# ---------------------------------------------------------------------------

def _build_aprop_pure_jax(r, g, dg, muC, lam, A4, A5, A13,
                           A71, A72, A81, A82, A9, A100, A101, A102,
                           A11, A12, A3_inv, n: int):
    """Build 8x8 propagator entirely in JAX, all-traced.

    Separates the sub-matrix assembly from the static precomputation so that
    the function can be used inside ``lax.scan`` after A-matrix constants have
    been captured from the closure.

    The A1/A2 building uses Python-level arithmetic on JAX scalars so that
    JIT can trace through it at compile time.  muC and lam are complex scalars.
    """
    # ---- A1, A2 from muC, lam (traces through complex arithmetic) ----
    n_int = int(n)
    s2n1 = math.sqrt(2 * n_int + 1)
    sn   = math.sqrt(n_int)
    sn1  = math.sqrt(n_int + 1)

    # Precomputed geometric scalars (static at trace time)
    fac_base  = 1.0 / (math.sqrt(3) * s2n1)
    fac2_base = math.sqrt((n_int - 1) / (2 * n_int - 1))
    fac3_base = math.sqrt((n_int - 1) / (2 * n_int + 1)) / math.sqrt(2)
    f4a_base  = math.sqrt((2*n_int+3)*(2*n_int+2) / (12*(2*n_int-1)*(2*n_int+1)))
    f4b_base  = math.sqrt(n_int*(2*n_int-1)*(n_int+1) / (3*(2*n_int+3)*(2*n_int+2)*(2*n_int+1)))
    fac5_base = math.sqrt((n_int + 2) / (2 * n_int + 1)) / math.sqrt(2)
    fac6_base = math.sqrt((n_int + 2) / (2 * n_int + 3))

    # muC, lam are complex JAX scalars; multiply by static floats
    fac    = (3 * lam + 2 * muC) * fac_base
    fac2   = 2 * muC * fac2_base
    fac3   = 2 * muC * fac3_base
    f4a    = 2 * muC * f4a_base
    f4b    = 2 * muC * f4b_base
    fac5   = 2 * muC * fac5_base
    fac6   = 2 * muC * fac6_base

    # Build A1 and A2 as (6,3) complex arrays using scatter (.at[].set)
    A1 = jnp.zeros((6, 3), dtype=jnp.complex128)
    A2 = jnp.zeros((6, 3), dtype=jnp.complex128)

    # Row 0
    A1 = A1.at[0, 0].set(fac * sn * (n_int - 1))
    A1 = A1.at[0, 2].set(fac * sn1 * (n_int + 2))
    A2 = A2.at[0, 0].set(-fac * sn)
    A2 = A2.at[0, 2].set(fac * sn1)
    # Row 1
    A1 = A1.at[1, 0].set(fac2 * n_int)
    A2 = A2.at[1, 0].set(fac2)
    # Row 2
    A1 = A1.at[2, 1].set(fac3 * (n_int + 1))
    A2 = A2.at[2, 1].set(fac3)
    # Row 3
    A1 = A1.at[3, 0].set(f4a * (n_int - 1))
    A1 = A1.at[3, 2].set(f4b * (n_int + 2))
    A2 = A2.at[3, 0].set(-f4a)
    A2 = A2.at[3, 2].set(f4b)
    # Row 4
    A1 = A1.at[4, 1].set(fac5 * n_int)
    A2 = A2.at[4, 1].set(-fac5)
    # Row 5
    A1 = A1.at[5, 2].set(fac6 * (n_int + 1))
    A2 = A2.at[5, 2].set(-fac6)

    # ---- Assemble Adotx, Ax ----
    Adotx = jnp.zeros((8, 8), dtype=jnp.complex128)
    Ax    = jnp.zeros((8, 8), dtype=jnp.complex128)

    # Block 1 (rows 0:3): A2 -> Adotx, A1 -> Ax  (role swap)
    Adotx = Adotx.at[:3, :3].set(A4 @ A2 @ A3_inv)
    Ax    = Ax   .at[:3, :3].set(-A4 @ A1 @ A3_inv / r)
    Ax    = Ax   .at[:3, 3:6].set(A13)

    # Block 2 (rows 3:6): momentum
    Adotx = Adotx.at[3:6, :3].set(-A5 @ A2 @ A3_inv / r)
    Adotx = Adotx.at[3:6, 3:6].set(A13)
    Ax    = Ax   .at[3:6, :3].set(
        A5 @ A1 @ A3_inv / r**2 + (g / r) * A71 + dg * A72
    )
    Ax    = Ax   .at[3:6, 6:8].set(A81 + A82 / r)

    # Block 3 (rows 6:8): Poisson
    Adotx = Adotx.at[6:8, :3].set(-A12)
    Adotx = Adotx.at[6:8, 6:8].set(A9)
    Ax    = Ax   .at[6:8, :3].set(A11 / r)
    Ax    = Ax   .at[6:8, 6:8].set(A100 + A101 / r + A102 / r**2)

    return jnp.linalg.solve(Adotx, Ax)


# ---------------------------------------------------------------------------
# Cash-Karp RK5 in JAX
# ---------------------------------------------------------------------------

def _make_cash_karp_jax(static_matrices: dict, n: int, Gg: float):
    """Return a JAX function that performs one RK5 Cash-Karp step.

    The static sub-matrices (A4, A5, A13, etc.) are captured in the closure.
    The per-layer density enters through the gravity computation.

    Returns
    -------
    cash_karp_jax : callable(r_start, dr, muC, lam, rho, M_inner, R_inner)
                    -> (inc, Aprop_at_r)
      where ``inc`` and ``Aprop_at_r`` are (8,8) complex128 JAX arrays.
    """
    A4    = static_matrices["A4"]
    A5    = static_matrices["A5"]
    A13   = static_matrices["A13"]
    A71   = static_matrices["A71"]
    A72   = static_matrices["A72"]
    A81   = static_matrices["A81"]
    A82   = static_matrices["A82"]
    A9    = static_matrices["A9"]
    A100  = static_matrices["A100"]
    A101  = static_matrices["A101"]
    A102  = static_matrices["A102"]
    A11   = static_matrices["A11"]
    A12   = static_matrices["A12"]
    A3_inv = static_matrices["A3_inv"]

    def _aprop(r, muC, lam, rho, M_inner, R_inner):
        """Build propagator at radius r, computing gravity internally."""
        M_r = M_inner + (4.0 / 3.0) * math.pi * rho * (r**3 - R_inner**3)
        g  = Gg * M_r / r**2
        dg = Gg * (2.0 * (4.0/3.0 * math.pi * rho * R_inner**3 - M_inner) / r**3
                   + 4.0/3.0 * math.pi * rho)
        return _build_aprop_pure_jax(
            r, g, dg, muC, lam,
            A4, A5, A13, A71, A72, A81, A82, A9,
            A100, A101, A102, A11, A12, A3_inv, n,
        )

    # Cash-Karp coefficients as Python scalars (static at trace time)
    ck_a = CK_A
    ck_b = CK_B
    ck_c = CK_C

    I8 = jnp.eye(8, dtype=jnp.complex128)

    def cash_karp_jax(r_start, dr, muC, lam, rho, M_inner, R_inner):
        # Stage 1 (CK_A[0] = 0)
        K1 = dr * _aprop(r_start, muC, lam, rho, M_inner, R_inner)
        # Stage 2
        K2 = dr * _aprop(r_start + ck_a[1] * dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[1][0] * K1)
        # Stage 3
        K3 = dr * _aprop(r_start + ck_a[2] * dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[2][0] * K1 + ck_b[2][1] * K2)
        # Stage 4
        K4 = dr * _aprop(r_start + ck_a[3] * dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[3][0] * K1 + ck_b[3][1] * K2 + ck_b[3][2] * K3)
        # Stage 5
        K5 = dr * _aprop(r_start + ck_a[4] * dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[4][0] * K1 + ck_b[4][1] * K2
                  + ck_b[4][2] * K3 + ck_b[4][3] * K4)
        # Stage 6
        K6 = dr * _aprop(r_start + ck_a[5] * dr, muC, lam, rho, M_inner, R_inner) \
             @ (I8 + ck_b[5][0] * K1 + ck_b[5][1] * K2
                  + ck_b[5][2] * K3 + ck_b[5][3] * K4 + ck_b[5][4] * K5)

        inc = (ck_c[0] * K1 + ck_c[2] * K3 + ck_c[3] * K4 + ck_c[5] * K6)
        return inc, K1 / dr   # Aprop_at_r = K1 / dr (K1 = dr * Aprop_stage1)

    return cash_karp_jax


# ---------------------------------------------------------------------------
# Full 1D forward propagation
# ---------------------------------------------------------------------------

def propagate_1d_jax(model, forcing, numerics):
    """JAX-based 1D radial integration from CMB to surface.

    Only the ``n=2`` forcing degree is supported (hard-wired).

    Parameters
    ----------
    model : InteriorModel (normalized, output of get_rheology)
    forcing : Forcing
    numerics : NumericsConfig (output of set_boundary_indices)

    Returns
    -------
    Y_all : (Nr+1, 8, 8) complex128 numpy array — fundamental matrix
    r_grid : (Nr+1,) float64 numpy array
    """
    n_layers = model.n_layers
    Nr = numerics.Nr
    Gg = model.Gg

    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_deg = f0.n
    if n_deg != 2:
        raise NotImplementedError(
            f"jax_propagator: only n=2 is supported; got n={n_deg}."
        )
    if any(int(model.ocean[i]) == 1 for i in range(n_layers)):
        raise NotImplementedError(
            "jax_propagator: ocean layers are not supported; use the NumPy "
            "solver (get_solution) for ocean-bearing models."
        )

    # ----- Build radial grid (NumPy; one-time overhead) -----
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

    # Per-layer static matrices (one set per unique rho; for single-layer models
    # this is just layer 1).  We reuse by layer index.
    cash_karp_by_layer = {}
    for i_layer in range(1, n_layers):
        rho_k = float(model.rho[i_layer])
        if i_layer not in cash_karp_by_layer:
            static = _precompute_static_n2(n_deg, rho_k, Gg)
            ck_fn = _make_cash_karp_jax(static, n_deg, Gg)
            cash_karp_by_layer[i_layer] = ck_fn

    # ----- Forward propagation (Python loop; JIT-able per step) -----
    Y = np.zeros((Nr + 1, 8, 8), dtype=np.complex128)
    Y[0] = np.eye(8, dtype=np.complex128)
    I8 = jnp.eye(8, dtype=jnp.complex128)

    prev_layer = 1
    for k_idx in range(1, Nr + 1):
        i_layer = layer_map[k_idx]
        r_curr = r_grid[k_idx]
        r_prev = r_grid[k_idx - 1]
        dr = r_curr - r_prev

        muC_k = complex(model.muC[i_layer])
        lam_k = complex(model.lam[i_layer])
        rho_k = float(model.rho[i_layer])
        R_inner_k = float(model.R[i_layer - 1])
        M_inner_k = M_at_boundary[i_layer]

        ck_fn = cash_karp_by_layer[i_layer]
        inc, _ = ck_fn(
            float(r_prev), float(dr),
            complex(muC_k), complex(lam_k),
            float(rho_k), float(M_inner_k), float(R_inner_k),
        )
        # Convert to numpy for the update (avoids accumulating a jnp chain)
        inc_np = np.array(inc)

        Y_old = Y[k_idx - 1].copy()
        if i_layer != prev_layer:
            # Density discontinuity correction
            Delta_rho_k = float(model.Delta_rho[i_layer])
            Y_old[7, :] += 4.0 * math.pi * Gg * Delta_rho_k * Y_old[0, :]

        Y[k_idx] = (np.eye(8, dtype=np.complex128) + inc_np) @ Y_old
        prev_layer = i_layer

    return Y, r_grid


def jax_get_love_k2(model, forcing, numerics):
    """Compute k2 for a 1D model using the JAX propagator.

    Calls ``propagate_1d_jax`` for the radial integration and then uses the
    NumPy boundary-condition assembly to extract the Love number.

    Returns
    -------
    k2 : complex — gravity Love number k₂ = Φ_surf − 1
    """
    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_deg = f0.n
    m_ord = f0.m

    Y_all, r_grid = propagate_1d_jax(model, forcing, numerics)

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

    # Physical solution at surface
    y_surf = Y_surf @ C
    k2 = complex(y_surf[6]) - 1.0
    return k2
