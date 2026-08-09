# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""JAX port of the coupled ocean-layer propagator (TASK-009).

Three-segment ``lax.scan`` integration for coupled 8N x 8N systems whose
model has a subsurface ocean: a solid below-ocean segment, an in-ocean
Laplace-only potential segment, and a solid shell segment.  This mirrors
the ground-truth NumPy path in ``solver._get_solution_coupled`` (ocean
branch) and the in-ocean propagator in ``ocean_coupled.build_aprop_ocean_coupled``
to ~1e-12 relative.

The elastic segments (below-ocean, shell) reuse the existing solid coupled
scan (``jax_coupled._make_jit_scan_coupled``/``_get_cached_scan``) on
sliced ``xs`` arrays, each restarted from the identity matrix -- exactly
matching the identity restarts NumPy applies at ocean entry and shell
entry (``solver.py:626-630``).  Only the in-ocean segment needs a new,
much simpler scan: the propagator there is zero except the per-mode
Poisson (Laplace) block.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .bc_ocean_coupled import assemble_bc_ocean_coupled
from .jax_coupled import _get_cached_scan, _grid_and_props
from .propagator import CK_A, CK_B, CK_C


def _detect_ocean(model, numerics):
    """Locate the ocean layer and its boundary grid-node indices.

    Mirrors the detection block in ``solver._get_solution_coupled``
    (0-based ``BCindices``, identical guard raises).  Returns
    ``(ocean_layer, ocean_start, ocean_end)``; ``ocean_layer == 0`` means
    the model has no ocean.
    """
    n_layers = model.n_layers
    for i in range(n_layers):
        if int(model.ocean[i]) == 1:
            if i < 2:
                raise NotImplementedError(
                    "Ocean directly above the core (layer index < 2) is "
                    "not supported (broken in MATLAB LOV3D as well)."
                )
            if i >= n_layers - 1:
                raise NotImplementedError(
                    "Ocean as the outermost layer is not supported "
                    "(errors in MATLAB LOV3D as well)."
                )
            if int(numerics.Nrlayer[i + 1]) == 0:
                # With no nodes in the layer above the ocean, the "shell
                # entry" restart lands in a different layer; NumPy and this
                # scan would silently diverge (and NumPy's own treatment of
                # that degenerate grid is discontinuous in the point count).
                raise ValueError(
                    "The layer directly above the ocean has zero grid "
                    "points; increase the radial resolution."
                )
            ocean_start = int(numerics.BCindices[i - 2])
            ocean_end = int(numerics.BCindices[i - 1])
            return i, ocean_start, ocean_end
    return 0, 0, 0


# ---------------------------------------------------------------------------
# In-ocean scan: Laplace-only Poisson block, no elastic rows.
# ---------------------------------------------------------------------------

def _make_jit_scan_ocean(static: dict):
    """Return a JIT-compiled scan over the in-ocean Laplace-only system.

    Ports ``ocean_coupled.build_aprop_ocean_coupled`` to traced jnp: the
    propagator is zero except the per-mode block
    ``A100 + A101/r + A102/r**2`` (already block-diagonal across modes in
    ``static``), with the degree-0 Poisson row rewritten exactly as the
    NumPy builder does.  Same Cash-Karp RK5 coefficients as the solid scan.
    """
    N2 = static["A9"].shape[0]
    N6 = static["A4"].shape[1]
    N8 = N6 + N2
    Gg = static["Gg"]
    deg0_modes = static["deg0_modes"]

    ck_a, ck_b, ck_c = CK_A, CK_B, CK_C
    I8N = jnp.eye(N8, dtype=jnp.complex128)

    def _ap(r, rho, M_inner, R_inner):
        """Build the in-ocean propagator at radius r (Laplace block only)."""
        Ap = jnp.zeros((N8, N8), dtype=jnp.complex128)
        Ap = Ap.at[N6:N8, N6:N8].set(
            static["A100"] + static["A101"] / r + static["A102"] / r**2
        )
        if deg0_modes:
            M_r = M_inner + (4.0 / 3.0) * math.pi * rho * (r**3 - R_inner**3)
            g = Gg * M_r / r**2
            for k in deg0_modes:
                b = N6 + 2 * k
                Ap = Ap.at[b, :].set(0)
                Ap = Ap.at[b, b].set(4.0 * math.pi * Gg / g)
        return Ap

    def _cash_karp(r_start, dr, rho, M_inner, R_inner):
        """One Cash-Karp RK5 step, in-ocean propagator."""
        args = (rho, M_inner, R_inner)
        K1 = dr * _ap(r_start, *args)
        K2 = dr * _ap(r_start + ck_a[1] * dr, *args) \
            @ (I8N + ck_b[1][0] * K1)
        K3 = dr * _ap(r_start + ck_a[2] * dr, *args) \
            @ (I8N + ck_b[2][0] * K1 + ck_b[2][1] * K2)
        K4 = dr * _ap(r_start + ck_a[3] * dr, *args) \
            @ (I8N + ck_b[3][0] * K1 + ck_b[3][1] * K2 + ck_b[3][2] * K3)
        K5 = dr * _ap(r_start + ck_a[4] * dr, *args) \
            @ (I8N + ck_b[4][0] * K1 + ck_b[4][1] * K2
                   + ck_b[4][2] * K3 + ck_b[4][3] * K4)
        K6 = dr * _ap(r_start + ck_a[5] * dr, *args) \
            @ (I8N + ck_b[5][0] * K1 + ck_b[5][1] * K2
                   + ck_b[5][2] * K3 + ck_b[5][3] * K4 + ck_b[5][4] * K5)
        return ck_c[0] * K1 + ck_c[2] * K3 + ck_c[3] * K4 + ck_c[5] * K6

    def step(Y_prev, xs):
        """Single in-ocean scan step: pure RK5 propagation, no drho term."""
        r_prev, r_curr, rho_k, M_inner_k, R_inner_k = xs
        dr = r_curr - r_prev
        inc = _cash_karp(r_prev, dr, rho_k, M_inner_k, R_inner_k)
        Y_new = (I8N + inc) @ Y_prev
        return Y_new, Y_new

    @jax.jit
    def run_scan(Y_init, xs_tuple):
        """JIT-compiled scan over the ocean segment's radial steps."""
        Y_final, Y_all = jax.lax.scan(step, Y_init, xs_tuple)
        return Y_final, Y_all

    return run_scan


_OCEAN_SCAN_CACHE: dict = {}


def _get_cached_scan_ocean(n_s, Coup, Gg):
    """Memoize the ocean-segment jitted scan, keyed like ``_get_cached_scan``."""
    key = (np.asarray(n_s).tobytes(), np.asarray(Coup).tobytes(), float(Gg))
    entry = _OCEAN_SCAN_CACHE.get(key)
    if entry is None:
        static, _ = _get_cached_scan(n_s, Coup, Gg)
        entry = _make_jit_scan_ocean(static)
        _OCEAN_SCAN_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SOLID_XS_NAMES = ("r_prev", "r_curr", "muC", "lam", "rho", "M_inner", "R_inner")
_OCEAN_XS_NAMES = ("r_prev", "r_curr", "rho", "M_inner", "R_inner")


def propagate_coupled_jax_scan_ocean(model, forcing, numerics, couplings, lateral):
    """JIT-compiled three-segment radial integration for ocean-bearing models.

    Mirrors the loop body of ``solver._get_solution_coupled`` for
    ocean-bearing models: identical grid, identical identity restarts at
    ocean entry and shell entry, identical in-ocean Laplace-only physics.

    Arguments mirror ``solver._get_solution_coupled``; ``forcing`` is
    unused here and kept for signature parity with the solid-path sibling.

    Returns
    -------
    Y_all : (Nr+1, 8N, 8N) complex128 numpy array
    r_grid : (Nr+1,) float64 numpy array
    """
    Nr = numerics.Nr
    Gg = model.Gg
    N = len(couplings.n_s)
    N8 = 8 * N

    ocean_layer, ocean_start, ocean_end = _detect_ocean(model, numerics)
    if ocean_layer == 0:
        raise ValueError(
            "propagate_coupled_jax_scan_ocean called on a model without "
            "an ocean layer"
        )

    r_grid, _, _, arrays = _grid_and_props(
        model, numerics, couplings, lateral,
    )

    # NumPy resets to identity (skipping the drho correction entirely) at
    # shell entry (solver.py:626-630); _grid_and_props unconditionally
    # records a drho entry there, so zero it before slicing the shell
    # segment. (The ocean-entry node needs nothing: the ocean segment's
    # step consumes no delta_rho at all.)
    delta_rho = np.array(arrays["delta_rho"], copy=True)
    delta_rho[ocean_end] = 0.0

    _, run_scan_solid = _get_cached_scan(couplings.n_s, couplings.Coup, Gg)
    run_scan_ocean = _get_cached_scan_ocean(couplings.n_s, couplings.Coup, Gg)

    def solid_xs(sl):
        vals = [jnp.array(arrays[name][sl]) for name in _SOLID_XS_NAMES]
        vals.append(jnp.array(delta_rho[sl]))
        vals.append(jnp.array(arrays["muC_amp"][sl]))
        vals.append(jnp.array(arrays["K_amp"][sl]))
        return tuple(vals)

    def ocean_xs(sl):
        return tuple(jnp.array(arrays[name][sl]) for name in _OCEAN_XS_NAMES)

    below_sl = slice(0, ocean_start)
    ocean_sl = slice(ocean_start, ocean_end)
    shell_sl = slice(ocean_end, Nr)

    Y_init = jnp.eye(N8, dtype=jnp.complex128)
    _, Y_below = run_scan_solid(Y_init, solid_xs(below_sl))
    _, Y_ocean = run_scan_ocean(Y_init, ocean_xs(ocean_sl))
    _, Y_shell = run_scan_solid(Y_init, solid_xs(shell_sl))
    jax.block_until_ready((Y_below, Y_ocean, Y_shell))

    Y_all = jnp.concatenate(
        [
            jnp.eye(N8, dtype=jnp.complex128)[None, :, :],
            Y_below, Y_ocean, Y_shell,
        ],
        axis=0,
    )
    return np.array(Y_all), r_grid


def jax_get_solution_coupled_scan_ocean(model, forcing, numerics, couplings, lateral):
    """Coupled ocean solve using the lax.scan JAX propagator.

    Mirrors the ocean branch of ``solver._get_solution_coupled``
    (boundary-condition assembly + three-segment recombination), built on
    top of ``propagate_coupled_jax_scan_ocean``.

    Returns
    -------
    y_sol : (Nr+1, 8N) complex128 numpy array
    r_grid : (Nr+1,) float64 numpy array
    Y : (Nr+1, 8N, 8N) complex128 numpy array
    Aprop_aux : (Nr+1, 3N, 8N) complex128 numpy array
    """
    Y, r_grid = propagate_coupled_jax_scan_ocean(
        model, forcing, numerics, couplings, lateral,
    )
    Nr = len(r_grid) - 1
    n_layers = model.n_layers
    Gg = model.Gg
    N = len(couplings.n_s)
    N8 = 8 * N

    ocean_layer, ocean_start, ocean_end = _detect_ocean(model, numerics)
    _, layer_map, _, _ = _grid_and_props(
        model, numerics, couplings, lateral,
    )

    rhoC = float(model.rho[0])
    Rc = float(model.R[0])
    Mc = (4.0 / 3.0) * math.pi * rhoC * Rc ** 3
    gc = Gg * Mc / Rc ** 2 if Rc > 0 else 0.0

    rho2 = float(model.Delta_rho[0]) + float(model.rho[1])
    rhoK_surface = float(model.rho[n_layers - 1])

    gO = float(model.gs[ocean_layer - 1])
    gI = float(model.gs[ocean_layer])
    rhoO = float(model.rho[ocean_layer])
    rho_below_ocean = float(model.rho[ocean_layer - 1])
    rho_above_ocean = float(model.rho[ocean_layer + 1])

    B, B2 = assemble_bc_ocean_coupled(
        Y[0], Y[Nr], Y[ocean_start], Y[ocean_end],
        couplings.n_s, couplings.m_s,
        gc, Rc, rho2, rhoK_surface, Gg,
        float(model.rho[1]),
        gO, gI, rhoO, rho_below_ocean, rho_above_ocean,
        forcing,
    )
    C = np.linalg.solve(B, B2)

    # Three-segment recombination (MATLAB get_solution.m:864-881).
    C_below = C[:N8]
    C_ocean = C[N8:2 * N8]
    C_shell = C[2 * N8:]

    y_sol = np.zeros((Nr + 1, N8), dtype=np.complex128)
    for k_idx in range(Nr + 1):
        il = layer_map[k_idx]
        if k_idx == ocean_end:
            # Ocean-ceiling node = the shell segment's identity origin.
            y_sol[k_idx] = C_shell.copy()
        elif il >= ocean_layer + 1:
            y_sol[k_idx] = Y[k_idx] @ C_shell
        elif il == ocean_layer:
            y_sol[k_idx] = Y[k_idx] @ C_ocean
        else:
            y_sol[k_idx] = Y[k_idx] @ C_below

    from .jax_coupled_aux import compute_aprop_aux_coupled
    Aprop_aux = compute_aprop_aux_coupled(
        model, forcing, numerics, couplings, lateral,
    )
    return y_sol, r_grid, Y, Aprop_aux
