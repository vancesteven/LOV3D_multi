"""JAX port of the coupled multi-mode propagator.

Static precompute, traceable 8N x 8N assembly, lax.scan radial integration,
and the public coupled solve API (``jax_get_solution_coupled_scan``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .boundary_conditions import assemble_bc_no_ocean_coupled
from .propagator import (
    CK_A,
    CK_B,
    CK_C,
    _COUP_ROWS,
    _a1a2_geometric,
    build_A3,
    build_A4,
    build_A5,
    build_others,
)


def _precompute_coupled_static(
    n_s: np.ndarray,
    Coup: np.ndarray,
    Gg: float,
) -> dict:
    """Precompute geometry-only tensors for coupled JAX assembly."""
    n_s = np.asarray(n_s, dtype=int)
    Coup = np.asarray(Coup)
    N = len(n_s)
    Nreo = Coup.shape[3]
    N2, N3, N6 = 2 * N, 3 * N, 6 * N

    G1_bulk = np.zeros((N6, N3), dtype=np.complex128)
    G2_bulk = np.zeros((N6, N3), dtype=np.complex128)
    G1_shear = np.zeros((N6, N3), dtype=np.complex128)
    G2_shear = np.zeros((N6, N3), dtype=np.complex128)
    G1_K = np.zeros((Nreo, N6, N3), dtype=np.complex128)
    G2_K = np.zeros((Nreo, N6, N3), dtype=np.complex128)
    G1_mu = np.zeros((Nreo, N6, N3), dtype=np.complex128)
    G2_mu = np.zeros((Nreo, N6, N3), dtype=np.complex128)

    A3_inv = np.zeros((N3, N3), dtype=np.complex128)
    A4 = np.zeros((N3, N6), dtype=np.complex128)
    A5 = np.zeros((N3, N6), dtype=np.complex128)
    A13 = np.eye(N3, dtype=np.complex128)
    A9 = np.eye(N2, dtype=np.complex128)
    A100 = np.zeros((N2, N2), dtype=np.complex128)
    A101 = np.zeros((N2, N2), dtype=np.complex128)
    A102 = np.zeros((N2, N2), dtype=np.complex128)
    P71 = np.zeros((N3, N3), dtype=np.complex128)
    P72 = np.zeros((N3, N3), dtype=np.complex128)
    P81 = np.zeros((N3, N2), dtype=np.complex128)
    P82 = np.zeros((N3, N2), dtype=np.complex128)
    P11 = np.zeros((N2, N3), dtype=np.complex128)
    P12 = np.zeros((N2, N3), dtype=np.complex128)

    geometric = []
    for k, n_value in enumerate(n_s):
        n = int(n_value)
        A1g, A2g = _a1a2_geometric(n)
        geometric.append((A1g, A2g))
        rows6 = slice(6 * k, 6 * k + 6)
        cols3 = slice(3 * k, 3 * k + 3)
        G1_bulk[6 * k, cols3] = A1g[0]
        G2_bulk[6 * k, cols3] = A2g[0]
        G1_shear[rows6, cols3] = A1g
        G2_shear[rows6, cols3] = A2g
        G1_shear[6 * k, cols3] = 0
        G2_shear[6 * k, cols3] = 0

        A3_inv[cols3, cols3] = np.linalg.inv(build_A3(n))
        A4[cols3, rows6] = build_A4(n)
        A5[cols3, rows6] = build_A5(n)

        others = build_others(n, rho=1.0, Gg=Gg)
        _, _, a71, a72, a81, a82, _, a100, a101, a102, a11, a12 = others
        cols2 = slice(2 * k, 2 * k + 2)
        P71[cols3, cols3] = a71
        P72[cols3, cols3] = a72
        P81[cols3, cols2] = a81
        P82[cols3, cols2] = a82
        A100[cols2, cols2] = a100
        A101[cols2, cols2] = a101
        A102[cols2, cols2] = a102
        P11[cols2, cols3] = a11
        P12[cols2, cols3] = a12

    for i in range(N):
        for j in range(N):
            source_cols = slice(3 * j, 3 * j + 3)
            A1g, A2g = geometric[j]
            for ireo in range(Nreo):
                if Coup[i, j, 26, ireo] == 0:
                    continue
                Cp = Coup[i, j, :26, ireo]
                G1_K[ireo, 6 * i, source_cols] += Cp[0] * A1g[0]
                G2_K[ireo, 6 * i, source_cols] += Cp[0] * A2g[0]
                for group, target in enumerate(_COUP_ROWS):
                    for source_index, source in enumerate(_COUP_ROWS):
                        slot = 1 + 5 * group + source_index
                        G1_mu[ireo, 6 * i + target, source_cols] += (
                            2 * Cp[slot] * A1g[source]
                        )
                        G2_mu[ireo, 6 * i + target, source_cols] += (
                            2 * Cp[slot] * A2g[source]
                        )

    arrays = {
        "G1_bulk": G1_bulk,
        "G2_bulk": G2_bulk,
        "G1_shear": G1_shear,
        "G2_shear": G2_shear,
        "G1_K": G1_K,
        "G2_K": G2_K,
        "G1_mu": G1_mu,
        "G2_mu": G2_mu,
        "A3_inv": A3_inv,
        "A4": A4,
        "A5": A5,
        "A13": A13,
        "A9": A9,
        "A100": A100,
        "A101": A101,
        "A102": A102,
        "P71": P71,
        "P72": P72,
        "P81": P81,
        "P82": P82,
        "P11": P11,
        "P12": P12,
    }
    static = {
        key: jnp.array(value, dtype=jnp.complex128)
        for key, value in arrays.items()
    }
    static["deg0_modes"] = [
        k for k, n_value in enumerate(n_s) if int(n_value) == 0
    ]
    # Needed by the degree-0 Poisson row; the traced builder has no Gg arg.
    static["Gg"] = float(Gg)
    return static


def _build_aprop_coupled_jax(
    r,
    g,
    dg,
    muC,
    lam,
    rho,
    muC_amp,
    K_amp,
    static: dict,
):
    """Assemble and solve the traceable coupled propagator system."""
    A1 = (
        (3 * lam + 2 * muC) * static["G1_bulk"]
        + 2 * muC * static["G1_shear"]
        + jnp.einsum("r,rij->ij", K_amp, static["G1_K"])
        + jnp.einsum("r,rij->ij", muC_amp, static["G1_mu"])
    )
    A2 = (
        (3 * lam + 2 * muC) * static["G2_bulk"]
        + 2 * muC * static["G2_shear"]
        + jnp.einsum("r,rij->ij", K_amp, static["G2_K"])
        + jnp.einsum("r,rij->ij", muC_amp, static["G2_mu"])
    )

    A71 = rho * static["P71"]
    A72 = rho * static["P72"]
    A81 = rho * static["P81"]
    A82 = rho * static["P82"]
    A11 = rho * static["P11"]
    A12 = rho * static["P12"]

    N3 = static["A3_inv"].shape[0]
    N6 = static["A4"].shape[1]
    N2 = static["A9"].shape[0]
    N8 = N6 + N2
    Adotx = jnp.zeros((N8, N8), dtype=jnp.complex128)
    Ax = jnp.zeros((N8, N8), dtype=jnp.complex128)

    A4_A2_A3inv = static["A4"] @ A2 @ static["A3_inv"]
    A4_A1_A3inv = static["A4"] @ A1 @ static["A3_inv"]
    A5_A2_A3inv = static["A5"] @ A2 @ static["A3_inv"]
    A5_A1_A3inv = static["A5"] @ A1 @ static["A3_inv"]
    Adotx = Adotx.at[:N3, :N3].set(A4_A2_A3inv)
    Ax = Ax.at[:N3, :N3].set(-A4_A1_A3inv / r)
    Ax = Ax.at[:N3, N3:N6].set(static["A13"])

    Adotx = Adotx.at[N3:N6, :N3].set(-A5_A2_A3inv / r)
    Adotx = Adotx.at[N3:N6, N3:N6].set(static["A13"])
    Ax = Ax.at[N3:N6, :N3].set(
        A5_A1_A3inv / r**2 + (g / r) * A71 + dg * A72
    )
    Ax = Ax.at[N3:N6, N6:N8].set(A81 + A82 / r)

    Adotx = Adotx.at[N6:N8, :N3].set(-A12)
    Adotx = Adotx.at[N6:N8, N6:N8].set(static["A9"])
    Ax = Ax.at[N6:N8, :N3].set(A11 / r)
    Ax = Ax.at[N6:N8, N6:N8].set(
        static["A100"] + static["A101"] / r + static["A102"] / r**2
    )

    for k in static["deg0_modes"]:
        for idx in (3 * k + 1, 3 * k + 2, N3 + 3 * k + 1, N3 + 3 * k + 2):
            Adotx = Adotx.at[idx, :].set(0)
            Ax = Ax.at[idx, :].set(0)
            Adotx = Adotx.at[idx, idx].set(1)
            Ax = Ax.at[idx, idx].set(1)
        phi_row = N6 + 2 * k
        Adotx = Adotx.at[phi_row, :].set(0)
        Ax = Ax.at[phi_row, :].set(0)
        Adotx = Adotx.at[phi_row, phi_row].set(1)
        Ax = Ax.at[phi_row, 3 * k].set(-4 * math.pi * static["Gg"])

    return jnp.linalg.solve(Adotx, Ax)


# ---------------------------------------------------------------------------
# Increment 3: lax.scan radial integration
# ---------------------------------------------------------------------------

def _make_jit_scan_coupled(static: dict):
    """Return a JIT-compiled scan over the coupled 8N x 8N system.

    Mirrors ``jax_scan._make_jit_scan``. ``static`` (from
    ``_precompute_coupled_static``) is closed over rather than passed as a
    jit argument: it is non-hashable, and tracing it would degrade the
    degree-0 row overrides to dynamic scatter ops.

    Returns run_scan(Y_init, xs_tuple) -> (Y_final, Y_all (Nr, 8N, 8N)).
    """
    N3 = static["A3_inv"].shape[0]
    N6 = static["A4"].shape[1]
    N2 = static["A9"].shape[0]
    N = N3 // 3
    N8 = N6 + N2
    Gg = static["Gg"]

    ck_a = CK_A
    ck_b = CK_B
    ck_c = CK_C
    I8N = jnp.eye(N8, dtype=jnp.complex128)

    dPhi_rows = jnp.array([6 * N + 2 * m + 1 for m in range(N)], dtype=jnp.int32)
    U_rows = jnp.array([3 * m for m in range(N)], dtype=jnp.int32)

    def _ap(r, muC, lam, rho, M_inner, R_inner, muC_amp, K_amp):
        """Build the coupled propagator at radius r given layer material."""
        M_r = M_inner + (4.0 / 3.0) * math.pi * rho * (r**3 - R_inner**3)
        g = Gg * M_r / r**2
        dg = Gg * (
            2.0 * (4.0 / 3.0 * math.pi * rho * R_inner**3 - M_inner) / r**3
            + 4.0 / 3.0 * math.pi * rho
        )
        return _build_aprop_coupled_jax(
            r, g, dg, muC, lam, rho, muC_amp, K_amp, static
        )

    def _cash_karp(r_start, dr, muC, lam, rho, M_inner, R_inner, muC_amp, K_amp):
        """One Cash-Karp RK5 step returning the 8N x 8N increment matrix."""
        args = (muC, lam, rho, M_inner, R_inner, muC_amp, K_amp)
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
        """Single scan step: per-mode density correction + RK5 propagation."""
        (r_prev, r_curr, muC_k, lam_k, rho_k, M_inner_k, R_inner_k,
         drho_k, muC_amp_k, K_amp_k) = xs
        dr = r_curr - r_prev

        # Density-discontinuity correction, per mode (nonzero only at
        # layer boundaries): dPhi_row(m) += 4*pi*Gg*drho * U_row(m).
        Y_corr = Y_prev.at[dPhi_rows, :].add(
            4.0 * jnp.pi * Gg * drho_k * Y_prev[U_rows, :]
        )

        inc = _cash_karp(
            r_prev, dr, muC_k, lam_k, rho_k, M_inner_k, R_inner_k,
            muC_amp_k, K_amp_k,
        )
        Y_new = (I8N + inc) @ Y_corr
        return Y_new, Y_new

    @jax.jit
    def run_scan(Y_init, xs_tuple):
        """JIT-compiled scan over all Nr radial steps."""
        Y_final, Y_all = jax.lax.scan(step, Y_init, xs_tuple)
        return Y_final, Y_all

    return run_scan


# ---------------------------------------------------------------------------
# Increment 4: public API
# ---------------------------------------------------------------------------

_SCAN_CACHE: dict = {}


def _get_cached_scan(n_s, Coup, Gg):
    """Memoize (static, run_scan) so repeated solves reuse the jitted scan.

    Without this, every call rebuilds the scan closure and jax.jit never
    hits its compilation cache, making small-N solves compile-bound.
    """
    key = (np.asarray(n_s).tobytes(), np.asarray(Coup).tobytes(), float(Gg))
    entry = _SCAN_CACHE.get(key)
    if entry is None:
        static = _precompute_coupled_static(n_s, Coup, Gg)
        entry = (static, _make_jit_scan_coupled(static))
        _SCAN_CACHE[key] = entry
    return entry


def _grid_and_props(model, numerics, couplings, lateral):
    """Build the coupled radial grid, layer map, masses, and scan inputs."""
    n_layers = model.n_layers
    Nr = numerics.Nr
    r_grid = np.zeros(Nr + 1)
    layer_map = np.zeros(Nr + 1, dtype=int)
    Rc = float(model.R[0])
    r_grid[0] = Rc

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

    Nreo = couplings.Coup.shape[3]
    arrays = {
        "r_prev": np.zeros(Nr), "r_curr": np.zeros(Nr),
        "muC": np.zeros(Nr, dtype=np.complex128),
        "lam": np.zeros(Nr, dtype=np.complex128),
        "rho": np.zeros(Nr), "M_inner": np.zeros(Nr),
        "R_inner": np.zeros(Nr), "delta_rho": np.zeros(Nr),
        "muC_amp": np.zeros((Nr, Nreo), dtype=np.complex128),
        "K_amp": np.zeros((Nr, Nreo), dtype=np.complex128),
    }
    for k_idx in range(1, Nr + 1):
        xs_k = k_idx - 1
        i_layer = layer_map[k_idx]
        prev_layer_k = 1 if k_idx == 1 else layer_map[k_idx - 1]
        arrays["r_prev"][xs_k] = r_grid[k_idx - 1]
        arrays["r_curr"][xs_k] = r_grid[k_idx]
        arrays["muC"][xs_k] = complex(model.muC[i_layer])
        arrays["lam"][xs_k] = complex(model.lam[i_layer])
        arrays["rho"][xs_k] = float(model.rho[i_layer])
        arrays["M_inner"][xs_k] = M_at_boundary[i_layer]
        arrays["R_inner"][xs_k] = float(model.R[i_layer - 1])
        arrays["muC_amp"][xs_k] = lateral.muC_amp[i_layer]
        arrays["K_amp"][xs_k] = lateral.K_amp[i_layer]
        if i_layer != prev_layer_k:
            arrays["delta_rho"][xs_k] = float(model.Delta_rho[i_layer])
    return r_grid, layer_map, M_at_boundary, arrays


def propagate_coupled_jax_scan(model, forcing, numerics, couplings, lateral):
    """JIT-compiled coupled radial integration using jax.lax.scan.

    Mirrors ``jax_scan.propagate_1d_jax_scan`` but integrates the 8N x 8N
    coupled system (multiple degree/order modes with lateral-variation
    coupling), matching the NumPy reference ``solver._get_solution_coupled``.

    Arguments mirror ``solver._get_solution_coupled``; ``forcing`` is unused
    here and kept for signature parity.

    Returns
    -------
    Y_all : (Nr+1, 8N, 8N) complex128 numpy array
    r_grid : (Nr+1,) float64 numpy array
    """
    n_layers = model.n_layers
    Nr = numerics.Nr
    Gg = model.Gg
    N = len(couplings.n_s)
    N8 = 8 * N

    # Ocean layers: supported by the NumPy coupled solver since TASK-007,
    # but not yet ported to the JAX scan (queued as TASK-009).
    for i in range(n_layers):
        if int(model.ocean[i]) == 1:
            raise NotImplementedError(
                "Ocean layers not yet supported in the coupled JAX scan; "
                "use solver.get_solution (NumPy) for ocean-bearing models."
            )

    r_grid, _, _, arrays = _grid_and_props(
        model, numerics, couplings, lateral,
    )

    # ----- Build (or reuse) the JIT-compiled scan -----
    _, run_scan = _get_cached_scan(couplings.n_s, couplings.Coup, Gg)

    Y_init = jnp.eye(N8, dtype=jnp.complex128)
    xs_tuple = tuple(jnp.array(arrays[name]) for name in (
        "r_prev", "r_curr", "muC", "lam", "rho", "M_inner", "R_inner",
        "delta_rho", "muC_amp", "K_amp",
    ))

    _, Y_steps = run_scan(Y_init, xs_tuple)
    jax.block_until_ready(Y_steps)

    # Prepend the identity initial matrix to get shape (Nr+1, 8N, 8N)
    Y_all_jax = jnp.concatenate(
        [jnp.eye(N8, dtype=jnp.complex128)[None, :, :], Y_steps], axis=0
    )
    return np.array(Y_all_jax), r_grid


def jax_get_solution_coupled_scan(model, forcing, numerics, couplings, lateral):
    """Coupled solve using the lax.scan JAX propagator.

    Mirrors the boundary-condition and solution-assembly block of
    ``solver._get_solution_coupled`` (lines 522-540), but built on top of
    ``propagate_coupled_jax_scan``, including the auxiliary propagator rows.

    Returns
    -------
    y_sol : (Nr+1, 8N) complex128 numpy array
    r_grid : (Nr+1,) float64 numpy array
    Y : (Nr+1, 8N, 8N) complex128 numpy array
    Aprop_aux : (Nr+1, 3N, 8N) complex128 numpy array
    """
    Y, r_grid = propagate_coupled_jax_scan(
        model, forcing, numerics, couplings, lateral,
    )
    Nr = len(r_grid) - 1
    n_layers = model.n_layers
    Gg = model.Gg

    rhoC = float(model.rho[0])
    Rc = float(model.R[0])
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
    y_sol = Y @ C

    from .jax_coupled_aux import compute_aprop_aux_coupled
    Aprop_aux = compute_aprop_aux_coupled(
        model, forcing, numerics, couplings, lateral,
    )
    return y_sol, r_grid, Y, Aprop_aux
