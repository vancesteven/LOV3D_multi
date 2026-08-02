"""JAX assembly helpers for the coupled multi-mode propagator.

This module covers the static precomputation and traceable 8N x 8N matrix
build used by the coupled ``lax.scan`` propagator.  Radial integration and
the public solver API are added separately.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .propagator import (
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
