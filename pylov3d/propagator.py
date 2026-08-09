# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Propagator matrix builders — translated from get_solution.m subfunctions.

Builds the 8×8 (single-mode) or 8N×8N (coupled multi-mode) propagator
matrix ``Aprop`` used in the radial ODE ``dy/dr = Aprop(r) · y``.

The state vector (per mode) is stored internally as
``[u_GSH(3), Sigma_SH(3), Phi(2)]`` with rearrangement to
``[U, V, W, R, S, T, Phi, dPhi/dr]`` done after integration.

For the coupled case, only A1 and A2 have off-diagonal (cross-mode)
blocks — all other sub-matrices are block-diagonal.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Sub-matrix builders (all return plain numpy arrays for assembly speed)
# ---------------------------------------------------------------------------

def build_A1_A2(n: int, muC: complex, lam: complex):
    """Constitutive: sigma = A1 · du/dr + A2/r · u  (GSH stress ← GSH disp).

    Returns A1 (6×3), A2 (6×3) for a single mode with degree *n*.
    """
    A1 = np.zeros((6, 3), dtype=np.complex128)
    A2 = np.zeros((6, 3), dtype=np.complex128)

    if n <= 0:
        # n=0 special case (only two non-zero entries)
        fac_nn0 = (3 * lam + 2 * muC) / math.sqrt(3) / math.sqrt(2 * n + 1)
        A1[0, 2] = fac_nn0 * math.sqrt(n + 1) * (n + 2)
        A2[0, 2] = fac_nn0 * math.sqrt(n + 1)
        A1[5, 2] = 2 * muC * math.sqrt((n + 2) / (2 * n + 3)) * (n + 1)
        A2[5, 2] = -2 * muC * math.sqrt((n + 2) / (2 * n + 3))
        return A1, A2

    s2n1 = math.sqrt(2 * n + 1)
    sn = math.sqrt(n)
    sn1 = math.sqrt(n + 1)

    # sigma_{n,n,0}
    fac = (3 * lam + 2 * muC) / math.sqrt(3) / s2n1
    A1[0, 0] = fac * sn * (n - 1)
    A1[0, 2] = fac * sn1 * (n + 2)
    A2[0, 0] = -fac * sn
    A2[0, 2] = fac * sn1

    # sigma_{n,n-2,2}
    fac2 = 2 * muC * math.sqrt((n - 1) / (2 * n - 1))
    A1[1, 0] = fac2 * n
    A2[1, 0] = fac2

    # sigma_{n,n-1,2}
    fac3 = 2 * muC / math.sqrt(2) * math.sqrt((n - 1) / (2 * n + 1))
    A1[2, 1] = fac3 * (n + 1)
    A2[2, 1] = fac3

    # sigma_{n,n,2}
    f4a = 2 * muC * math.sqrt((2*n+3)*(2*n+2) / (12*(2*n-1)*(2*n+1)))
    f4b = 2 * muC * math.sqrt(n*(2*n-1)*(n+1) / (3*(2*n+3)*(2*n+2)*(2*n+1)))
    A1[3, 0] = f4a * (n - 1)
    A1[3, 2] = f4b * (n + 2)
    A2[3, 0] = -f4a
    A2[3, 2] = f4b

    # sigma_{n,n+1,2}
    fac5 = 2 * muC / math.sqrt(2) * math.sqrt((n + 2) / (2 * n + 1))
    A1[4, 1] = fac5 * n
    A2[4, 1] = -fac5

    # sigma_{n,n+2,2}
    fac6 = 2 * muC * math.sqrt((n + 2) / (2 * n + 3))
    A1[5, 2] = fac6 * (n + 1)
    A2[5, 2] = -fac6

    return A1, A2


def build_A3(n: int):
    """Displacement transform: [U, V, W] = A3 · [u_{n-1}, u_n, u_{n+1}].

    Returns A3 (3×3).
    """
    A3 = np.zeros((3, 3), dtype=np.complex128)
    if n > 0:
        s2n1 = math.sqrt(2 * n + 1)
        sn = math.sqrt(n)
        sn1 = math.sqrt(n + 1)
        # U
        A3[0, 0] = sn / s2n1
        A3[0, 2] = -sn1 / s2n1
        # V
        A3[1, 0] = 1 / (s2n1 * sn)
        A3[1, 2] = 1 / (s2n1 * sn1)
        # W
        A3[2, 1] = 1j / math.sqrt(n * (n + 1))
    else:
        # n=0 special case
        A3[0, 2] = -math.sqrt(n + 1) / math.sqrt(2 * n + 1)
        A3[2, 1] = 1.0
        A3[1, 0] = 1.0
    return A3


def build_A4(n: int):
    """Stress transform: [R, S, T] = A4 · sigma_GSH  (3×6)."""
    A4 = np.zeros((3, 6), dtype=np.complex128)
    if n <= 0:
        A4[0, 0] = -1 / math.sqrt(3)
        A4[0, 5] = math.sqrt((n+1)*(n+2) / ((2*n+1)*(2*n+3)))
        return A4

    nn = n * (n + 1)
    s2n1 = 2 * n + 1

    # R
    A4[0, 0] = -1 / math.sqrt(3)
    A4[0, 1] = math.sqrt(n * (n - 1) / ((2*n - 1) * s2n1))
    A4[0, 3] = -math.sqrt(n) / s2n1 * (
        math.sqrt((2*n+3)*(2*n+2) / (12*(2*n-1)))
        + math.sqrt((2*n-1)*(n+1)**2 / (3*(2*n+2)*(2*n+3)))
    )
    A4[0, 5] = math.sqrt((n+1)*(n+2) / (s2n1*(2*n+3)))

    # S
    A4[1, 1] = math.sqrt((n - 1) / ((2*n - 1) * n * s2n1))
    A4[1, 3] = (
        -math.sqrt((2*n+3)*(2*n+2) / (12*n*(2*n-1)*s2n1**2))
        + math.sqrt(n*(2*n-1) / (3*s2n1**2*(2*n+2)*(2*n+3)))
    )
    A4[1, 5] = -math.sqrt((n + 2) / ((2*n + 3) * (n + 1) * s2n1))

    # T
    A4[2, 2] = 1j * math.sqrt(n - 1) / math.sqrt(2 * n * (n + 1) * s2n1)
    A4[2, 4] = -1j * math.sqrt(n + 2) / math.sqrt(2 * n * (n + 1) * s2n1)

    return A4


def build_A5(n: int):
    """Stress divergence: part of momentum equation (3×6).

    Note: the MATLAB code negates A5 at the end (``A5 = -A5``).
    """
    A5 = np.zeros((3, 6), dtype=np.complex128)

    if n <= 0:
        A5[0, 5] = (n + 3) * math.sqrt((n+1)*(n+2) / ((2*n+3)*(2*n+1)))
        return -A5

    s2n1 = 2 * n + 1

    # R row
    A5[0, 1] = -(n - 2) * math.sqrt(n*(n-1) / ((2*n-1)*s2n1))
    A5[0, 3] = 1/s2n1 * (
        -(n+1) * math.sqrt((2*n+3)*(2*n+2)*n / (12*(2*n-1)))
        + n * math.sqrt(n*(n+1)**2*(2*n-1) / (3*(2*n+2)*(2*n+3)))
    )
    A5[0, 5] = (n + 3) * math.sqrt((n+1)*(n+2) / ((2*n+3)*s2n1))

    # S row
    A5[1, 0] = -1 / math.sqrt(3)
    A5[1, 1] = -(n - 2) * math.sqrt((n-1) / ((2*n-1)*s2n1*n))
    A5[1, 3] = -1/s2n1 * (
        (n+1) * math.sqrt((2*n+3)*(2*n+2) / ((2*n-1)*12*n))
        + n * math.sqrt(n*(2*n-1) / (3*(2*n+2)*(2*n+3)))
    )
    A5[1, 5] = -(n + 3) * math.sqrt((n+2) / ((2*n+3)*s2n1*(n+1)))

    # T row
    A5[2, 2] = -(n - 1) * math.sqrt(n - 1) / math.sqrt(2*n*(n+1)*s2n1) * 1j
    A5[2, 4] = -(n + 2) * math.sqrt(n + 2) / math.sqrt(2*n*(n+1)*s2n1) * 1j

    return -A5  # MATLAB negates A5


def build_others(n: int, rho: float, Gg: float):
    """Gravity, potential, and identity matrices.

    Returns (A13, A6, A71, A72, A81, A82, A9, A100, A101, A102, A11, A12).
    """
    A13 = np.eye(3, dtype=np.complex128)
    A6 = np.zeros((3, 3), dtype=np.complex128)  # always zero

    A71 = np.zeros((3, 3), dtype=np.complex128)
    A72 = np.zeros((3, 3), dtype=np.complex128)
    A81 = np.zeros((3, 2), dtype=np.complex128)
    A82 = np.zeros((3, 2), dtype=np.complex128)
    A9 = np.eye(2, dtype=np.complex128)
    A100 = np.zeros((2, 2), dtype=np.complex128)
    A101 = np.zeros((2, 2), dtype=np.complex128)
    A102 = np.zeros((2, 2), dtype=np.complex128)
    A11 = np.zeros((2, 3), dtype=np.complex128)
    A12 = np.zeros((2, 3), dtype=np.complex128)

    nn = n * (n + 1)

    # A71, A72 — gravity-displacement coupling
    A71[0, 0] = -2 * rho
    A71[0, 1] = rho * nn
    A72[0, 0] = rho
    A71[1, 0] = rho

    if n > 0:
        # A81, A82 — momentum-potential coupling
        A81[0, 1] = rho       # R from dΦ/dr
        A82[1, 0] = rho       # S from Φ/r

        # A100-A102 — potential ODE
        A100[0, 1] = 1.0      # dΦ/dr = Φ' (trivial)
        A101[1, 1] = -2.0     # -2/r · Φ'
        A102[1, 0] = nn       # n(n+1)/r² · Φ

        # A11, A12 — Poisson source
        A11[1, 0] = -2 * 4 * math.pi * Gg * rho
        A11[1, 1] = 4 * math.pi * Gg * rho * nn
        A12[1, 0] = -4 * math.pi * Gg * rho
    else:
        # n=0 special case (Longman 1963)
        A100[0, 0] = 1.0
        A100[1, 1] = 1.0

    return A13, A6, A71, A72, A81, A82, A9, A100, A101, A102, A11, A12


# ---------------------------------------------------------------------------
# Propagator assembly
# ---------------------------------------------------------------------------

def build_aprop(
    r: float,
    g: float,
    dg: float,
    n: int,
    muC: complex,
    lam: complex,
    rho: float,
    Gg: float,
) -> np.ndarray:
    """Build the 8×8 propagator matrix at radius *r*.

    The ODE is ``dy/dr = Aprop · y`` with state
    ``y = [U, V, W, R, S, T, Φ, dΦ/dr]`` (SH form, 1 mode).

    Parameters
    ----------
    r : float     — normalized radius
    g : float     — normalized gravity at *r*
    dg : float    — dg/dr at *r*
    n : int       — spherical harmonic degree
    muC : complex — normalized complex shear modulus
    lam : complex — normalized Lamé λ
    rho : float   — normalized density
    Gg : float    — normalized gravitational constant
    """
    Nmodes = 1

    # Build sub-matrices
    A1, A2 = build_A1_A2(n, muC, lam)
    A3 = build_A3(n)
    A3_inv = np.linalg.inv(A3)
    A4 = build_A4(n)
    A5 = build_A5(n)
    A13, A6, A71, A72, A81, A82, A9, A100, A101, A102, A11, A12 = \
        build_others(n, rho, Gg)

    N3 = 3 * Nmodes  # 3
    N6 = 6 * Nmodes  # 6
    N8 = 8 * Nmodes  # 8
    N2 = 2 * Nmodes  # 2

    # Assemble Adotx and Ax (8×8)
    Adotx = np.zeros((N8, N8), dtype=np.complex128)
    Ax = np.zeros((N8, N8), dtype=np.complex128)

    # Block 1: constitutive (rows 0:3)
    # NOTE: A1 (angular /r term, carries the (n-1) factor) and A2 (pure-moduli
    # d/dr term) map to the Ax (/r) and Adotx (d/dr) blocks respectively — i.e.
    # A2 -> Adotx, A1 -> Ax.  build_A1_A2 returns them in (A1, A2) order; the
    # correct assignment swaps their roles here.
    Adotx[:N3, :N3] = A4 @ A2 @ A3_inv
    Ax[:N3, :N3] = -A4 @ A1 @ A3_inv / r
    Ax[:N3, N3:N6] = A13

    # Block 2: momentum (rows 3:6)
    Adotx[N3:N6, :N3] = -A5 @ A2 @ A3_inv / r + A6
    Adotx[N3:N6, N3:N6] = A13
    Ax[N3:N6, :N3] = A5 @ A1 @ A3_inv / r**2 + g / r * A71 + dg * A72
    Ax[N3:N6, N6:N8] = A81 + A82 / r

    # Block 3: Poisson (rows 6:8)
    Adotx[N6:N8, :N3] = -A12
    Adotx[N6:N8, N6:N8] = A9
    Ax[N6:N8, :N3] = A11 / r
    Ax[N6:N8, N6:N8] = A100 + A101 / r + A102 / r**2

    # Solve: Aprop = Adotx \ Ax
    Aprop = np.linalg.solve(Adotx, Ax)

    return Aprop


def build_aprop_ocean(r: float, n: int, Gg: float, gK: float = 0.0) -> np.ndarray:
    """Build the 8×8 propagator inside a liquid (inviscid) ocean layer.

    Translates the ``ocean_flag==1`` branch of ``get_Aprop`` in
    ``get_solution.m`` (lines 1924–1935): displacements and stresses are
    undefined in the ocean, so every row is zero except the Poisson block,
    which reduces to the Laplace equation for the potential.

    ``gK`` (gravity at *r*) is only used by the degree-0 special case.
    """
    _, _, _, _, _, _, _, A100, A101, A102, _, _ = build_others(n, 0.0, Gg)

    Aprop = np.zeros((8, 8), dtype=np.complex128)
    Aprop[6:8, 6:8] = A100 + A101 / r + A102 / r**2
    if n == 0:
        # MATLAB get_solution.m:1928-1931 (Longman degree-0 row)
        Aprop[6, :] = 0
        Aprop[6, 6] = 4 * math.pi * Gg / gK
    return Aprop


def compute_gravity(
    r: float,
    rho_layer: float,
    M_inner: float,
    R_inner: float,
    Gg: float,
) -> tuple[float, float]:
    """Compute gravity and its radial derivative at radius *r*.

    Parameters
    ----------
    r : float         — current normalized radius
    rho_layer : float — normalized density of current layer
    M_inner : float   — normalized mass enclosed within R_inner
    R_inner : float   — normalized radius of inner boundary
    Gg : float        — normalized gravitational constant

    Returns
    -------
    g, dg : gravity and dg/dr at *r*
    """
    M_r = M_inner + (4 / 3) * math.pi * rho_layer * (r**3 - R_inner**3)
    g = Gg * M_r / r**2
    dg = Gg * (2 * (4/3 * math.pi * rho_layer * R_inner**3 - M_inner) / r**3
               + 4/3 * math.pi * rho_layer)
    return g, dg


# ---------------------------------------------------------------------------
# Coupled (multi-mode) propagator
# ---------------------------------------------------------------------------

def _a1a2_geometric(n: int):
    """Stress–displacement geometric factors with unit material properties.

    Returns (A1g, A2g) each (6, 3).
    - Row 0 uses (3λ+2μ)=1 → multiply by actual (3λ+2μ) for diagonal,
      or K_nm for coupling.
    - Rows 1–5 use 2μ=1 → multiply by actual 2μ for diagonal,
      or 2·mu_nm for coupling.
    """
    A1 = np.zeros((6, 3), dtype=np.complex128)
    A2 = np.zeros((6, 3), dtype=np.complex128)

    if n <= 0:
        s2n1 = math.sqrt(max(2 * n + 1, 1))
        sn1 = math.sqrt(n + 1)
        fac0 = 1.0 / math.sqrt(3) / s2n1

        A1[0, 2] = fac0 * sn1 * (n + 2)
        A2[0, 2] = fac0 * sn1
        if 2 * n + 3 > 0:
            fac6 = math.sqrt((n + 2) / (2 * n + 3))
            A1[5, 2] = fac6 * (n + 1)
            A2[5, 2] = -fac6
        return A1, A2

    s2n1 = math.sqrt(2 * n + 1)
    sn = math.sqrt(n)
    sn1 = math.sqrt(n + 1)

    # Row 0: σ_{n,n,0}
    fac0 = 1.0 / math.sqrt(3) / s2n1
    A1[0, 0] = fac0 * sn * (n - 1)
    A1[0, 2] = fac0 * sn1 * (n + 2)
    A2[0, 0] = -fac0 * sn
    A2[0, 2] = fac0 * sn1

    # Row 1: σ_{n,n-2,2}
    if 2 * n - 1 > 0:
        f1 = math.sqrt((n - 1) / (2 * n - 1))
        A1[1, 0] = f1 * n
        A2[1, 0] = f1

    # Row 2: σ_{n,n-1,2}
    f2 = math.sqrt((n - 1) / (2 * n + 1)) / math.sqrt(2)
    A1[2, 1] = f2 * (n + 1)
    A2[2, 1] = f2

    # Row 3: σ_{n,n,2}
    if 2 * n - 1 > 0:
        f3a = math.sqrt((2*n+3)*(2*n+2) / (12*(2*n-1)*(2*n+1)))
        f3b = math.sqrt(n*(2*n-1)*(n+1) / (3*(2*n+3)*(2*n+2)*(2*n+1)))
        A1[3, 0] = f3a * (n - 1)
        A1[3, 2] = f3b * (n + 2)
        A2[3, 0] = -f3a
        A2[3, 2] = f3b

    # Row 4: σ_{n,n+1,2}
    f4 = math.sqrt((n + 2) / (2 * n + 1)) / math.sqrt(2)
    A1[4, 1] = f4 * n
    A2[4, 1] = -f4

    # Row 5: σ_{n,n+2,2}
    f5 = math.sqrt((n + 2) / (2 * n + 3))
    A1[5, 2] = f5 * (n + 1)
    A2[5, 2] = -f5

    return A1, A2


# Coupling slot → (target_row, source_formula) mapping
# Group g → target stress row;  sub s → source stress row (with na)
_COUP_ROWS = [3, 1, 5, 2, 4]  # σ_{nn2}, σ_{nn-22}, σ_{nn+22}, σ_{nn-12}, σ_{nn+12}


def _coupling_A1_A2(na: int, K_nm: complex, mu_nm: complex, Cp: np.ndarray):
    """Build 6×3 coupling contribution for source mode with degree na.

    Parameters
    ----------
    na : source mode degree
    K_nm : bulk-modulus perturbation amplitude for this rheology mode
    mu_nm : complex-shear-modulus perturbation amplitude
    Cp : (26,) coupling coefficients (slots 0-25)

    Returns (A1c, A2c) each (6, 3).
    """
    A1g, A2g = _a1a2_geometric(na)
    A1c = np.zeros((6, 3), dtype=np.complex128)
    A2c = np.zeros((6, 3), dtype=np.complex128)

    # Slot 0: σ_{n,n,0} ← K_nm weighted
    if Cp[0] != 0:
        A1c[0, :] += K_nm * Cp[0] * A1g[0, :]
        A2c[0, :] += K_nm * Cp[0] * A2g[0, :]

    # Slots 1-25: grouped by target row
    for g in range(5):
        target = _COUP_ROWS[g]
        for s in range(5):
            source = _COUP_ROWS[s]
            slot = 1 + 5 * g + s
            if Cp[slot] != 0:
                A1c[target, :] += 2 * mu_nm * Cp[slot] * A1g[source, :]
                A2c[target, :] += 2 * mu_nm * Cp[slot] * A2g[source, :]

    return A1c, A2c


def build_A1_A2_coupled(
    n_s: np.ndarray,
    muC: complex,
    lam: complex,
    Coup: np.ndarray,
    muC_amp: np.ndarray,
    K_amp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build coupled constitutive matrices A1, A2 (6N×3N).

    Parameters
    ----------
    n_s : (N,) mode degrees
    muC, lam : mean (0,0) complex shear modulus and Lamé lambda
    Coup : (N, N, 27, Nreo) coupling tensor
    muC_amp : (Nreo,) complex shear modulus amplitudes for this layer
    K_amp : (Nreo,) bulk modulus amplitudes for this layer
    """
    N = len(n_s)
    Nreo = Coup.shape[3]
    A1 = np.zeros((6 * N, 3 * N), dtype=np.complex128)
    A2 = np.zeros((6 * N, 3 * N), dtype=np.complex128)

    for i in range(N):
        n = int(n_s[i])
        # Diagonal: standard single-mode A1/A2
        A1d, A2d = build_A1_A2(n, muC, lam)
        A1[6*i:6*i+6, 3*i:3*i+3] = A1d
        A2[6*i:6*i+6, 3*i:3*i+3] = A2d

        # Off-diagonal coupling
        for ireo in range(Nreo):
            km = K_amp[ireo]
            mm = muC_amp[ireo]
            if km == 0 and mm == 0:
                continue

            for j in range(N):
                if Coup[i, j, 26, ireo] == 0:
                    continue
                na = int(n_s[j])
                Cp = Coup[i, j, :26, ireo]
                A1c, A2c = _coupling_A1_A2(na, km, mm, Cp)
                A1[6*i:6*i+6, 3*j:3*j+3] += A1c
                A2[6*i:6*i+6, 3*j:3*j+3] += A2c

    return A1, A2


def build_aprop_coupled(
    r: float,
    g: float,
    dg: float,
    n_s: np.ndarray,
    muC: complex,
    lam: complex,
    rho: float,
    Gg: float,
    Coup: np.ndarray,
    muC_amp: np.ndarray,
    K_amp: np.ndarray,
) -> np.ndarray:
    """Build the 8N×8N coupled propagator matrix.

    Translates ``get_Aprop`` from ``get_solution.m`` lines 1877–1936.

    Parameters
    ----------
    r, g, dg : radius, gravity, dg/dr
    n_s : (N,) mode degrees
    muC, lam, rho, Gg : mean material properties (normalized)
    Coup : (N, N, 27, Nreo) coupling tensor
    muC_amp : (Nreo,) muC amplitudes for this layer
    K_amp : (Nreo,) K amplitudes for this layer
    """
    N = len(n_s)
    N3 = 3 * N
    N6 = 6 * N
    N8 = 8 * N
    N2 = 2 * N

    # --- Coupled sub-matrices (off-diagonal for A1/A2 only) ---
    A1, A2 = build_A1_A2_coupled(n_s, muC, lam, Coup, muC_amp, K_amp)

    # --- Block-diagonal sub-matrices ---
    A3 = np.zeros((N3, N3), dtype=np.complex128)
    A4 = np.zeros((N3, N6), dtype=np.complex128)
    A5 = np.zeros((N3, N6), dtype=np.complex128)
    A13 = np.eye(N3, dtype=np.complex128)
    A6 = np.zeros((N3, N3), dtype=np.complex128)
    A71 = np.zeros((N3, N3), dtype=np.complex128)
    A72 = np.zeros((N3, N3), dtype=np.complex128)
    A81 = np.zeros((N3, N2), dtype=np.complex128)
    A82 = np.zeros((N3, N2), dtype=np.complex128)
    A9 = np.eye(N2, dtype=np.complex128)
    A100 = np.zeros((N2, N2), dtype=np.complex128)
    A101 = np.zeros((N2, N2), dtype=np.complex128)
    A102 = np.zeros((N2, N2), dtype=np.complex128)
    A11 = np.zeros((N2, N3), dtype=np.complex128)
    A12 = np.zeros((N2, N3), dtype=np.complex128)

    for k in range(N):
        n = int(n_s[k])

        A3[3*k:3*k+3, 3*k:3*k+3] = build_A3(n)
        A4[3*k:3*k+3, 6*k:6*k+6] = build_A4(n)
        A5[3*k:3*k+3, 6*k:6*k+6] = build_A5(n)

        a13_k, a6_k, a71_k, a72_k, a81_k, a82_k, a9_k, \
            a100_k, a101_k, a102_k, a11_k, a12_k = build_others(n, rho, Gg)

        A71[3*k:3*k+3, 3*k:3*k+3] = a71_k
        A72[3*k:3*k+3, 3*k:3*k+3] = a72_k
        A81[3*k:3*k+3, 2*k:2*k+2] = a81_k
        A82[3*k:3*k+3, 2*k:2*k+2] = a82_k
        A100[2*k:2*k+2, 2*k:2*k+2] = a100_k
        A101[2*k:2*k+2, 2*k:2*k+2] = a101_k
        A102[2*k:2*k+2, 2*k:2*k+2] = a102_k
        A11[2*k:2*k+2, 3*k:3*k+3] = a11_k
        A12[2*k:2*k+2, 3*k:3*k+3] = a12_k

    # --- Invert A3 block-diagonal ---
    A3_inv = np.zeros_like(A3)
    for k in range(N):
        A3_inv[3*k:3*k+3, 3*k:3*k+3] = np.linalg.inv(
            A3[3*k:3*k+3, 3*k:3*k+3]
        )

    # --- Assemble Adotx, Ax (8N×8N) ---
    Adotx = np.zeros((N8, N8), dtype=np.complex128)
    Ax = np.zeros((N8, N8), dtype=np.complex128)

    # Block 1: constitutive (rows 0:3N)
    # NOTE: A1/A2 role swap — see the same block in build_aprop (scalar) for the
    # rationale.  A2 -> Adotx (d/dr), A1 -> Ax (/r).
    Adotx[:N3, :N3] = A4 @ A2 @ A3_inv
    Ax[:N3, :N3] = -A4 @ A1 @ A3_inv / r
    Ax[:N3, N3:N6] = A13

    # Block 2: momentum (rows 3N:6N)
    Adotx[N3:N6, :N3] = -A5 @ A2 @ A3_inv / r + A6
    Adotx[N3:N6, N3:N6] = A13
    Ax[N3:N6, :N3] = A5 @ A1 @ A3_inv / r**2 + (g / r) * A71 + dg * A72
    Ax[N3:N6, N6:N8] = A81 + A82 / r

    # Block 3: Poisson (rows 6N:8N)
    Adotx[N6:N8, :N3] = -A12
    Adotx[N6:N8, N6:N8] = A9
    Ax[N6:N8, :N3] = A11 / r
    Ax[N6:N8, N6:N8] = A100 + A101 / r + A102 / r**2

    # --- Degree-0 special case ---
    for k in range(N):
        n = int(n_s[k])
        if n == 0:
            # V, W rows → trivial propagation (dV/dr = V/r, dW/dr = W/r)
            for idx in [3*k + 1, 3*k + 2]:  # V and W displacement indices
                Adotx[idx, :] = 0
                Ax[idx, :] = 0
                Adotx[idx, idx] = 1.0
                Ax[idx, idx] = 1.0
            for idx in [N3 + 3*k + 1, N3 + 3*k + 2]:  # S and T stress indices
                Adotx[idx, :] = 0
                Ax[idx, :] = 0
                Adotx[idx, idx] = 1.0
                Ax[idx, idx] = 1.0
            # Poisson first row for this mode: link to radial displacement
            phi_row = N6 + 2 * k
            Ax[phi_row, :] = 0
            Adotx[phi_row, :] = 0
            Adotx[phi_row, phi_row] = 1.0
            Ax[phi_row, 3 * k] = -4 * math.pi * Gg

    # Solve: Aprop = Adotx \ Ax
    Aprop = np.linalg.solve(Adotx, Ax)
    return Aprop


# ---------------------------------------------------------------------------
# Cash-Karp RK5 coefficients
# ---------------------------------------------------------------------------

# Node coefficients
CK_A = [0.0, 1/5, 3/10, 3/5, 1.0, 7/8]

# Butcher tableau
CK_B = [
    [],
    [1/5],
    [3/40, 9/40],
    [3/10, -9/10, 6/5],
    [-11/54, 5/2, -70/27, 35/27],
    [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
]

# 5th-order solution weights
CK_C = [37/378, 0.0, 250/621, 125/594, 0.0, 512/1771]
