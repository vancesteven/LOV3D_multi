"""Propagator matrix builders — translated from get_solution.m subfunctions.

Builds the 8×8 propagator matrix ``Aprop`` used in the radial ODE
``dy/dr = Aprop(r) · y``.  The state vector (per mode) is stored internally
as ``[u_GSH(3), Sigma_SH(3), Phi(2)]`` with the rearrangement to
``[U, V, W, R, S, T, Phi, dPhi/dr]`` done after integration.

For the 1D (spherically symmetric) case with a single mode the system is
8×8.  Toroidal components (W, T) decouple from the spheroidal ones but are
still propagated for completeness.
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
    Adotx[:N3, :N3] = A4 @ A1 @ A3_inv
    Ax[:N3, :N3] = -A4 @ A2 @ A3_inv / r
    Ax[:N3, N3:N6] = A13

    # Block 2: momentum (rows 3:6)
    Adotx[N3:N6, :N3] = -A5 @ A1 @ A3_inv / r + A6
    Adotx[N3:N6, N3:N6] = A13
    Ax[N3:N6, :N3] = A5 @ A2 @ A3_inv / r**2 + g / r * A71 + dg * A72
    Ax[N3:N6, N6:N8] = A81 + A82 / r

    # Block 3: Poisson (rows 6:8)
    Adotx[N6:N8, :N3] = -A12
    Adotx[N6:N8, N6:N8] = A9
    Ax[N6:N8, :N3] = A11 / r
    Ax[N6:N8, N6:N8] = A100 + A101 / r + A102 / r**2

    # Solve: Aprop = Adotx \ Ax
    Aprop = np.linalg.solve(Adotx, Ax)

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
