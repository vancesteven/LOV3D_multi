# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Coupling coefficient computation for lateral variations.

Translates ``get_couplings.m``, ``get_active_modes.m``, and
``next_coupling.m`` from the MATLAB LOV3D codebase.

When material properties vary laterally (as spherical-harmonic expansions),
different harmonic modes become coupled through angular integrals involving
Wigner 3j and 6j symbols.  This module computes:

1. Which modes are excited (``get_active_modes``)
2. The 27 coupling coefficients per (equation-mode, source-mode, rheology-mode)
   triplet (``coupling_coefficients``, ``get_couplings``)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .wigner import wigner3j, wigner6j


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class Couplings(NamedTuple):
    """Coupling coefficients for the laterally heterogeneous problem.

    Attributes
    ----------
    n_s : (Nsol,) int — degrees of solution modes
    m_s : (Nsol,) int — orders of solution modes
    order : (Nsol,) int — perturbation order at which each mode first appears
    Coup : (Nsol, Nsol, 27, Nreo) float — coupling tensor.
        Coup[i, j, k, ireo] is the k-th coupling coefficient between
        equation-mode i and source-mode j for rheology-mode ireo.
        Slot 26 (index 26, 1-indexed: 27) is a sparsity flag:
        1 if any of slots 0-25 are nonzero, 0 otherwise.
    """
    n_s: np.ndarray
    m_s: np.ndarray
    order: np.ndarray
    Coup: np.ndarray


# ---------------------------------------------------------------------------
# Mode selection (next_coupling + get_active_modes)
# ---------------------------------------------------------------------------

def next_coupling(
    n0: int, m0: int, st: int, n1: int, m1: int, order: int,
) -> list[tuple[int, int, int, int]]:
    """Compute modes activated by coupling (n0,m0) with rheology (n1,m1).

    Translates ``next_coupling.m``.

    Parameters
    ----------
    n0, m0 : degree/order of the input mode
    st : 1=spheroidal, 2=toroidal
    n1, m1 : degree/order of the rheology mode
    order : perturbation order label

    Returns
    -------
    List of (n_new, m_new, st_new, order) tuples.
    """
    modes = []
    m_new = m0 + m1

    # Even-parity couplings: |n0-n1|, |n0-n1|+2, ..., n0+n1
    num1 = (n0 + n1 - abs(n0 - n1)) // 2
    for j in range(num1 + 1):
        n_new = abs(n0 - n1) + 2 * j
        st_new = st  # same parity
        if abs(m_new) < n_new + 1:
            modes.append((n_new, m_new, st_new, order))

    # Odd-parity couplings: |n0-n1|+1, |n0-n1|+3, ...
    num2 = (n0 + n1 - abs(n0 - n1)) // 2 - 1
    for j in range(max(0, num2 + 1)):
        n_new = abs(n0 - n1) + 2 * j + 1
        st_new = 2 if st == 1 else 1  # parity flip
        if abs(m_new) < n_new + 1 and (abs(m_new) + abs(m0) + abs(m1)) != 0:
            modes.append((n_new, m_new, st_new, order))

    return modes


def get_active_modes(
    perturbation_order: int,
    variations: np.ndarray,
    forcing_n: int,
    forcing_m: int,
) -> np.ndarray:
    """Compute all modes excited by lateral variations up to a given order.

    Translates ``get_active_modes.m``.

    Parameters
    ----------
    perturbation_order : int
        Maximum coupling order (default 2).
    variations : (Nreo, 2) array
        Unique rheology (degree, order) pairs across all layers.
    forcing_n, forcing_m : int
        Forcing degree and order.

    Returns
    -------
    active_modes : (Nsol, 3) int array
        Columns: [degree, order, min_perturbation_order].
    """
    # Unique rheology (n, m) pairs
    rheo = np.unique(variations[:, :2].astype(int), axis=0)

    # Start with forcing mode: (n, m, ST=1 spheroidal, order=0)
    # modes list: [(n, m, ST, order), ...]
    all_modes = [(forcing_n, forcing_m, 1, 0)]

    # Iteratively discover new modes order by order
    for current_order in range(1, perturbation_order + 1):
        # Find modes that were added at the previous order
        prev_order_modes = [m for m in all_modes if m[3] == current_order - 1]

        new_modes = []
        for n0, m0, st, _ in prev_order_modes:
            for rh in rheo:
                n1, m1 = int(rh[0]), int(rh[1])
                coupled = next_coupling(n0, m0, st, n1, m1, current_order)
                new_modes.extend(coupled)

        # Add only new (n, m) pairs
        existing_nm = {(m[0], m[1]) for m in all_modes}
        for nm in new_modes:
            if (nm[0], nm[1]) not in existing_nm:
                all_modes.append(nm)
                existing_nm.add((nm[0], nm[1]))

    # Build output: unique (n, m) with minimum order
    nm_to_order = {}
    for n, m, st, order in all_modes:
        key = (n, m)
        if key not in nm_to_order or order < nm_to_order[key]:
            nm_to_order[key] = order

    result = np.array(
        [(n, m, o) for (n, m), o in sorted(nm_to_order.items())],
        dtype=int,
    )
    return result


# ---------------------------------------------------------------------------
# Coupling coefficient computation
# ---------------------------------------------------------------------------

def _coupling_Y(
    nc: np.ndarray, mc: np.ndarray,
    nc1: np.ndarray, nc2: np.ndarray,
    na: np.ndarray, ma: np.ndarray,
    na1: np.ndarray, na2: np.ndarray,
    nb: np.ndarray, mb: np.ndarray,
) -> np.ndarray:
    """Angular coupling via Wigner 6j × 6j × 3j × 3j.

    Translates ``couplingY`` in ``get_couplings.m`` lines 238-250.
    All inputs are length-234 integer arrays.
    """
    ones = np.ones(len(nc), dtype=int)
    zeros = np.zeros(len(nc), dtype=int)

    Lam_arg = (
        (2 * nc2 + 1).astype(float) * (2 * nc1 + 1) * (2 * nc + 1)
        * (2 * na2 + 1) * (2 * na1 + 1) * (2 * na + 1)
        * (2 * nb + 1)
    )
    Lam = np.sqrt(np.maximum(Lam_arg, 0.0))

    phase = (-1.0) ** (na + na1 + nc + nc1 + mc)

    w6j_1 = wigner6j(na, na1, ones, nc1, nc, nb)
    w6j_2 = wigner6j(na1, na2, ones, nc2, nc1, nb)
    w3j_1 = wigner3j(na2, nc2, nb, zeros, zeros, zeros)
    w3j_2 = wigner3j(na, nc, nb, ma, -mc, mb)

    return np.nan_to_num(phase * Lam * w6j_1 * w6j_2 * w3j_1 * w3j_2, nan=0.0)


def _coupling_T(
    nc: np.ndarray, nc2: np.ndarray,
    lc: np.ndarray, mc: np.ndarray,
    na: np.ndarray, na2: np.ndarray,
    la: np.ndarray, ma: np.ndarray,
    nb: np.ndarray, mb: np.ndarray,
    nc1: np.ndarray, na1: np.ndarray,
) -> np.ndarray:
    """Tensor coupling: sum over 9 intermediate terms per coupling slot.

    Translates ``couplingT`` in ``get_couplings.m`` lines 224-236.
    All inputs are length-234 integer arrays.

    Returns
    -------
    (26,) array — one value per coupling slot (summed over 9 internal terms).
    """
    ones = np.ones(234, dtype=int)

    Lam_a = np.sqrt(np.maximum((2 * la + 1).astype(float) * (2 * na1 + 1), 0.0))
    Lam_c = np.sqrt(np.maximum((2 * lc + 1).astype(float) * (2 * nc1 + 1), 0.0))

    phase = (-1.0) ** (na + na2 + la + nc + nc2 + lc)

    w6j_1 = wigner6j(ones, la, ones, na, na1, na2)
    w6j_2 = wigner6j(ones, lc, ones, nc, nc1, nc2)

    C_Y = _coupling_Y(nc, mc, nc1, nc2, na, ma, na1, na2, nb, mb)

    C_aux = np.nan_to_num(phase * Lam_a * Lam_c * w6j_1 * w6j_2 * C_Y, nan=0.0)

    # Reshape to (9, 26) and sum over 9 — gives 26 coupling slots
    C_int = C_aux.reshape(9, 26, order='F')
    return C_int.sum(axis=0)


def coupling_coefficients(
    n: int, m: int, na: int, ma: int, nb: int, mb: int,
) -> np.ndarray:
    """Compute 27 coupling coefficients for mode triple (n,m)-(na,ma)-(nb,mb).

    Translates ``coupling_coefficients`` in ``get_couplings.m`` lines 186-222.

    Parameters
    ----------
    n, m : degree/order of the equation mode
    na, ma : degree/order of the source mode
    nb, mb : degree/order of the rheology mode

    Returns
    -------
    (27,) float array.  Slots 0-25 are the 26 coupling coefficients;
    slot 26 is 1 if any are nonzero, 0 otherwise.
    """
    Cp = np.zeros(27)

    # Build the 234-element vectorised quantum-number arrays
    # nc1 pattern: [n-1, n, n+1] repeated 78 times = 234
    nc1v = np.array([n - 1, n, n + 1])
    nc1 = np.tile(nc1v, 78)

    # na1 pattern: each of [na-1, na, na+1] repeated 3 times, then tiled 26 times
    na1v = np.repeat(np.array([na - 1, na, na + 1]), 3)
    na1 = np.tile(na1v, 26)

    # lc, la: [0, then 25×2] → each repeated 9 times = 234
    lc_base = np.concatenate(([0], np.full(25, 2)))
    lc = np.repeat(lc_base, 9)
    la = np.repeat(lc_base, 9)

    # nc2: [n, then 5 groups of {n, n-2, n+2, n-1, n+1} each repeated 5 times]
    nc2_base = np.array([n, n - 2, n + 2, n - 1, n + 1])
    nc2_inner = np.concatenate(([n], np.repeat(nc2_base, 5)))  # length 26
    nc2 = np.repeat(nc2_inner, 9)

    # na2: [na, then {na, na-2, na+2, na-1, na+1} tiled 5 times]
    na2_base = np.array([na, na - 2, na + 2, na - 1, na + 1])
    na2_inner = np.concatenate(([na], np.tile(na2_base, 5)))  # length 26
    na2 = np.repeat(na2_inner, 9)

    # Fixed arrays (all 234 elements)
    nc_vec = np.full(234, n)
    mc_vec = np.full(234, m)
    na_vec = np.full(234, na)
    ma_vec = np.full(234, ma)
    nb_vec = np.full(234, nb)
    mb_vec = np.full(234, mb)

    Cp[:26] = _coupling_T(
        nc_vec, nc2, lc, mc_vec,
        na_vec, na2, la, ma_vec,
        nb_vec, mb_vec,
        nc1, na1,
    )

    Cp[26] = 1.0 if np.max(np.abs(Cp[:26])) > 0 else 0.0
    return Cp


def get_couplings(
    variations: np.ndarray,
    forcing_n: int,
    forcing_m: int,
    perturbation_order: int = 2,
) -> Couplings:
    """Compute the full coupling tensor for a set of rheology variations.

    Translates ``get_couplings`` in ``get_couplings.m`` (lines 82-183).

    Parameters
    ----------
    variations : (Nreo, 2) array
        Unique rheology (degree, order) pairs.
    forcing_n, forcing_m : int
        Forcing harmonic degree and order.
    perturbation_order : int
        Maximum perturbation order (default 2).

    Returns
    -------
    Couplings
        Contains active-mode lists and the 4D coupling tensor.
    """
    Nreo = len(variations)

    # Get active modes
    active = get_active_modes(
        perturbation_order, variations, forcing_n, forcing_m,
    )
    n_s = active[:, 0]
    m_s = active[:, 1]
    orders = active[:, 2]
    Nsol = len(n_s)

    nb_arr = variations[:, 0].astype(int)
    mb_arr = variations[:, 1].astype(int)

    Cp = np.zeros((Nsol, Nsol, 27, Nreo))

    for ireo in range(Nreo):
        nb = int(nb_arr[ireo])
        mb = int(mb_arr[ireo])

        for imod1 in range(Nsol):
            n_eq = int(n_s[imod1])
            m_eq = int(m_s[imod1])

            for imod2 in range(Nsol):
                na = int(n_s[imod2])
                ma = int(m_s[imod2])

                # Selection rules: triangle inequality + m-conservation
                if na >= abs(n_eq - nb) and na <= n_eq + nb and ma == m_eq - mb:
                    Cp[imod1, imod2, :, ireo] = coupling_coefficients(
                        n_eq, m_eq, na, ma, nb, mb,
                    )

    return Couplings(
        n_s=n_s,
        m_s=m_s,
        order=orders,
        Coup=Cp,
    )
