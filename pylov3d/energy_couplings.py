# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Energy coupling coefficients for multi-mode tidal dissipation.

Translates ``get_energy_couplings.m``.  The energy coupling tensor
``EC[i1, i2, i3, i4, i5]`` relates pairs of solution modes (i1, i2)
to energy spectrum modes (i3) via products of Wigner 3j, 6j, and 9j
symbols.  Indices i4, i5 correspond to the six second-rank tensor
components (n2 = n−2 … n+2 plus the isotropic n).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .wigner import wigner3j, wigner6j, wigner9j


class EnergyCouplings(NamedTuple):
    """Energy coupling coefficients.

    Attributes
    ----------
    EC : (N_sol, N_sol, N_en_nonzero, 6, 6) float
        Coupling tensor.
    n_s : (N_sol,) int
        Solution mode degrees.
    m_s : (N_sol,) int
        Solution mode orders.
    n_en : (N_en_nonzero,) int
        Energy spectrum degrees (only non-zero entries).
    m_en : (N_en_nonzero,) int
        Energy spectrum orders.
    """
    EC: np.ndarray
    n_s: np.ndarray
    m_s: np.ndarray
    n_en: np.ndarray
    m_en: np.ndarray


# ---------------------------------------------------------------------------
# Core coupling coefficient
# ---------------------------------------------------------------------------

def _couplings_coefficient(
    na, na2, la, ma, nb, nb2, lb, mb, nc, mc, na1, nb1,
) -> np.ndarray:
    """Vectorized Wigner product for energy coupling.

    All inputs are int arrays of equal length (subset of the 324-element
    quantum number vectors). Returns an array whose length is ``len(na) / 9``.

    MATLAB's ``reshape(Caux, 9, [])`` is column-major. NumPy's default reshape
    is row-major, so the translation must use ``order='F'`` here. Otherwise
    the wrong groups of nine Wigner-product terms are summed into each tensor
    coefficient even though the individual Wigner values are correct.
    """
    ones = np.ones_like(na)
    zeros = np.zeros_like(na)

    # Use np.maximum to avoid NaN from sqrt of negative quantum numbers.
    # Invalid entries are zeroed by the Wigner symbols; in MATLAB sqrt(-x)
    # gives a complex number that is harmlessly multiplied by zero.
    def _sq(x):
        return np.sqrt(np.maximum(x, 0.0))

    Lam_a = _sq((2 * la + 1.0) * (2 * na1 + 1.0))
    Lam_b = _sq((2 * lb + 1.0) * (2 * nb1 + 1.0))

    CC = (
        (-1.0) ** (mc + nb + nb2)
        * _sq((2 * na2 + 1.0) * (2 * na1 + 1.0) * (2 * na + 1.0))
        * _sq((2 * nb2 + 1.0) * (2 * nb1 + 1.0) * (2 * nb + 1.0))
        * _sq(2 * nc + 1.0)
        * wigner3j(na2, nb2, nc, zeros, zeros, zeros)
        * wigner3j(na, nb, nc, ma, mb, -mc)
        * wigner9j(na, na1, ones, nc, na2, nb2, nb, ones, nb1)
    )

    C_aux = (
        (-1.0) ** (na + na2 + la + nb + nb2 + lb)
        * Lam_a * Lam_b
        * wigner6j(ones, la, ones, na, na1, na2)
        * wigner6j(ones, lb, ones, nb, nb1, nb2)
        * CC
    )

    # MATLAB: C_int = reshape(Caux,9,[]); C = sum(C_int);
    # Preserve MATLAB/Fortran column-major grouping explicitly.
    C_int = C_aux.reshape(9, -1, order="F")
    return C_int.sum(axis=0)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_energy_couplings(
    n_sol: np.ndarray,
    m_sol: np.ndarray,
    Nenergy: int = 8,
) -> EnergyCouplings:
    """Build energy coupling tensor for all solution × energy mode pairs.

    Parameters
    ----------
    n_sol, m_sol : (N_sol,) int
        Degrees and orders of solution modes.
    Nenergy : int
        Maximum degree of energy spectrum expansion (default 8).

    Returns
    -------
    EnergyCouplings
        Tensor trimmed to non-zero energy modes.
    """
    n_sol = np.asarray(n_sol, dtype=int)
    m_sol = np.asarray(m_sol, dtype=int)
    N_sol = len(n_sol)

    # Build energy mode list: all (n, m) for 0 ≤ n ≤ Nenergy, −n ≤ m ≤ n
    n_en_list, m_en_list = [], []
    for n in range(Nenergy + 1):
        for m in range(-n, n + 1):
            n_en_list.append(n)
            m_en_list.append(m)
    n_en = np.array(n_en_list, dtype=int)
    m_en = np.array(m_en_list, dtype=int)
    N_en = len(n_en)

    # la, lb vectors for the 6 second-rank tensor components
    la = np.array([2, 2, 2, 2, 2, 0])
    lb = np.array([2, 2, 2, 2, 2, 0])
    # Full 324-element vectors (6 × 6 × 9 = 324)
    la_full = np.repeat(la, 54)
    lb_full = np.tile(np.repeat(lb, 9), 6)

    EC = np.zeros((N_sol, N_sol, N_en, 6, 6))

    for i1 in range(N_sol):
        na_val = int(n_sol[i1])
        ma_val = int(m_sol[i1])

        for i2 in range(N_sol):
            nb_val = int(n_sol[i2])
            mb_val = int(m_sol[i2])

            n_lo = abs(na_val - nb_val)
            n_hi = na_val + nb_val
            m_target = ma_val + mb_val

            mask = (n_en >= n_lo) & (n_en <= n_hi) & (m_en == m_target)
            ind_modes = np.where(mask)[0]
            if len(ind_modes) == 0:
                continue

            for idx in ind_modes:
                nc_val = int(n_en[idx])
                mc_val = int(m_en[idx])

                na2_6 = np.array([na_val - 2, na_val - 1, na_val,
                                  na_val + 1, na_val + 2, na_val])
                nb2_6 = np.array([nb_val - 2, nb_val - 1, nb_val,
                                  nb_val + 1, nb_val + 2, nb_val])
                na2_full = np.repeat(na2_6, 54)
                nb2_full = np.tile(np.repeat(nb2_6, 9), 6)
                nc_full = np.full(324, nc_val)

                nz_mask = (
                    ((na2_full + nb2_full + nc_full) % 2 == 0)
                    & (nb2_full >= np.abs(na2_full - nc_full))
                    & (nb2_full <= na2_full + nc_full)
                )
                nz_idx = np.where(nz_mask)[0]
                if len(nz_idx) == 0:
                    continue

                n_nz = len(nz_idx)

                i4_vec = nz_idx // 54
                i5_vec = (nz_idx // 9) % 6
                i4_out = i4_vec[::9]
                i5_out = i5_vec[::9]

                na_arr = np.full(n_nz, na_val)
                ma_arr = np.full(n_nz, ma_val)
                nb_arr = np.full(n_nz, nb_val)
                mb_arr = np.full(n_nz, mb_val)
                mc_arr = np.full(n_nz, mc_val)
                nc_arr = nc_full[nz_idx]
                na2_arr = na2_full[nz_idx]
                nb2_arr = nb2_full[nz_idx]
                la_arr = la_full[nz_idx]
                lb_arr = lb_full[nz_idx]

                na1v = np.array([na_val - 1, na_val, na_val + 1])
                nb1v = np.array([nb_val - 1, nb_val, nb_val + 1])
                rep_na1 = n_nz // 9
                rep_nb1 = n_nz // 3
                na1_arr = np.tile(np.repeat(na1v, 3), rep_na1)
                nb1_arr = np.tile(nb1v, rep_nb1)

                ec_vec = _couplings_coefficient(
                    na_arr, na2_arr, la_arr, ma_arr,
                    nb_arr, nb2_arr, lb_arr, mb_arr,
                    nc_arr, mc_arr, na1_arr, nb1_arr,
                )

                # MATLAB uses column-major sub2ind/reshape; this Python pair of
                # row-major linear indexing plus row-major reshape places the
                # value at the same logical [i4,i5] matrix coordinate.
                store = np.zeros(36)
                linear_idx = i4_out * 6 + i5_out
                store[linear_idx] = ec_vec
                EC[i1, i2, idx, :, :] = store.reshape(6, 6)

    nz_en = []
    for i in range(N_en):
        if np.any(np.abs(EC[:, :, i, :, :]) > 0):
            nz_en.append(i)

    if len(nz_en) == 0:
        return EnergyCouplings(
            EC=np.zeros((N_sol, N_sol, 0, 6, 6)),
            n_s=n_sol, m_s=m_sol,
            n_en=np.array([], dtype=int), m_en=np.array([], dtype=int),
        )

    nz_en = np.array(nz_en)
    return EnergyCouplings(
        EC=EC[:, :, nz_en, :, :],
        n_s=n_sol, m_s=m_sol,
        n_en=n_en[nz_en], m_en=m_en[nz_en],
    )
