# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Wigner 3j, 6j, and 9j symbol computation — wraps py3nj.

py3nj uses the integer ``2j`` convention: all angular momentum arguments
are passed as ``2*j``.  The wrapper functions here accept *physical* j
values (integer or half-integer) and handle the conversion internally.

For the coupling coefficient calculation we need vectorised evaluation
of 3j and 6j symbols over arrays of quantum numbers.
"""

from __future__ import annotations

import numpy as np
import py3nj


def wigner3j(
    j1: int | np.ndarray,
    j2: int | np.ndarray,
    j3: int | np.ndarray,
    m1: int | np.ndarray,
    m2: int | np.ndarray,
    m3: int | np.ndarray,
) -> float | np.ndarray:
    r"""Wigner 3j symbol.

    .. math::
        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

    Parameters are physical angular-momentum values (integers for our use
    case).  Accepts scalars or arrays of equal length.

    Returns 0 when selection rules are violated (triangle inequality,
    m₁+m₂+m₃=0) or when any j value is negative.
    """
    j1 = np.asarray(j1, dtype=np.int64)
    j2 = np.asarray(j2, dtype=np.int64)
    j3 = np.asarray(j3, dtype=np.int64)
    m1 = np.asarray(m1, dtype=np.int64)
    m2 = np.asarray(m2, dtype=np.int64)
    m3 = np.asarray(m3, dtype=np.int64)

    scalar = j1.ndim == 0
    j1, j2, j3 = np.atleast_1d(j1), np.atleast_1d(j2), np.atleast_1d(j3)
    m1, m2, m3 = np.atleast_1d(m1), np.atleast_1d(m2), np.atleast_1d(m3)

    valid = (j1 >= 0) & (j2 >= 0) & (j3 >= 0)
    result = np.zeros(len(j1), dtype=np.float64)

    if np.any(valid):
        result[valid] = py3nj.wigner3j(
            2 * j1[valid], 2 * j2[valid], 2 * j3[valid],
            2 * m1[valid], 2 * m2[valid], 2 * m3[valid],
        )

    return float(result[0]) if scalar else result


def wigner6j(
    j1: int | np.ndarray,
    j2: int | np.ndarray,
    j3: int | np.ndarray,
    j4: int | np.ndarray,
    j5: int | np.ndarray,
    j6: int | np.ndarray,
) -> float | np.ndarray:
    r"""Wigner 6j symbol.

    .. math::
        \begin{Bmatrix} j_1 & j_2 & j_3 \\ j_4 & j_5 & j_6 \end{Bmatrix}

    Parameters are physical angular-momentum values.
    Accepts scalars or arrays of equal length.

    Returns 0 when any j value is negative.
    """
    j1 = np.asarray(j1, dtype=np.int64)
    j2 = np.asarray(j2, dtype=np.int64)
    j3 = np.asarray(j3, dtype=np.int64)
    j4 = np.asarray(j4, dtype=np.int64)
    j5 = np.asarray(j5, dtype=np.int64)
    j6 = np.asarray(j6, dtype=np.int64)

    scalar = j1.ndim == 0
    j1, j2, j3 = np.atleast_1d(j1), np.atleast_1d(j2), np.atleast_1d(j3)
    j4, j5, j6 = np.atleast_1d(j4), np.atleast_1d(j5), np.atleast_1d(j6)

    valid = (j1 >= 0) & (j2 >= 0) & (j3 >= 0) & (j4 >= 0) & (j5 >= 0) & (j6 >= 0)
    result = np.zeros(len(j1), dtype=np.float64)

    if np.any(valid):
        result[valid] = py3nj.wigner6j(
            2 * j1[valid], 2 * j2[valid], 2 * j3[valid],
            2 * j4[valid], 2 * j5[valid], 2 * j6[valid],
        )

    return float(result[0]) if scalar else result


def wigner9j(
    j1: int | np.ndarray, j2: int | np.ndarray, j3: int | np.ndarray,
    j4: int | np.ndarray, j5: int | np.ndarray, j6: int | np.ndarray,
    j7: int | np.ndarray, j8: int | np.ndarray, j9: int | np.ndarray,
) -> float | np.ndarray:
    r"""Wigner 9j symbol.

    .. math::
        \begin{Bmatrix} j_1 & j_2 & j_3 \\ j_4 & j_5 & j_6 \\
                         j_7 & j_8 & j_9 \end{Bmatrix}

    Parameters are physical angular-momentum values.
    Accepts scalars or arrays of equal length.

    Returns 0 when any j value is negative.
    """
    args = [np.asarray(x, dtype=np.int64) for x in
            (j1, j2, j3, j4, j5, j6, j7, j8, j9)]
    scalar = args[0].ndim == 0
    args = [np.atleast_1d(a) for a in args]

    valid = args[0] >= 0
    for a in args[1:]:
        valid &= a >= 0
    result = np.zeros(len(args[0]), dtype=np.float64)

    if np.any(valid):
        idx = valid
        result[idx] = py3nj.wigner9j(*(2 * a[idx] for a in args))

    return float(result[0]) if scalar else result
