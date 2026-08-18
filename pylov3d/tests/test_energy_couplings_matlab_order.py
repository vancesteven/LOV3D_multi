# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Regression for MATLAB column-major grouping in energy couplings.

MATLAB ``reshape(Caux, 9, [])`` groups consecutive blocks in Fortran/
column-major order.  NumPy's default C-order reshape produces different
nine-term contractions and previously corrupted the radial energy profile
while leaving the propagated fields unchanged.
"""

import numpy as np

import pylov3d.energy_couplings as ecmod


def test_couplings_coefficient_uses_matlab_column_major_grouping(monkeypatch):
    """Pin the reshape semantics independently of Wigner-symbol numerics."""

    # Isolate the reshape/grouping behavior by replacing each Wigner symbol
    # with unity.  The remaining prefactors vary across the 324 entries, so
    # C- and Fortran-order grouping produce measurably different results.
    monkeypatch.setattr(ecmod, "wigner3j", lambda *args: np.ones_like(np.asarray(args[0]), dtype=float))
    monkeypatch.setattr(ecmod, "wigner6j", lambda *args: np.ones_like(np.asarray(args[0]), dtype=float))
    monkeypatch.setattr(ecmod, "wigner9j", lambda *args: np.ones_like(np.asarray(args[0]), dtype=float))

    # Keep all quantum-number prefactors positive and integer-valued while
    # varying them enough to distinguish the two memory-order conventions.
    j = np.arange(324)
    na = 2 + (j // 108) % 2
    nb = 3 + (j // 81) % 2
    na2 = 2 + (j // 54) % 3
    nb2 = 2 + (j // 27) % 3
    la = 1 + (j // 18) % 2
    lb = 1 + (j // 9) % 2
    ma = np.zeros(324, dtype=int)
    mb = np.zeros(324, dtype=int)
    nc = 2 + (j // 36) % 2
    mc = np.zeros(324, dtype=int)
    na1 = 2 + (j // 3) % 3
    nb1 = 2 + j % 3

    got = ecmod._couplings_coefficient(
        na, na2, la, ma,
        nb, nb2, lb, mb,
        nc, mc, na1, nb1,
    )

    sq = np.sqrt
    lam_a = sq((2 * la + 1.0) * (2 * na1 + 1.0))
    lam_b = sq((2 * lb + 1.0) * (2 * nb1 + 1.0))
    caux = (
        (-1.0) ** (mc + nb + nb2)
        * sq((2 * na2 + 1.0) * (2 * na1 + 1.0) * (2 * na + 1.0))
        * sq((2 * nb2 + 1.0) * (2 * nb1 + 1.0) * (2 * nb + 1.0))
        * sq(2 * nc + 1.0)
        * (-1.0) ** (na + na2 + la + nb + nb2 + lb)
        * lam_a
        * lam_b
    )

    expected_matlab = caux.reshape(9, -1, order="F").sum(axis=0)
    wrong_c_order = caux.reshape(9, -1, order="C").sum(axis=0)

    np.testing.assert_allclose(got, expected_matlab, rtol=0.0, atol=0.0)
    assert not np.allclose(got, wrong_c_order)
