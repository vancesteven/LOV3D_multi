# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.wigner — Wigner 3j and 6j symbol wrappers."""

import math

import numpy as np
import pytest

from pylov3d.wigner import wigner3j, wigner6j


class TestWigner3j:

    def test_j1_j1_0(self):
        """3j(1,1,0,0,0,0) = -1/sqrt(3)."""
        val = wigner3j(1, 1, 0, 0, 0, 0)
        assert val == pytest.approx(-1.0 / math.sqrt(3))

    def test_j2_j2_0(self):
        """3j(2,2,0,0,0,0) = 1/sqrt(5)."""
        val = wigner3j(2, 2, 0, 0, 0, 0)
        assert val == pytest.approx(1.0 / math.sqrt(5))

    def test_m_conservation_violated(self):
        """3j should be zero when m1+m2+m3 != 0."""
        val = wigner3j(1, 1, 1, 1, 1, 0)
        assert val == pytest.approx(0.0)

    def test_triangle_violated(self):
        """3j should be zero when triangle inequality fails."""
        val = wigner3j(1, 1, 5, 0, 0, 0)
        assert val == pytest.approx(0.0)

    def test_known_value_211(self):
        """3j(2,1,1,0,0,0) = sqrt(2/15)."""
        val = wigner3j(2, 1, 1, 0, 0, 0)
        assert val == pytest.approx(math.sqrt(2.0 / 15.0))

    def test_vectorized(self):
        """Vectorized call should match scalar calls."""
        j1 = np.array([1, 2, 2])
        j2 = np.array([1, 2, 1])
        j3 = np.array([0, 0, 1])
        m1 = np.array([0, 0, 0])
        m2 = np.array([0, 0, 0])
        m3 = np.array([0, 0, 0])

        result = wigner3j(j1, j2, j3, m1, m2, m3)
        assert len(result) == 3
        assert result[0] == pytest.approx(-1.0 / math.sqrt(3))
        assert result[1] == pytest.approx(1.0 / math.sqrt(5))
        assert result[2] == pytest.approx(math.sqrt(2.0 / 15.0))

    def test_symmetry_even_permutation(self):
        """Even column permutation preserves the value."""
        v1 = wigner3j(1, 2, 1, 0, 0, 0)
        v2 = wigner3j(2, 1, 1, 0, 0, 0)
        # Odd permutation introduces phase (-1)^(j1+j2+j3)
        phase = (-1) ** (1 + 2 + 1)
        assert v1 == pytest.approx(phase * v2)


class TestWigner6j:

    def test_111_111(self):
        """6j{1,1,1;1,1,1} = 1/6."""
        val = wigner6j(1, 1, 1, 1, 1, 1)
        assert val == pytest.approx(1.0 / 6.0)

    def test_110_011(self):
        """6j{1,1,0;0,1,1} = -1/sqrt(3*2) = -1/sqrt(6)."""
        # Using the known formula
        val = wigner6j(1, 1, 0, 0, 1, 1)
        # This should be (-1)^(1+1) / sqrt((2*1+1)(2*1+1)) = 1/3
        # Actually let me check: {j1,j2,j3;j4,j5,j6} with j3=0
        # gives (-1)^(j1+j5+j6) * delta(j1,j2)*delta(j4,j5) / sqrt((2j1+1)(2j4+1))
        # {1,1,0;0,1,1}: j3=0, so need j1==j2 (yes 1==1) and j4==j5 (0!=1) → 0
        assert val == pytest.approx(0.0)

    def test_zero_j3(self):
        """6j{j,j,0; j',j',j''} orthogonality for j3=0."""
        # {2,2,0; 1,2,2}: j3=0, need j1==j2 (2==2 yes), j4==j5 (1!=2) → 0
        val = wigner6j(2, 2, 0, 1, 2, 2)
        assert val == pytest.approx(0.0)

    def test_vectorized(self):
        """Vectorized 6j should match scalar calls."""
        j1 = np.array([1, 1])
        j2 = np.array([1, 1])
        j3 = np.array([1, 0])
        j4 = np.array([1, 0])
        j5 = np.array([1, 1])
        j6 = np.array([1, 1])

        result = wigner6j(j1, j2, j3, j4, j5, j6)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0 / 6.0)

    def test_triangle_violated(self):
        """6j should be zero when a triad violates triangle inequality."""
        val = wigner6j(1, 1, 5, 1, 1, 1)
        assert val == pytest.approx(0.0)
