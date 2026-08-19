# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from pylov3d.matlab_sph import (
    complex_modes_to_stokes,
    equiangular_grid,
    grid_to_stokes,
    normalized_legendre,
    stokes_to_grid,
)


def test_equiangular_grid_shape():
    lat, lon = equiangular_grid(5)
    assert lat.shape == (10,)
    assert lon.shape == (20,)
    assert np.isclose(lat[0], -81.0)
    assert np.isclose(lat[-1], 81.0)
    assert np.isclose(lon[0], -171.0)
    assert np.isclose(lon[-1], 171.0)


def test_legendre_reference_values():
    P = normalized_legendre(2, np.array([0.0, 0.5]))
    np.testing.assert_allclose(P[0, 0], [1.0, 1.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(P[1, 1, 0], np.sqrt(3.0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(P[1, 0, 1], np.sqrt(3.0) * 0.5, rtol=0, atol=1e-15)


def test_stokes_roundtrip_axisymmetric():
    lmax = 6
    c = np.zeros((lmax + 1, lmax + 1))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = 0.12
    c[4, 0] = -0.03
    _, _, z = stokes_to_grid(c, s, lmax)
    c2, s2 = grid_to_stokes(z, lmax)
    np.testing.assert_allclose(c2[:5, :5], c[:5, :5], rtol=0, atol=2e-12)
    np.testing.assert_allclose(s2, 0.0, rtol=0, atol=2e-12)


def test_stokes_roundtrip_cos_sin():
    lmax = 8
    c = np.zeros((lmax + 1, lmax + 1))
    s = np.zeros_like(c)
    c[2, 2] = 0.17
    s[3, 1] = -0.08
    c[5, 3] = 0.025
    _, _, z = stokes_to_grid(c, s, lmax)
    c2, s2 = grid_to_stokes(z, lmax)
    np.testing.assert_allclose(c2, c, rtol=0, atol=2e-12)
    np.testing.assert_allclose(s2, s, rtol=0, atol=2e-12)


def test_complex_input_conversion_matches_matlab_formula():
    modes = [
        (2, 0, 0.3 + 0j),
        (2, 2, 0.2 - 0.1j),
        (2, -2, 0.2 + 0.1j),
    ]
    c, s = complex_modes_to_stokes(modes, 4)
    np.testing.assert_allclose(c[2, 0], 0.3)
    np.testing.assert_allclose(c[2, 2], np.sqrt(2.0) * 0.2)
    np.testing.assert_allclose(s[2, 2], np.sqrt(2.0) * 0.1)
