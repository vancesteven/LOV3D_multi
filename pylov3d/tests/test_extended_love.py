# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.extended_love (Berne et al. 2026 item B1).

Fast lane: a two-layer elastic model with a fluid core. The core radius
is 0.5 R: a tiny core makes the degree-6 coupled modes ill-conditioned,
since r^l spans (R/r_core)^l. The Enceladus MATLAB-validated path is
also checked.
Slow lane: the Mars TASK-016 model against the native-MATLAB full coupled
spectra for forcings (2,0), (2,1) and (2,2)
(``data/tests/mars/mars_lateral_cross_check.mat``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from pylov3d.extended_love import (
    complex_from_real_matrix,
    complex_modes_for_degrees,
    complex_to_real_coefficients,
    extended_love_tensor,
    real_labels_for_degrees,
    to_real_basis,
)
from pylov3d.love import get_love
from pylov3d.mapping import fully_normalized_legendre
from pylov3d.mars_lateral import _real_sh_to_complex_mu_variable
from pylov3d.types import make_forcing, make_interior_model, make_numerics

TD = 86400.0
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model():
    return make_interior_model(
        R0_km=[500.0, 1000.0], rho0=[5000.0, 3000.0], mu0=[0.0, 1e10],
    )


@pytest.fixture(scope="module")
def numerics():
    return make_numerics(n_layers=2, method="variable", Nrbase=50,
                         perturbation_order=2)


@pytest.fixture(scope="module")
def k2_1d(model, numerics):
    love, _, _ = get_love(model, make_forcing(TD, 2, 0, 1.0), numerics)
    return complex(love.k[0])


# Non-axisymmetric, real (conjugate-paired) structure with a sine part.
MIXED_EPS = 0.02
MIXED_MU = {1: [
    (1, 1, MIXED_EPS * (1 - 0.5j)),
    (1, -1, -MIXED_EPS * (1 + 0.5j)),
    (2, 0, MIXED_EPS),
]}


@pytest.fixture(scope="module")
def mixed_tensor(model, numerics):
    return extended_love_tensor(model, numerics, TD, mu_variable=MIXED_MU)


# ---------------------------------------------------------------------------
# (a) zero lateral amplitude
# ---------------------------------------------------------------------------

class TestZeroAmplitude:

    @pytest.mark.parametrize("mu_variable", [
        None,
        {1: [(1, 1, 0j), (1, -1, 0j), (2, 0, 0.0)]},
    ])
    def test_diagonal_equals_1d_k2(self, model, numerics, k2_1d, mu_variable):
        T = extended_love_tensor(model, numerics, TD, mu_variable=mu_variable)
        assert T.forcing_modes == tuple((2, m) for m in range(-2, 3))
        for j, (lp, mp) in enumerate(T.forcing_modes):
            for i, mode in enumerate(T.response_modes):
                expected = k2_1d if mode == (lp, mp) else 0.0
                assert abs(T.K[i, j] - expected) <= 1e-12, (mode, (lp, mp))
        # All five diagonal entries present.
        for mp in range(-2, 3):
            assert abs(T.entry(2, mp, 2, mp) - k2_1d) <= 1e-12


# ---------------------------------------------------------------------------
# (b) complex <-> real conversion
# ---------------------------------------------------------------------------

class TestRealConversion:

    def test_T_unitary_and_roundtrip(self):
        degrees = [0, 1, 2, 3, 4]
        modes = complex_modes_for_degrees(degrees)
        labels = real_labels_for_degrees(degrees)
        T = complex_from_real_matrix(modes, labels)
        assert T.shape == (25, 25)
        np.testing.assert_allclose(T.conj().T @ T, np.eye(25), atol=1e-15)
        rng = np.random.default_rng(0)
        r = rng.standard_normal(25) + 1j * rng.standard_normal(25)
        np.testing.assert_allclose(T.conj().T @ (T @ r), r, atol=1e-14)

    def test_T_matches_mu_variable_convention(self):
        """``T`` reproduces ``_real_sh_to_complex_mu_variable`` (the
        MATLAB-validated real->complex map), including the sine branch."""
        real = {(2, 0): 0.3, (2, 1): -0.7, (2, -1): 0.2, (3, 2): 0.4,
                (3, -2): -1.1, (3, 3): 0.0, (3, -3): 0.6}
        ref = {(n, m): a for n, m, a in _real_sh_to_complex_mu_variable(real)}
        labels = real_labels_for_degrees([2, 3])
        rvec = np.array([
            real.get((l, m if cs == "C" else -m), 0.0) for cs, l, m in labels
        ])
        modes = complex_modes_for_degrees([2, 3])
        a = complex_from_real_matrix(modes, labels) @ rvec
        for md, val in zip(modes, a):
            assert abs(val - ref.get(md, 0.0)) <= 1e-15, md
        # Inverse through the dict helper.
        back = complex_to_real_coefficients(dict(zip(modes, a)))
        for (cs, l, m), v in back.items():
            key = (l, m) if cs == "C" else (l, -m)
            assert abs(v - real.get(key, 0.0)) <= 1e-15

    def test_T_matches_synthesis(self):
        """Independent of the formula: synthesize complex ``Y`` via
        ``complex_sh_synthesis`` and compare with the real cos/sin field."""
        from pylov3d.mars_lateral import complex_sh_synthesis

        lat = np.linspace(-80, 80, 17)
        lon = np.linspace(0, 350, 36)
        P = fully_normalized_legendre(3, np.sin(np.radians(lat)))
        phi = np.radians(lon)
        labels = real_labels_for_degrees([3])
        modes = complex_modes_for_degrees([3])
        T = complex_from_real_matrix(modes, labels)
        for j, (cs, l, m) in enumerate(labels):
            trig = np.cos(m * phi) if cs == "C" else np.sin(m * phi)
            real_field = P[l, m][:, None] * trig[None, :]
            entries = [(md[0], md[1], T[i, j]) for i, md in enumerate(modes)
                       if T[i, j] != 0]
            field = complex_sh_synthesis(entries, lat, lon)
            np.testing.assert_allclose(field, real_field, atol=1e-13)

    def test_1d_diagonal_preserved(self, model, numerics, k2_1d):
        R = to_real_basis(extended_love_tensor(model, numerics, TD))
        assert R.forcing_labels == real_labels_for_degrees([2])
        np.testing.assert_allclose(R.K, k2_1d * np.eye(5), atol=1e-12, rtol=0)

    def test_needs_closed_forcing_set(self, model, numerics):
        T = extended_love_tensor(model, numerics, TD, forcing_orders=[0, 1])
        with pytest.raises(ValueError, match="partner"):
            to_real_basis(T)

    def test_elastic_real_structure_gives_real_tensor(self, mixed_tensor):
        R = to_real_basis(mixed_tensor)
        scale = np.abs(R.K).max()
        assert np.abs(R.K.imag).max() <= 1e-9 * scale
        # Off-diagonal content is present and not negligible.
        blk = [i for i, lab in enumerate(R.response_labels) if lab[1] == 2]
        off = R.K[blk, :] - np.diag(np.diag(R.K[blk, :]))
        assert np.abs(off).max() > 1e-6

    def test_conjugate_identity(self, mixed_tensor):
        """K[(l,-m),(l',-m')] = (-1)^(m+m') conj K[(l,m),(l',m')] (elastic,
        real structure); derivation in the module docstring."""
        T = mixed_tensor
        scale = np.abs(T.K).max()
        worst = 0.0
        for i, (l, m) in enumerate(T.response_modes):
            for j, (lp, mp) in enumerate(T.forcing_modes):
                mirror = T.entry(l, -m, lp, -mp)
                worst = max(worst, abs(mirror - (-1) ** (m + mp) * np.conj(T.K[i, j])))
        assert worst <= 1e-9 * scale


# ---------------------------------------------------------------------------
# (c) m'=0 column vs the existing verified single-forcing path
# ---------------------------------------------------------------------------

def _enceladus_model():
    """Same construction as test_matlab_validation.TestEnceladusBenchmark."""
    G = 6.67e-11
    r_ratio = 0.91
    rho_ratio = 1610 / 1000
    rho_r = (rho_ratio - 1 + r_ratio**3) / r_ratio**3
    rho_core = rho_r * 1000
    R_surface = 252.1e3
    return make_interior_model(
        R0_km=[r_ratio * R_surface / 1e3, R_surface / 1e3],
        rho0=[rho_core, 1000],
        mu0=[0.0, 3.3e9],
        Ks0=[1e20, 100 * 3.3e9],
        Delta_rho0=[0.0, rho_core - 1000],
    )


class TestColumnConsistency:

    def test_enceladus_q11_m0_column(self):
        path = REPO / "data" / "tests" / "enceladus" / "Q_11.mat"
        if not path.exists():
            pytest.skip(f"MATLAB reference not found: {path}")
        mat = scipy.io.loadmat(path)
        amp_sph = float(mat["amp"].ravel()[4])
        amp_c = amp_sph / math.sqrt(2) / math.sqrt(4 * math.pi)
        mu_variable = {1: [(1, 1, amp_c), (1, -1, -amp_c)]}
        model = _enceladus_model()
        numerics = make_numerics(n_layers=2, method="fixed", Nrbase=100,
                                 perturbation_order=2)

        T = extended_love_tensor(model, numerics, 1.0, forcing_orders=[0],
                                 mu_variable=mu_variable)
        love, _, _ = get_love(model, make_forcing(1.0, 2, 0, 1.0), numerics,
                              mu_variable=mu_variable)
        col = T.column(0)
        assert set(col) == {(int(n), int(m)) for n, m in zip(love.n, love.m)}
        for n, m, k in zip(love.n, love.m, love.k):
            assert abs(col[(int(n), int(m))] - k) <= 1e-10

        # Same column vs MATLAB, at the tolerances of
        # test_matlab_validation.py::test_lateral_love_spectra.
        k_Q = mat["k_Q"]
        for n, m, order, k_ml in zip(k_Q[1:, 0].astype(int), k_Q[1:, 1].astype(int),
                                     k_Q[1:, 2].astype(int), k_Q[1:, 3 + 4]):
            if (n, m) not in col or abs(k_ml) < 1e-8:
                continue
            tol = {1: 0.01, 2: 0.05}.get(order, 0.10)
            if order == 0:
                continue  # amplitude-independent uniform row
            assert abs(col[(n, m)] - k_ml) / abs(k_ml) < tol, (n, m)


# ---------------------------------------------------------------------------
# (d) axisymmetric structure conserves m
# ---------------------------------------------------------------------------

class TestAxisymmetric:

    def test_m_conserved(self, model, numerics, k2_1d):
        mu = {1: [(2, 0, 0.02), (1, 0, 0.01)]}
        T = extended_love_tensor(model, numerics, TD, mu_variable=mu)
        n_off = 0
        for i, (l, m) in enumerate(T.response_modes):
            for j, (lp, mp) in enumerate(T.forcing_modes):
                if m != mp:
                    n_off += 1
                    assert T.K[i, j] == 0.0
        assert n_off > 0
        # Structure is felt: the diagonals shift, and split by |m'|.
        diag = np.array([T.entry(2, mp, 2, mp) for mp in range(-2, 3)])
        assert np.all(np.abs(diag - k2_1d) > 1e-6)
        assert abs(diag[0] - diag[4]) <= 1e-12 and abs(diag[1] - diag[3]) <= 1e-12
        assert abs(diag[2] - diag[4]) > 1e-8
        # Real basis: block-diagonal in m, and C/S of equal m identical.
        R = to_real_basis(T)
        for i, (cs, l, m) in enumerate(R.response_labels):
            for j, (cs2, lp, mp) in enumerate(R.forcing_labels):
                if m != mp or cs != cs2:
                    assert abs(R.K[i, j]) <= 1e-15


# ---------------------------------------------------------------------------
# Phase convention: rotation covariance
# ---------------------------------------------------------------------------

def _real_basis_values(labels, t, phi):
    lmax = max(l for _, l, _ in labels)
    P = fully_normalized_legendre(lmax, t)
    out = []
    for cs, l, m in labels:
        trig = np.cos(m * phi) if cs == "C" else np.sin(m * phi)
        out.append(P[l, m] * trig)
    return np.array(out)


def _real_rotation_matrix(degree, Rmat, nlat=24, nlon=48):
    """``D[i, j] = <b_i, b_j o R^{-1}>`` over the 4pi-normalized real basis,
    by exact Gauss-Legendre x trapezoid quadrature. ``(D r)`` is the real
    coefficient vector of the rotated field ``f(R^{-1} x)``."""
    labels = real_labels_for_degrees([degree])
    t, w = np.polynomial.legendre.leggauss(nlat)
    phi = np.arange(nlon) * 2 * np.pi / nlon
    T, PHI = np.meshgrid(t, phi, indexing="ij")
    W = np.repeat(w[:, None], nlon, axis=1) * (2 * np.pi / nlon)
    st = np.sqrt(1 - T**2)
    xyz = np.stack([st * np.cos(PHI), st * np.sin(PHI), T])
    src = np.einsum("ij,j...->i...", Rmat.T, xyz)  # R^{-1} x
    t2 = np.clip(src[2], -1, 1)
    phi2 = np.arctan2(src[1], src[0])
    B = _real_basis_values(labels, T, PHI)
    Brot = _real_basis_values(labels, t2, phi2)
    D = np.einsum("iab,jab,ab->ij", B, Brot, W) / (4 * np.pi)
    return labels, D


class TestRotationCovariance:
    """Derivation-free check of the phase/normalization convention.

    Solve with axisymmetric structure ``eps * Pbar_10`` (along z), then
    with the same field rotated onto the x axis (``R_y(+90deg)``). The
    rotated field's real coefficients come from quadrature, and its
    ``mu_variable`` from the MATLAB-validated real->complex map. Physics
    requires ``K_real(rot) = D K_real(z) D^T``. A Condon-Shortley slip
    flips the sign of every odd ``m - m'`` complex entry. The negative
    control below shows that breaks the relation at O(eps).
    """

    EPS = 0.01

    @pytest.fixture(scope="class")
    @classmethod
    def tensors(cls, model, numerics):
        c = math.cos(math.pi / 2)
        s = math.sin(math.pi / 2)
        Rmat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # z -> x
        lab1, D1 = _real_rotation_matrix(1, Rmat)
        rot1 = D1 @ np.array([cls.EPS, 0.0, 0.0])  # C10 = eps
        real_rot = {}
        for (cs, l, m), v in zip(lab1, rot1):
            if abs(v) > 1e-14:
                real_rot[(l, m if cs == "C" else -m)] = float(v)
        mu_z = {1: [(1, 0, cls.EPS)]}
        mu_x = {1: _real_sh_to_complex_mu_variable(real_rot)}
        Tz = extended_love_tensor(model, numerics, TD, mu_variable=mu_z)
        Tx = extended_love_tensor(model, numerics, TD, mu_variable=mu_x)
        return Rmat, rot1, Tz, Tx

    @staticmethod
    def _predict(Rmat, Kz_real, Tz_real):
        degrees = sorted({l for _, l, _ in Tz_real.response_labels})
        blocks = [_real_rotation_matrix(l, Rmat)[1] for l in degrees]
        n = sum(b.shape[0] for b in blocks)
        Dresp = np.zeros((n, n))
        o = 0
        for b in blocks:
            Dresp[o:o + b.shape[0], o:o + b.shape[0]] = b
            o += b.shape[0]
        Dforc = _real_rotation_matrix(2, Rmat)[1]
        return Dresp @ Kz_real @ Dforc.T

    def _compare(self, Rmat, Tz, Tx, flip_cs=False):
        if flip_cs:
            # Reinterpret the rotated solve's K as if the solver basis
            # carried a Condon-Shortley phase: K -> (-1)^(m+m') K. (The
            # z-axis tensor is m-diagonal, so it is invariant under this.)
            Kx = Tx.K * np.array([[(-1) ** abs(m + mp) for (_, mp) in Tx.forcing_modes]
                                  for (_, m) in Tx.response_modes])
            Tx = type(Tx)(Tx.response_modes, Tx.forcing_modes, Kx, Tx.spectra)
        Rz = to_real_basis(Tz)
        Rx = to_real_basis(Tx)
        pred = self._predict(Rmat, Rz.K, Rz)
        rows = [Rz.response_labels.index(lab) for lab in Rx.response_labels]
        diff = Rx.K - pred[rows, :]
        # Scale: the first-order (off-diagonal) coupling magnitude.
        off = Rx.K.copy()
        for j, lab in enumerate(Rx.forcing_labels):
            off[Rx.response_labels.index(lab), j] = 0.0
        return np.abs(diff).max(), np.abs(off).max()

    def test_rotated_field_is_x_axis(self, tensors):
        _, rot1, _, _ = tensors
        np.testing.assert_allclose(rot1, [0.0, self.EPS, 0.0], atol=1e-15)

    def test_covariance(self, tensors):
        Rmat, _, Tz, Tx = tensors
        err, off = self._compare(Rmat, Tz, Tx)
        assert off > 1e-5  # coupling is present
        # Residual is O(eps^2) truncation of the rotated (non-closed) mode set.
        assert err <= 1e-3 * off, (err, off)  # measured 3.5e-5

    def test_cs_flip_breaks_covariance(self, tensors):
        Rmat, _, Tz, Tx = tensors
        err, off = self._compare(Rmat, Tz, Tx, flip_cs=True)
        assert err >= 0.5 * off, (err, off)


# ---------------------------------------------------------------------------
# Slow: Mars full coupled spectra vs native MATLAB, forcings (2,0),(2,1),(2,2)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mars_columns_match_matlab():
    from pylov3d.mars import MARS_FORCING_TD, build_mars_model
    from pylov3d.mars_lateral import mu_variable_from_topography

    path = REPO / "data" / "tests" / "mars" / "mars_lateral_cross_check.mat"
    d = scipy.io.loadmat(path, squeeze_me=True)
    T = extended_love_tensor(
        build_mars_model(),
        make_numerics(n_layers=4, method="combination", Nrbase=30,
                      perturbation_order=2),
        MARS_FORCING_TD,
        forcing_orders=(0, 1, 2),
        mu_variable=mu_variable_from_topography(lmax=4),
    )
    for r in d["results"]:
        mp = int(r["forcing_m"])
        n = np.atleast_1d(r["n"]).astype(int)
        m = np.atleast_1d(r["m"]).astype(int)
        k = np.atleast_1d(r["k"]).astype(complex)
        col = T.column(mp)
        assert set(col) == set(zip(n.tolist(), m.tolist()))
        for ni, mi, ki in zip(n, m, k):
            kp = col[(int(ni), int(mi))]
            if (ni, mi) == (2, mp):
                assert abs(kp - ki) <= 1e-10 * abs(ki)
            else:
                # Off-diagonal: absolute, relative to the forcing-mode k.
                assert abs(kp - ki) <= 1e-10 * abs(r["k2_forcing"]), (mp, ni, mi)
