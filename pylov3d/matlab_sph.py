# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
"""MATLAB-faithful spherical-harmonic transforms used by LOV3D.

This module ports ``src/SPH_Tools/Legendre.m``, ``SPH_LatLon.m`` and
``LatLon_SPH.m`` closely enough to diagnose MATLAB/Python rheology-spectrum
parity without involving the radial solver.

The MATLAB routines use fully normalized real Stokes harmonics on a
cell-centred equiangular grid with 2*N latitude rows and 4*N longitude
columns for ``lmax=N``.  This is deliberately kept separate from the
Gauss-Legendre/SciPy transform used elsewhere in pylov3d until parity is
established.
"""
from __future__ import annotations

import numpy as np


def normalized_legendre(lmax: int, t: np.ndarray) -> np.ndarray:
    """Port of ``SPH_Tools/Legendre.m``.

    Returns ``P[l,m,k]`` in the MATLAB fully-normalized, no-CS recurrence.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    P = np.zeros((lmax + 1, lmax + 1, t.size), dtype=float)
    P[0, 0, :] = 1.0
    if lmax == 0:
        return P

    u = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    P[1, 1, :] = np.sqrt(3.0) * u
    for ell in range(2, lmax + 1):
        P[ell, ell, :] = (
            np.sqrt((2.0 * ell + 1.0) / (2.0 * ell))
            * u
            * P[ell - 1, ell - 1, :]
        )

    for m in range(0, lmax):
        P[m + 1, m, :] = np.sqrt(2.0 * m + 3.0) * t * P[m, m, :]
        for ell in range(m + 2, lmax + 1):
            a = np.sqrt(
                (2.0 * ell - 1.0) * (2.0 * ell + 1.0)
                / ((ell - m) * (ell + m))
            )
            b = np.sqrt(
                (2.0 * ell + 1.0)
                * (ell + m - 1.0)
                * (ell - m - 1.0)
                / ((2.0 * ell - 3.0) * (ell + m) * (ell - m))
            )
            P[ell, m, :] = a * t * P[ell - 1, m, :] - b * P[ell - 2, m, :]
    return P


def equiangular_grid(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB ``SPH_LatLon`` cell-centred latitude/longitude arrays."""
    if lmax < 1:
        raise ValueError("lmax must be >= 1")
    delta = 180.0 / (2.0 * lmax)
    lat = -90.0 + delta / 2.0 + delta * np.arange(2 * lmax)
    lon = -180.0 + delta / 2.0 + delta * np.arange(4 * lmax)
    return lat, lon


def stokes_to_grid(
    clm: np.ndarray,
    slm: np.ndarray,
    lmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ``SPH_LatLon.m`` for real fully-normalized Stokes coefficients."""
    clm = np.asarray(clm)
    slm = np.asarray(slm)
    if lmax is None:
        lmax = min(clm.shape[0], clm.shape[1]) - 1
    lat, lon = equiangular_grid(lmax)
    colat = np.deg2rad(90.0 - lat)
    lon_rad = np.deg2rad(lon)
    P = normalized_legendre(lmax, np.cos(colat))

    z = np.zeros((lat.size, lon.size), dtype=float)
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            phase = (-1.0) ** m
            radial_c = phase * float(clm[ell, m]) * P[ell, m, :]
            radial_s = phase * float(slm[ell, m]) * P[ell, m, :]
            z += radial_c[:, None] * np.cos(m * lon_rad)[None, :]
            z += radial_s[:, None] * np.sin(m * lon_rad)[None, :]
    return lat, lon, z


def _dh_weights(N: int) -> np.ndarray:
    """The latitude quadrature weights constructed in ``LatLon_SPH.m``."""
    q = np.zeros(2 * N, dtype=float)
    h = np.arange(N, dtype=float)
    odd = 2.0 * h + 1.0
    for j in range(N + 1):
        angle = (j + 0.5) * np.pi / (2.0 * N)
        a = np.sum(np.sin(odd * angle) / odd)
        q[j] = np.sin(angle) * a / N
    # MATLAB: qj(N+1:2*N)=flip(qj(1:N)); the q[N] value is overwritten.
    q[N:2 * N] = q[:N][::-1]
    return q


def grid_to_stokes(z: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``LatLon_SPH.m`` for an exactly matching equiangular grid."""
    z = np.asarray(z, dtype=float)
    expected = (2 * lmax, 4 * lmax)
    if z.shape != expected:
        raise ValueError(f"expected grid shape {expected}, got {z.shape}")

    # MATLAB flips latitude before the transform.
    z_aux = np.flipud(z)
    lat, _ = equiangular_grid(lmax)
    colat = 90.0 - lat
    colat = np.flip(colat)
    t = np.cos(np.deg2rad(colat))
    P = normalized_legendre(lmax, t)
    q = _dh_weights(lmax)

    aa = np.zeros((2 * lmax, lmax + 1), dtype=float)
    bb = np.zeros_like(aa)
    for j in range(2 * lmax):
        # MATLAB shifts longitudes from [-pi,pi) to [0,2pi) before fft.
        row = np.concatenate((z_aux[j, 2 * lmax : 4 * lmax], z_aux[j, : 2 * lmax]))
        y = np.fft.fft(row)
        aa[j, :] = np.real(y[: lmax + 1])
        bb[j, :] = -np.imag(y[: lmax + 1])

    clm = np.zeros((lmax + 1, lmax + 1), dtype=float)
    slm = np.zeros_like(clm)
    scale = (1.0 / lmax) / 4.0
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            phase = (-1.0) ** m
            kernel = q * P[ell, m, :]
            clm[ell, m] = phase * scale * np.sum(kernel * aa[:, m])
            slm[ell, m] = phase * scale * np.sum(kernel * bb[:, m])
    return clm, slm


def complex_modes_to_stokes(
    modes: list[tuple[int, int, complex]], lmax: int
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``get_rheology.m`` complex-input -> real-Stokes conversion."""
    clm = np.zeros((lmax + 1, lmax + 1), dtype=float)
    slm = np.zeros_like(clm)
    for ell, m, amp in modes:
        if ell > lmax or m < 0:
            continue
        if m == 0:
            clm[ell, 0] = float(np.real(amp))
        else:
            clm[ell, m] = np.sqrt(2.0) * float(np.real(amp))
            slm[ell, m] = -np.sqrt(2.0) * float(np.imag(amp))
    return clm, slm


def stokes_pair_to_complex(
    clm_r: np.ndarray,
    slm_r: np.ndarray,
    clm_i: np.ndarray,
    slm_i: np.ndarray,
    lmax: int,
) -> dict[tuple[int, int], complex]:
    """Port the real/imag Stokes -> complex-mode block in ``get_rheology.m``."""
    out: dict[tuple[int, int], complex] = {}
    root2 = np.sqrt(2.0)
    for ell in range(lmax + 1):
        # m=0
        out[(ell, 0)] = complex(clm_r[ell, 0], clm_i[ell, 0])
        for m in range(1, ell + 1):
            phase = (-1.0) ** m
            r_pos = clm_r[ell, m] / root2 - phase * 1j * slm_r[ell, m] / root2
            r_neg = phase * clm_r[ell, m] / root2 + 1j * slm_r[ell, m] / root2
            i_pos = clm_i[ell, m] / root2 - phase * 1j * slm_i[ell, m] / root2
            i_neg = phase * clm_i[ell, m] / root2 + 1j * slm_i[ell, m] / root2
            out[(ell, m)] = r_pos + 1j * i_pos
            out[(ell, -m)] = r_neg + 1j * i_neg
    return out


def maxwell_rheology_from_fractional_grid(
    dmu: np.ndarray,
    deta: np.ndarray,
    mu_mean: float,
    maxwell_mean: float,
    lmax: int,
) -> tuple[complex, dict[tuple[int, int], complex]]:
    """Apply the MATLAB raw-grid Maxwell transform and return complex SH modes.

    ``dmu`` and ``deta`` are fractional deviations from their scalar means,
    so the physical relative fields are ``1+dmu`` and ``1+deta``.
    """
    dmu = np.asarray(dmu, dtype=float)
    deta = np.asarray(deta, dtype=float)
    expected = (2 * lmax, 4 * lmax)
    if dmu.shape != expected or deta.shape != expected:
        raise ValueError(f"expected dmu/deta shape {expected}")

    mu_rel = 1.0 + dmu
    eta_rel = 1.0 + deta
    maxwell_field = eta_rel / mu_rel
    cmu = mu_mean * mu_rel / (1.0 - 1j / (maxwell_field * maxwell_mean))

    cr, sr = grid_to_stokes(np.real(cmu), lmax)
    ci, si = grid_to_stokes(np.imag(cmu), lmax)
    modes = stokes_pair_to_complex(cr, sr, ci, si, lmax)
    return modes[(0, 0)], modes


def filter_rheology_modes(
    modes: dict[tuple[int, int], complex],
    cutoff: float,
    minimum_log_value: float = -14.0,
) -> list[tuple[int, int, complex, float, float]]:
    """MATLAB ``get_rheology.m`` real/imag log-amplitude filtering."""
    mu00 = modes[(0, 0)]
    rows: list[tuple[int, int, complex, float, float]] = []
    tiny = np.finfo(float).tiny
    for (ell, m), amp in modes.items():
        if ell == 0:
            continue
        lr = np.log10(max(abs(amp.real / mu00.real), tiny))
        li = np.log10(max(abs(amp.imag / mu00.imag), tiny))
        rows.append((ell, m, amp, lr, li))
    if not rows:
        return []
    max_log = max(max(r[3], r[4]) for r in rows)
    if max_log <= minimum_log_value:
        return []
    return [
        row for row in rows
        if (row[3] - max_log >= -cutoff) or (row[4] - max_log >= -cutoff)
    ]
