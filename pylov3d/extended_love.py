# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

r"""Extended Love-number tensor K^{l'm'}_{lm} (Berne et al. 2026 item B1).

For a laterally heterogeneous body, a unit tidal forcing at one harmonic
``(l', m')`` produces a gravitational response at many harmonics ``(l, m)``.
The *extended Love numbers* collect these into a matrix::

    K[(l, m), (l', m')] = gravitational response at (l, m)
                          per unit forcing at (l', m')

:func:`extended_love_tensor` builds it column by column: one coupled solve
(:func:`pylov3d.love.get_love`, one forcing per call) per forcing order
``m' = -n_f..n_f`` at forcing degree ``n_f`` (default 2), and stacks the
resulting :class:`~pylov3d.types.LoveSpectra` over the union of the active
response modes. :func:`to_real_basis` then converts the tensor to the real,
4pi-normalized cos/sin basis of gravity Stokes coefficients.

Column definition (inherited from ``extract_love_numbers``)
-----------------------------------------------------------
Each solve imposes unit forcing: the forcing boundary condition
(``boundary_conditions.py``, ``B2[7] = 2n+1``) does not read
``Forcing.F`` at all, so every column is per unit forcing coefficient.
For the forced mode, ``K = Phi_surf - 1`` (the forcing's own unit
potential removed); for every other mode ``K = Phi_surf``. Both are
coefficients of the same surface potential in the same basis, so
``K`` is a plain matrix of potential-coefficient ratios. The sign and
phase of ``Im K`` are the solver's (the same time convention as ``k2``
from ``get_love``).

A response mode that is not in a given forcing's active set
(:func:`pylov3d.couplings.get_active_modes`) gets ``K = 0``. That zero
is either an exact selection-rule zero (e.g. ``m`` conserved for
axisymmetric structure) or truncation at ``numerics.perturbation_order``.
It is not a solved-for value.

Complex basis used by the solver (derived, not assumed)
-------------------------------------------------------
The solver's mode ``(l, m)`` stands for the complex harmonic::

    Y_l^0    =          Pbar_l^0(cos th)
    Y_l^{+m} =          Pbar_l^m(cos th) exp(+i m ph) / sqrt(2)     (m > 0)
    Y_l^{-m} = (-1)^m * Pbar_l^m(cos th) exp(-i m ph) / sqrt(2)

where ``Pbar_l^m`` is 4pi-fully-normalized with **no Condon-Shortley
phase** (:func:`pylov3d.mapping.fully_normalized_legendre`, a port of
LOV3D ``src/SPH_Tools/Legendre.m``). Provenance:

* LOV3D ``src/get_map.m`` (~lines 196-201) synthesizes both the rheology
  and the solution fields with exactly this ``Y``.
* ``pylov3d.mars_lateral._real_sh_to_complex_mu_variable`` generalizes
  MATLAB ``get_rheology.m``'s real->complex map,
  ``amp(+m) = (C - iS)/sqrt(2)``, ``amp(-m) = (-1)^m (C + iS)/sqrt(2)``.
  Combined with :func:`pylov3d.mars_lateral.complex_sh_synthesis`, that
  map is round-tripped against ``pylov3d.mapping.sh_to_latlon`` in
  ``test_mars_lateral.py::TestCSHRoundTrip``. The same ``mu_variable``
  path is end-to-end MATLAB-validated (Weber Moon, Mars TASK-016).
* ``pylov3d.mars_detectability.sh_basis_norm`` documents the same basis
  for the gravity-observable derivation.

The ``(-1)^m`` on ``Y^{-m}`` gives ``Y^{-m} = (-1)^m conj(Y^{+m})``, a
relation that holds with or without a Condon-Shortley phase. What rules
out a CS phase is the ``+m`` element having no ``(-1)^m`` factor, i.e.
the real cosine field ``C Pbar cos(m ph)`` has ``amp(+m) = +C/sqrt(2)``.
Assuming a CS phase would flip the sign of every entry whose ``m - m'``
is odd. ``test_extended_love.py::TestRotationCovariance`` checks the
convention without relying on this derivation. It rotates an
axisymmetric structure onto the x axis and requires
``K_real' = D K_real D^T``, with the real rotation matrices ``D`` built
by quadrature in the real basis. A wrong CS convention fails that check
at first order in the lateral amplitude.

The overall 4pi-vs-orthonormal scale (``1/sqrt(4 pi)``) multiplies
forcing and response alike, so it cancels in ``K``. It matters only for
``mu_variable`` amplitudes; see ``test_matlab_validation.py``,
``amp/sqrt(4 pi)``.

Real basis (Stokes coefficients) and the conversion
---------------------------------------------------
The real basis is the geodesy one used for gravity Stokes coefficients,
e.g. GMM-3 or MRO120 (``pylov3d.sh_data``, ``pylov3d.mapping``)::

    V(R, th, ph) = (GM/R) sum_lm Pbar_l^m(cos th)
                   [C_lm cos(m ph) + S_lm sin(m ph)]

This is 4pi-normalized with no Condon-Shortley phase, the same
``Pbar`` as above. Label real coefficients ``('C', l, m)`` for
``m >= 0`` and ``('S', l, m)`` for ``m >= 1``. The complex coefficient
vector ``a`` (solver basis) and the real vector ``r`` are related by
``a = T r``, where for each ``(l, m > 0)``::

    a(l, +m) = (C_lm - i S_lm) / sqrt(2)
    a(l, -m) = (-1)^m (C_lm + i S_lm) / sqrt(2)
    a(l,  0) = C_l0

This is the same map as ``_real_sh_to_complex_mu_variable``, applied to
potential coefficients. ``T`` is unitary: every element of both bases
has ``int |.|^2 dOmega = 4 pi``, so ``T^{-1} = T^H``, which gives::

    C_lm = (a(l,+m) + (-1)^m a(l,-m)) / sqrt(2)
    S_lm = i (a(l,+m) - (-1)^m a(l,-m)) / sqrt(2)

The real-basis tensor is therefore::

    K_real = T_resp^H  K  T_forc

It maps real forcing coefficients (``C``/``S`` of the tide-raising
potential at degree ``n_f``, normalized by ``GM/R`` like the Stokes
coefficients) to real response Stokes coefficients::

    [dC; dS]_(l, m) = sum_(l', m') K_real[(l,m), (l',m')] [C; S]^tide_(l', m')

With no lateral structure, ``K_real`` is ``diag(k_{n_f})``. That is the
IERS/Eanes relation ``dC_nm - i dS_nm = k_nm/(2n+1) (GM_j/GM)
(R/r_j)^(n+1) Pbar_nm(sin phi_j) exp(-i m lam_j)``. The ``1/(2n+1)`` and
ephemeris factors belong to the tidal forcing coefficients, not to
``K``; see the ``pylov3d.mars_detectability`` docstring, sec. 1.

``K_real`` acts on complex *phasors* of ``C(t)`` and ``S(t)`` at the
forcing frequency ``2 pi / Td``. For an elastic body with real lateral
structure (a conjugate-paired ``mu_variable``), ``K_real`` is real, and
``K`` satisfies the exact identity::

    K[(l, -m), (l', -m')] = (-1)^(m + m') conj(K[(l, m), (l', m')])

Both are tested in ``pylov3d/tests/test_extended_love.py``. For
anelastic rheology, ``K_real`` is complex and carries the phase lags.

MATLAB anchor status
--------------------
``data/tests/mars/mars_lateral_cross_check.mat`` (``results(1..3)``)
holds the native-MATLAB full coupled spectra for Mars forcings (2,0),
(2,1) and (2,2). Those are the ``m' = 0, 1, 2`` columns of ``K``,
including the off-diagonal entries. ``m' = -1, -2`` have no direct
MATLAB run. They follow from the conjugate identity above, which is
exact for that elastic model and tested on the Python side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .love import get_love
from .types import InteriorModel, LoveSpectra, NumericsConfig, make_forcing

Mode = tuple[int, int]
RealLabel = tuple[str, int, int]


@dataclass(frozen=True)
class ExtendedLoveTensor:
    """Extended Love numbers in the solver's complex basis.

    ``K[i, j]`` is the response at ``response_modes[i]`` to unit forcing
    at ``forcing_modes[j]``. ``spectra[j]`` is the raw
    :class:`LoveSpectra` of solve ``j``.
    """

    response_modes: tuple[Mode, ...]
    forcing_modes: tuple[Mode, ...]
    K: np.ndarray
    spectra: tuple[LoveSpectra, ...]

    def entry(self, l: int, m: int, lp: int, mp: int) -> complex:
        """``K[(l, m), (lp, mp)]``. Returns 0 if ``(l, m)`` was not solved for."""
        j = self.forcing_modes.index((lp, mp))
        try:
            i = self.response_modes.index((l, m))
        except ValueError:
            return 0.0 + 0.0j
        return complex(self.K[i, j])

    def column(self, mp: int, lp: int | None = None) -> dict[Mode, complex]:
        """The ``(l', m')`` column as a ``{(l, m): K}`` dict (active modes only)."""
        lp = self.forcing_modes[0][0] if lp is None else lp
        spec = self.spectra[self.forcing_modes.index((lp, mp))]
        return {
            (int(n), int(m)): complex(k)
            for n, m, k in zip(spec.n, spec.m, spec.k)
        }


@dataclass(frozen=True)
class RealExtendedLoveTensor:
    """Extended Love numbers in the real 4pi-normalized cos/sin basis.

    Labels are ``('C', l, m)`` (``m >= 0``) or ``('S', l, m)`` (``m >= 1``).
    """

    response_labels: tuple[RealLabel, ...]
    forcing_labels: tuple[RealLabel, ...]
    K: np.ndarray

    def entry(self, row: RealLabel, col: RealLabel) -> complex:
        return complex(
            self.K[self.response_labels.index(row), self.forcing_labels.index(col)]
        )


# ---------------------------------------------------------------------------
# Tensor assembly
# ---------------------------------------------------------------------------

def extended_love_tensor(
    interior_model: InteriorModel,
    numerics: NumericsConfig,
    Td: float,
    *,
    n_forcing: int = 2,
    forcing_orders: Sequence[int] | None = None,
    mu_variable: dict | None = None,
    eta_variable: dict | None = None,
    K_variable: dict | None = None,
) -> ExtendedLoveTensor:
    """Solve once per forcing order and assemble ``K[(l,m),(n_forcing,m')]``.

    Parameters
    ----------
    interior_model, numerics, mu_variable, eta_variable, K_variable
        Passed unchanged to :func:`pylov3d.love.get_love`. The same
        lateral structure is used for every column.
    Td : float
        Forcing period [s], the same for every column.
    n_forcing : int
        Forcing degree ``l'`` (default 2).
    forcing_orders : sequence of int, optional
        Orders ``m'`` to solve. Defaults to ``-n_forcing..n_forcing``.
        :func:`to_real_basis` needs every ``m'`` present in the set to
        have its ``-m'`` partner too.

    Returns
    -------
    ExtendedLoveTensor
        Rows are the sorted union of every solve's active modes, ordered
        by ``(l, m)``. An entry is 0 where a mode is not active for that
        column (see the module docstring).
    """
    if forcing_orders is None:
        forcing_orders = range(-n_forcing, n_forcing + 1)
    forcing_modes = tuple((n_forcing, int(mp)) for mp in forcing_orders)
    for _, mp in forcing_modes:
        if abs(mp) > n_forcing:
            raise ValueError(f"|m'|={abs(mp)} exceeds forcing degree {n_forcing}")

    spectra = []
    for lp, mp in forcing_modes:
        forcing = make_forcing(Td=Td, n=lp, m=mp, F=1.0)
        love, _, _ = get_love(
            interior_model, forcing, numerics,
            mu_variable=mu_variable,
            eta_variable=eta_variable,
            K_variable=K_variable,
        )
        spectra.append(love)

    modes = sorted({
        (int(n), int(m)) for s in spectra for n, m in zip(s.n, s.m)
    })
    row = {mode: i for i, mode in enumerate(modes)}
    K = np.zeros((len(modes), len(forcing_modes)), dtype=np.complex128)
    for j, s in enumerate(spectra):
        for n, m, k in zip(s.n, s.m, s.k):
            K[row[(int(n), int(m))], j] = complex(k)

    return ExtendedLoveTensor(
        response_modes=tuple(modes),
        forcing_modes=forcing_modes,
        K=K,
        spectra=tuple(spectra),
    )


# ---------------------------------------------------------------------------
# Complex <-> real basis
# ---------------------------------------------------------------------------

def real_labels_for_degrees(degrees: Sequence[int]) -> tuple[RealLabel, ...]:
    """Complete real labels ``C_l0, C_l1, S_l1, ..., C_ll, S_ll`` for each degree."""
    labels: list[RealLabel] = []
    for l in sorted(set(int(d) for d in degrees)):
        labels.append(("C", l, 0))
        for m in range(1, l + 1):
            labels.append(("C", l, m))
            labels.append(("S", l, m))
    return tuple(labels)


def complex_modes_for_degrees(degrees: Sequence[int]) -> tuple[Mode, ...]:
    """Complete complex modes ``(l, -l..l)`` for each degree, sorted."""
    return tuple(
        (l, m) for l in sorted(set(int(d) for d in degrees))
        for m in range(-l, l + 1)
    )


def complex_from_real_matrix(
    complex_modes: Sequence[Mode],
    real_labels: Sequence[RealLabel],
) -> np.ndarray:
    """The matrix ``T`` with ``a = T r`` (module docstring).

    Rows are indexed by ``complex_modes`` and columns by ``real_labels``.
    Over complete degree blocks ``T`` is square and unitary.
    """
    row = {tuple(md): i for i, md in enumerate(complex_modes)}
    T = np.zeros((len(complex_modes), len(real_labels)), dtype=np.complex128)
    s2 = 1.0 / math.sqrt(2.0)
    for j, (cs, l, m) in enumerate(real_labels):
        if m == 0:
            if cs != "C":
                raise ValueError(f"no S coefficient at m=0: {(cs, l, m)}")
            if (l, 0) in row:
                T[row[(l, 0)], j] = 1.0
            continue
        sign = (-1) ** m
        if cs == "C":
            cp, cm = s2, sign * s2
        elif cs == "S":
            cp, cm = -1j * s2, sign * 1j * s2
        else:
            raise ValueError(f"bad real label {(cs, l, m)}")
        if (l, m) in row:
            T[row[(l, m)], j] = cp
        if (l, -m) in row:
            T[row[(l, -m)], j] = cm
    return T


def complex_to_real_coefficients(
    a: dict[Mode, complex],
) -> dict[RealLabel, complex]:
    """Convert complex-basis coefficients to real ``C/S`` coefficients (``r = T^H a``).

    Missing partner modes are treated as zero.
    """
    degrees = sorted({l for l, _ in a})
    modes = complex_modes_for_degrees(degrees)
    labels = real_labels_for_degrees(degrees)
    T = complex_from_real_matrix(modes, labels)
    avec = np.array([a.get(md, 0.0) for md in modes], dtype=np.complex128)
    r = T.conj().T @ avec
    return {lab: complex(v) for lab, v in zip(labels, r)}


def to_real_basis(tensor: ExtendedLoveTensor) -> RealExtendedLoveTensor:
    """Convert to the real Stokes basis: ``K_real = T_resp^H K T_forc``.

    Response rows are padded to complete degree blocks. Padded rows are
    modes not in any solve's active set, so they are 0 (module docstring).
    The forcing set must be closed under ``m' -> -m'``. Otherwise a real
    ``C``/``S`` forcing column cannot be built from the solved columns.
    """
    fset = set(tensor.forcing_modes)
    for lp, mp in tensor.forcing_modes:
        if (lp, -mp) not in fset:
            raise ValueError(
                f"forcing set lacks ({lp},{-mp}), the partner of ({lp},{mp}); "
                "the real C/S forcing basis needs both"
            )

    resp_degrees = sorted({l for l, _ in tensor.response_modes})
    resp_modes = complex_modes_for_degrees(resp_degrees)
    resp_labels = real_labels_for_degrees(resp_degrees)

    # Embed the solved rows into complete degree blocks (zeros elsewhere).
    idx = {md: i for i, md in enumerate(resp_modes)}
    K_full = np.zeros((len(resp_modes), len(tensor.forcing_modes)), dtype=np.complex128)
    for i, md in enumerate(tensor.response_modes):
        K_full[idx[md], :] = tensor.K[i, :]

    forc_degrees = sorted({l for l, _ in tensor.forcing_modes})
    forc_labels = tuple(
        lab for lab in real_labels_for_degrees(forc_degrees)
        if (lab[1], lab[2]) in fset
    )

    T_resp = complex_from_real_matrix(resp_modes, resp_labels)
    T_forc = complex_from_real_matrix(tensor.forcing_modes, forc_labels)
    K_real = T_resp.conj().T @ K_full @ T_forc

    return RealExtendedLoveTensor(
        response_labels=resp_labels,
        forcing_labels=forc_labels,
        K=K_real,
    )
