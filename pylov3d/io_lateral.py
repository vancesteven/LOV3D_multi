# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Io viscoelastic + lateral-rheology model and heating-pattern helpers (TASK-046).

Reproduces the upstream MATLAB ``tests/Consistency_test_Energy.m`` Io
four-layer model, its degree-2 tidal-heating pattern, and the mapping of
that pattern into fractional shear-modulus (``mu``) and viscosity (``eta``)
lateral variations on the asthenosphere layer -- **numerically**, not
approximately (same recursion inputs, same grid, same nonlinear
Q -> Phi -> (eta_rel, mu_rel) chain).

Basis-convention finding (read before extending)
---------------------------------------------------------------------------
``pylov3d.mars_lateral._real_sh_to_complex_mu_variable`` /
``complex_sh_synthesis`` implement a complex spherical-harmonic basis built
on ``pylov3d.mapping.fully_normalized_legendre`` (real, 4pi-normalized, no
Condon-Shortley phase). That basis is exactly what
``process_lateral_variations``'s **elastic** branch consumes directly (it
just multiplies the raw amplitude by ``mu_i`` and forwards it) -- this is
the already-MATLAB-validated Mars/Moon crust convention.

The **viscoelastic** branch is different: it synthesizes the physical grid
field via ``pylov3d.rheology._sh_synthesis``, which calls
``scipy.special.sph_harm_y`` -- an orthonormal, Condon-Shortley-phase
complex spherical-harmonic basis, *not* the ``fully_normalized_legendre``
basis. Verified numerically (see ``pylov3d/tests/test_io_lateral.py::
test_real_sh_to_complex_mu_variable_wrong_basis_for_viscoelastic``): feeding
``_real_sh_to_complex_mu_variable``'s output straight into
``_sh_synthesis`` reproduces the target field only after rescaling by the
*constant* ``1/sqrt(4*pi)`` for ``m=0`` (and would need an additional
Condon-Shortley sign flip for odd ``m``), i.e. the two conventions are
genuinely different bases, not a relabelling of the same one.
``pylov3d.mars_lateral``'s own module docstring already flags this
("Different convention from the sph_harm_y-based rheology._sh_synthesis
... do not conflate the two"); this module is the first place in the
repository that actually feeds a nonzero ``eta_variable`` (viscoelastic
lateral) input, so the distinction becomes load-bearing here for the first
time -- no prior test exercises it (``pylov3d/tests/test_lateral_rheology.py
::TestLateralViscoelastic`` only ever uses ``m=0``, where a constant
rescaling is invisible to a "nonzero"/"finite" check).

Consequently, this module builds the asthenosphere's ``mu_variable`` /
``eta_variable`` entries by evaluating the target field analytically on
``pylov3d.rheology._sh_grid``'s own quadrature nodes and analysing it with
``pylov3d.rheology._sh_analysis`` (the exact synthesis/analysis inverse
pair ``process_lateral_variations`` itself uses) -- not via
``_real_sh_to_complex_mu_variable``. ``pylov3d.mars_mantle
.project_temperature_real_sh`` (the real, 4pi-normalized, weighted
least-squares projector this task's spec names) is still used, but only
for the human-readable degree-by-degree amplitude spectrum report: since
both bases are complete orthonormal-up-to-a-constant expansions of the same
function space, Parseval's theorem makes the per-degree power *ratios*
(e.g. degree-4 relative to degree-2) basis-independent even though the
absolute per-coefficient values are not; see the same test module for the
explicit degree-2-dominance / degree-4-secondary check.

Mean-offset finding
---------------------------------------------------------------------------
``process_lateral_variations._unify_modes`` unconditionally strips any
``(0,0)`` entry from ``mu_variable``/``eta_variable`` ("the mean is handled
separately" -- the layer's own ``mu0``/``eta0`` scalar already *is* the
mean reference). Because ``mu_rel``/``eta_rel`` are nonlinear (rational)
functions of the underlying zero-mean degree-2 heating pattern, ``dmu``'s
and ``deta``'s own spherical means are *not* exactly zero even though the
pattern they are built from is (measured: mean(dmu)/degree-2 amplitude
about 2%, mean(deta)/degree-2 amplitude about 6%). That small offset is
therefore silently dropped by ``process_lateral_variations`` regardless of
what this module feeds it -- there is no way to communicate it through the
existing ``mu_variable``/``eta_variable`` API without changing
``build_io_model``'s fixed ``mu0``/``eta0`` (which would break the
spec-table match this module is required to preserve exactly). Reported,
not corrected; see ``pylov3d/tests/test_io_lateral.py::TestBasisConvention
::test_nonzero_field_mean_is_a_documented_api_gap``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import lpmv

from .mars_mantle import RealSH, project_temperature_real_sh
from .rheology import _sh_analysis, _sh_grid
from .types import Forcing, InteriorModel, make_forcing, make_interior_model, make_numerics

# ---------------------------------------------------------------------------
# Io model constants (verbatim from tests/Consistency_test_Energy.m)
# ---------------------------------------------------------------------------

# 0-based layer indices: core=0, mantle=1, asthenosphere=2, crust=3.
IO_CORE_LAYER_INDEX = 0
IO_MANTLE_LAYER_INDEX = 1
IO_ASTHENOSPHERE_LAYER_INDEX = 2
IO_CRUST_LAYER_INDEX = 3

IO_R0_KM: tuple[float, float, float, float] = (965.0, 1591.6, 1791.6, 1821.6)
IO_RHO0: tuple[float, float, float, float] = (5150.0, 3244.0, 3244.0, 3244.0)
IO_MU0: tuple[float, float, float, float] = (0.0, 6e10, 7.8e5, 6.5e10)

# The MATLAB Consistency_test_Energy.m value (200e12 Pa). NOT the 200e16
# used by the pre-existing pylov3d/tests/test_energy.py fixture -- see the
# module docstring of scripts/io_energy_consistency.py for the audit of
# that discrepancy.
IO_KS0: tuple[float, float, float, float] = (0.0, 200e12, 200e12, 200e12)
IO_ETA0: tuple[float | None, float, float, float] = (None, 1e20, 1e11, 1e23)

IO_OMEGA0 = 4.1086e-05  # Io's orbital frequency [rad/s]
IO_FORCING_TD = 2.0 * math.pi / IO_OMEGA0  # Io's orbital period [s]

# (n, m, F) for the three eccentricity-tide forcing components.
IO_FORCING_COMPONENTS: tuple[tuple[int, int, float], ...] = (
    (2, 0, 3.0 / 4.0 * math.sqrt(1.0 / 5.0)),
    (2, -2, -7.0 / 8.0 * math.sqrt(6.0 / 5.0)),
    (2, 2, 1.0 / 8.0 * math.sqrt(6.0 / 5.0)),
)


def build_io_model() -> InteriorModel:
    """Build the 4-layer Io ``InteriorModel`` (core to surface).

    Matches ``tests/Consistency_test_Energy.m`` lines 143-168 exactly,
    including the MATLAB ``Ks0 = 200e12 Pa`` solid-layer bulk modulus (see
    :data:`IO_KS0`). ``Delta_rho0`` is left to
    :func:`pylov3d.types.make_interior_model`'s auto-fill, which sets the
    core's contrast to ``rho0[0] - rho0[1]`` -- the only entry
    :func:`pylov3d.rheology.normalize` actually reads (layers >= 1 always
    recompute ``rho[i-1] - rho[i]`` from the density array itself; see
    ``rheology.normalize``), so this is bit-for-bit equivalent to the
    MATLAB script's single ``Interior_Model(1).Delta_rho0`` assignment.
    """
    return make_interior_model(
        R0_km=list(IO_R0_KM),
        rho0=list(IO_RHO0),
        mu0=list(IO_MU0),
        Ks0=list(IO_KS0),
        eta0=list(IO_ETA0),
    )


def build_io_forcings(F: tuple[float, float, float] | None = None) -> list[Forcing]:
    """Build the three (2,0)/(2,-2)/(2,2) eccentricity-tide forcings.

    Parameters
    ----------
    F : optional override of the three forcing amplitudes (same order as
        :data:`IO_FORCING_COMPONENTS`); defaults to the MATLAB values.
    """
    amps = F if F is not None else tuple(f for _n, _m, f in IO_FORCING_COMPONENTS)
    return [
        make_forcing(Td=IO_FORCING_TD, n=n, m=m, F=amp)
        for (n, m, _f0), amp in zip(IO_FORCING_COMPONENTS, amps)
    ]


def io_default_numerics(Nrbase: int, n_layers: int = 4):
    """Convenience wrapper around :func:`pylov3d.types.make_numerics`.

    ``method='combination'``, ``perturbation_order=2``, ``rheology_cutoff=2.0``
    (the ``make_numerics`` default) and ``Nenergy=12`` match
    ``Consistency_test_Energy.m``'s ``Numerics`` block (``Numerics.Nenergy
    = 12``, the maximum degree to which energy dissipation is expanded).
    """
    return make_numerics(
        n_layers=n_layers, method="combination", Nrbase=Nrbase,
        perturbation_order=2, Nenergy=12,
    )


# ---------------------------------------------------------------------------
# Degree-2 heating pattern (tests/Consistency_test_Energy.m lines 39-96)
# ---------------------------------------------------------------------------

# MATLAB fac() per associated-Legendre order (0, 1, 2); order-1 is zero.
_FAC_ORDER0 = -33.0 / 7.0
_FAC_ORDER2 = 9.0 / 14.0

Q_MEAN = 2.3  # W m^-2
MELT_C = 0.01  # D2-0 model: Phi_diff = c * Q_diff
PHI_MEAN = 0.1
B_ETA = 20.0
B_MU = 67.0 / 15.0


def io_matlab_p2_legendre(colat: np.ndarray) -> dict[int, np.ndarray]:
    r"""MATLAB ``legendre(2, cos(colat))`` rows :math:`m=0,1,2`.

    MATLAB's ``legendre(N, X)`` returns the *unnormalized* associated
    Legendre functions :math:`P_N^m(X)` **with** the Condon-Shortley phase
    :math:`(-1)^m` built in -- the same convention as
    ``scipy.special.lpmv(m, n, x)`` (verified numerically in
    ``pylov3d/tests/test_io_lateral.py::test_lpmv_matches_matlab_legendre``,
    e.g. :math:`P_2^1(0.5) = -1.299038...` both ways). Do **not** use
    :func:`pylov3d.mapping.fully_normalized_legendre` here -- that is a
    *different*, no-Condon-Shortley, 4pi-normalized convention (see this
    module's docstring).
    """
    x = np.cos(colat)
    return {m: lpmv(m, 2, x) for m in (0, 1, 2)}


def io_heating_grid(l_max: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the MATLAB lat/lon grid and ``z(lat, lon)`` heating pattern.

    Faithful, numerical port of ``tests/Consistency_test_Energy.m`` lines
    39-96 (the "Starting-pattern" block) -- not an approximation of it.
    ``l_max=100`` gives ``delta = 180 / (2*(2*l_max - 1))`` degrees per the
    spec.

    Grid asymmetry note (replicated verbatim, not a bug in this port): the
    MATLAB script computes ``latM = 90 - delta_lon/2`` -- it reuses the
    *longitude* step for the latitude upper bound, not ``delta_lat``.
    Because ``delta_lon`` and ``delta_lat`` are defined by the identical
    formula (``180/(2*(2*l_max-1))``), the two are numerically equal, so
    this line has no actual effect on the grid; it is a code-style artifact
    of the original script, preserved here for exact reproduction.

    Returns
    -------
    lat_deg : (nlat,) float, ascending, cell-centered, degrees
    lon_deg : (nlon,) float, ascending, cell-centered, degrees
    z : (nlat, nlon) float
        The dimensionless heating-pattern factor (spatial mean == 1 to
        floating-point precision, since only zero-mean degree-2 terms
        enter ``gv``).
    """
    delta_lon = 180.0 / (2.0 * (2 * l_max - 1))
    delta_lat = 180.0 / (2.0 * (2 * l_max - 1))
    lon0 = -180.0 + delta_lon / 2.0
    lonM = 180.0 - delta_lon / 2.0
    lat0 = -90.0 + delta_lat / 2.0
    latM = 90.0 - delta_lon / 2.0  # MATLAB reuses delta_lon here; see docstring

    n_lon = int(round((lonM - lon0) / delta_lon)) + 1
    n_lat = int(round((latM - lat0) / delta_lat)) + 1
    lon_deg = lon0 + delta_lon * np.arange(n_lon)
    lat_deg = lat0 + delta_lat * np.arange(n_lat)

    colat = np.radians(90.0 - lat_deg)
    lon_rad = np.radians(lon_deg)
    z = _io_z_pattern(colat[:, None], lon_rad[None, :])
    return lat_deg, lon_deg, z


def _io_z_pattern(colat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Evaluate the MATLAB ``z`` heating-pattern factor at arbitrary points.

    ``colat``, ``lon`` in radians, broadcastable (e.g. ``colat`` shape
    ``(nlat, 1)`` and ``lon`` shape ``(1, nlon)`` for a grid, or matching
    1-D shapes for scattered evaluation such as quadrature nodes).
    Analytically identical to the ``gv``/``z_val`` block of
    ``tests/Consistency_test_Energy.m`` lines 61-84: only the order-0 and
    order-2 associated-Legendre terms are nonzero (``fac(order=1) == 0``),
    so ``gv = fac0 * P_2^0(cos colat) + fac2 * P_2^2(cos colat) * cos(2*lon)``
    with ``phase = 0``, and both sign factors ``(-1)^(m_idx-1)`` are ``+1``
    for ``m_idx in {1, 3}`` (orders 0 and 2).
    """
    P0 = lpmv(0, 2, np.cos(colat))
    P2 = lpmv(2, 2, np.cos(colat))
    gv = _FAC_ORDER0 * P0 + _FAC_ORDER2 * P2 * np.cos(2.0 * lon)
    return (21.0 / 5.0 + 0.5 * gv) / (21.0 / 5.0)


def _io_dmu_deta(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``z`` -> fractional ``(dmu, deta)`` via the MATLAB Q/Phi/eta/mu chain.

    ``dmu = mu_rel - 1``, ``deta = eta_rel - 1`` (fractional, the
    ``process_lateral_variations`` convention: field = 1 + fractional).
    Matches ``tests/Consistency_test_Energy.m`` lines 98-114.
    """
    Q_diff = (z - 1.0) * Q_MEAN
    Phi_diff = MELT_C * Q_diff
    eta_rel = np.exp(-B_ETA * Phi_diff)
    mu_rel = (1.0 + B_MU * PHI_MEAN) / (1.0 + B_MU * (Phi_diff + PHI_MEAN))
    return mu_rel - 1.0, eta_rel - 1.0


def io_mu_eta_grids(l_max: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB-grid ``(lat_deg, lon_deg, dmu, deta)`` fractional fields."""
    lat_deg, lon_deg, z = io_heating_grid(l_max=l_max)
    dmu, deta = _io_dmu_deta(z)
    return lat_deg, lon_deg, dmu, deta


# ---------------------------------------------------------------------------
# Real 4pi-normalized degree spectrum (reporting only; see module docstring)
# ---------------------------------------------------------------------------

def _degree_power_spectrum(real_coeffs: RealSH, lmax: int) -> dict[int, float]:
    """RMS amplitude per degree: ``sqrt(sum_m c_nm^2)`` (4pi real convention).

    Same convention as ``pylov3d.mars_mantle.unit_rms_coefficients``'s
    docstring ("spherical mean square is exactly the sum of squared real
    C/S coefficients").
    """
    spectrum: dict[int, float] = {}
    for n in range(1, lmax + 1):
        acc = 0.0
        for m in range(0, n + 1):
            acc += real_coeffs.get((n, m), 0.0) ** 2
            if m >= 1:
                acc += real_coeffs.get((n, -m), 0.0) ** 2
        spectrum[n] = math.sqrt(acc)
    return spectrum


# ---------------------------------------------------------------------------
# Solver-facing entries: sph_harm_y-basis analysis (see module docstring)
# ---------------------------------------------------------------------------

def io_pattern_lateral_fields(
    lmax_sh: int = 12, l_max_report: int = 100,
) -> dict:
    """Build the Io asthenosphere ``mu_variable``/``eta_variable`` entries.

    Two independent computations, deliberately kept separate (see module
    docstring):

    1. **Solver-facing entries** (``mu_variable_entries``,
       ``eta_variable_entries``): the target ``(dmu, deta)`` field is
       evaluated analytically on ``pylov3d.rheology._sh_grid(lmax_sh)``'s
       own quadrature nodes and decomposed with
       ``pylov3d.rheology._sh_analysis`` -- the exact inverse of the
       ``_sh_synthesis`` call ``process_lateral_variations`` uses
       internally for viscoelastic layers, so these entries reproduce the
       intended physical field exactly (to quadrature truncation at
       ``lmax_sh``), independent of any cross-basis normalization.
    2. **Reporting spectrum** (``mu_degree_spectrum``,
       ``eta_degree_spectrum``): the MATLAB-grid field
       (:func:`io_mu_eta_grids`, ``l_max_report``) projected with
       ``pylov3d.mars_mantle.project_temperature_real_sh`` (real,
       4pi-normalized). Per-degree power *ratios* from this projection
       equal those of the sph_harm_y decomposition by Parseval's theorem
       (both are complete orthonormal-up-to-a-constant expansions of the
       same field), even though the absolute values differ by basis.

    Returns
    -------
    dict with keys ``mu_variable_entries``, ``eta_variable_entries``
    (``list[(n, m, complex)]``, ready for
    ``{IO_ASTHENOSPHERE_LAYER_INDEX: entries}``), ``mu_degree_spectrum``,
    ``eta_degree_spectrum`` (``dict[int, float]``), ``mu_real_sh``,
    ``eta_real_sh`` (the raw real 4pi-normalized coefficient dicts), and
    the report-grid ``lat_deg``/``lon_deg``/``dmu_grid``/``deta_grid``.
    """
    # --- 1. Solver-facing entries (sph_harm_y basis, exact inverse pair) --
    theta, phi, weights = _sh_grid(lmax_sh)
    colat_grid = theta[:, None]
    lon_grid = phi[None, :]
    z_gl = _io_z_pattern(colat_grid, lon_grid)
    dmu_gl, deta_gl = _io_dmu_deta(z_gl)

    mu_sh = _sh_analysis(dmu_gl.astype(complex), theta, phi, weights, lmax_sh)
    eta_sh = _sh_analysis(deta_gl.astype(complex), theta, phi, weights, lmax_sh)

    # Drop floating-point noise coefficients (the pattern is analytically
    # even-degree-only; odd degrees and m=1 come back at ~1e-16..1e-30
    # relative to the degree-2 term from quadrature roundoff). A relative
    # cutoff, not an absolute one, so this scales with either field's own
    # amplitude; process_lateral_variations applies its own
    # (post-synthesis) rheology_cutoff filter regardless, so this is a
    # performance/tidiness step, not a physics decision.
    _NOISE_REL = 1e-9
    mu_max = max((abs(v) for v in mu_sh.values()), default=0.0)
    eta_max = max((abs(v) for v in eta_sh.values()), default=0.0)
    mu_entries = [
        (n, m, amp) for (n, m), amp in sorted(mu_sh.items())
        if n > 0 and abs(amp) > _NOISE_REL * mu_max
    ]
    eta_entries = [
        (n, m, amp) for (n, m), amp in sorted(eta_sh.items())
        if n > 0 and abs(amp) > _NOISE_REL * eta_max
    ]

    # --- 2. Reporting spectrum (real 4pi-normalized convention) ----------
    lat_deg, lon_deg, dmu_grid, deta_grid = io_mu_eta_grids(l_max=l_max_report)
    mu_real_sh = project_temperature_real_sh(dmu_grid, lat_deg, lon_deg, lmax=lmax_sh)
    eta_real_sh = project_temperature_real_sh(deta_grid, lat_deg, lon_deg, lmax=lmax_sh)

    return {
        "mu_variable_entries": mu_entries,
        "eta_variable_entries": eta_entries,
        "mu_real_sh": mu_real_sh,
        "eta_real_sh": eta_real_sh,
        "mu_degree_spectrum": _degree_power_spectrum(mu_real_sh, lmax_sh),
        "eta_degree_spectrum": _degree_power_spectrum(eta_real_sh, lmax_sh),
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "dmu_grid": dmu_grid,
        "deta_grid": deta_grid,
    }


def io_mu_eta_variable(
    lmax_sh: int = 12, l_max_report: int = 100,
) -> tuple[dict[int, list], dict[int, list], dict]:
    """Convenience wrapper: ``(mu_variable, eta_variable, diagnostics)``.

    ``mu_variable``/``eta_variable`` are ready to pass to
    ``process_lateral_variations``/``get_love`` as
    ``{IO_ASTHENOSPHERE_LAYER_INDEX: entries}``.
    """
    fields = io_pattern_lateral_fields(lmax_sh=lmax_sh, l_max_report=l_max_report)
    mu_variable = {IO_ASTHENOSPHERE_LAYER_INDEX: fields["mu_variable_entries"]}
    eta_variable = {IO_ASTHENOSPHERE_LAYER_INDEX: fields["eta_variable_entries"]}
    return mu_variable, eta_variable, fields
