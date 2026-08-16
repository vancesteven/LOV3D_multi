# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Pure data contracts for the Mars upper-mantle thermal pilot.

The source product is Data Set S1 of Plesa et al. (2018).  It contains a
present-day temperature map at 150 km depth; it does not contain the 400 km
slice originally proposed in TASK-042.  The raw product is intentionally not
bundled here because its redistribution terms have not been cleared.

This module stops at data preparation and algebraic diagnostics.  It does not
run a radial or coupled Love-number solve.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from .couplings import get_active_modes
from .mapping import fully_normalized_legendre
from .mars_lateral import (
    CRUST_LAYER_INDEX,
    _real_sh_to_complex_mu_variable,
    complex_sh_synthesis,
)


UPPER_MANTLE_LAYER_INDEX = 2
"""Upper-mantle index in :func:`pylov3d.mars.build_mars_model`."""

PLESAS1_NLAT = 180
PLESAS1_NLON = 360
PLESAS1_NROWS = PLESAS1_NLAT * PLESAS1_NLON
PLESAS1_NCOLS = 8
PLESAS1_T150_COLUMN = 6

# TASK-042's initial olivine constitutive prior, divided by the committed
# upper-mantle reference shear modulus.
BETA_MU_PER_K = -1.92485e-4

EXPECTED_L2_ACTIVE_MODES = 43

RealSH = dict[tuple[int, int], float]
ComplexSHEntries = list[tuple[int, int, complex]]


class PlesaS1Grid(NamedTuple):
    """Validated Data Set S1 fields on a latitude-descending regular grid."""

    lon_e_deg: np.ndarray
    lat_deg: np.ndarray
    crustal_thickness_km: np.ndarray
    surface_heat_flow_mw_m2: np.ndarray
    elastic_thickness_1e14_km: np.ndarray
    elastic_thickness_1e17_km: np.ndarray
    temperature_150km_k: np.ndarray
    depth_1370k_isotherm_km: np.ndarray


class PositivityDiagnostics(NamedTuple):
    """Grid and coefficient bounds for ``mu/mu_bar = 1 + f``."""

    minimum_mu_factor: float
    maximum_mu_factor: float
    minimum_lat_deg: float
    minimum_lon_e_deg: float
    maximum_imaginary_residual: float
    coefficient_upper_bound: float
    grid_margin: float
    coefficient_margin: float
    grid_passes: bool
    coefficient_passes: bool
    passes: bool


class DistinguishabilityMetrics(NamedTuple):
    """Whitened thermal-versus-crust Jacobian geometry."""

    thermal_whitened: np.ndarray
    crust_whitened: np.ndarray
    thermal_norm: float
    crust_norm: float
    correlation: float
    orthogonal_thermal: np.ndarray
    orthogonal_norm_per_unit: float
    max_orthogonal_sigma: float
    correlation_warning: bool
    passes_one_sigma: bool


def load_plesa_s1(path: str | Path) -> PlesaS1Grid:
    """Load and validate the 64,800 x 8 ASCII Plesa Data Set S1 product.

    Columns are longitude east [deg], latitude [deg], crustal thickness [km],
    surface heat flow [mW/m2], elastic thickness at strain rates 1e-14 and
    1e-17 1/s [km], temperature at 150 km depth [K], and depth to the 1370 K
    isotherm [km].  Comment lines beginning with ``#`` are handled by
    :func:`numpy.loadtxt`.

    The archive's north-pole row is longitude-rotated, so validation uses the
    complete coordinate set rather than assuming a row-major input ordering.
    Returned arrays always use latitude 90..-89 and longitude 0..359 east.
    """

    rows = np.loadtxt(Path(path), comments="#", dtype=float)
    if rows.shape != (PLESAS1_NROWS, PLESAS1_NCOLS):
        raise ValueError(
            "Plesa S1 must have shape "
            f"({PLESAS1_NROWS}, {PLESAS1_NCOLS}), got {rows.shape}"
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("Plesa S1 contains non-finite values")

    lon = rows[:, 0]
    lat = rows[:, 1]
    lon_i = np.rint(lon).astype(int)
    lat_i = np.rint(lat).astype(int)
    if not np.allclose(lon, lon_i, rtol=0.0, atol=1e-10):
        raise ValueError("Plesa S1 longitude coordinates must be integer degrees")
    if not np.allclose(lat, lat_i, rtol=0.0, atol=1e-10):
        raise ValueError("Plesa S1 latitude coordinates must be integer degrees")
    if np.any((lon_i < 0) | (lon_i >= PLESAS1_NLON)):
        raise ValueError("Plesa S1 longitude must span 0..359 degrees east")
    if np.any((lat_i > 90) | (lat_i < -89)):
        raise ValueError("Plesa S1 latitude must span 90..-89 degrees")

    lat_index = 90 - lat_i
    flat_index = lat_index * PLESAS1_NLON + lon_i
    if np.unique(flat_index).size != PLESAS1_NROWS:
        raise ValueError("Plesa S1 coordinate grid has duplicates or missing cells")

    gridded = np.empty((PLESAS1_NLAT, PLESAS1_NLON, 6), dtype=float)
    gridded[lat_index, lon_i, :] = rows[:, 2:]
    lat_out = np.arange(90.0, -90.0, -1.0)
    lon_out = np.arange(0.0, 360.0, 1.0)
    return PlesaS1Grid(
        lon_e_deg=lon_out,
        lat_deg=lat_out,
        crustal_thickness_km=gridded[:, :, 0],
        surface_heat_flow_mw_m2=gridded[:, :, 1],
        elastic_thickness_1e14_km=gridded[:, :, 2],
        elastic_thickness_1e17_km=gridded[:, :, 3],
        temperature_150km_k=gridded[:, :, PLESAS1_T150_COLUMN - 2],
        depth_1370k_isotherm_km=gridded[:, :, 5],
    )


def area_weighted_mean(field: np.ndarray, lat_deg: np.ndarray) -> float:
    """Area-weighted mean on an equally spaced longitude grid."""

    values = np.asarray(field, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    if values.ndim != 2 or lat.ndim != 1 or values.shape[0] != lat.size:
        raise ValueError("field must be (nlat, nlon) and match lat_deg")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(lat)):
        raise ValueError("field and latitude coordinates must be finite")
    if np.any(np.abs(lat) > 90.0):
        raise ValueError("latitude coordinates must lie within [-90, 90]")

    weights = np.clip(np.cos(np.radians(lat)), 0.0, None)
    denominator = float(np.sum(weights) * values.shape[1])
    if denominator <= 0.0:
        raise ValueError("latitude grid has zero total area weight")
    return float(np.sum(values * weights[:, None]) / denominator)


def remove_area_weighted_mean(field: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Return a float copy of ``field`` with its spherical mean removed."""

    values = np.asarray(field, dtype=float)
    return values - area_weighted_mean(values, lat_deg)


def _real_mode_keys(lmax: int, include_mean: bool = False) -> list[tuple[int, int]]:
    if lmax < 1:
        raise ValueError("lmax must be at least 1")
    keys: list[tuple[int, int]] = [(0, 0)] if include_mean else []
    for n in range(1, lmax + 1):
        keys.append((n, 0))
        for m in range(1, n + 1):
            keys.extend(((n, m), (n, -m)))
    return keys


def project_temperature_real_sh(
    temperature_k: np.ndarray,
    lat_deg: np.ndarray,
    lon_e_deg: np.ndarray,
    lmax: int = 4,
) -> RealSH:
    """Project a temperature grid into real, 4pi-normalized SH coefficients.

    The area-weighted mean is removed first.  A weighted least-squares
    projection is used because S1 is an equal-angle node grid rather than a
    Gauss-Legendre quadrature grid.  Positive ``m`` keys are cosine terms and
    negative ``m`` keys are sine terms, matching :func:`mapping.sh_to_latlon`.
    Degrees 1 through ``lmax`` are returned; degree zero is never returned.
    """

    field = np.asarray(temperature_k, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_e_deg, dtype=float)
    if field.ndim != 2 or lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("temperature must be 2-D with 1-D latitude/longitude")
    if field.shape != (lat.size, lon.size):
        raise ValueError("temperature shape must match latitude/longitude")
    if not np.all(np.isfinite(lon)):
        raise ValueError("longitude coordinates must be finite")

    centered = remove_area_weighted_mean(field, lat)
    keys = _real_mode_keys(lmax, include_mean=True)
    P = fully_normalized_legendre(lmax, np.sin(np.radians(lat)))
    lon_rad = np.radians(lon)
    design = np.empty((field.size, len(keys)), dtype=float)
    for column, (n, m) in enumerate(keys):
        mm = abs(m)
        angular = np.cos(mm * lon_rad) if m >= 0 else np.sin(mm * lon_rad)
        design[:, column] = (P[n, mm, :, None] * angular[None, :]).ravel()

    sqrt_weights = np.sqrt(np.clip(np.cos(np.radians(lat)), 0.0, None))
    point_weights = np.repeat(sqrt_weights, lon.size)
    coefficients, *_ = np.linalg.lstsq(
        design * point_weights[:, None],
        centered.ravel() * point_weights,
        rcond=None,
    )
    return {
        key: float(coefficients[index])
        for index, key in enumerate(keys)
        if key != (0, 0)
    }


def unit_rms_coefficients(coefficients: Mapping[tuple[int, int], float], cutoff: int) -> RealSH:
    """Truncate at ``cutoff`` and normalize to unit spherical RMS.

    In the repository's real 4pi-normalized basis, spherical mean square is
    exactly the sum of squared real C/S coefficients.
    """

    if cutoff < 1:
        raise ValueError("cutoff must be at least 1")
    selected: RealSH = {}
    for (n, m), amplitude in coefficients.items():
        if n == 0:
            raise ValueError("degree-zero coefficients are forbidden in the template")
        if n < 0 or abs(m) > n:
            raise ValueError(f"invalid spherical-harmonic mode ({n}, {m})")
        value = float(amplitude)
        if not math.isfinite(value):
            raise ValueError("template coefficients must be finite")
        if n <= cutoff:
            selected[(n, m)] = value
    rms = math.sqrt(sum(value * value for value in selected.values()))
    if rms <= 0.0:
        raise ValueError("truncated template has zero RMS")
    return {mode: value / rms for mode, value in selected.items()}


def unit_rms_coefficients_by_cutoff(
    coefficients: Mapping[tuple[int, int], float],
    cutoffs: Sequence[int] = (1, 2, 3, 4),
) -> dict[int, RealSH]:
    """Return independently unit-normalized templates for each cutoff."""

    return {int(cutoff): unit_rms_coefficients(coefficients, int(cutoff)) for cutoff in cutoffs}


def thermal_fractional_coefficients(
    unit_template: Mapping[tuple[int, int], float],
    amplitude_k: float,
    beta_mu_per_k: float = BETA_MU_PER_K,
) -> RealSH:
    """Map signed RMS temperature amplitude to fractional ``delta mu / mu``."""

    amplitude = float(amplitude_k)
    beta = float(beta_mu_per_k)
    if not math.isfinite(amplitude) or not math.isfinite(beta):
        raise ValueError("thermal amplitude and beta_mu must be finite")
    template: RealSH = {}
    for (n, m), value_in in unit_template.items():
        if n <= 0 or abs(m) > n:
            raise ValueError(f"invalid template mode ({n}, {m}); degree zero is forbidden")
        value = float(value_in)
        if not math.isfinite(value):
            raise ValueError("template coefficients must be finite")
        template[(n, m)] = value
    template_rms = math.sqrt(sum(value * value for value in template.values()))
    if not math.isclose(template_rms, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"unit_template must have unit RMS, got {template_rms:.16g}")
    scale = beta * amplitude
    return {mode: scale * value for mode, value in template.items()}


def thermal_mu_variable(
    unit_template: Mapping[tuple[int, int], float],
    amplitude_k: float,
    beta_mu_per_k: float = BETA_MU_PER_K,
) -> dict[int, ComplexSHEntries]:
    """Build the solver-facing upper-mantle ``mu_variable`` dictionary."""

    fractional = thermal_fractional_coefficients(unit_template, amplitude_k, beta_mu_per_k)
    return {UPPER_MANTLE_LAYER_INDEX: _real_sh_to_complex_mu_variable(fractional)}


def positivity_diagnostics(
    fractional_coefficients: Mapping[tuple[int, int], float],
    epsilon: float = 1e-6,
    nlat: int = 181,
    nlon: int = 360,
) -> PositivityDiagnostics:
    """Evaluate grid and rigorous coefficient positivity margins.

    The pole-inclusive grid tests the actual low-degree field.  The second
    guard uses ``|Y_nm| <= sqrt(2n+1)``, a direct consequence of the 4pi
    addition theorem, and therefore does not rely on the sampled extrema.
    Both guards must pass.
    """

    eps = float(epsilon)
    if not (0.0 < eps < 1.0):
        raise ValueError("epsilon must lie strictly between zero and one")
    if nlat < 2 or nlon < 4:
        raise ValueError("positivity grid must have nlat >= 2 and nlon >= 4")

    real_coefficients: RealSH = {}
    for (n, m), amplitude in fractional_coefficients.items():
        if n <= 0 or abs(m) > n:
            raise ValueError(f"invalid lateral mode ({n}, {m}); degree zero is forbidden")
        value = float(amplitude)
        if not math.isfinite(value):
            raise ValueError("fractional coefficients must be finite")
        real_coefficients[(n, m)] = value

    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    entries = _real_sh_to_complex_mu_variable(real_coefficients)
    fractional_grid = complex_sh_synthesis(entries, lat, lon)
    mu_factor = 1.0 + fractional_grid.real
    minimum_index = np.unravel_index(int(np.argmin(mu_factor)), mu_factor.shape)
    minimum = float(mu_factor[minimum_index])
    maximum = float(np.max(mu_factor))
    imag_residual = float(np.max(np.abs(fractional_grid.imag)))

    coefficient_bound = float(
        sum(abs(value) * math.sqrt(2 * n + 1) for (n, _m), value in real_coefficients.items())
    )
    grid_margin = minimum - eps
    coefficient_margin = 1.0 - eps - coefficient_bound
    grid_passes = grid_margin > 0.0 and imag_residual < 1e-10
    coefficient_passes = coefficient_margin > 0.0
    return PositivityDiagnostics(
        minimum_mu_factor=minimum,
        maximum_mu_factor=maximum,
        minimum_lat_deg=float(lat[minimum_index[0]]),
        minimum_lon_e_deg=float(lon[minimum_index[1]]),
        maximum_imaginary_residual=imag_residual,
        coefficient_upper_bound=coefficient_bound,
        grid_margin=grid_margin,
        coefficient_margin=coefficient_margin,
        grid_passes=grid_passes,
        coefficient_passes=coefficient_passes,
        passes=grid_passes and coefficient_passes,
    )


def merge_mantle_crust_mu_variable(
    mantle_mu_variable: Mapping[int, Sequence[tuple[int, int, complex]]],
    crust_mu_variable: Mapping[int, Sequence[tuple[int, int, complex]]],
) -> dict[int, ComplexSHEntries]:
    """Merge the two expected Mars lateral layers without combining amplitudes."""

    if set(mantle_mu_variable) != {UPPER_MANTLE_LAYER_INDEX}:
        raise ValueError(f"mantle mu_variable must contain only layer {UPPER_MANTLE_LAYER_INDEX}")
    if set(crust_mu_variable) != {CRUST_LAYER_INDEX}:
        raise ValueError(f"crust mu_variable must contain only layer {CRUST_LAYER_INDEX}")
    return {
        UPPER_MANTLE_LAYER_INDEX: list(mantle_mu_variable[UPPER_MANTLE_LAYER_INDEX]),
        CRUST_LAYER_INDEX: list(crust_mu_variable[CRUST_LAYER_INDEX]),
    }


def active_modes_for_mu_variable(
    mu_variable: Mapping[int, Sequence[tuple[int, int, complex]]],
    forcing: tuple[int, int] = (2, 0),
    perturbation_order: int = 2,
) -> np.ndarray:
    """Return coupled mode closure using the production selection algorithm."""

    support: set[tuple[int, int]] = set()
    for entries in mu_variable.values():
        entries_list = list(entries)
        if not any(abs(complex(amplitude)) > 0.0 for _n, _m, amplitude in entries_list):
            continue
        for n, m, _amplitude in entries_list:
            if n > 0:
                support.add((int(n), int(m)))
    if not support:
        return np.array([[int(forcing[0]), int(forcing[1]), 0]], dtype=int)
    variations = np.asarray(sorted(support), dtype=int)
    return get_active_modes(
        int(perturbation_order), variations, int(forcing[0]), int(forcing[1])
    )


def l2_mode_closure(
    mu_variable: Mapping[int, Sequence[tuple[int, int, complex]]],
    forcing: tuple[int, int] = (2, 0),
    perturbation_order: int = 2,
) -> np.ndarray:
    """Return closure for a degree-2-or-lower lateral model."""

    if any(n > 2 for entries in mu_variable.values() for n, _m, _a in entries):
        raise ValueError("L=2 closure received a mode above degree 2")
    return active_modes_for_mu_variable(mu_variable, forcing, perturbation_order)


def jacobian_distinguishability(
    thermal_jacobian: np.ndarray,
    crust_jacobian: np.ndarray,
    covariance: np.ndarray,
    max_abs_thermal_amplitude: float,
    correlation_threshold: float = 0.95,
) -> DistinguishabilityMetrics:
    """Whiten two Jacobians and measure thermal signal orthogonal to crust.

    The thermal pilot passes the algebraic one-sigma gate when the largest
    admitted thermal amplitude times the whitened orthogonal norm is at least
    one.  The correlation threshold is a warning, not the decision rule.
    """

    thermal = np.asarray(thermal_jacobian, dtype=float)
    crust = np.asarray(crust_jacobian, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if thermal.ndim != 1 or crust.ndim != 1 or thermal.shape != crust.shape:
        raise ValueError("thermal and crust Jacobians must be same-length vectors")
    if cov.shape != (thermal.size, thermal.size):
        raise ValueError("covariance shape must match the Jacobian length")
    if not np.all(np.isfinite(thermal)) or not np.all(np.isfinite(crust)):
        raise ValueError("Jacobians must be finite")
    if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-12, atol=0.0):
        raise ValueError("covariance must be finite and symmetric")
    amplitude = float(max_abs_thermal_amplitude)
    threshold = float(correlation_threshold)
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("max_abs_thermal_amplitude must be finite and non-negative")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("correlation_threshold must lie in [0, 1]")

    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    thermal_w = np.linalg.solve(chol, thermal)
    crust_w = np.linalg.solve(chol, crust)
    thermal_norm = float(np.linalg.norm(thermal_w))
    crust_norm = float(np.linalg.norm(crust_w))
    if thermal_norm == 0.0:
        raise ValueError("thermal Jacobian has zero whitened norm")
    if crust_norm == 0.0:
        raise ValueError("crust Jacobian has zero whitened norm")

    correlation = float(np.dot(thermal_w, crust_w) / (thermal_norm * crust_norm))
    # Clip only roundoff outside the mathematical interval.
    correlation = float(np.clip(correlation, -1.0, 1.0))
    projection = float(np.dot(thermal_w, crust_w) / np.dot(crust_w, crust_w))
    orthogonal = thermal_w - projection * crust_w
    orthogonal_norm = float(np.linalg.norm(orthogonal))
    max_sigma = amplitude * orthogonal_norm
    return DistinguishabilityMetrics(
        thermal_whitened=thermal_w,
        crust_whitened=crust_w,
        thermal_norm=thermal_norm,
        crust_norm=crust_norm,
        correlation=correlation,
        orthogonal_thermal=orthogonal,
        orthogonal_norm_per_unit=orthogonal_norm,
        max_orthogonal_sigma=max_sigma,
        correlation_warning=abs(correlation) >= threshold,
        passes_one_sigma=max_sigma >= 1.0,
    )
