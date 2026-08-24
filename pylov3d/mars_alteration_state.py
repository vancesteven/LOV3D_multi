# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Common physical state for Mars alteration forward models.

Gravity, seismic properties and tides should consume the same candidate
alteration field rather than independent hand-tuned perturbations. This module
provides the first narrow bridge: a hydrated fraction of locally reactive rock
is mapped to rho, K and mu, then the radially resolved elastic field is reduced
to the effective lateral shell properties consumed by LOV3D.

Gravity should continue to use the full radial density field through
``mars_alteration_gravity``. The tidal reduction is a first-order layer-average
because the current Mars LOV3D reference model represents the crust as one
radial layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars_alteration_gravity import _remove_matlab_half_cell_phase
from .mars_lateral import _real_sh_to_complex_mu_variable
from .matlab_sph import grid_to_stokes


@dataclass(frozen=True)
class AlterationEndmembers:
    rho_dry_kg_m3: float
    rho_hydrated_kg_m3: float
    mu_dry_Pa: float
    mu_hydrated_Pa: float
    K_dry_Pa: float
    K_hydrated_Pa: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ElasticLayerState:
    """Mean elastic properties plus lateral fractional SH perturbations."""

    mean_mu_Pa: float
    mean_K_Pa: float
    mu_variable: dict[int, list[tuple[int, int, complex]]]
    K_variable: dict[int, list[tuple[int, int, complex]]]


def alteration_bulk_fraction(hydrated_fraction, reactive_fraction) -> np.ndarray:
    """Return bulk-rock fraction occupied by hydrated reactive material."""
    fh = np.asarray(hydrated_fraction, dtype=float)
    fr = np.asarray(reactive_fraction, dtype=float)
    for name, arr in (("hydrated_fraction", fh), ("reactive_fraction", fr)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1):
            raise ValueError(f"{name} must be finite and lie in [0,1]")
    try:
        return np.asarray(fh * fr, dtype=float)
    except ValueError as exc:
        raise ValueError("hydrated_fraction and reactive_fraction must be broadcastable") from exc


def _mix_modulus(x: np.ndarray, dry: float, hydrated: float, law: str) -> np.ndarray:
    voigt = (1.0 - x) * dry + x * hydrated
    if law == "voigt":
        return voigt
    reuss = 1.0 / ((1.0 - x) / dry + x / hydrated)
    if law == "reuss":
        return reuss
    if law == "hill":
        return 0.5 * (voigt + reuss)
    raise ValueError("mixing_law must be 'voigt', 'reuss', or 'hill'")


def alteration_material_fields(
    hydrated_fraction,
    reactive_fraction,
    endmembers: AlterationEndmembers,
    *,
    mixing_law: str = "hill",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho, mu, K)`` for one candidate alteration state.

    Density uses direct volume mixing. Elastic moduli use the requested
    Voigt/Reuss/Hill bound. This is intentionally a transparent material bridge,
    not a substitute for the planned Perple_X/rock-physics calculation.
    """
    x = alteration_bulk_fraction(hydrated_fraction, reactive_fraction)
    rho = (1.0 - x) * endmembers.rho_dry_kg_m3 + x * endmembers.rho_hydrated_kg_m3
    mu = _mix_modulus(x, endmembers.mu_dry_Pa, endmembers.mu_hydrated_Pa, mixing_law)
    K = _mix_modulus(x, endmembers.K_dry_Pa, endmembers.K_hydrated_Pa, mixing_law)
    return np.asarray(rho), np.asarray(mu), np.asarray(K)


def radial_volume_average(field: np.ndarray, radius_edges_m: np.ndarray) -> np.ndarray:
    """Average a ``(..., nz)`` field over radius using exact dV/dOmega weights."""
    a = np.asarray(field, dtype=float)
    r = np.asarray(radius_edges_m, dtype=float)
    if a.ndim < 1 or r.ndim != 1 or r.size != a.shape[-1] + 1:
        raise ValueError("radius_edges_m must have length nz+1 for field[..., nz]")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
        raise ValueError("field and radius edges must be finite")
    if np.any(np.diff(r) <= 0) or np.any(r <= 0):
        raise ValueError("radius edges must be positive and increase outward")
    weights = (r[1:] ** 3 - r[:-1] ** 3) / 3.0
    return np.sum(a * weights, axis=-1) / np.sum(weights)


def _physical_grid_to_stokes(grid: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(grid, dtype=float)
    expected = (2 * lmax, 4 * lmax)
    if arr.shape != expected:
        raise ValueError(f"physical grid must have shape {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("physical grid must be finite")
    c, s = grid_to_stokes(arr, lmax)
    return _remove_matlab_half_cell_phase(c, s, lmax)


def _fractional_complex_modes(
    grid: np.ndarray,
    mean: float,
    lmax: int,
    *,
    relative_floor: float = 1e-13,
) -> list[tuple[int, int, complex]]:
    """Return fractional solver modes, dropping pure transform roundoff."""
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("mean material property must be positive")
    if relative_floor < 0:
        raise ValueError("relative_floor must be non-negative")
    c, s = _physical_grid_to_stokes(grid, lmax)
    real_coeffs: dict[tuple[int, int], float] = {}
    for degree in range(1, lmax + 1):
        for order in range(degree + 1):
            cv = float(c[degree, order] / mean)
            if abs(cv) > relative_floor:
                real_coeffs[(degree, order)] = cv
            if order > 0:
                sv = float(s[degree, order] / mean)
                if abs(sv) > relative_floor:
                    real_coeffs[(degree, -order)] = sv
    return _real_sh_to_complex_mu_variable(real_coeffs)


def elastic_layer_state_from_3d(
    mu_Pa: np.ndarray,
    K_Pa: np.ndarray,
    radius_edges_m: np.ndarray,
    lmax: int,
    *,
    layer_index: int = 3,
) -> ElasticLayerState:
    """Reduce a 3D elastic field to one LOV3D layer's mean+lateral state.

    Input arrays use shape ``(2*lmax, 4*lmax, nz)``. The radial reduction is
    exact volume-per-solid-angle weighting. Degree zero becomes the layer mean;
    degree >=1 coefficients are normalized by that mean and converted to the
    complex harmonic convention expected by ``process_lateral_variations``.
    """
    mu = np.asarray(mu_Pa, dtype=float)
    K = np.asarray(K_Pa, dtype=float)
    if mu.shape != K.shape or mu.ndim != 3:
        raise ValueError("mu_Pa and K_Pa must be identical 3-D arrays")
    expected_surface = (2 * lmax, 4 * lmax)
    if mu.shape[:2] != expected_surface:
        raise ValueError(f"surface grid must have shape {expected_surface}")
    if np.any(mu <= 0) or np.any(K <= 0) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(K)):
        raise ValueError("elastic moduli must be finite and positive")

    mu_grid = radial_volume_average(mu, radius_edges_m)
    K_grid = radial_volume_average(K, radius_edges_m)
    c_mu, _ = _physical_grid_to_stokes(mu_grid, lmax)
    c_K, _ = _physical_grid_to_stokes(K_grid, lmax)
    mean_mu = float(c_mu[0, 0])
    mean_K = float(c_K[0, 0])

    mu_modes = _fractional_complex_modes(mu_grid, mean_mu, lmax)
    K_modes = _fractional_complex_modes(K_grid, mean_K, lmax)
    return ElasticLayerState(
        mean_mu_Pa=mean_mu,
        mean_K_Pa=mean_K,
        mu_variable={layer_index: mu_modes} if mu_modes else {},
        K_variable={layer_index: K_modes} if K_modes else {},
    )
