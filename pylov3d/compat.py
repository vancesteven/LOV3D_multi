# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""PlanetProfile compatibility adapters.

The current PlanetProfile ``PlanetStruct`` stores shell properties in
``rho_kgm3``, ``eta_Pas`` and ``Seismic.{GS_GPa,KS_GPa}``, with shell outer
radii in ``r_m[:-1]`` ordered from the surface inward. pylov3d uses the
opposite convention: one property value per homogeneous shell, ordered from
core outward, and moduli in Pa.

This module performs that conversion explicitly and refuses to silently decimate
a high-resolution PlanetProfile model. If a full profile contains more shells
than pylov3d's static ``MAX_LAYERS`` limit, populate ``Planet.Reduced`` first or
supply another scientifically controlled reduced model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .constants import MAX_LAYERS
from .types import InteriorModel, make_interior_model


class _SeismicProtocol(Protocol):
    GS_GPa: np.ndarray | None
    KS_GPa: np.ndarray | None
    VP_kms: np.ndarray | None
    VS_kms: np.ndarray | None


class PlanetStructProtocol(Protocol):
    """Subset of the current PlanetProfile ``PlanetStruct`` used here."""

    r_m: np.ndarray
    rho_kgm3: np.ndarray
    eta_Pas: np.ndarray | None
    phase: np.ndarray | None
    Seismic: _SeismicProtocol
    Reduced: object


@dataclass(frozen=True)
class PlanetProfileShells:
    """Neutral shell representation extracted from PlanetProfile."""

    outer_radius_m: np.ndarray
    rho_kgm3: np.ndarray
    mu_Pa: np.ndarray
    K_Pa: np.ndarray
    eta_Pas: np.ndarray
    phase: np.ndarray | None
    source: str


def _as_1d(value, name: str, n: int | None = None) -> np.ndarray:
    if value is None:
        raise ValueError(f"PlanetProfile field {name} is required")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f"PlanetProfile field {name} has length {arr.size}, expected {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"PlanetProfile field {name} must be finite")
    return arr


def _derive_bulk_from_seismic(rho_kgm3: np.ndarray, vp_kms, vs_kms) -> np.ndarray:
    vp = _as_1d(vp_kms, "Reduced.Seismic.VP_kms", rho_kgm3.size) * 1.0e3
    vs = _as_1d(vs_kms, "Reduced.Seismic.VS_kms", rho_kgm3.size) * 1.0e3
    K = rho_kgm3 * (vp * vp - (4.0 / 3.0) * vs * vs)
    if np.any(K <= 0) or not np.all(np.isfinite(K)):
        raise ValueError("derived PlanetProfile bulk modulus is non-positive or non-finite")
    return K


def _shell_outer_radii(r_m, n: int, name: str) -> np.ndarray:
    r = _as_1d(r_m, name)
    if r.size == n + 1:
        r = r[:-1]
    elif r.size != n:
        raise ValueError(f"{name} has length {r.size}; expected {n} shell radii or {n+1} boundaries")
    if np.any(r <= 0):
        raise ValueError(f"{name} shell outer radii must be positive")
    return r


def _phase_array(value, n: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size != n:
        raise ValueError(f"PlanetProfile field {name} has length {arr.size}, expected {n}")
    return arr


def _sort_core_to_surface(shells: PlanetProfileShells) -> PlanetProfileShells:
    order = np.argsort(shells.outer_radius_m)
    r = shells.outer_radius_m[order]
    if np.any(np.diff(r) <= 0):
        raise ValueError("PlanetProfile shell outer radii must be unique")
    phase = None if shells.phase is None else shells.phase[order]
    return PlanetProfileShells(
        outer_radius_m=r,
        rho_kgm3=shells.rho_kgm3[order],
        mu_Pa=shells.mu_Pa[order],
        K_Pa=shells.K_Pa[order],
        eta_Pas=shells.eta_Pas[order],
        phase=phase,
        source=shells.source,
    )


def _extract_reduced(planet) -> PlanetProfileShells | None:
    reduced = getattr(planet, "Reduced", None)
    if reduced is None or getattr(reduced, "r_m", None) is None:
        return None
    r_raw = np.asarray(reduced.r_m).reshape(-1)
    if r_raw.size == 0:
        return None

    rho = _as_1d(getattr(reduced, "rho_kgm3", None), "Reduced.rho_kgm3")
    n = rho.size
    r = _shell_outer_radii(reduced.r_m, n, "Reduced.r_m")
    seismic = getattr(reduced, "Seismic", None)
    if seismic is None:
        raise ValueError("Planet.Reduced.Seismic is required for pylov3d conversion")
    mu = _as_1d(getattr(seismic, "GS_GPa", None), "Reduced.Seismic.GS_GPa", n) * 1.0e9
    K = _derive_bulk_from_seismic(rho, getattr(seismic, "VP_kms", None), getattr(seismic, "VS_kms", None))

    eta_raw = getattr(reduced, "eta_Pas", None)
    if eta_raw is None:
        eta = np.full(n, np.nan)
    else:
        eta = np.asarray(eta_raw, dtype=float).reshape(-1)
        if eta.size != n:
            raise ValueError(f"Reduced.eta_Pas has length {eta.size}, expected {n}")
    phase = _phase_array(getattr(reduced, "phase", None), n, "Reduced.phase")
    return _sort_core_to_surface(PlanetProfileShells(r, rho, mu, K, eta, phase, "Reduced"))


def _extract_full(planet) -> PlanetProfileShells:
    rho = _as_1d(getattr(planet, "rho_kgm3", None), "rho_kgm3")
    n = rho.size
    r = _shell_outer_radii(getattr(planet, "r_m", None), n, "r_m")
    seismic = getattr(planet, "Seismic", None)
    if seismic is None:
        raise ValueError("Planet.Seismic is required for pylov3d conversion")
    mu = _as_1d(getattr(seismic, "GS_GPa", None), "Seismic.GS_GPa", n) * 1.0e9
    K = _as_1d(getattr(seismic, "KS_GPa", None), "Seismic.KS_GPa", n) * 1.0e9

    eta_raw = getattr(planet, "eta_Pas", None)
    if eta_raw is None:
        eta = np.full(n, np.nan)
    else:
        eta = np.asarray(eta_raw, dtype=float).reshape(-1)
        if eta.size != n:
            raise ValueError(f"PlanetProfile field eta_Pas has length {eta.size}, expected {n}")
    phase = _phase_array(getattr(planet, "phase", None), n, "phase")
    return _sort_core_to_surface(PlanetProfileShells(r, rho, mu, K, eta, phase, "full"))


def planetstruct_shells(
    planet: PlanetStructProtocol,
    *,
    prefer_reduced: bool = True,
) -> PlanetProfileShells:
    """Extract a validated, core-to-surface shell model from PlanetProfile.

    If ``prefer_reduced`` is true and ``Planet.Reduced`` is populated, the
    reduced profile is used. Reduced PlanetProfile currently carries VP, VS and
    GS but not KS, so K is reconstructed from ``rho*(VP^2 - 4 VS^2/3)``.
    """
    shells = _extract_reduced(planet) if prefer_reduced else None
    if shells is None:
        shells = _extract_full(planet)
    if shells.outer_radius_m.size > MAX_LAYERS:
        raise ValueError(
            f"PlanetProfile {shells.source} model has {shells.outer_radius_m.size} shells, "
            f"but pylov3d MAX_LAYERS={MAX_LAYERS}. Produce a controlled Planet.Reduced "
            "model first; this adapter will not silently decimate the structure."
        )
    if np.any(shells.rho_kgm3 <= 0) or np.any(shells.K_Pa <= 0) or np.any(shells.mu_Pa < 0):
        raise ValueError("PlanetProfile density/K must be positive and shear modulus non-negative")
    return shells


def planetstruct_to_interior_model(
    planet: PlanetStructProtocol,
    ocean_layer: Optional[int] = None,
    *,
    prefer_reduced: bool = True,
    fluid_mu_tol_Pa: float = 1.0,
) -> InteriorModel:
    """Convert a current PlanetProfile profile to pylov3d ``InteriorModel``.

    The returned shells are ordered core to surface. PlanetProfile seismic
    moduli are converted from GPa to Pa, while density and viscosity retain SI
    units. Fluid shells are identified by near-zero shear modulus; the deepest
    shell (index 0) remains the central-core layer and is not marked as an
    ``ocean`` boundary-condition shell.

    ``ocean_layer`` is an optional manual override in the *converted*,
    core-to-surface indexing convention.
    """
    if fluid_mu_tol_Pa < 0:
        raise ValueError("fluid_mu_tol_Pa must be non-negative")
    shells = planetstruct_shells(planet, prefer_reduced=prefer_reduced)
    n = shells.outer_radius_m.size

    ocean = (shells.mu_Pa <= fluid_mu_tol_Pa).astype(int)
    if n:
        ocean[0] = 0
    if ocean_layer is not None:
        if not 0 <= ocean_layer < n:
            raise ValueError(f"ocean_layer must lie in [0,{n-1}]")
        ocean[ocean_layer] = 1

    mu = shells.mu_Pa.copy()
    eta_list: list[float | None] = []
    for i in range(n):
        if ocean[i] or mu[i] <= fluid_mu_tol_Pa:
            mu[i] = 0.0
            eta_list.append(None)
            continue
        eta = float(shells.eta_Pas[i])
        eta_list.append(eta if np.isfinite(eta) and eta > 0 else None)

    return make_interior_model(
        R0_km=(shells.outer_radius_m / 1.0e3).tolist(),
        rho0=shells.rho_kgm3.tolist(),
        mu0=mu.tolist(),
        Ks0=shells.K_Pa.tolist(),
        eta0=eta_list,
        ocean=ocean.tolist(),
    )


def pyalma3_to_interior_model(*args, **kwargs):
    """Deprecated placeholder retained for API compatibility."""
    raise NotImplementedError(
        "Direct PyALMA3-to-pylov3d conversion is not implemented. "
        "Use PlanetProfile or construct an InteriorModel explicitly."
    )
