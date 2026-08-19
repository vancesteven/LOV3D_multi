# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Mars mid-crust seismic constraints for proposal-facing joint inference.

This module provides the *observation/likelihood layer*, not a complete rock-
physics interpretation.  Wright, Morzfeld & Manga (2024, PNAS,
doi:10.1073/pnas.2409983121) used Vp, Vs and bulk density for the InSight
mid-crust (approximately 11.5--20 km depth) together with a Berryman self-
consistent + Gassmann-Biot rock-physics model.  Their adopted data vector is

    Vp  = 4.1 +/- 0.2 km/s
    Vs  = 2.5 +/- 0.3 km/s
    rho = 2589 +/- 157 kg/m^3

The purpose here is to pin those observables and expose a stable likelihood
interface.  Forward models for dry/hydrated mineral assemblages, fractured
porosity, liquid saturation, pore aspect ratio, or metamorphic alteration can
all map into this same three-component observable without being conflated.

The default likelihood assumes independent Gaussian errors because Wright et
al. tabulate separate one-sigma uncertainties.  A full covariance matrix can
be supplied when a future data product justifies it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeismicConstraint:
    """Observed isotropic elastic state and covariance."""

    vp_m_s: float
    vs_m_s: float
    rho_kg_m3: float
    covariance: np.ndarray
    depth_top_km: float | None = None
    depth_bottom_km: float | None = None
    label: str = ""

    @property
    def vector(self) -> np.ndarray:
        return np.asarray([self.vp_m_s, self.vs_m_s, self.rho_kg_m3], dtype=float)


WRIGHT_2024_MIDCRUST = SeismicConstraint(
    vp_m_s=4.1e3,
    vs_m_s=2.5e3,
    rho_kg_m3=2589.0,
    covariance=np.diag([0.2e3**2, 0.3e3**2, 157.0**2]),
    depth_top_km=11.5,
    depth_bottom_km=20.0,
    label="Wright et al. (2024) InSight mid-crust",
)


def isotropic_velocities(K_pa: float, mu_pa: float, rho_kg_m3: float) -> tuple[float, float]:
    """Return ``(Vp, Vs)`` [m/s] for isotropic elastic moduli and density.

    Uses ``Vp^2=(K+4*mu/3)/rho`` and ``Vs^2=mu/rho``.  This is a constitutive
    conversion only; it does not include porosity/fluid effective-medium
    physics.
    """
    K = float(K_pa)
    mu = float(mu_pa)
    rho = float(rho_kg_m3)
    if K <= 0 or mu < 0 or rho <= 0:
        raise ValueError("K and rho must be positive and mu must be non-negative")
    vp = np.sqrt((K + 4.0 * mu / 3.0) / rho)
    vs = np.sqrt(mu / rho)
    return float(vp), float(vs)


def moduli_from_velocities(vp_m_s: float, vs_m_s: float, rho_kg_m3: float) -> tuple[float, float]:
    """Return isotropic ``(K, mu)`` [Pa] from ``Vp, Vs, rho``."""
    vp = float(vp_m_s)
    vs = float(vs_m_s)
    rho = float(rho_kg_m3)
    if vp <= 0 or vs < 0 or rho <= 0:
        raise ValueError("Vp and rho must be positive and Vs must be non-negative")
    mu = rho * vs**2
    K = rho * (vp**2 - 4.0 * vs**2 / 3.0)
    if K <= 0:
        raise ValueError("velocities imply a non-positive bulk modulus")
    return float(K), float(mu)


def seismic_residual(
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
    constraint: SeismicConstraint = WRIGHT_2024_MIDCRUST,
) -> np.ndarray:
    """Return model-minus-observation vector ``[Vp, Vs, rho]``."""
    return np.asarray([vp_m_s, vs_m_s, rho_kg_m3], dtype=float) - constraint.vector


def seismic_chi2(
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
    constraint: SeismicConstraint = WRIGHT_2024_MIDCRUST,
) -> float:
    """Gaussian chi-square for a predicted mid-crust state."""
    r = seismic_residual(vp_m_s, vs_m_s, rho_kg_m3, constraint)
    cov = np.asarray(constraint.covariance, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError("seismic covariance must be 3x3")
    return float(r @ np.linalg.solve(cov, r))


def seismic_loglike(
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
    constraint: SeismicConstraint = WRIGHT_2024_MIDCRUST,
    *,
    normalized: bool = False,
) -> float:
    """Gaussian log likelihood for ``Vp, Vs, rho``.

    With ``normalized=False`` (default), returns ``-chi2/2``.  Set
    ``normalized=True`` to include the Gaussian normalization term.
    """
    chi2 = seismic_chi2(vp_m_s, vs_m_s, rho_kg_m3, constraint)
    if not normalized:
        return -0.5 * chi2
    cov = np.asarray(constraint.covariance, dtype=float)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("seismic covariance must be positive definite")
    return float(-0.5 * (chi2 + logdet + 3.0 * np.log(2.0 * np.pi)))
