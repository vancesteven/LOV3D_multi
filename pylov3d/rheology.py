"""Normalization and Maxwell rheology — translated from get_rheology.m.

For Milestone 1 (1D, no lateral variations) this module provides:

1. ``normalize`` — Non-dimensionalize the interior model using surface-layer
   reference values.
2. ``compute_complex_rheology`` — Maxwell viscoelastic complex shear modulus
   and Lamé parameter λ.

Both are combined in the convenience function ``get_rheology``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from .constants import G, MAX_LAYERS
from .types import InteriorModel, Forcing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rheology(
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
) -> InteriorModel:
    """Normalize and compute complex rheology (1D uniform model).

    This is the main entry-point, combining normalization and Maxwell
    rheology in a single call.

    Parameters
    ----------
    model : InteriorModel
        Dimensional interior model (output of ``make_interior_model``).
    forcing : Forcing or list[Forcing]
        Tidal forcing.  Only the period ``Td`` from the first component is
        used for normalization.
    """
    if isinstance(forcing, list):
        Td = forcing[0].Td
    else:
        Td = forcing.Td

    model = normalize(model, Td)
    model = compute_complex_rheology(model)
    return model


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(model: InteriorModel, Td: float) -> InteriorModel:
    """Non-dimensionalize the interior model.

    Reference values are taken from the outermost layer (index n_layers-1):
      R_ref   = R0_surface [km]
      rho_ref = rho0_surface [kg/m^3]
      mu_ref  = mu0_surface [Pa]
      t_ref   = Td [s]

    The normalized gravitational constant is:
      Gg = G * (R0_surface*1e3)^2 * rho0_surface^2 / mu0_surface
    """
    n = model.n_layers
    last = n - 1  # surface layer index

    R0_surf = float(model.R0[last])
    rho0_surf = float(model.rho0[last])
    mu0_surf = float(model.mu0[last])

    # Normalized gravitational constant
    Gg = G * (R0_surf * 1e3) ** 2 * rho0_surf ** 2 / mu0_surf

    # Build normalized arrays in plain Python, then convert once
    R = [0.0] * MAX_LAYERS
    rho = [0.0] * MAX_LAYERS
    mu = [0.0] * MAX_LAYERS
    Ks = [0.0] * MAX_LAYERS
    eta = [float('nan')] * MAX_LAYERS
    MaxTime = [float('nan')] * MAX_LAYERS
    elastic = [0] * MAX_LAYERS
    gs = [0.0] * MAX_LAYERS
    Delta_rho = [0.0] * MAX_LAYERS
    rho_av = [0.0] * MAX_LAYERS

    # Accumulate mass for average density / gravity
    M = 0.0  # normalized cumulative mass

    for i in range(n):
        R0_i = float(model.R0[i])
        rho0_i = float(model.rho0[i])

        R[i] = R0_i / R0_surf
        rho[i] = rho0_i / rho0_surf

        if i == 0:
            # Core
            M = (4.0 / 3.0) * math.pi * rho[i] * R[i] ** 3
            Delta_rho[i] = float(model.Delta_rho0[i]) / rho0_surf
        else:
            mu0_i = float(model.mu0[i])
            Ks0_i = float(model.Ks0[i])
            eta0_i = float(model.eta0[i])

            mu[i] = mu0_i / mu0_surf
            Ks[i] = Ks0_i / mu0_surf

            # Viscosity / Maxwell time
            if math.isnan(eta0_i):
                # Elastic layer
                eta[i] = float('nan')
                MaxTime[i] = float('nan')
                elastic[i] = 1
            else:
                eta[i] = eta0_i / (mu0_surf * Td)
                MaxTime[i] = 2.0 * math.pi * eta0_i / (mu0_i * Td)
                elastic[i] = 0

            # Density contrast with layer below
            Delta_rho[i] = rho[i - 1] - rho[i]

            # Accumulate mass
            M += (4.0 / 3.0) * math.pi * rho[i] * (R[i] ** 3 - R[i - 1] ** 3)

        # Average density and gravity at this radius
        rho_av[i] = M / ((4.0 / 3.0) * math.pi * R[i] ** 3) if R[i] > 0 else 0.0
        gs[i] = Gg * M / R[i] ** 2 if R[i] > 0 else 0.0

    return model._replace(
        R=jnp.array(R, dtype=jnp.float64),
        rho=jnp.array(rho, dtype=jnp.float64),
        mu=jnp.array(mu, dtype=jnp.float64),
        Ks=jnp.array(Ks, dtype=jnp.float64),
        eta=jnp.array(eta, dtype=jnp.float64),
        MaxTime=jnp.array(MaxTime, dtype=jnp.float64),
        elastic=jnp.array(elastic, dtype=jnp.int32),
        gs=jnp.array(gs, dtype=jnp.float64),
        Delta_rho=jnp.array(Delta_rho, dtype=jnp.float64),
        rho_av=jnp.array(rho_av, dtype=jnp.float64),
        Gg=Gg,
    )


# ---------------------------------------------------------------------------
# Complex rheology
# ---------------------------------------------------------------------------

def compute_complex_rheology(model: InteriorModel) -> InteriorModel:
    """Compute Maxwell complex shear modulus and Lamé λ.

    For each layer (i >= 1):
      - Viscoelastic: muC = mu / (1 - i/MaxTime)
      - Elastic:      muC = mu  (real)
      - Lambda:       lam = Ks - 2/3 * muC
    """
    n = model.n_layers

    muC = [0j] * MAX_LAYERS
    lam = [0j] * MAX_LAYERS

    for i in range(1, n):
        mu_i = float(model.mu[i])
        Ks_i = float(model.Ks[i])
        MaxTime_i = float(model.MaxTime[i])
        elastic_i = int(model.elastic[i])

        if elastic_i:
            muC[i] = complex(mu_i, 0.0)
        else:
            # muC = mu * (1 - 1j/MaxTime)^{-1}
            muC[i] = mu_i / (1.0 - 1j / MaxTime_i)

        lam[i] = Ks_i - (2.0 / 3.0) * muC[i]

    return model._replace(
        muC=jnp.array(muC, dtype=jnp.complex128),
        lam=jnp.array(lam, dtype=jnp.complex128),
    )
