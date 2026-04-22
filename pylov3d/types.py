"""JAX-compatible data structures for pylov3d.

All data structures are plain NamedTuple-based for JAX pytree compatibility.
Arrays use static padding to MAX_LAYERS / MAX_MODES / MAX_NR_TOTAL dimensions
with boolean masks indicating active entries.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .constants import MAX_LAYERS, MAX_NR_TOTAL


# ---------------------------------------------------------------------------
# Input structures
# ---------------------------------------------------------------------------

class InteriorModel(NamedTuple):
    """Multi-layer planetary interior model.

    All arrays have shape (MAX_LAYERS,). Layers are ordered from core (index 0)
    to surface (index n_layers-1).  Unused entries are zero-padded.
    """
    n_layers: int

    # Dimensional input properties
    R0: jnp.ndarray        # (MAX_LAYERS,) radii [km]
    rho0: jnp.ndarray      # (MAX_LAYERS,) density [kg/m^3]
    mu0: jnp.ndarray       # (MAX_LAYERS,) shear modulus [Pa]
    Ks0: jnp.ndarray       # (MAX_LAYERS,) bulk modulus [Pa]
    eta0: jnp.ndarray      # (MAX_LAYERS,) viscosity [Pa s] (NaN => elastic)
    ocean: jnp.ndarray     # (MAX_LAYERS,) int, 0=solid, 1=ocean
    Delta_rho0: jnp.ndarray  # (MAX_LAYERS,) density contrast [kg/m^3]

    # Normalized properties (filled by rheology.normalize)
    R: jnp.ndarray         # (MAX_LAYERS,) R0 / R0_surface
    rho: jnp.ndarray       # (MAX_LAYERS,) rho0 / rho0_surface
    mu: jnp.ndarray        # (MAX_LAYERS,) mu0 / mu0_surface
    Ks: jnp.ndarray        # (MAX_LAYERS,) Ks0 / mu0_surface
    eta: jnp.ndarray       # (MAX_LAYERS,) eta0 / (mu0_surface * Td)
    muC: jnp.ndarray       # (MAX_LAYERS,) complex128, complex shear modulus
    lam: jnp.ndarray       # (MAX_LAYERS,) complex128, Lame parameter lambda
    MaxTime: jnp.ndarray   # (MAX_LAYERS,) normalized Maxwell time
    elastic: jnp.ndarray   # (MAX_LAYERS,) int, 1=elastic, 0=viscoelastic
    gs: jnp.ndarray        # (MAX_LAYERS,) normalized surface gravity at R
    Delta_rho: jnp.ndarray # (MAX_LAYERS,) normalized density contrast
    rho_av: jnp.ndarray    # (MAX_LAYERS,) normalized average density within R
    Gg: float              # normalized gravitational constant


class Forcing(NamedTuple):
    """Tidal forcing specification for a single harmonic component."""
    Td: float              # Forcing period [s]
    n: int                 # Spherical harmonic degree
    m: int                 # Spherical harmonic order
    F: complex             # Forcing amplitude


class NumericsConfig(NamedTuple):
    """Numerical discretization parameters."""
    n_layers: int          # Number of layers (including core)
    method: str            # 'variable', 'fixed', 'combination', 'manual'
    Nrbase: int            # Base radial points
    Nr: int                # Total radial points (computed)
    Nrlayer: jnp.ndarray   # (MAX_LAYERS,) int, points per layer
    BCindices: jnp.ndarray # (MAX_LAYERS,) int, boundary point indices
    perturbation_order: int
    rheology_cutoff: float
    Nenergy: int


# ---------------------------------------------------------------------------
# Output structures
# ---------------------------------------------------------------------------

class LoveSpectra(NamedTuple):
    """Love number spectra output."""
    nf: int                # Forcing degree
    mf: int                # Forcing order
    n: jnp.ndarray         # (n_modes,) response degrees
    m: jnp.ndarray         # (n_modes,) response orders
    k: jnp.ndarray         # (n_modes,) complex128, gravity Love number
    h: jnp.ndarray         # (n_modes,) complex128, radial displacement Love
    l: jnp.ndarray         # (n_modes,) complex128, tangential displacement Love


class RadialSolution(NamedTuple):
    """Full radial solution."""
    r: jnp.ndarray         # (Nr+1,) radial positions
    y: jnp.ndarray         # (Nr+1, 8, n_modes) solution components per mode
    n_s: jnp.ndarray       # (n_modes,) degrees of solution modes
    m_s: jnp.ndarray       # (n_modes,) orders of solution modes


class EnergySpectra(NamedTuple):
    """Tidal dissipation energy spectra."""
    n: jnp.ndarray         # (n_modes,) degrees
    m: jnp.ndarray         # (n_modes,) orders
    energy_integral: jnp.ndarray  # (n_modes,) radially integrated dissipation
    energy_profile: jnp.ndarray   # (Nr+1, n_modes) dissipation vs radius


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_interior_model(
    R0_km: list[float],
    rho0: list[float],
    mu0: list[float],
    Ks0: list[float] | None = None,
    eta0: list[float | None] | None = None,
    ocean: list[int] | None = None,
    Delta_rho0: list[float] | None = None,
) -> InteriorModel:
    """Build an InteriorModel from per-layer lists.

    Parameters
    ----------
    R0_km : radii from core to surface [km]
    rho0 : densities [kg/m^3]
    mu0 : shear moduli [Pa] (ignored for core, use 0)
    Ks0 : bulk moduli [Pa] (None => incompressible, set to 1e7*mu0_surface)
    eta0 : viscosities [Pa s] (None => elastic)
    ocean : 0 or 1 per layer
    Delta_rho0 : density contrasts [kg/m^3] (auto-computed if None)
    """
    n = len(R0_km)
    assert n <= MAX_LAYERS

    def _pad(vals, dtype=jnp.float64, fill=0.0):
        arr = jnp.zeros(MAX_LAYERS, dtype=dtype)
        return arr.at[:n].set(jnp.array(vals, dtype=dtype))

    R0_arr = _pad(R0_km)
    rho0_arr = _pad(rho0)
    mu0_arr = _pad(mu0)

    if Ks0 is None:
        # Incompressible: large bulk modulus
        Ks0_vals = [1e7 * mu0[-1]] * n
    else:
        Ks0_vals = Ks0
    Ks0_arr = _pad(Ks0_vals)

    if eta0 is None:
        eta0_vals = [float('nan')] * n
    else:
        eta0_vals = [float('nan') if v is None else v for v in eta0]
    eta0_arr = _pad(eta0_vals)

    if ocean is None:
        ocean_vals = [0] * n
    else:
        ocean_vals = ocean
    ocean_arr = _pad(ocean_vals, dtype=jnp.int32, fill=0)

    # Auto-compute density contrasts
    if Delta_rho0 is None:
        drho = [0.0] * n
        for i in range(1, n):
            drho[i] = rho0[i - 1] - rho0[i]
        # Core: default to rho_core - rho_layer2
        if n > 1:
            drho[0] = rho0[0] - rho0[1]
    else:
        drho = Delta_rho0
    Delta_rho0_arr = _pad(drho)

    zeros = jnp.zeros(MAX_LAYERS, dtype=jnp.float64)
    zeros_c = jnp.zeros(MAX_LAYERS, dtype=jnp.complex128)
    zeros_i = jnp.zeros(MAX_LAYERS, dtype=jnp.int32)

    return InteriorModel(
        n_layers=n,
        R0=R0_arr, rho0=rho0_arr, mu0=mu0_arr, Ks0=Ks0_arr,
        eta0=eta0_arr, ocean=ocean_arr, Delta_rho0=Delta_rho0_arr,
        # Normalized (to be filled by rheology.normalize)
        R=zeros, rho=zeros, mu=zeros, Ks=zeros, eta=zeros,
        muC=zeros_c, lam=zeros_c, MaxTime=zeros,
        elastic=zeros_i, gs=zeros, Delta_rho=zeros, rho_av=zeros,
        Gg=0.0,
    )


def make_forcing(Td: float, n: int, m: int, F: complex) -> Forcing:
    """Create a single Forcing component."""
    return Forcing(Td=Td, n=n, m=m, F=F)


def make_numerics(
    n_layers: int,
    method: str = "combination",
    Nrbase: int = 200,
    perturbation_order: int = 2,
    rheology_cutoff: float = 2.0,
    Nenergy: int = 8,
) -> NumericsConfig:
    """Create a NumericsConfig with defaults (Nr/Nrlayer/BCindices computed later)."""
    return NumericsConfig(
        n_layers=n_layers,
        method=method,
        Nrbase=Nrbase,
        Nr=0,
        Nrlayer=jnp.zeros(MAX_LAYERS, dtype=jnp.int32),
        BCindices=jnp.zeros(MAX_LAYERS, dtype=jnp.int32),
        perturbation_order=perturbation_order,
        rheology_cutoff=rheology_cutoff,
        Nenergy=Nenergy,
    )
