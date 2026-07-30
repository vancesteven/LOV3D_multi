"""PlanetProfile compatibility adapter.

Provides conversion functions to use pylov3d with PlanetProfile interior models.
"""
from typing import Protocol, Optional
import numpy as np
from .types import InteriorModel, make_interior_model


class PlanetStructProtocol(Protocol):
    """Minimal PlanetProfile.PlanetStruct interface for type checking."""

    R_m: np.ndarray  # Layer radii in meters
    rho_kgm3: np.ndarray  # Density in kg/m³
    G_Pa: Optional[np.ndarray]  # Shear modulus in Pa
    eta_Pas: Optional[np.ndarray]  # Viscosity in Pa·s
    Ocean: dict  # Ocean layer info


def planetstruct_to_interior_model(
    planet: PlanetStructProtocol,
    ocean_layer: Optional[int] = None,
) -> InteriorModel:
    """Convert PlanetProfile PlanetStruct to pylov3d InteriorModel.

    Parameter Mapping
    -----------------

    | PlanetProfile        | pylov3d           | Conversion        |
    |----------------------|-------------------|-------------------|
    | R_m                  | R0_km             | / 1000            |
    | rho_kgm3             | rho0              | direct            |
    | G_Pa                 | mu0               | * 1e9 (Pa)        |
    | eta_Pas              | eta0              | direct            |
    | Ocean['comp']        | (ocean detection) | layer index       |

    Parameters
    ----------
    planet : PlanetStructProtocol
        PlanetProfile interior structure with R_m, rho_kgm3, G_Pa, eta_Pas
    ocean_layer : int, optional
        Layer index for ocean (0-indexed). If None, auto-detect from planet.Ocean

    Returns
    -------
    InteriorModel
        pylov3d interior model ready for get_love()

    Notes
    -----
    - Layer boundaries are defined by outer radius R0_km
    - Ocean layers: set mu0=None, eta0=None for that layer
    - Core (fluid): set mu0=0.0 or None, eta0=None
    - Rigid layers: set eta0=None (infinite viscosity)

    Examples
    --------
    >>> # Assuming 'planet' is a PlanetProfile.PlanetStruct
    >>> model = planetstruct_to_interior_model(planet)
    >>> forcing = make_forcing(Td=planet.Tb_yr * 365.25 * 86400, n=2, m=0)
    >>> love, _, _ = get_love(model, forcing, numerics)
    """
    # Convert radii: meters → kilometers
    R0_km = planet.R_m / 1000.0

    # Density: direct copy (both kg/m³)
    rho0 = planet.rho_kgm3.copy()

    # Shear modulus: GPa → Pa (if available)
    if planet.G_Pa is not None:
        mu0 = planet.G_Pa * 1e9  # GPa to Pa
    else:
        mu0 = np.zeros_like(rho0)

    # Viscosity: direct copy (both Pa·s)
    if planet.eta_Pas is not None:
        eta0 = planet.eta_Pas.copy()
    else:
        eta0 = np.full_like(rho0, None, dtype=object)

    # Auto-detect ocean layer if not specified
    if ocean_layer is None and hasattr(planet, 'Ocean') and planet.Ocean:
        # PlanetProfile marks ocean with Ocean['comp'] != 'none'
        if 'comp' in planet.Ocean and planet.Ocean['comp'] != 'none':
            # Typically ocean is layer with lowest G_Pa
            if planet.G_Pa is not None:
                ocean_layer = np.argmin(np.where(planet.G_Pa > 0, planet.G_Pa, np.inf))

    # Build ocean array (0=solid, 1=ocean)
    ocean_arr = np.zeros(len(R0_km), dtype=int)
    if ocean_layer is not None:
        ocean_arr[ocean_layer] = 1
        # Ocean layers: set mu=0 (inviscid)
        mu0[ocean_layer] = 0.0

    # Convert to lists for make_interior_model
    R0_km_list = R0_km.tolist()
    rho0_list = rho0.tolist()

    # Convert numpy types to native Python
    mu0_list = []
    for mu in mu0:
        if mu is None or (isinstance(mu, (int, float, np.number)) and mu == 0):
            mu0_list.append(0.0)  # Core or ocean: G=0 → mu0=0.0
        else:
            mu0_list.append(float(mu))

    eta0_list = []
    for eta in eta0:
        if eta is None:
            eta0_list.append(None)  # Elastic (None → NaN in make_interior_model)
        else:
            eta0_list.append(float(eta))

    ocean_list = ocean_arr.tolist()

    return make_interior_model(
        R0_km=R0_km_list,
        rho0=rho0_list,
        mu0=mu0_list,
        eta0=eta0_list,
        ocean=ocean_list,
    )


def pyalma3_to_interior_model(*args, **kwargs):
    """Convert PyALMA3 model to pylov3d InteriorModel.

    Placeholder for future PyALMA3 cross-validation work.

    Raises
    ------
    NotImplementedError
        PyALMA3 integration not yet implemented (Milestone 4)
    """
    raise NotImplementedError(
        "PyALMA3 integration planned for Milestone 4. "
        "For now, use planetstruct_to_interior_model() for PlanetProfile models."
    )
