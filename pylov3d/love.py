"""Love number extraction — pipeline orchestrator.

Combines grid setup, rheology, solver, and Love number extraction
into a single high-level API.  Translated from get_Love.m.
"""

from __future__ import annotations

import numpy as np

from .types import (
    InteriorModel,
    Forcing,
    NumericsConfig,
    LoveSpectra,
    RadialSolution,
)
from .grid import set_boundary_indices
from .rheology import get_rheology
from .solver import get_solution


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def get_love(
    interior_model: InteriorModel,
    forcing: Forcing | list[Forcing],
    numerics: NumericsConfig,
) -> tuple[LoveSpectra, RadialSolution, InteriorModel]:
    """Full pipeline: grid → rheology → integration → Love numbers.

    Parameters
    ----------
    interior_model : InteriorModel
        Raw (dimensional) model from ``make_interior_model``.
    forcing : Forcing or list[Forcing]
        Tidal forcing specification(s).
    numerics : NumericsConfig
        Grid configuration from ``make_numerics``.

    Returns
    -------
    love : LoveSpectra
        Love numbers (k, h, l) for the forced mode.
    y_rad : RadialSolution
        Full radial solution including r grid, state vector, and
        fundamental matrix.
    model : InteriorModel
        Normalized model (output of ``get_rheology``).
    """
    # 1. Grid setup
    numerics, model = set_boundary_indices(numerics, interior_model)

    # 2. Rheology (normalization + complex modulus)
    model = get_rheology(model, forcing)

    # 3. Radial integration + boundary conditions
    y_sol, r_grid, Y, Aprop_aux = get_solution(model, forcing, numerics)

    # 4. Extract Love numbers
    love = extract_love_numbers(y_sol, model, forcing)

    # 5. Build RadialSolution
    f0 = forcing[0] if isinstance(forcing, list) else forcing
    y_rad = RadialSolution(
        r=r_grid,
        y=y_sol,
        n_s=np.array([f0.n]),
        m_s=np.array([f0.m]),
    )

    return love, y_rad, model


# ---------------------------------------------------------------------------
# Love number extraction
# ---------------------------------------------------------------------------

def extract_love_numbers(
    y_sol: np.ndarray,
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
) -> LoveSpectra:
    """Extract Love numbers from the surface solution.

    For the forced mode (matching n, m):
        k = Φ_surf − 1
        h = −gs · U_surf
        l = −gs · V_surf

    For other modes (lateral variations, M2+):
        k = Φ_surf
        h = −gs · U_surf
        l = −gs · V_surf

    Parameters
    ----------
    y_sol : (Nr+1, 8) complex
        Physical solution from ``get_solution``.
    model : InteriorModel
        Normalized model.
    forcing : Forcing or list[Forcing]
        Tidal forcing (used to identify the forced mode).

    Returns
    -------
    LoveSpectra
        Contains k, h, l arrays (length 1 for M1).

    Notes
    -----
    State vector indices: U=0, V=1, W=2, R=3, S=4, T=5, Φ=6, dΦ/dr=7.
    Matches MATLAB get_Love.m lines 637–650.
    """
    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_f = f0.n
    m_f = f0.m

    # Surface values (last radial point)
    U_surf = complex(y_sol[-1, 0])
    V_surf = complex(y_sol[-1, 1])
    Phi_surf = complex(y_sol[-1, 6])

    # Normalized surface gravity at outermost layer
    gs_surface = float(model.gs[model.n_layers - 1])

    # Love numbers for the forced mode
    k = Phi_surf - 1.0
    h = -gs_surface * U_surf
    l = -gs_surface * V_surf

    return LoveSpectra(
        nf=n_f,
        mf=m_f,
        n=np.array([n_f]),
        m=np.array([m_f]),
        k=np.array([k]),
        h=np.array([h]),
        l=np.array([l]),
    )
