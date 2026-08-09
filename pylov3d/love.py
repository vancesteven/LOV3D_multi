# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

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
    mu_variable: dict | None = None,
    eta_variable: dict | None = None,
    K_variable: dict | None = None,
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
    mu_variable : dict, optional
        Lateral shear modulus variations.  ``{layer_idx: [(n, m, amp), ...]}``.
    eta_variable : dict, optional
        Lateral viscosity variations.  Same format as *mu_variable*.
    K_variable : dict, optional
        Lateral bulk modulus variations.  Same format as *mu_variable*.

    Returns
    -------
    love : LoveSpectra
        Love numbers (k, h, l) for all coupled modes.
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

    # 3. Lateral variations + couplings (if any)
    couplings = None
    lateral = None
    if mu_variable or eta_variable or K_variable:
        from .rheology import process_lateral_variations
        from .couplings import get_couplings

        model, lateral = process_lateral_variations(
            model, forcing,
            mu_variable=mu_variable,
            eta_variable=eta_variable,
            K_variable=K_variable,
        )

        f0 = forcing[0] if isinstance(forcing, list) else forcing
        couplings = get_couplings(
            lateral.variations, f0.n, f0.m,
            perturbation_order=numerics.perturbation_order,
        )

    # 4. Radial integration + boundary conditions
    y_sol, r_grid, Y, Aprop_aux = get_solution(
        model, forcing, numerics, couplings=couplings, lateral=lateral,
    )

    # 5. Extract Love numbers
    love = extract_love_numbers(y_sol, model, forcing, couplings=couplings)

    # 6. Build RadialSolution
    if couplings is not None:
        n_s = couplings.n_s.copy()
        m_s = couplings.m_s.copy()
    else:
        f0 = forcing[0] if isinstance(forcing, list) else forcing
        n_s = np.array([f0.n])
        m_s = np.array([f0.m])

    y_rad = RadialSolution(r=r_grid, y=y_sol, n_s=n_s, m_s=m_s)

    return love, y_rad, model


# ---------------------------------------------------------------------------
# Love number extraction
# ---------------------------------------------------------------------------

def extract_love_numbers(
    y_sol: np.ndarray,
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
    couplings=None,
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
    y_sol : (Nr+1, 8) or (Nr+1, 8N) complex
        Physical solution from ``get_solution``.
    model : InteriorModel
        Normalized model.
    forcing : Forcing or list[Forcing]
        Tidal forcing (used to identify the forced mode).
    couplings : Couplings, optional
        Mode coupling info.  When provided, extracts Love numbers for all
        coupled modes from the (Nr+1, 8N) state vector.

    Returns
    -------
    LoveSpectra
        Contains k, h, l arrays (length N for coupled, 1 for 1D).

    Notes
    -----
    1D state indices: U=0, V=1, W=2, R=3, S=4, T=5, Φ=6, dΦ/dr=7.
    Coupled state ordering: [displacements(3N), stresses(3N), potentials(2N)].
    Matches MATLAB get_Love.m lines 637–650.
    """
    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_f = f0.n
    m_f = f0.m
    gs_surface = float(model.gs[model.n_layers - 1])

    if couplings is None:
        # 1D case (existing behavior)
        U_surf = complex(y_sol[-1, 0])
        V_surf = complex(y_sol[-1, 1])
        Phi_surf = complex(y_sol[-1, 6])

        k = Phi_surf - 1.0
        h = -gs_surface * U_surf
        l_love = -gs_surface * V_surf

        return LoveSpectra(
            nf=n_f, mf=m_f,
            n=np.array([n_f]), m=np.array([m_f]),
            k=np.array([k]), h=np.array([h]), l=np.array([l_love]),
        )

    # Coupled case
    N = len(couplings.n_s)
    N6 = 6 * N

    k_arr = np.zeros(N, dtype=np.complex128)
    h_arr = np.zeros(N, dtype=np.complex128)
    l_arr = np.zeros(N, dtype=np.complex128)

    for idx in range(N):
        n_k = int(couplings.n_s[idx])
        m_k = int(couplings.m_s[idx])

        U_k = complex(y_sol[-1, 3 * idx])
        V_k = complex(y_sol[-1, 3 * idx + 1])
        Phi_k = complex(y_sol[-1, N6 + 2 * idx])

        is_forcing = (n_k == n_f and m_k == m_f)

        k_arr[idx] = Phi_k - (1.0 if is_forcing else 0.0)
        h_arr[idx] = -gs_surface * U_k
        l_arr[idx] = -gs_surface * V_k

    return LoveSpectra(
        nf=n_f, mf=m_f,
        n=couplings.n_s.copy(), m=couplings.m_s.copy(),
        k=k_arr, h=h_arr, l=l_arr,
    )
