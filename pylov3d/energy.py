"""Tidal dissipation energy computation — translated from get_energy.m.

Computes GSH stress and strain at every radial point, the radial
dissipation profile, and the total dissipation.

For the 1D single-mode case the total dissipation equals Im(k₂) × norms.
For the coupled multi-mode case, the energy coupling tensor (Wigner 9j
products from ``energy_couplings.py``) contracts stress and strain across
mode pairs to yield the angular energy spectrum.
"""

from __future__ import annotations

import math

import numpy as np

from .types import InteriorModel, Forcing, NumericsConfig, EnergySpectra
from .propagator import build_A1_A2, build_A3


# ---------------------------------------------------------------------------
# Strain–displacement matrices (GSH form, purely geometric)
# ---------------------------------------------------------------------------

def build_A14_A15(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Build strain–displacement GSH matrices.

    .. math::
        \varepsilon = A_{14}\,\dot u_\mathrm{gsh} + \frac{A_{15}}{r}\,u_\mathrm{gsh}

    where :math:`u_\mathrm{gsh} = [u_{n,n-1},\; u_{n,n},\; u_{n,n+1}]`.

    The six strain components are ordered
    ``[ε_{n,n,0}, ε_{n,n-2,2}, ε_{n,n-1,2}, ε_{n,n,2}, ε_{n,n+1,2}, ε_{n,n+2,2}]``.

    Translated from ``get_A14A15`` in ``get_solution.m`` (lines 1629–1704).

    Parameters
    ----------
    n : int
        Spherical harmonic degree.

    Returns
    -------
    A14 : (6, 3) float
        Coefficient of :math:`du_\mathrm{gsh}/dr`.
    A15 : (6, 3) float
        Coefficient of :math:`u_\mathrm{gsh}/r`.
    """
    A14 = np.zeros((6, 3))
    A15 = np.zeros((6, 3))

    if n > 0:
        s2n1 = math.sqrt(2 * n + 1)
        sn = math.sqrt(n)
        sn1 = math.sqrt(n + 1)

        # ε_{n,n,0}
        c0 = 1.0 / math.sqrt(3) / s2n1
        A14[0, 0] = -c0 * sn
        A14[0, 2] = c0 * sn1
        A15[0, 0] = c0 * (n - 1) * sn
        A15[0, 2] = c0 * sn1 * (n + 2)

        # ε_{n,n-2,2}
        c1 = math.sqrt((n - 1) / (2 * n - 1))
        A14[1, 0] = c1
        A15[1, 0] = c1 * n

        # ε_{n,n-1,2}
        c2 = 1.0 / math.sqrt(2) * math.sqrt((n - 1) / (2 * n + 1))
        A14[2, 1] = c2
        A15[2, 1] = c2 * (n + 1)

        # ε_{n,n,2}
        c3a = math.sqrt((2*n+3) * (2*n+2) / (12 * (2*n-1) * (2*n+1)))
        c3b = math.sqrt(n * (2*n-1) * (n+1) / (3 * (2*n+3) * (2*n+2) * (2*n+1)))
        A14[3, 0] = -c3a
        A14[3, 2] = c3b
        A15[3, 0] = c3a * (n - 1)
        A15[3, 2] = c3b * (n + 2)

        # ε_{n,n+1,2}
        c4 = 1.0 / math.sqrt(2) * math.sqrt((n + 2) / (2 * n + 1))
        A14[4, 1] = -c4
        A15[4, 1] = c4 * n

        # ε_{n,n+2,2}
        c5 = math.sqrt((n + 2) / (2 * n + 3))
        A14[5, 2] = -c5
        A15[5, 2] = c5 * (n + 1)
    else:
        # n = 0: only ε_{0,0,0} and ε_{0,2,2} survive
        c0 = 1.0 / math.sqrt(3) / math.sqrt(2 * n + 1)
        A14[0, 2] = c0 * math.sqrt(n + 1)
        A15[0, 2] = c0 * math.sqrt(n + 1) * (n + 2)

        c5 = math.sqrt((n + 2) / (2 * n + 3))
        A14[5, 2] = -c5
        A15[5, 2] = c5 * (n + 1)

    return A14, A15


# ---------------------------------------------------------------------------
# Stress and strain at every radial point
# ---------------------------------------------------------------------------

def compute_stress_strain(
    y_sol: np.ndarray,
    r_grid: np.ndarray,
    Aprop_aux: np.ndarray,
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
    numerics: NumericsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute GSH displacement, stress, and strain at all radial points.

    Uses the constitutive relation :math:`\sigma = A_1\,\dot u + A_2/r\,u`
    and the kinematic relation :math:`\varepsilon = A_{14}\,\dot u + A_{15}/r\,u`
    where :math:`\dot u = A_3^{-1}\,[\text{Aprop}_{0:3}\,y]`.

    Matches MATLAB ``get_solution.m`` lines 906–999.

    Parameters
    ----------
    y_sol : (Nr+1, 8) complex
    r_grid : (Nr+1,) float
    Aprop_aux : (Nr+1, 3, 8) complex — first 3 rows of Aprop
    model : InteriorModel (normalized)
    forcing : Forcing or list[Forcing]
    numerics : NumericsConfig

    Returns
    -------
    u_gsh : (Nr+1, 3) complex — GSH displacement
    stress : (Nr+1, 6) complex — GSH stress
    strain : (Nr+1, 6) complex — GSH strain
    """
    f0 = forcing[0] if isinstance(forcing, list) else forcing
    n_deg = f0.n
    Nr = numerics.Nr

    # Inverse displacement transform
    A3 = build_A3(n_deg)
    A3_inv = np.linalg.inv(A3)

    # Strain–displacement matrices (geometric, material-independent)
    A14, A15 = build_A14_A15(n_deg)

    # Allocate output arrays
    u_gsh = np.zeros((Nr + 1, 3), dtype=np.complex128)
    stress = np.zeros((Nr + 1, 6), dtype=np.complex128)
    strain = np.zeros((Nr + 1, 6), dtype=np.complex128)

    # Reconstruct layer map (same logic as solver.py)
    layer_map = np.zeros(Nr + 1, dtype=int)
    layer_map[0] = 0
    k = 1
    for i_layer in range(1, model.n_layers):
        npts = int(numerics.Nrlayer[i_layer])
        for j in range(npts):
            layer_map[k] = i_layer
            k += 1

    # Cache A1/A2 per layer (material-dependent)
    A1_cache = {}
    A2_cache = {}
    for i_layer in range(1, model.n_layers):
        if int(model.ocean[i_layer]) == 1:
            continue
        muC_k = complex(model.muC[i_layer])
        lam_k = complex(model.lam[i_layer])
        A1_cache[i_layer], A2_cache[i_layer] = build_A1_A2(n_deg, muC_k, lam_k)

    for k_idx in range(Nr + 1):
        r = r_grid[k_idx]
        i_layer = layer_map[k_idx]

        # Skip core (fluid) and ocean layers
        if i_layer == 0 or int(model.ocean[i_layer]) == 1:
            continue
        if i_layer not in A1_cache:
            continue

        # GSH displacement: u_gsh = A3_inv @ [U, V, W]
        UVW = y_sol[k_idx, :3]
        u_gsh[k_idx] = A3_inv @ UVW

        # Displacement radial derivative:
        # x_dot = Aprop_aux @ y  gives [dU/dr, dV/dr, dW/dr]
        x_dot = Aprop_aux[k_idx] @ y_sol[k_idx]
        u_dot_gsh = A3_inv @ x_dot

        # GSH stress: σ = A1 @ u_dot + A2/r @ u
        A1 = A1_cache[i_layer]
        A2 = A2_cache[i_layer]
        if r > 0:
            stress[k_idx] = A1 @ u_dot_gsh + (A2 @ u_gsh[k_idx]) / r
        else:
            stress[k_idx] = A1 @ u_dot_gsh

        # GSH strain: ε = A14 @ u_dot + A15/r @ u
        if r > 0:
            strain[k_idx] = A14 @ u_dot_gsh + (A15 @ u_gsh[k_idx]) / r
        else:
            strain[k_idx] = A14 @ u_dot_gsh

    return u_gsh, stress, strain


# ---------------------------------------------------------------------------
# Energy spectra
# ---------------------------------------------------------------------------

def get_energy(
    y_sol: np.ndarray,
    r_grid: np.ndarray,
    Aprop_aux: np.ndarray,
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
    numerics: NumericsConfig,
) -> EnergySpectra:
    r"""Compute tidal dissipation energy.

    For M1 (1D, single mode), the radial dissipation density is

    .. math::
        \dot e(r) = \operatorname{Im}\!\bigl(\sigma^*\!:\varepsilon\bigr)\,r^2

    and the total dissipation is obtained by trapezoidal integration,
    matching the MATLAB formula (``get_energy.m`` line 547).

    Parameters
    ----------
    y_sol, r_grid, Aprop_aux : from ``get_solution``
    model : InteriorModel (normalized)
    forcing : Forcing or list[Forcing]
    numerics : NumericsConfig

    Returns
    -------
    EnergySpectra
        ``energy_profile`` is the radial dissipation density at each point;
        ``energy_integral`` is the trapezoidal radial integral.
    """
    Nr = numerics.Nr

    # Compute GSH stress and strain
    _, stress, strain = compute_stress_strain(
        y_sol, r_grid, Aprop_aux, model, forcing, numerics,
    )

    # Dissipation density: Im(σ* : ε) at each radial point
    # σ* : ε = Σ_j conj(σ_j) · ε_j
    dissipation = np.zeros(Nr + 1)
    for k in range(Nr + 1):
        product = np.sum(np.conj(stress[k]) * strain[k])
        dissipation[k] = product.imag

    # Trapezoidal radial integration matching MATLAB:
    #   radialdiff = ((r(1:end-1)+r(2:end))/2)^2 * (r(2:end)-r(1:end-1))
    #   energy_s = sum(radialdiff * (energy(2:end) + energy(1:end-1))/2)
    r_mid = (r_grid[:-1] + r_grid[1:]) / 2
    dr = r_grid[1:] - r_grid[:-1]
    weights = r_mid ** 2 * dr
    energy_avg = (dissipation[:-1] + dissipation[1:]) / 2
    total = np.sum(weights * energy_avg)

    return EnergySpectra(
        n=np.array([0]),
        m=np.array([0]),
        energy_integral=np.array([total]),
        energy_profile=dissipation.reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# Global dissipation from Love numbers (exact for 1D)
# ---------------------------------------------------------------------------

def global_dissipation(
    k_love: complex,
    omega: float,
    R_m: float,
    rho_surface: float,
    n: int = 2,
) -> float:
    r"""Total tidal dissipation from :math:`\operatorname{Im}(k_n)`.

    .. math::
        \dot E = -\frac{(2n+1)^2}{4\pi G}\,\omega\,R^{2n-1}\,
                 \operatorname{Im}(k_n)\,|\Phi_n|^2

    For practical use the formula simplifies to

    .. math::
        \dot E = -\operatorname{Im}(k_n)\,\frac{(2n+1)}{2}\,
                 \frac{\omega\,R^5\,g_s^2}{G}

    when the forcing potential amplitude is :math:`g_s R / (2n+1)`.

    Parameters
    ----------
    k_love : complex
        Love number :math:`k_n` (Im < 0 for dissipative bodies).
    omega : float
        Forcing angular frequency [rad/s].
    R_m : float
        Surface radius [m].
    rho_surface : float
        Surface-layer density [kg/m^3].
    n : int
        Harmonic degree (default 2).

    Returns
    -------
    float
        Total dissipation rate [W].
    """
    from .constants import G
    gs = G * (4.0 / 3.0) * math.pi * rho_surface * R_m
    return -k_love.imag * (2 * n + 1) / 2.0 * omega * R_m ** 5 * gs ** 2 / G


# ---------------------------------------------------------------------------
# Coupled stress and strain (multi-mode)
# ---------------------------------------------------------------------------

def compute_stress_strain_coupled(
    y_sol: np.ndarray,
    r_grid: np.ndarray,
    Aprop_aux: np.ndarray,
    model: InteriorModel,
    n_s: np.ndarray,
    numerics: NumericsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute GSH displacement, stress, and strain for a coupled solution.

    Extends ``compute_stress_strain`` to the multi-mode case.  Builds
    block-diagonal A3_inv (3N×3N), A14/A15 (6N×3N), and per-layer
    A1/A2 (6N×3N) matrices where each mode's block uses the 1D
    constitutive relation for that mode's degree.

    Translates MATLAB ``get_solution.m`` lines 906–998.

    Parameters
    ----------
    y_sol : (Nr+1, 8N) complex
        Coupled physical solution.
    r_grid : (Nr+1,) float
    Aprop_aux : (Nr+1, 3N, 8N) complex
        First 3N rows of the coupled propagator at each point.
    model : InteriorModel (normalized)
    n_s : (N,) int
        Degrees of the solution modes.
    numerics : NumericsConfig

    Returns
    -------
    u_gsh : (Nr+1, 3N) complex — GSH displacements (3 per mode)
    stress : (Nr+1, 6N) complex — GSH stress (6 per mode)
    strain : (Nr+1, 6N) complex — GSH strain (6 per mode)
    """
    N = len(n_s)
    Nr = numerics.Nr

    # Build block-diagonal A3_inv (3N × 3N)
    A3_inv_full = np.zeros((3 * N, 3 * N), dtype=np.complex128)
    for i in range(N):
        A3_i = build_A3(int(n_s[i]))
        A3_inv_full[3*i:3*i+3, 3*i:3*i+3] = np.linalg.inv(A3_i)

    # Build block-diagonal A14, A15 (6N × 3N)
    A14_full = np.zeros((6 * N, 3 * N))
    A15_full = np.zeros((6 * N, 3 * N))
    for i in range(N):
        A14_i, A15_i = build_A14_A15(int(n_s[i]))
        A14_full[6*i:6*i+6, 3*i:3*i+3] = A14_i
        A15_full[6*i:6*i+6, 3*i:3*i+3] = A15_i

    # Extract displacement indices: U_k = 3*k, V_k = 3*k+1, W_k = 3*k+2
    disp_idx = np.zeros(3 * N, dtype=int)
    for i in range(N):
        disp_idx[3*i:3*i+3] = [8*i, 8*i+1, 8*i+2]

    # Compute u = A3_inv @ U for all radial points at once
    U_all = y_sol[:, disp_idx]  # (Nr+1, 3N)
    u_gsh = (A3_inv_full @ U_all.T).T  # (Nr+1, 3N)

    # Build full y_temp for Aprop_aux multiplication:
    # reorder [displacements(3N), stresses(3N), potentials(2N)]
    state_idx = np.zeros(8 * N, dtype=int)
    for i in range(N):
        state_idx[3*i:3*i+3] = [3*i, 3*i+1, 3*i+2]           # disp
        state_idx[3*N+3*i:3*N+3*i+3] = [3*N+3*i, 3*N+3*i+1, 3*N+3*i+2]  # stress
        state_idx[6*N+2*i:6*N+2*i+2] = [6*N+2*i, 6*N+2*i+1]  # pot

    # Reconstruct layer map (needed for skipping core/ocean)
    layer_map = np.zeros(Nr + 1, dtype=int)
    layer_map[0] = 0
    k = 1
    for i_layer in range(1, model.n_layers):
        npts = int(numerics.Nrlayer[i_layer])
        for j in range(npts):
            layer_map[k] = i_layer
            k += 1

    # Build block-diagonal A1/A2 per layer
    A1_cache = {}
    A2_cache = {}
    for i_layer in range(1, model.n_layers):
        if int(model.ocean[i_layer]) == 1:
            continue
        muC_k = complex(model.muC[i_layer])
        lam_k = complex(model.lam[i_layer])
        A1_blk = np.zeros((6 * N, 3 * N), dtype=np.complex128)
        A2_blk = np.zeros((6 * N, 3 * N), dtype=np.complex128)
        for i in range(N):
            A1_i, A2_i = build_A1_A2(int(n_s[i]), muC_k, lam_k)
            A1_blk[6*i:6*i+6, 3*i:3*i+3] = A1_i
            A2_blk[6*i:6*i+6, 3*i:3*i+3] = A2_i
        A1_cache[i_layer] = A1_blk
        A2_cache[i_layer] = A2_blk

    # Compute u_dot, stress, and strain per radial point
    # (u_dot only computed for solid, non-core layers — matches MATLAB)
    u_dot = np.zeros((Nr + 1, 3 * N), dtype=np.complex128)
    stress = np.zeros((Nr + 1, 6 * N), dtype=np.complex128)
    strain = np.zeros((Nr + 1, 6 * N), dtype=np.complex128)

    for k_idx in range(Nr + 1):
        r = r_grid[k_idx]
        i_layer = layer_map[k_idx]

        # Skip core and ocean layers
        if i_layer == 0 or int(model.ocean[i_layer]) == 1:
            continue
        if i_layer not in A1_cache:
            continue

        # u_dot = A3_inv @ Aprop_aux @ y
        x_dot = Aprop_aux[k_idx] @ y_sol[k_idx]
        u_dot[k_idx] = A3_inv_full @ x_dot

        # Stress: σ = A1 @ u_dot + A2/r @ u
        A1 = A1_cache[i_layer]
        A2 = A2_cache[i_layer]
        if r > 0:
            stress[k_idx] = A1 @ u_dot[k_idx] + (A2 @ u_gsh[k_idx]) / r
        else:
            stress[k_idx] = A1 @ u_dot[k_idx]

        # Strain: ε = A14 @ u_dot + A15/r @ u
        if r > 0:
            strain[k_idx] = A14_full @ u_dot[k_idx] + (A15_full @ u_gsh[k_idx]) / r
        else:
            strain[k_idx] = A14_full @ u_dot[k_idx]

    return u_gsh, stress, strain


# ---------------------------------------------------------------------------
# Coupled energy spectra
# ---------------------------------------------------------------------------

def get_energy_coupled(
    y_solutions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    forcings: list[Forcing],
    model: InteriorModel,
    numerics: NumericsConfig,
    n_s: np.ndarray,
    m_s: np.ndarray,
    Nenergy: int = 8,
) -> EnergySpectra:
    r"""Compute coupled tidal dissipation energy spectra.

    Combines solutions across forcings, computes stress/strain per mode,
    and contracts with the energy coupling tensor.

    Translates MATLAB ``get_energy.m`` lines 201–553.

    Parameters
    ----------
    y_solutions : list of (y_sol, r_grid, Aprop_aux) tuples
        One per forcing, from ``get_solution``.
    forcings : list of Forcing
        Forcing for each solution.
    model : InteriorModel (normalized)
    numerics : NumericsConfig
    n_s, m_s : (N,) int
        Degrees and orders of the coupled solution modes.
    Nenergy : int
        Maximum degree for energy spectrum expansion.

    Returns
    -------
    EnergySpectra
    """
    from .energy_couplings import get_energy_couplings

    N = len(n_s)
    Nf = len(forcings)
    r_grid = y_solutions[0][1]
    Nr = numerics.Nr

    # --- (1) Combine solutions across forcings ---
    # Sum F_j × y_j for each mode
    y_combined = np.zeros((Nr + 1, 8 * N), dtype=np.complex128)
    Aprop_combined = y_solutions[0][2].copy()  # Same for all forcings

    for j in range(Nf):
        y_j, _, _ = y_solutions[j]
        y_combined += forcings[j].F * y_j

    # --- (2) Compute stress and strain per mode ---
    _, stress_all, strain_all = compute_stress_strain_coupled(
        y_combined, r_grid, Aprop_combined, model, n_s, numerics,
    )
    # stress_all: (Nr+1, 6N), strain_all: (Nr+1, 6N)
    # Reshape to (Nr+1, 6, N) — 6 GSH components per mode
    stress_3d = stress_all.reshape(Nr + 1, N, 6).transpose(0, 2, 1)  # (Nr+1, 6, N)
    strain_3d = strain_all.reshape(Nr + 1, N, 6).transpose(0, 2, 1)  # (Nr+1, 6, N)

    # --- (3) Build stressP/strainP and stressN/strainN ---
    # MATLAB reorders stress columns: [14,15,16,17,18,13] → [n-2,n-1,n,n+1,n+2,n]
    # In our layout, GSH stress components are already [ε_{n,0}, ε_{n-2,2},
    # ε_{n-1,2}, ε_{n,2}, ε_{n+1,2}, ε_{n+2,2}].
    # MATLAB cols 13-18 = [σ_{n,0}, σ_{n-2,2}, σ_{n-1,2}, σ_{n,2}, σ_{n+1,2}, σ_{n+2,2}]
    # MATLAB reorder [14,15,16,17,18,13] → [σ_{n-2}, σ_{n-1}, σ_{n}, σ_{n+1}, σ_{n+2}, σ_{n,0}]
    # This maps indices [1,2,3,4,5,0] in our 0-based ordering.
    reorder = [1, 2, 3, 4, 5, 0]
    stressP = stress_3d[:, reorder, :]  # (Nr+1, 6, N)
    strainP = strain_3d[:, reorder, :]  # (Nr+1, 6, N)

    # stressN/strainN: conjugate of the -m mode
    stressN = np.zeros_like(stressP)
    strainN = np.zeros_like(strainP)
    for i in range(N):
        n_i, m_i = int(n_s[i]), int(m_s[i])
        # Find the mode with same n, opposite m
        neg_idx = np.where((n_s == n_i) & (m_s == -m_i))[0]
        if len(neg_idx) > 0:
            stressN[:, :, i] = np.conj(stressP[:, :, neg_idx[0]])
            strainN[:, :, i] = np.conj(strainP[:, :, neg_idx[0]])
        elif m_i == 0:
            # m=0: conjugate of self
            stressN[:, :, i] = np.conj(stressP[:, :, i])
            strainN[:, :, i] = np.conj(strainP[:, :, i])

    # --- (4) Get energy coupling coefficients ---
    ec = get_energy_couplings(n_s, m_s, Nenergy=Nenergy)
    EC = ec.EC          # (N, N, N_en, 6, 6)
    n_en = ec.n_en      # (N_en,)
    m_en = ec.m_en      # (N_en,)
    N_en = len(n_en)

    # --- (5) Contract: energy density at each radial point ---
    # Find non-zero entries in EC
    nz = np.nonzero(EC)
    nz_i1, nz_i2, nz_k, nz_i3, nz_i4 = nz
    n_nz = len(nz_i1)

    # Build n2a and n2b arrays for phase factors
    # n2a[i3] for mode i1: [n-2, n-1, n, n+1, n+2, n][i3]
    n2a_offset = np.array([-2, -1, 0, 1, 2, 0])
    n2b_offset = np.array([-2, -1, 0, 1, 2, 0])

    # Compute all terms
    energy = np.zeros((Nr + 1, N_en))

    # Vectorized: compute sol_eq for all non-zero entries
    sol_eq = np.zeros((Nr + 1, n_nz), dtype=np.complex128)
    for idx in range(n_nz):
        i1 = nz_i1[idx]
        i2 = nz_i2[idx]
        k = nz_k[idx]
        i3 = nz_i3[idx]
        i4 = nz_i4[idx]

        n2a = int(n_s[i1]) + n2a_offset[i3]
        n2b = int(n_s[i2]) + n2b_offset[i4]

        ec_val = EC[i1, i2, k, i3, i4]

        phase1 = (-1) ** (n2a + int(n_s[i1]) - int(m_s[i1]))
        phase2 = (-1) ** (n2b + int(n_s[i2]) - int(m_s[i2]))

        term1 = (1j * 2 * math.pi * phase1 * ec_val
                 * stressN[:, i3, i1] * strainP[:, i4, i2])
        term2 = (1j * 2 * math.pi * phase2 * ec_val
                 * stressP[:, i3, i1] * strainN[:, i4, i2])

        sol_eq[:, idx] = term1 - term2

    # Sum contributions per energy mode (take real part; imaginary is
    # numerically negligible for physical energy).
    for k in range(N_en):
        mask = nz_k == k
        if np.any(mask):
            energy[:, k] = np.sum(sol_eq[:, mask], axis=1).real

    # --- (6) Trapezoidal radial integration ---
    r_mid = (r_grid[:-1] + r_grid[1:]) / 2
    dr = r_grid[1:] - r_grid[:-1]
    weights = r_mid ** 2 * dr

    energy_integral = np.zeros(N_en)
    for k in range(N_en):
        energy_avg = (energy[:-1, k] + energy[1:, k]) / 2
        energy_integral[k] = np.sum(weights * energy_avg)

    return EnergySpectra(
        n=n_en,
        m=m_en,
        energy_integral=energy_integral,
        energy_profile=energy,
    )
