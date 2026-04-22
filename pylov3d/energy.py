"""Tidal dissipation energy computation — translated from get_energy.m.

For Milestone 1 (1D, single mode), computes the GSH stress and strain
at every radial point, the radial dissipation profile, and the total
dissipation.

The stress and strain are computed from the physical solution vector
via the constitutive relation (A1/A2 matrices) and the kinematic
relation (A14/A15 matrices).

Full angular energy coupling (Wigner 3j products) is an M2 concern;
for M1 the total dissipation can be obtained exactly from Im(k₂).
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
