"""Normalization and Maxwell rheology — translated from get_rheology.m.

Provides:

1. ``normalize`` — Non-dimensionalize the interior model using surface-layer
   reference values.
2. ``compute_complex_rheology`` — Maxwell viscoelastic complex shear modulus
   and Lamé parameter λ.
3. ``process_lateral_variations`` — Process per-layer lateral variations in
   mu, eta, K into the complex-shear-modulus SH representation needed by
   the coupled solver.

Functions 1–2 are combined in the convenience function ``get_rheology``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import jax.numpy as jnp
from scipy.special import sph_harm_y

from .constants import G, MAX_LAYERS
from .types import InteriorModel, Forcing, LateralRheology


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


# ---------------------------------------------------------------------------
# Spherical-harmonic synthesis/analysis (low-degree utility)
# ---------------------------------------------------------------------------

def _sh_grid(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a Gauss-Legendre × uniform-longitude grid.

    Returns (theta, phi, weights) where theta is colatitude (GL nodes),
    phi is longitude (uniform), and weights are GL quadrature weights.
    """
    N_theta = max(4 * (lmax + 1), 8)
    N_phi = 2 * N_theta

    # Gauss-Legendre nodes in [-1, 1] → colatitude
    x, w = np.polynomial.legendre.leggauss(N_theta)
    # x are cos(theta), sorted -1 to 1; we want theta 0..pi
    theta = np.arccos(x[::-1])
    weights = w[::-1]

    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    return theta, phi, weights


def _sh_synthesis(
    coeffs: list[tuple[int, int, complex]],
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Evaluate a complex SH expansion on a (theta, phi) grid.

    Parameters
    ----------
    coeffs : [(n, m, amplitude), ...]
        Complex SH coefficients.  Amplitudes are *fractional* perturbations
        (the mean = 1 is assumed to include the (0,0) coefficient).
    theta : (Nθ,) colatitude grid
    phi : (Nφ,) longitude grid

    Returns (Nθ, Nφ) complex field.
    """
    Nth, Nph = len(theta), len(phi)
    field = np.ones((Nth, Nph), dtype=complex)  # start with mean = 1

    for n, m, amp in coeffs:
        if amp == 0:
            continue
        # sph_harm_y(n, m, theta_colat, phi_lon) — vectorised
        Ynm = sph_harm_y(n, m, theta[:, None], phi[None, :])
        field += amp * Ynm

    return field


def _sh_analysis(
    field: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    lmax: int,
) -> dict[tuple[int, int], complex]:
    """Decompose a field on a GL grid into complex SH coefficients.

    Returns dict mapping (n, m) → complex coefficient.
    """
    dphi = 2 * np.pi / len(phi)
    coeffs = {}
    for n in range(lmax + 1):
        for m in range(-n, n + 1):
            Ynm_conj = np.conj(sph_harm_y(n, m, theta[:, None], phi[None, :]))
            integrand = field * Ynm_conj  # (Nth, Nph)
            # Integrate over phi (trapezoidal)
            phi_sum = np.sum(integrand, axis=1) * dphi  # (Nth,)
            # Integrate over theta (GL quadrature; weights include sinθ factor)
            coeffs[(n, m)] = np.sum(phi_sum * weights)
    return coeffs


# ---------------------------------------------------------------------------
# Lateral variation helpers
# ---------------------------------------------------------------------------

def _ensure_conjugate_pairs(
    modes: list[tuple[int, int, complex]],
) -> list[tuple[int, int, complex]]:
    """Ensure each (n, +m) has a matching (n, -m) conjugate for real fields.

    For a real scalar field, c(n, -m) = (-1)^m * conj(c(n, m)).
    If only +m is provided, the -m component is auto-added.
    """
    nm_set = {(n, m) for n, m, _ in modes}
    result = list(modes)
    for n, m, amp in modes:
        if m != 0 and (n, -m) not in nm_set:
            warnings.warn(
                f"Auto-adding conjugate mode (n={n}, m={-m}) for real field.",
                stacklevel=3,
            )
            result.append((n, -m, (-1) ** m * np.conj(amp)))
            nm_set.add((n, -m))
    return result


def _unify_modes(
    mu_modes: list[tuple[int, int, complex]] | None,
    eta_modes: list[tuple[int, int, complex]] | None,
    K_modes: list[tuple[int, int, complex]] | None,
) -> tuple[list[tuple[int, int]], dict, dict, dict]:
    """Collect unique (n, m) across mu/eta/K and pad missing entries with 0.

    Returns
    -------
    nm_list : sorted list of unique (n, m) pairs (excluding (0,0))
    mu_map, eta_map, K_map : dict (n,m) → amplitude
    """
    mu_map: dict[tuple[int, int], complex] = {}
    eta_map: dict[tuple[int, int], complex] = {}
    K_map: dict[tuple[int, int], complex] = {}

    for modes, target in [(mu_modes, mu_map), (eta_modes, eta_map), (K_modes, K_map)]:
        if modes is not None:
            for n, m, amp in modes:
                key = (n, m)
                target[key] = target.get(key, 0.0) + amp

    all_nm = sorted(set(mu_map) | set(eta_map) | set(K_map))
    # Remove (0,0) if present — the mean is handled separately
    all_nm = [(n, m) for n, m in all_nm if n > 0]

    for nm in all_nm:
        mu_map.setdefault(nm, 0.0)
        eta_map.setdefault(nm, 0.0)
        K_map.setdefault(nm, 0.0)

    return all_nm, mu_map, eta_map, K_map


# ---------------------------------------------------------------------------
# Main lateral variation processing
# ---------------------------------------------------------------------------

def process_lateral_variations(
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
    mu_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    eta_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    K_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    rheology_cutoff: float = 2.0,
) -> tuple[InteriorModel, LateralRheology]:
    """Process lateral variations in rheological properties.

    Translates the lateral-variation section of ``get_rheology.m``
    (lines 370–960).

    Parameters
    ----------
    model : InteriorModel
        Already normalized model (output of ``get_rheology``).
    forcing : Forcing or list[Forcing]
        Tidal forcing (for MaxTime computation).
    mu_variable : dict mapping layer_index → [(n, m, amplitude), ...]
        Fractional complex SH amplitudes δμ/μ₀.
    eta_variable : same format for viscosity δη/η₀.
    K_variable : same format for bulk modulus δK/K₀.
    rheology_cutoff : float
        Decades below maximum to retain (default 2.0).

    Returns
    -------
    model : InteriorModel
        Updated with corrected muC, lam for non-uniform layers.
    lateral : LateralRheology
        Coupling data for the coupled solver.
    """
    n_layers = model.n_layers
    mu_variable = mu_variable or {}
    eta_variable = eta_variable or {}
    K_variable = K_variable or {}

    # Per-layer processing ------------------------------------------------
    uniform = np.ones(n_layers, dtype=bool)
    # Temporary storage: per-layer dict of (n,m) → muC_amplitude
    layer_muC = {}  # layer_idx → {(n,m): complex}
    layer_K = {}    # layer_idx → {(n,m): complex}

    # Updated muC and lam for the model
    muC_new = np.array(model.muC, dtype=complex)
    lam_new = np.array(model.lam, dtype=complex)

    for ilayer in range(1, n_layers):  # skip core (layer 0)
        mu_modes = mu_variable.get(ilayer)
        eta_modes = eta_variable.get(ilayer)
        K_modes = K_variable.get(ilayer)

        has_variations = (
            (mu_modes is not None and any(abs(a) > 0 for _, _, a in mu_modes))
            or (eta_modes is not None and any(abs(a) > 0 for _, _, a in eta_modes))
            or (K_modes is not None and any(abs(a) > 0 for _, _, a in K_modes))
        )

        if not has_variations:
            layer_muC[ilayer] = {}
            layer_K[ilayer] = {}
            continue

        # Ensure conjugate pairs for real fields
        if mu_modes:
            mu_modes = _ensure_conjugate_pairs(mu_modes)
        if eta_modes:
            eta_modes = _ensure_conjugate_pairs(eta_modes)
        if K_modes:
            K_modes = _ensure_conjugate_pairs(K_modes)

        # Unify (n,m) across properties
        nm_list, mu_map, eta_map, K_map = _unify_modes(mu_modes, eta_modes, K_modes)

        if not nm_list:
            layer_muC[ilayer] = {}
            layer_K[ilayer] = {}
            continue

        mu_i = float(model.mu[ilayer])
        Ks_i = float(model.Ks[ilayer])
        is_elastic = bool(model.elastic[ilayer])
        MaxTime_i = float(model.MaxTime[ilayer])

        if is_elastic:
            # Elastic: muC_nm = mu * mu_amplitude_nm
            uniform[ilayer] = False
            layer_muC[ilayer] = {
                nm: mu_i * mu_map[nm] for nm in nm_list
            }
            layer_K[ilayer] = {nm: 0.0 for nm in nm_list}
        else:
            # Viscoelastic: grid-based nonlinear computation
            lmax = max(n for n, m in nm_list)

            # Build coefficient lists for synthesis
            mu_coeffs = [(n, m, mu_map[(n, m)]) for n, m in nm_list
                         if abs(mu_map[(n, m)]) > 0]
            eta_coeffs = [(n, m, eta_map[(n, m)]) for n, m in nm_list
                          if abs(eta_map[(n, m)]) > 0]

            theta, phi, weights = _sh_grid(lmax)

            # Synthesize mu and eta fields (fractional: 1 + δμ/μ₀, 1 + δη/η₀)
            mu_field = _sh_synthesis(mu_coeffs, theta, phi)
            eta_field = _sh_synthesis(eta_coeffs, theta, phi)

            # MaxTime field (normalized)
            MaxTime_field = np.real(eta_field) / np.real(mu_field)

            # Complex shear modulus on the grid
            # Cmu = mu * (1 + δμ) / (1 - i/(MaxTime_field * MaxTime_mean))
            Cmu_field = mu_i * mu_field / (
                1.0 - 1j / (MaxTime_field * MaxTime_i)
            )

            # Analyse back to SH
            analysis_lmax = min(2 * lmax, 2 * lmax)
            muC_sh = _sh_analysis(Cmu_field, theta, phi, weights, analysis_lmax)

            # Mean (0,0) component replaces the model's muC for this layer.
            # SH coefficient c_00 = sqrt(4π) × spatial_mean, so divide.
            Y00 = 1.0 / np.sqrt(4.0 * np.pi)
            mu00 = muC_sh.get((0, 0), complex(mu_i, 0.0)) * Y00
            muC_new[ilayer] = mu00
            lam_new[ilayer] = Ks_i - (2.0 / 3.0) * mu00

            # Filter significant modes
            all_amplitudes = {
                (n, m): muC_sh.get((n, m), 0.0)
                for n, m in muC_sh if n > 0
            }
            if all_amplitudes:
                max_rel = max(np.log10(max(abs(v) for v in all_amplitudes.values())
                              / max(abs(np.real(mu00)), 1e-30)), -30.0)
                significant = {
                    nm: amp for nm, amp in all_amplitudes.items()
                    if np.log10(max(abs(amp), 1e-300)
                                / max(abs(np.real(mu00)), 1e-30)) >= max_rel - rheology_cutoff
                }
            else:
                significant = {}

            if significant:
                uniform[ilayer] = False
                layer_muC[ilayer] = significant
                layer_K[ilayer] = {nm: 0.0 for nm in significant}
            else:
                # All modes below cutoff — revert to uniform
                uniform[ilayer] = True
                layer_muC[ilayer] = {}
                layer_K[ilayer] = {}

    # Collect global unique (n,m) across all non-uniform layers -----------
    all_nm = set()
    for ilayer in range(1, n_layers):
        all_nm.update(layer_muC.get(ilayer, {}).keys())

    if not all_nm:
        # No lateral variations survived — return trivial lateral rheology
        variations = np.zeros((1, 2), dtype=int)
        muC_amp = np.zeros((n_layers, 1), dtype=complex)
        K_amp_arr = np.zeros((n_layers, 1), dtype=complex)
        return model, LateralRheology(
            variations=variations, muC_amp=muC_amp, K_amp=K_amp_arr,
            uniform=uniform,
        )

    sorted_nm = sorted(all_nm)
    Nreo = len(sorted_nm)
    variations = np.array(sorted_nm, dtype=int)

    # Pad each layer to the global (n,m) set
    muC_amp = np.zeros((n_layers, Nreo), dtype=complex)
    K_amp_arr = np.zeros((n_layers, Nreo), dtype=complex)

    for ilayer in range(1, n_layers):
        if uniform[ilayer]:
            continue
        for j, nm in enumerate(sorted_nm):
            muC_amp[ilayer, j] = layer_muC.get(ilayer, {}).get(nm, 0.0)
            K_amp_arr[ilayer, j] = layer_K.get(ilayer, {}).get(nm, 0.0)

    # Update model with corrected muC, lam
    model = model._replace(
        muC=jnp.array(muC_new, dtype=jnp.complex128),
        lam=jnp.array(lam_new, dtype=jnp.complex128),
    )

    return model, LateralRheology(
        variations=variations,
        muC_amp=muC_amp,
        K_amp=K_amp_arr,
        uniform=uniform,
    )
