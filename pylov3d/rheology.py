# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Normalization and Maxwell rheology — translated from get_rheology.m."""

from __future__ import annotations

import math
import warnings

import numpy as np
import jax.numpy as jnp
from scipy.special import sph_harm_y

from .constants import G, MAX_LAYERS
from .types import InteriorModel, Forcing, LateralRheology


def get_rheology(model: InteriorModel, forcing: Forcing | list[Forcing]) -> InteriorModel:
    """Normalize and compute complex rheology (1D uniform model)."""
    Td = forcing[0].Td if isinstance(forcing, list) else forcing.Td
    model = normalize(model, Td)
    return compute_complex_rheology(model)


def normalize(model: InteriorModel, Td: float) -> InteriorModel:
    """Non-dimensionalize the interior model."""
    n = model.n_layers
    last = n - 1
    R0_surf = float(model.R0[last])
    rho0_surf = float(model.rho0[last])
    mu0_surf = float(model.mu0[last])
    Gg = G * (R0_surf * 1e3) ** 2 * rho0_surf ** 2 / mu0_surf

    R = [0.0] * MAX_LAYERS
    rho = [0.0] * MAX_LAYERS
    mu = [0.0] * MAX_LAYERS
    Ks = [0.0] * MAX_LAYERS
    eta = [float("nan")] * MAX_LAYERS
    MaxTime = [float("nan")] * MAX_LAYERS
    elastic = [0] * MAX_LAYERS
    gs = [0.0] * MAX_LAYERS
    Delta_rho = [0.0] * MAX_LAYERS
    rho_av = [0.0] * MAX_LAYERS
    M = 0.0

    for i in range(n):
        R0_i = float(model.R0[i])
        rho0_i = float(model.rho0[i])
        R[i] = R0_i / R0_surf
        rho[i] = rho0_i / rho0_surf

        if i == 0:
            M = (4.0 / 3.0) * math.pi * rho[i] * R[i] ** 3
            Delta_rho[i] = float(model.Delta_rho0[i]) / rho0_surf
        else:
            mu0_i = float(model.mu0[i])
            Ks0_i = float(model.Ks0[i])
            eta0_i = float(model.eta0[i])
            mu[i] = mu0_i / mu0_surf
            Ks[i] = Ks0_i / mu0_surf
            if math.isnan(eta0_i):
                elastic[i] = 1
            else:
                eta[i] = eta0_i / (mu0_surf * Td)
                MaxTime[i] = 2.0 * math.pi * eta0_i / (mu0_i * Td)
                elastic[i] = 0
            Delta_rho[i] = rho[i - 1] - rho[i]
            M += (4.0 / 3.0) * math.pi * rho[i] * (R[i] ** 3 - R[i - 1] ** 3)

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


def compute_complex_rheology(model: InteriorModel) -> InteriorModel:
    """Compute Maxwell complex shear modulus and Lamé lambda."""
    n = model.n_layers
    muC = [0j] * MAX_LAYERS
    lam = [0j] * MAX_LAYERS
    for i in range(1, n):
        mu_i = float(model.mu[i])
        Ks_i = float(model.Ks[i])
        MaxTime_i = float(model.MaxTime[i])
        if int(model.elastic[i]):
            muC[i] = complex(mu_i, 0.0)
        else:
            muC[i] = mu_i / (1.0 - 1j / MaxTime_i)
        lam[i] = Ks_i - (2.0 / 3.0) * muC[i]
    return model._replace(
        muC=jnp.array(muC, dtype=jnp.complex128),
        lam=jnp.array(lam, dtype=jnp.complex128),
    )


def _sh_grid(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a Gauss-Legendre by uniform-longitude grid."""
    N_theta = max(4 * (lmax + 1), 8)
    N_phi = 2 * N_theta
    x, w = np.polynomial.legendre.leggauss(N_theta)
    theta = np.arccos(x[::-1])
    weights = w[::-1]
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    return theta, phi, weights


def _sh_synthesis(
    coeffs: list[tuple[int, int, complex]],
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Evaluate a complex SH expansion on a theta/phi grid."""
    field = np.ones((len(theta), len(phi)), dtype=complex)
    for n, m, amp in coeffs:
        if amp == 0:
            continue
        field += amp * sph_harm_y(n, m, theta[:, None], phi[None, :])
    return field


def _sh_analysis(
    field: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    lmax: int,
) -> dict[tuple[int, int], complex]:
    """Decompose a field on a GL grid into complex SH coefficients."""
    dphi = 2 * np.pi / len(phi)
    coeffs = {}
    for n in range(lmax + 1):
        for m in range(-n, n + 1):
            Ynm_conj = np.conj(sph_harm_y(n, m, theta[:, None], phi[None, :]))
            phi_sum = np.sum(field * Ynm_conj, axis=1) * dphi
            coeffs[(n, m)] = np.sum(phi_sum * weights)
    return coeffs


def _ensure_conjugate_pairs(
    modes: list[tuple[int, int, complex]],
) -> list[tuple[int, int, complex]]:
    """Ensure each positive-m coefficient has the real-field partner."""
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
    """Collect unique nonzero-degree modes across mu, eta and K."""
    mu_map: dict[tuple[int, int], complex] = {}
    eta_map: dict[tuple[int, int], complex] = {}
    K_map: dict[tuple[int, int], complex] = {}
    for modes, target in [(mu_modes, mu_map), (eta_modes, eta_map), (K_modes, K_map)]:
        if modes is not None:
            for n, m, amp in modes:
                key = (n, m)
                target[key] = target.get(key, 0.0) + amp
    all_nm = sorted(set(mu_map) | set(eta_map) | set(K_map))
    all_nm = [(n, m) for n, m in all_nm if n > 0]
    for nm in all_nm:
        mu_map.setdefault(nm, 0.0)
        eta_map.setdefault(nm, 0.0)
        K_map.setdefault(nm, 0.0)
    return all_nm, mu_map, eta_map, K_map


def process_lateral_variations(
    model: InteriorModel,
    forcing: Forcing | list[Forcing],
    mu_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    eta_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    K_variable: dict[int, list[tuple[int, int, complex]]] | None = None,
    rheology_cutoff: float = 2.0,
) -> tuple[InteriorModel, LateralRheology]:
    """Process lateral variations in rheological properties.

    The viscoelastic branch follows MATLAB ``get_rheology.m``: coefficient
    inputs are synthesized on a fixed degree-30 working representation,
    transformed through the nonlinear Maxwell law, re-expanded through
    degree 59, and filtered by real and imaginary amplitudes separately.
    """
    n_layers = model.n_layers
    mu_variable = mu_variable or {}
    eta_variable = eta_variable or {}
    K_variable = K_variable or {}

    uniform = np.ones(n_layers, dtype=bool)
    layer_muC = {}
    layer_K = {}
    muC_new = np.array(model.muC, dtype=complex)
    lam_new = np.array(model.lam, dtype=complex)

    for ilayer in range(1, n_layers):
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

        if mu_modes:
            mu_modes = _ensure_conjugate_pairs(mu_modes)
        if eta_modes:
            eta_modes = _ensure_conjugate_pairs(eta_modes)
        if K_modes:
            K_modes = _ensure_conjugate_pairs(K_modes)
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
            uniform[ilayer] = False
            layer_muC[ilayer] = {nm: mu_i * mu_map[nm] for nm in nm_list}
            layer_K[ilayer] = {nm: 0.0 for nm in nm_list}
        else:
            # MATLAB parity: get_rheology.m uses a fixed degree-30 work field
            # for coefficient inputs, then analyses the nonlinear complex
            # modulus through degree 2*30-1 = 59.
            working_lmax = 30
            analysis_lmax = 2 * working_lmax - 1
            mu_coeffs = [
                (n, m, mu_map[(n, m)]) for n, m in nm_list
                if abs(mu_map[(n, m)]) > 0
            ]
            eta_coeffs = [
                (n, m, eta_map[(n, m)]) for n, m in nm_list
                if abs(eta_map[(n, m)]) > 0
            ]
            theta, phi, weights = _sh_grid(working_lmax)
            mu_field = _sh_synthesis(mu_coeffs, theta, phi)
            eta_field = _sh_synthesis(eta_coeffs, theta, phi)
            MaxTime_field = np.real(eta_field) / np.real(mu_field)
            Cmu_field = mu_i * mu_field / (
                1.0 - 1j / (MaxTime_field * MaxTime_i)
            )
            muC_sh = _sh_analysis(Cmu_field, theta, phi, weights, analysis_lmax)

            Y00 = 1.0 / np.sqrt(4.0 * np.pi)
            mu00 = muC_sh.get((0, 0), complex(mu_i, 0.0)) * Y00
            muC_new[ilayer] = mu00
            lam_new[ilayer] = Ks_i - (2.0 / 3.0) * mu00

            # MATLAB filters real and imaginary spectra independently relative
            # to the strongest nonzero component, then takes their union.
            rows: list[tuple[tuple[int, int], complex, float, float]] = []
            tiny = np.finfo(float).tiny
            for (n, m), amp in muC_sh.items():
                if n == 0:
                    continue
                rr = abs(amp.real * Y00 / max(abs(mu00.real), tiny))
                ii = abs(amp.imag * Y00 / max(abs(mu00.imag), tiny))
                lr = np.log10(max(rr, tiny))
                li = np.log10(max(ii, tiny))
                rows.append(((n, m), amp, lr, li))

            if rows:
                max_log = max(max(lr, li) for _, _, lr, li in rows)
                significant = {
                    nm: amp for nm, amp, lr, li in rows
                    if (lr - max_log >= -rheology_cutoff)
                    or (li - max_log >= -rheology_cutoff)
                }
            else:
                significant = {}

            if significant:
                uniform[ilayer] = False
                layer_muC[ilayer] = significant
                layer_K[ilayer] = {nm: 0.0 for nm in significant}
            else:
                uniform[ilayer] = True
                layer_muC[ilayer] = {}
                layer_K[ilayer] = {}

    all_nm = set()
    for ilayer in range(1, n_layers):
        all_nm.update(layer_muC.get(ilayer, {}).keys())

    if not all_nm:
        variations = np.zeros((1, 2), dtype=int)
        muC_amp = np.zeros((n_layers, 1), dtype=complex)
        K_amp_arr = np.zeros((n_layers, 1), dtype=complex)
        return model, LateralRheology(
            variations=variations,
            muC_amp=muC_amp,
            K_amp=K_amp_arr,
            uniform=uniform,
        )

    sorted_nm = sorted(all_nm)
    variations = np.array(sorted_nm, dtype=int)
    muC_amp = np.zeros((n_layers, len(sorted_nm)), dtype=complex)
    K_amp_arr = np.zeros((n_layers, len(sorted_nm)), dtype=complex)
    for ilayer in range(1, n_layers):
        if uniform[ilayer]:
            continue
        for j, nm in enumerate(sorted_nm):
            muC_amp[ilayer, j] = layer_muC.get(ilayer, {}).get(nm, 0.0)
            K_amp_arr[ilayer, j] = layer_K.get(ilayer, {}).get(nm, 0.0)

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
