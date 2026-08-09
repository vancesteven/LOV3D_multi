# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for static and traced JAX coupled-propagator assembly."""

import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.jax_coupled import (
    _build_aprop_coupled_jax,
    _precompute_coupled_static,
)
from pylov3d.propagator import build_aprop_coupled, compute_gravity
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.types import make_forcing, make_interior_model, make_numerics


def _relative_error(actual, expected):
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected))) / scale


@pytest.fixture(scope="module")
def coupled_case():
    """Io three-layer lateral-variation pipeline and radial samples."""
    raw_model = make_interior_model(
        R0_km=[800.0, 1600.0, 1821.6],
        rho0=[5150.0, 3300.0, 3000.0],
        mu0=[0.0, 60e9, 65e9],
        Ks0=[200e9, 200e9, 200e9],
        eta0=[None, 1e19, None],
    )
    forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)
    numerics, raw_model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(raw_model, forcing)
    model, lateral = process_lateral_variations(
        model,
        forcing,
        mu_variable={1: [(2, 0, 0.1)]},
    )
    couplings = get_couplings(lateral.variations, 2, 0)
    static = _precompute_coupled_static(
        couplings.n_s, couplings.Coup, model.Gg,
    )

    radii = np.asarray(model.R[:model.n_layers], dtype=float)
    densities = np.asarray(model.rho[:model.n_layers], dtype=float)
    masses = np.zeros(model.n_layers)
    masses[0] = 4.0 / 3.0 * math.pi * densities[0] * radii[0] ** 3
    masses[1] = masses[0]
    for layer in range(2, model.n_layers):
        masses[layer] = masses[layer - 1] + 4.0 / 3.0 * math.pi * (
            densities[layer - 1]
            * (radii[layer - 1] ** 3 - radii[layer - 2] ** 3)
        )

    samples = [
        (radii[0], 1),
        ((radii[0] + radii[1]) / 2.0, 1),
        (radii[-1], model.n_layers - 1),
    ]
    return model, lateral, couplings, static, masses, samples


def _build_pair(case, radius, layer, muC_amp=None, K_amp=None):
    model, lateral, couplings, static, masses, _ = case
    rho = float(model.rho[layer])
    g, dg = compute_gravity(
        radius,
        rho,
        masses[layer],
        float(model.R[layer - 1]),
        model.Gg,
    )
    if muC_amp is None:
        muC_amp = lateral.muC_amp[layer]
    if K_amp is None:
        K_amp = lateral.K_amp[layer]
    args = (
        radius,
        g,
        dg,
        complex(model.muC[layer]),
        complex(model.lam[layer]),
        rho,
        np.asarray(muC_amp, dtype=np.complex128),
        np.asarray(K_amp, dtype=np.complex128),
    )
    actual = np.asarray(_build_aprop_coupled_jax(*args, static))
    expected = build_aprop_coupled(
        radius,
        g,
        dg,
        couplings.n_s,
        args[3],
        args[4],
        rho,
        model.Gg,
        couplings.Coup,
        args[6],
        args[7],
    )
    return actual, expected, args


def test_direct_equivalence_at_three_radii(coupled_case):
    """JAX assembly matches NumPy from the CMB through the surface."""
    *_, samples = coupled_case
    for radius, layer in samples:
        actual, expected, _ = _build_pair(coupled_case, radius, layer)
        assert _relative_error(actual, expected) < 1e-10

    # The current lateral pipeline produces zero K amplitudes, so exercise
    # the real coupling tensor with an explicit complex bulk perturbation too.
    radius, layer = samples[1]
    K_amp = np.asarray(coupled_case[1].K_amp[layer]).copy()
    K_amp[0] = 0.02 - 0.01j
    actual, expected, _ = _build_pair(
        coupled_case, radius, layer, K_amp=K_amp,
    )
    assert _relative_error(actual, expected) < 1e-10


def test_zero_amplitude_reduction(coupled_case):
    """Zero amplitudes reproduce the block-diagonal NumPy system."""
    *_, samples = coupled_case
    nreo = coupled_case[2].Coup.shape[3]
    zeros = np.zeros(nreo, dtype=np.complex128)
    radius, layer = samples[1]
    actual, expected, _ = _build_pair(
        coupled_case, radius, layer, muC_amp=zeros, K_amp=zeros,
    )
    assert _relative_error(actual, expected) < 1e-10

    N = len(coupled_case[2].n_s)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rows = np.r_[3 * i:3 * i + 3, 3 * N + 3 * i:3 * N + 3 * i + 3,
                         6 * N + 2 * i:6 * N + 2 * i + 2]
            cols = np.r_[3 * j:3 * j + 3, 3 * N + 3 * j:3 * N + 3 * j + 3,
                         6 * N + 2 * j:6 * N + 2 * j + 2]
            assert np.max(np.abs(actual[np.ix_(rows, cols)])) < 1e-12


def test_jit_smoke(coupled_case):
    """The traced build compiles and preserves its numerical result."""
    *_, static, _, samples = coupled_case
    radius, layer = samples[1]
    _, expected, args = _build_pair(coupled_case, radius, layer)
    compiled = jax.jit(
        lambda r, g, dg, muC, lam, rho, mu_amp, K_amp:
        _build_aprop_coupled_jax(
            r, g, dg, muC, lam, rho, mu_amp, K_amp, static,
        )
    )
    actual = np.asarray(compiled(*args))
    assert _relative_error(actual, expected) < 1e-12


def test_multi_reo_coupling(coupled_case):
    """Distinct per-rheology-mode amplitudes exercise the reo contraction axis.

    The Io fixture has Nreo=1, so without this test the einsum over the reo
    axis could be replaced by a hardcoded index 0 and still pass.
    """
    model, _, couplings, _, masses, samples = coupled_case
    Coup2 = np.concatenate([couplings.Coup, 0.5 * couplings.Coup], axis=3)
    static2 = _precompute_coupled_static(couplings.n_s, Coup2, model.Gg)

    radius, layer = samples[1]
    rho = float(model.rho[layer])
    g, dg = compute_gravity(
        radius, rho, masses[layer], float(model.R[layer - 1]), model.Gg,
    )
    muC_amp = np.array([0.1, 0.03 - 0.02j], dtype=np.complex128)
    K_amp = np.array([0.02 - 0.01j, 0.005j], dtype=np.complex128)

    actual = np.asarray(_build_aprop_coupled_jax(
        radius, g, dg,
        complex(model.muC[layer]), complex(model.lam[layer]),
        rho, muC_amp, K_amp, static2,
    ))
    expected = build_aprop_coupled(
        radius, g, dg, couplings.n_s,
        complex(model.muC[layer]), complex(model.lam[layer]),
        rho, model.Gg, Coup2, muC_amp, K_amp,
    )
    assert _relative_error(actual, expected) < 1e-10


def test_degree_zero_overrides(coupled_case):
    """Degree-0 V/W/S/T and Poisson rows retain their special equations."""
    model, _, couplings, _, _, samples = coupled_case
    zero_modes = np.flatnonzero(couplings.n_s == 0)
    assert zero_modes.size > 0
    radius, layer = samples[1]
    actual, expected, _ = _build_pair(coupled_case, radius, layer)
    assert _relative_error(actual, expected) < 1e-10

    N = len(couplings.n_s)
    for k in zero_modes:
        for row in (3 * k + 1, 3 * k + 2, 3 * N + 3 * k + 1,
                    3 * N + 3 * k + 2):
            wanted = np.zeros(8 * N, dtype=np.complex128)
            wanted[row] = 1.0
            np.testing.assert_allclose(actual[row], wanted, atol=1e-12)
        phi_row = 6 * N + 2 * k
        wanted = np.zeros(8 * N, dtype=np.complex128)
        wanted[3 * k] = -4 * math.pi * model.Gg
        np.testing.assert_allclose(actual[phi_row], wanted, atol=1e-12)
