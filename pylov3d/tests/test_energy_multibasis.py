# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-forcing native-basis coupled energy contraction."""

import numpy as np

from pylov3d.couplings import get_couplings
from pylov3d.energy_multibasis import get_energy_coupled_multibasis
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.solver import get_solution
from pylov3d.types import make_forcing, make_interior_model, make_numerics


def _model_and_lateral(forcings):
    model = make_interior_model(
        R0_km=[800.0, 1600.0, 1821.6],
        rho0=[5150.0, 3300.0, 3000.0],
        mu0=[0.0, 60e9, 65e9],
        Ks0=[0.0, 200e12, 200e12],
        eta0=[None, 1e19, 1e23],
    )
    numerics = make_numerics(
        n_layers=3, method="combination", Nrbase=12, perturbation_order=2,
    )
    numerics, model = set_boundary_indices(numerics, model)
    model = get_rheology(model, forcings)
    model, lateral = process_lateral_variations(
        model,
        forcings,
        mu_variable={1: [(2, 0, 0.05)]},
        eta_variable={1: [(2, 0, 0.05)]},
        rheology_cutoff=numerics.rheology_cutoff,
    )
    return model, numerics, lateral


def _solve(model, numerics, lateral, forcing, couplings):
    y, r, _, aprop = get_solution(
        model, forcing, numerics, couplings=couplings, lateral=lateral,
    )
    return y, r, aprop


def _energy_for_single_forcing(amplitude: float):
    forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=amplitude)
    model, numerics, lateral = _model_and_lateral([forcing])
    c = get_couplings(
        lateral.variations, forcing.n, forcing.m,
        perturbation_order=numerics.perturbation_order,
    )
    sol = _solve(model, numerics, lateral, forcing, c)
    return get_energy_coupled_multibasis(
        [sol], [forcing], model, numerics, [c.n_s], [c.m_s], Nenergy=4,
    )


def test_single_basis_energy_has_quadratic_forcing_scaling():
    """Energy must scale as F^2 for a fixed linear tidal solution.

    This replaces the old equality-to-get_energy_coupled check.  The legacy
    coupled-energy routine reconstructs stress with an older diagonal-only
    constitutive path, so exact agreement ceased to be a valid scientific
    invariant once TASK-046 made the multibasis reconstruction use the same
    coupled A1/A2 matrices as the forward solver.
    """
    e1 = _energy_for_single_forcing(1.0)
    ehalf = _energy_for_single_forcing(0.5)

    np.testing.assert_array_equal(ehalf.n, e1.n)
    np.testing.assert_array_equal(ehalf.m, e1.m)
    np.testing.assert_allclose(
        ehalf.energy_integral, 0.25 * e1.energy_integral,
        rtol=1e-10, atol=1e-13,
    )
    np.testing.assert_allclose(
        ehalf.energy_profile, 0.25 * e1.energy_profile,
        rtol=1e-10, atol=1e-13,
    )


def test_distinct_forcing_closures_are_supported_without_truncation():
    forcings = [
        make_forcing(Td=1.769 * 86400, n=2, m=0, F=0.6),
        make_forcing(Td=1.769 * 86400, n=2, m=2, F=0.2),
    ]
    model, numerics, lateral = _model_and_lateral(forcings)

    couplings = [
        get_couplings(
            lateral.variations, f.n, f.m,
            perturbation_order=numerics.perturbation_order,
        )
        for f in forcings
    ]
    assert not (
        np.array_equal(couplings[0].n_s, couplings[1].n_s)
        and np.array_equal(couplings[0].m_s, couplings[1].m_s)
    )

    solutions = [
        _solve(model, numerics, lateral, f, c)
        for f, c in zip(forcings, couplings)
    ]
    energy = get_energy_coupled_multibasis(
        solutions,
        forcings,
        model,
        numerics,
        [c.n_s for c in couplings],
        [c.m_s for c in couplings],
        Nenergy=4,
    )

    assert np.all(np.isfinite(energy.energy_integral))
    assert np.all(np.isfinite(energy.energy_profile))
    monopole = np.where((energy.n == 0) & (energy.m == 0))[0]
    assert len(monopole) == 1
    assert abs(energy.energy_integral[monopole[0]]) > 0.0