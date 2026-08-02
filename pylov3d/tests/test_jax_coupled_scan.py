"""Tests for the coupled lax.scan radial propagator and public API."""

import numpy as np
import pytest

from pylov3d.couplings import Couplings, get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.jax_coupled import (
    _SCAN_CACHE,
    _get_cached_scan,
    jax_get_solution_coupled_scan,
    propagate_coupled_jax_scan,
)
from pylov3d.jax_scan import propagate_1d_jax_scan
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.solver import _get_solution_coupled
from pylov3d.types import (
    LateralRheology,
    make_forcing,
    make_interior_model,
    make_numerics,
)


def _relative_error(actual, expected):
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected))) / scale


@pytest.fixture(scope="module")
def coupled_solution():
    """Io three-layer lateral-variation case: NumPy reference + JAX scan."""
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
        model, forcing, mu_variable={1: [(2, 0, 0.1)]},
    )
    couplings = get_couplings(lateral.variations, 2, 0)

    y_ref, r_ref, Y_ref, Aprop_aux_ref = _get_solution_coupled(
        model, forcing, numerics, couplings, lateral,
    )
    y_jax, r_jax, Y_jax, Aprop_aux_jax = jax_get_solution_coupled_scan(
        model, forcing, numerics, couplings, lateral,
    )

    return {
        "model": model,
        "forcing": forcing,
        "numerics": numerics,
        "couplings": couplings,
        "lateral": lateral,
        "y_ref": y_ref,
        "r_ref": r_ref,
        "Y_ref": Y_ref,
        "y_jax": y_jax,
        "r_jax": r_jax,
        "Y_jax": Y_jax,
        "Aprop_aux_ref": Aprop_aux_ref,
        "Aprop_aux_jax": Aprop_aux_jax,
    }


def test_Y_matches_reference(coupled_solution):
    """Y_all from the JAX scan matches the NumPy fundamental matrix."""
    Y_ref = coupled_solution["Y_ref"]
    Y_jax = coupled_solution["Y_jax"]
    assert Y_jax.shape == Y_ref.shape

    rel_err = _relative_error(Y_jax, Y_ref)
    assert rel_err < 1e-12, f"Y max relative error {rel_err:.3e} >= 1e-12"


def test_y_sol_matches_reference(coupled_solution):
    """y_sol from the JAX scan matches the NumPy physical solution."""
    y_ref = coupled_solution["y_ref"]
    y_jax = coupled_solution["y_jax"]
    assert y_jax.shape == y_ref.shape
    np.testing.assert_allclose(
        coupled_solution["r_jax"], coupled_solution["r_ref"],
    )

    rel_err = _relative_error(y_jax, y_ref)
    assert rel_err < 1e-12, f"y_sol max relative error {rel_err:.3e} >= 1e-12"


def test_aprop_aux_matches_reference(coupled_solution):
    """Auxiliary propagator rows match NumPy at every radial point."""
    expected = coupled_solution["Aprop_aux_ref"]
    actual = coupled_solution["Aprop_aux_jax"]
    assert actual.shape == expected.shape
    rel_err = _relative_error(actual, expected)
    assert rel_err < 1e-12, f"Aprop_aux max relative error {rel_err:.3e}"


def test_viscoelastic_response_is_complex(coupled_solution):
    """Layer 1's eta=1e19 makes the forcing-mode response complex."""
    couplings = coupled_solution["couplings"]
    y_jax = coupled_solution["y_jax"]
    N = len(couplings.n_s)
    N6 = 6 * N

    k_f = np.where(
        (couplings.n_s == 2) & (couplings.m_s == 0)
    )[0][0]
    phi_surf = y_jax[-1, N6 + 2 * k_f]

    assert abs(phi_surf.real) > 0
    assert abs(phi_surf.imag) > 1e-6 * abs(phi_surf.real), (
        f"expected nontrivial imaginary part, got {phi_surf!r}"
    )


def test_jit_scan_is_cached_and_deterministic(coupled_solution):
    """Repeated solves reuse the memoized jitted scan and match exactly."""
    model = coupled_solution["model"]
    forcing = coupled_solution["forcing"]
    numerics = coupled_solution["numerics"]
    couplings = coupled_solution["couplings"]
    lateral = coupled_solution["lateral"]

    Y1, r1 = propagate_coupled_jax_scan(
        model, forcing, numerics, couplings, lateral,
    )
    n_entries = len(_SCAN_CACHE)
    _, scan1 = _get_cached_scan(couplings.n_s, couplings.Coup, model.Gg)

    Y2, r2 = propagate_coupled_jax_scan(
        model, forcing, numerics, couplings, lateral,
    )
    _, scan2 = _get_cached_scan(couplings.n_s, couplings.Coup, model.Gg)

    assert scan1 is scan2, "second solve rebuilt the jitted scan"
    assert len(_SCAN_CACHE) == n_entries
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(Y1, Y2)


def test_nonzero_K_amp_selection(coupled_solution):
    """Hand-injected per-layer K_amp exercises the bulk path end to end.

    The current lateral pipeline produces identically zero K_amp, so without
    this test the per-layer K_amp selection in propagate_coupled_jax_scan
    would be dead code to the suite.
    """
    model = coupled_solution["model"]
    forcing = coupled_solution["forcing"]
    numerics = coupled_solution["numerics"]
    couplings = coupled_solution["couplings"]
    lateral = coupled_solution["lateral"]

    K_amp = np.zeros_like(lateral.K_amp)
    K_amp[1, :] = 0.02 - 0.01j
    K_amp[2, :] = 0.005j
    lateral_K = lateral._replace(K_amp=K_amp)

    y_ref, _, _, Aprop_aux_ref = _get_solution_coupled(
        model, forcing, numerics, couplings, lateral_K,
    )
    y_jax, _, _, Aprop_aux_jax = jax_get_solution_coupled_scan(
        model, forcing, numerics, couplings, lateral_K,
    )

    rel_err = _relative_error(y_jax, y_ref)
    assert rel_err < 1e-12, f"K_amp case relative error {rel_err:.3e}"
    aux_err = _relative_error(Aprop_aux_jax, Aprop_aux_ref)
    assert aux_err < 1e-12, f"K_amp Aprop_aux relative error {aux_err:.3e}"
    # The injected K_amp must actually change the solution, or this test
    # would pass with the K path dropped entirely.
    assert _relative_error(y_jax, coupled_solution["y_jax"]) > 1e-6


def test_single_mode_matches_1d_scan():
    """N=1 with zero coupling reduces to the 1D scan propagator."""
    raw_model = make_interior_model(
        R0_km=[800.0, 1600.0, 1821.6],
        rho0=[5150.0, 3300.0, 3000.0],
        mu0=[0.0, 60e9, 65e9],
        eta0=[None, 1e19, None],
    )
    forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=3, method="combination", Nrbase=20)
    numerics, raw_model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(raw_model, forcing)

    n_s = np.array([2])
    m_s = np.array([0])
    Coup = np.zeros((1, 1, 27, 1))
    couplings = Couplings(
        n_s=n_s, m_s=m_s, order=np.array([0]), Coup=Coup,
    )
    lateral = LateralRheology(
        variations=np.zeros((1, 2), dtype=int),
        muC_amp=np.zeros((model.n_layers, 1), dtype=complex),
        K_amp=np.zeros((model.n_layers, 1), dtype=complex),
        uniform=np.ones(model.n_layers, dtype=bool),
    )

    Y_1d, r_1d = propagate_1d_jax_scan(model, forcing, numerics)
    Y_coup, r_coup = propagate_coupled_jax_scan(
        model, forcing, numerics, couplings, lateral,
    )

    np.testing.assert_allclose(r_coup, r_1d)
    rel_err = _relative_error(Y_coup, Y_1d)
    assert rel_err < 1e-9, f"N=1 max relative error {rel_err:.3e} >= 1e-9"
