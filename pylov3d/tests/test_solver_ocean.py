"""Tests for the 1D ocean-layer solve path (TASK-005).

The ocean path was previously untested and carried two divergences from the
MATLAB reference plus an index-convention bug:

1. Inside the ocean the propagator must reduce to the Laplace equation for
   the potential (MATLAB get_solution.m:1924-1935); the elastic system with
   the ocean's muC=0 was being integrated instead.
2. The ocean-ceiling node is the shell's identity origin in the solution
   recombination (get_solution.m:879).
3. BCindices were stored 1-based (MATLAB convention) but consumed 0-based,
   shifting ocean_start/ocean_end one node up and making the 24x24 BC
   matrix singular (rank 22: V/W columns of the sub-ocean block dropped).

Physics anchor: an inviscid ocean is the mu -> 0 limit of an elastic layer,
so the ocean path must agree with the ordinary no-ocean solve as the layer
rigidity vanishes.  MATLAB numerical cross-validation is TASK-008.
"""

import numpy as np
import pytest

from pylov3d.grid import set_boundary_indices
from pylov3d.love import get_love
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.types import make_forcing, make_interior_model, make_numerics

# Europa-like: core / rocky mantle / ocean / ice shell
GEOM = dict(
    R0_km=[700.0, 1430.0, 1550.0, 1560.8],
    rho0=[5150.0, 3300.0, 1000.0, 920.0],
    eta0=[None, None, None, None],
)
MU = [0.0, 40e9, 0.0, 3.3e9]
OCEAN_LAYER = 2


def _forcing():
    return make_forcing(Td=3.06822e5, n=2, m=0, F=1.0)


def _numerics():
    return make_numerics(n_layers=4, method="combination", Nrbase=100)


@pytest.fixture(scope="module")
def ocean_solution():
    model = make_interior_model(mu0=MU, ocean=[0, 0, 1, 0], **GEOM)
    forcing = _forcing()
    numerics, model = set_boundary_indices(_numerics(), model)
    model = get_rheology(model, forcing)
    y_sol, r_grid, Y, aux = get_solution(model, forcing, numerics)
    return model, forcing, numerics, y_sol, r_grid, Y, aux


def _layer_map(numerics):
    Nr = numerics.Nr
    lm = np.zeros(Nr + 1, dtype=int)
    k = 1
    for il in range(1, 4):
        for _ in range(int(numerics.Nrlayer[il])):
            lm[k] = il
            k += 1
    return lm


def _ocean_nodes(numerics):
    ocean_start = int(numerics.BCindices[OCEAN_LAYER - 2])
    ocean_end = int(numerics.BCindices[OCEAN_LAYER - 1])
    return ocean_start, ocean_end


def test_bcindices_are_zero_based_boundary_nodes(ocean_solution):
    """BCindices must point at the interface nodes (last node of the layer
    below), the convention the solver consumes them with."""
    _, _, numerics, _, r_grid, _, _ = ocean_solution
    model_raw = make_interior_model(mu0=MU, ocean=[0, 0, 1, 0], **GEOM)
    lm = _layer_map(numerics)
    ocean_start, ocean_end = _ocean_nodes(numerics)

    assert lm[ocean_start] == OCEAN_LAYER - 1     # ocean floor: mantle node
    assert lm[ocean_start + 1] == OCEAN_LAYER     # next node is in the ocean
    assert lm[ocean_end] == OCEAN_LAYER           # ocean ceiling node
    assert lm[ocean_end + 1] == OCEAN_LAYER + 1   # next node is in the shell
    # Interface nodes sit exactly on the layer radii.
    R = [float(r) for r in model_raw.R0[:4]]
    np.testing.assert_allclose(r_grid[ocean_start] * R[3], R[1], rtol=1e-12)
    np.testing.assert_allclose(r_grid[ocean_end] * R[3], R[2], rtol=1e-12)


def test_ocean_solve_is_elastic_and_reasonable(ocean_solution):
    """Elastic ocean-bearing Europa: k2 real and in the expected range."""
    _, _, numerics, y_sol, r_grid, _, _ = ocean_solution
    Nr = len(r_grid) - 1
    k2 = complex(y_sol[Nr][6]) - 1.0
    assert abs(k2.imag) < 1e-10
    # Ocean decoupling makes k2 an order of magnitude above the no-ocean
    # value (~0.02); literature Europa-with-ocean k2 is ~0.25-0.3.
    assert 0.15 < k2.real < 0.40


def test_displacement_rows_frozen_inside_ocean(ocean_solution):
    """The in-ocean propagator is Laplace-only: after the identity restart,
    the U..T rows of the fundamental matrix must stay exactly identity."""
    _, _, numerics, _, _, Y, _ = ocean_solution
    ocean_start, ocean_end = _ocean_nodes(numerics)
    I8 = np.eye(8, dtype=np.complex128)
    for k in range(ocean_start + 1, ocean_end + 1):
        np.testing.assert_array_equal(Y[k][:6, :], I8[:6, :])


def test_aprop_aux_zero_inside_ocean(ocean_solution):
    """Stress/strain recovery rows are undefined (zero) in the ocean."""
    _, _, numerics, _, _, _, aux = ocean_solution
    ocean_start, ocean_end = _ocean_nodes(numerics)
    assert np.max(np.abs(aux[ocean_start + 1: ocean_end + 1])) == 0.0


def test_potential_continuous_at_ocean_ceiling(ocean_solution):
    """y_sol at the ocean-ceiling node is the shell's identity origin
    (MATLAB get_solution.m:879); the potential must be continuous across
    the ocean/shell representations."""
    _, _, numerics, y_sol, _, _, _ = ocean_solution
    _, ocean_end = _ocean_nodes(numerics)
    # Phi cannot distinguish the segments here: BC row 23 forces the ocean
    # and shell representations of the potential to agree at this node.
    # U can: the ocean representation has U = 0 (undefined in the fluid),
    # the shell representation carries the shell's basal displacement.
    U = y_sol[:, 0].real
    assert abs(U[ocean_end]) > 1e-6
    assert abs(U[ocean_end] - U[ocean_end + 1]) < 1e-2 * abs(U[ocean_end])
    # And the potential remains continuous across the interface.
    phi = y_sol[:, 6].real
    step_interface = abs(phi[ocean_end] - phi[ocean_end - 1])
    step_ocean = abs(phi[ocean_end - 1] - phi[ocean_end - 2])
    assert step_interface < 2 * step_ocean


def test_fluid_limit_matches_ocean_path(ocean_solution):
    """mu -> 0 elastic layer converges to the true ocean solve.

    Measured convergence: mu=1e5 -> k2=0.2261, mu=1e4 -> 0.2650,
    mu=1e3 -> 0.2698 vs ocean 0.2703 (grid-independent at Nrbase=100/200).
    """
    _, forcing, numerics, y_sol, r_grid, _, _ = ocean_solution
    Nr = len(r_grid) - 1
    k2_ocean = (complex(y_sol[Nr][6]) - 1.0).real

    k2_soft = []
    for mu_o in (1e4, 1e3):
        model_s = make_interior_model(mu0=[0.0, 40e9, mu_o, 3.3e9], **GEOM)
        love_s, _, _ = get_love(model_s, forcing, _numerics())
        k2_soft.append(complex(love_s.k[0]).real)

    # Monotone approach and sub-percent agreement at mu=1e3 Pa.
    assert abs(k2_soft[1] - k2_ocean) < abs(k2_soft[0] - k2_ocean)
    assert abs(k2_soft[1] - k2_ocean) / k2_ocean < 5e-3


def test_regression_pin_europa_like_k2(ocean_solution):
    """Pin the ocean-path k2 (to be cross-checked against MATLAB in
    TASK-008; update the reference there if MATLAB disagrees)."""
    _, _, _, y_sol, r_grid, _, _ = ocean_solution
    Nr = len(r_grid) - 1
    k2 = complex(y_sol[Nr][6]) - 1.0
    np.testing.assert_allclose(k2.real, 0.270346, atol=2e-4)


def test_jax_paths_reject_ocean_models():
    """The 1D JAX solvers do not implement the ocean equations and must
    refuse ocean-bearing models instead of silently mis-solving them."""
    from pylov3d.jax_propagator import propagate_1d_jax
    from pylov3d.jax_scan import propagate_1d_jax_scan

    model = make_interior_model(mu0=MU, ocean=[0, 0, 1, 0], **GEOM)
    forcing = _forcing()
    numerics, model = set_boundary_indices(_numerics(), model)
    model = get_rheology(model, forcing)

    with pytest.raises(NotImplementedError, match="ocean"):
        propagate_1d_jax_scan(model, forcing, numerics)
    with pytest.raises(NotImplementedError, match="ocean"):
        propagate_1d_jax(model, forcing, numerics)
