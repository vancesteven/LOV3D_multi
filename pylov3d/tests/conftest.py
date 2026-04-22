"""Shared fixtures for pylov3d tests.

The Io 4-layer model from ``scripts/multiple_layers_example.m`` is the
primary integration-test reference.
"""

import math

import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics


@pytest.fixture
def io_model():
    """Io 4-layer interior model (Steinke et al. 2020a, Model A)."""
    return make_interior_model(
        R0_km=[965.0, 1591.6, 1791.6, 1821.6],
        rho0=[5150.0, 3244.0, 3244.0, 3244.0],
        mu0=[0.0, 6e10, 7.8e5, 6.5e10],
        Ks0=[0.0, 200e16, 200e16, 200e16],
        eta0=[None, 1e20, 1e11, 1e23],
        ocean=[0, 0, 0, 0],
        Delta_rho0=[5150.0 - 3244.0, 5150.0 - 3244.0, 0.0, 0.0],
    )


@pytest.fixture
def io_forcing():
    """Three forcing components for Io eccentricity tide."""
    omega0 = 4.1086e-05
    Td = 2 * math.pi / omega0
    return [
        make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5)),
        make_forcing(Td=Td, n=2, m=-2, F=-7 / 8 * math.sqrt(6 / 5)),
        make_forcing(Td=Td, n=2, m=2, F=1 / 8 * math.sqrt(6 / 5)),
    ]


@pytest.fixture
def io_numerics():
    """Numerics config matching the Io example script."""
    return make_numerics(
        n_layers=4,
        method="combination",
        Nrbase=200,
        perturbation_order=2,
        rheology_cutoff=2.0,
        Nenergy=12,
    )
