# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.types — data structures and factory helpers."""


import jax.numpy as jnp
import pytest

from pylov3d.constants import MAX_LAYERS
from pylov3d.types import (
    make_interior_model,
    make_forcing,
)


class TestMakeInteriorModel:

    def test_basic_creation(self, io_model):
        assert io_model.n_layers == 4
        assert io_model.R0.shape == (MAX_LAYERS,)
        assert io_model.R0.dtype == jnp.float64

    def test_radii_padded(self, io_model):
        assert float(io_model.R0[0]) == 965.0
        assert float(io_model.R0[3]) == 1821.6
        # Padding is zero
        assert float(io_model.R0[4]) == 0.0

    def test_density(self, io_model):
        assert float(io_model.rho0[0]) == 5150.0
        assert float(io_model.rho0[1]) == 3244.0

    def test_elastic_layer_has_nan_eta(self, io_model):
        assert jnp.isnan(io_model.eta0[0])

    def test_viscosity_set(self, io_model):
        assert float(io_model.eta0[1]) == 1e20
        assert float(io_model.eta0[2]) == 1e11

    def test_incompressible_default(self):
        model = make_interior_model(
            R0_km=[100.0, 200.0],
            rho0=[5000.0, 3000.0],
            mu0=[0.0, 1e10],
        )
        # Ks0 should default to 1e7 * mu0_surface
        expected_Ks = 1e7 * 1e10
        assert float(model.Ks0[0]) == pytest.approx(expected_Ks)
        assert float(model.Ks0[1]) == pytest.approx(expected_Ks)

    def test_auto_delta_rho(self):
        model = make_interior_model(
            R0_km=[100.0, 200.0, 300.0],
            rho0=[5000.0, 3500.0, 3000.0],
            mu0=[0.0, 1e10, 1e10],
        )
        # Core: rho0[0] - rho0[1]
        assert float(model.Delta_rho0[0]) == pytest.approx(1500.0)
        # Layer 1: rho0[0] - rho0[1]
        assert float(model.Delta_rho0[1]) == pytest.approx(1500.0)
        # Layer 2: rho0[1] - rho0[2]
        assert float(model.Delta_rho0[2]) == pytest.approx(500.0)

    def test_normalized_fields_initially_zero(self, io_model):
        assert jnp.all(io_model.R == 0.0)
        assert jnp.all(io_model.mu == 0.0)
        assert io_model.Gg == 0.0

    def test_ocean_flag(self):
        model = make_interior_model(
            R0_km=[100.0, 200.0, 300.0],
            rho0=[5000.0, 1000.0, 3000.0],
            mu0=[0.0, 0.0, 1e10],
            ocean=[0, 1, 0],
        )
        assert int(model.ocean[0]) == 0
        assert int(model.ocean[1]) == 1
        assert int(model.ocean[2]) == 0


class TestMakeForcing:

    def test_creation(self):
        f = make_forcing(Td=1.0e5, n=2, m=0, F=0.75 + 0j)
        assert f.Td == 1.0e5
        assert f.n == 2
        assert f.m == 0
        assert f.F == 0.75

    def test_io_forcing(self, io_forcing):
        assert len(io_forcing) == 3
        assert io_forcing[0].n == 2
        assert io_forcing[0].m == 0
        assert io_forcing[1].m == -2
        assert io_forcing[2].m == 2


class TestMakeNumerics:

    def test_defaults(self, io_numerics):
        assert io_numerics.n_layers == 4
        assert io_numerics.method == "combination"
        assert io_numerics.Nrbase == 200
        assert io_numerics.Nr == 0  # computed later
        assert io_numerics.Nrlayer.shape == (MAX_LAYERS,)

    def test_perturbation_order(self, io_numerics):
        assert io_numerics.perturbation_order == 2
