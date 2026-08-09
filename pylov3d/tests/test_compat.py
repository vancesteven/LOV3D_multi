# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.compat — PlanetProfile adapter."""
import numpy as np
import pytest
from pylov3d.compat import planetstruct_to_interior_model, pyalma3_to_interior_model


class MockPlanetStruct:
    """Mock PlanetProfile.PlanetStruct for testing."""

    def __init__(self, n_layers=4, has_ocean=False):
        self.R_m = np.array([965e3, 1591.6e3, 1791.6e3, 1821.6e3])[:n_layers]
        self.rho_kgm3 = np.array([5150.0, 3244.0, 3244.0, 3244.0])[:n_layers]
        self.G_Pa = np.array([0.0, 60.0, 0.00078, 65.0])[:n_layers]  # GPa
        self.eta_Pas = np.array([None, 1e20, 1e11, 1e23], dtype=object)[:n_layers]

        if has_ocean:
            self.Ocean = {'comp': 'water', 'thickness_km': 200.0}
        else:
            self.Ocean = {}


class TestPlanetStructToInteriorModel:
    """Test PlanetProfile → pylov3d conversion."""

    def test_basic_conversion(self):
        """Convert 4-layer Io model."""
        planet = MockPlanetStruct(n_layers=4)
        model = planetstruct_to_interior_model(planet)

        # InteriorModel fields are padded arrays up to MAX_LAYERS
        # Check first 4 values are non-zero (the actual layers)
        assert model.n_layers == 4
        assert model.R0[0] > 0
        assert model.R0[3] > 0
        assert model.rho0[0] > 0
        assert model.rho0[3] > 0

    def test_radii_converted_to_km(self):
        """Radii converted from meters to kilometers."""
        planet = MockPlanetStruct(n_layers=4)
        model = planetstruct_to_interior_model(planet)

        # Field is R0, check first 4 values (rest are padded zeros)
        np.testing.assert_allclose(
            model.R0[:4],
            [965.0, 1591.6, 1791.6, 1821.6],
            rtol=1e-10
        )

    def test_density_unchanged(self):
        """Density passed through unchanged."""
        planet = MockPlanetStruct(n_layers=4)
        model = planetstruct_to_interior_model(planet)

        # Check first 4 values
        np.testing.assert_allclose(
            model.rho0[:4],
            [5150.0, 3244.0, 3244.0, 3244.0],
            rtol=1e-10
        )

    def test_shear_modulus_gpa_to_pa(self):
        """Shear modulus converted from GPa to Pa."""
        planet = MockPlanetStruct(n_layers=4)
        model = planetstruct_to_interior_model(planet)

        # Core: G=0 → mu0=0.0
        assert model.mu0[0] == 0.0

        # Deep mantle: G=60 GPa → mu0=6e10 Pa
        np.testing.assert_allclose(model.mu0[1], 60.0e9, rtol=1e-10)

        # Asthenosphere: G=0.00078 GPa → mu0=7.8e5 Pa
        np.testing.assert_allclose(model.mu0[2], 7.8e5, rtol=1e-2)

        # Lithosphere: G=65 GPa → mu0=6.5e10 Pa
        np.testing.assert_allclose(model.mu0[3], 65.0e9, rtol=1e-10)

    def test_viscosity_unchanged(self):
        """Viscosity passed through unchanged."""
        planet = MockPlanetStruct(n_layers=4)
        model = planetstruct_to_interior_model(planet)

        # None → NaN (elastic), actual values passed through
        assert np.isnan(model.eta0[0])  # Core: elastic (None)
        np.testing.assert_allclose(model.eta0[1], 1e20, rtol=1e-10)
        np.testing.assert_allclose(model.eta0[2], 1e11, rtol=1e-10)
        np.testing.assert_allclose(model.eta0[3], 1e23, rtol=1e-10)

    def test_ocean_layer_detection(self):
        """Ocean layer auto-detected and set to fluid."""
        planet = MockPlanetStruct(n_layers=4, has_ocean=True)
        model = planetstruct_to_interior_model(planet)

        # Ocean has lowest shear modulus (G=0.00078)
        # Should be marked in ocean array and mu0 set to 0
        assert model.ocean[2] == 1
        assert model.mu0[2] == 0.0

    def test_manual_ocean_override(self):
        """Manual ocean layer specification."""
        planet = MockPlanetStruct(n_layers=4, has_ocean=False)
        model = planetstruct_to_interior_model(planet, ocean_layer=2)

        assert model.ocean[2] == 1
        assert model.mu0[2] == 0.0

    def test_two_layer_model(self):
        """Simple 2-layer model."""
        planet = MockPlanetStruct(n_layers=2)
        model = planetstruct_to_interior_model(planet)

        # Check first 2 layers are populated
        assert model.n_layers == 2
        assert model.R0[0] > 0
        assert model.R0[1] > 0
        assert model.rho0[0] > 0
        assert model.rho0[1] > 0


class TestPyALMA3Conversion:
    """Test PyALMA3 conversion placeholder."""

    def test_not_implemented(self):
        """PyALMA3 conversion raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Milestone 4"):
            pyalma3_to_interior_model()
