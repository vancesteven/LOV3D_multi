"""Tests for pylov3d.grid — radial discretization methods."""

import jax.numpy as jnp
import pytest

from pylov3d.grid import set_boundary_indices
from pylov3d.types import make_interior_model, make_numerics


@pytest.fixture
def io_4layer():
    """Io 4-layer model + numerics for grid tests."""
    model = make_interior_model(
        R0_km=[965.0, 1591.6, 1791.6, 1821.6],
        rho0=[5150.0, 3244.0, 3244.0, 3244.0],
        mu0=[0.0, 6e10, 7.8e5, 6.5e10],
    )
    return model


class TestCombination:

    def test_nr_total(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)
        numerics, _ = set_boundary_indices(numerics, io_4layer)
        # Each layer gets floor(frac*200) + 200 points
        # Total should be > 3*200 = 600
        assert numerics.Nr > 600
        assert numerics.Nr == int(jnp.sum(numerics.Nrlayer))

    def test_core_has_zero_points(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)
        numerics, _ = set_boundary_indices(numerics, io_4layer)
        assert int(numerics.Nrlayer[0]) == 0

    def test_bcindices_monotonic(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)
        numerics, _ = set_boundary_indices(numerics, io_4layer)
        # BCindices for layers 1..n_layers-1 should be monotonically increasing
        bc = [int(numerics.BCindices[i]) for i in range(3)]
        assert bc[0] < bc[1] < bc[2]

    def test_last_bcindex_equals_nr_plus_1(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)
        numerics, _ = set_boundary_indices(numerics, io_4layer)
        # Last BC index = 1 + sum(Nrlayer) = 1 + Nr
        assert int(numerics.BCindices[2]) == numerics.Nr + 1

    def test_matlab_reference(self, io_4layer):
        """Match MATLAB combination method for Io with Nrbase=200."""
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)
        numerics, _ = set_boundary_indices(numerics, io_4layer)
        # Fractional thicknesses:
        # layer 2: (1591.6 - 965) / (1821.6 - 965) = 626.6/856.6 ≈ 0.7315
        # layer 3: (1791.6 - 1591.6) / 856.6 = 200/856.6 ≈ 0.2335
        # layer 4: (1821.6 - 1791.6) / 856.6 = 30/856.6 ≈ 0.0350
        # Nrlayer: [0, floor(0.7315*200)+200, floor(0.2335*200)+200, floor(0.0350*200)+200]
        #        = [0, 146+200, 46+200, 7+200] = [0, 346, 246, 207]
        assert int(numerics.Nrlayer[0]) == 0
        assert int(numerics.Nrlayer[1]) == 346
        assert int(numerics.Nrlayer[2]) == 246
        assert int(numerics.Nrlayer[3]) == 207
        assert numerics.Nr == 799


class TestVariable:

    def test_each_layer_gets_nrbase(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="variable", Nrbase=100)
        numerics, _ = set_boundary_indices(numerics, io_4layer)
        for i in range(1, 4):
            assert int(numerics.Nrlayer[i]) == 100
        assert numerics.Nr == 300


class TestFixed:

    def test_total_equals_nrbase(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="fixed", Nrbase=300)
        numerics, model = set_boundary_indices(numerics, io_4layer)
        assert numerics.Nr == 300

    def test_adjusts_radii(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="fixed", Nrbase=300)
        _, model = set_boundary_indices(numerics, io_4layer)
        # Interior radii may be adjusted; surface radius stays the same
        # The adjusted R0 should still be close to originals
        assert abs(float(model.R0[3]) - 1821.6) < 5.0  # surface layer


class TestManual:

    def test_manual_setting(self, io_4layer):
        nrlayer = [0, 100, 50, 150]
        numerics = make_numerics(n_layers=4, method="manual", Nrbase=200)
        numerics, _ = set_boundary_indices(numerics, io_4layer, Nrlayer_manual=nrlayer)
        assert numerics.Nr == 300
        assert int(numerics.Nrlayer[1]) == 100
        assert int(numerics.Nrlayer[2]) == 50
        assert int(numerics.Nrlayer[3]) == 150

    def test_manual_missing_raises(self, io_4layer):
        numerics = make_numerics(n_layers=4, method="manual", Nrbase=200)
        with pytest.raises(ValueError):
            set_boundary_indices(numerics, io_4layer)


class TestSingleLayer:

    def test_two_layer_model(self):
        model = make_interior_model(
            R0_km=[100.0, 200.0],
            rho0=[5000.0, 3000.0],
            mu0=[0.0, 1e10],
        )
        numerics = make_numerics(n_layers=2, method="combination", Nrbase=100)
        numerics, _ = set_boundary_indices(numerics, model)
        assert numerics.Nr == 100
        assert int(numerics.Nrlayer[1]) == 100
