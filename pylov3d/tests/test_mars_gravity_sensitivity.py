import math

import pytest

from pylov3d.mars_gravity_sensitivity import (
    slab_gravity_mgal,
    sinusoidal_sheet_gravity_mgal,
    thickness_for_gravity_mgal,
)


def test_zero_thickness_zero_signal():
    assert slab_gravity_mgal(420.0, 0.0) == 0.0


def test_signal_scales_linearly_with_density_and_thickness():
    base = slab_gravity_mgal(420.0, 10.0)
    assert slab_gravity_mgal(840.0, 10.0) == pytest.approx(2.0 * base)
    assert slab_gravity_mgal(420.0, 20.0) == pytest.approx(2.0 * base)


def test_upward_continuation_at_one_e_folding():
    wavelength = 1000.0
    altitude = wavelength / (2.0 * math.pi)
    surface = slab_gravity_mgal(420.0, 10.0)
    got = sinusoidal_sheet_gravity_mgal(420.0, 10.0, wavelength, altitude)
    assert got == pytest.approx(surface / math.e)


def test_thickness_inverse_round_trip():
    target = 10.0
    H = thickness_for_gravity_mgal(420.0, target)
    assert slab_gravity_mgal(420.0, H) == pytest.approx(target)


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        sinusoidal_sheet_gravity_mgal(420.0, 10.0, 0.0, 300.0)
