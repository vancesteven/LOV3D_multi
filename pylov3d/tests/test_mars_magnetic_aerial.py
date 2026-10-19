import math

import pytest

from pylov3d.mars_magnetic_aerial import (
    relative_low_altitude_gain,
    upward_continuation_factor,
    wavelength_for_retained_fraction,
)


def test_surface_factor_is_unity():
    assert upward_continuation_factor(10_000.0, 0.0) == pytest.approx(1.0)


def test_known_e_fold_scale():
    wavelength = 2.0 * math.pi * 100.0
    assert upward_continuation_factor(wavelength, 100.0) == pytest.approx(math.e ** -1)


def test_low_altitude_gain_exceeds_unity():
    assert relative_low_altitude_gain(10_000.0, 100.0, 100_000.0) > 1.0


def test_half_amplitude_wavelength_round_trip():
    h = 100.0
    wavelength = wavelength_for_retained_fraction(h, 0.5)
    assert upward_continuation_factor(wavelength, h) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 1.0),
        (1.0, -1.0),
    ],
)
def test_upward_continuation_validation(args):
    with pytest.raises(ValueError):
        upward_continuation_factor(*args)


def test_gain_validation():
    with pytest.raises(ValueError):
        relative_low_altitude_gain(1000.0, 1000.0, 100.0)


def test_fraction_validation():
    with pytest.raises(ValueError):
        wavelength_for_retained_fraction(100.0, 1.0)
