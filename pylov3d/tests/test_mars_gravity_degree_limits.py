import math

import pytest

from pylov3d.mars_gravity_degree_limits import (
    thickness_for_degree_fraction,
    thickness_for_degree_rms,
)


def test_fraction_scaling():
    got = thickness_for_degree_fraction(10_000.0, 2.0, 1.0, 0.1)
    assert got == pytest.approx(500.0)


def test_fraction_equal_norm():
    got = thickness_for_degree_fraction(10_000.0, 0.25, 0.5, 1.0)
    assert got == pytest.approx(20_000.0)


def test_rms_scaling():
    got = thickness_for_degree_rms(10_000.0, 4.0, 1.0)
    assert got == pytest.approx(2500.0)


def test_zero_signal_returns_inf():
    assert math.isinf(thickness_for_degree_fraction(1.0, 0.0, 1.0, 0.1))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(trial_thickness_m=0, trial_signal_coefficient=1, degree_norm=1, target_fraction=0.1),
        dict(trial_thickness_m=1, trial_signal_coefficient=1, degree_norm=0, target_fraction=0.1),
        dict(trial_thickness_m=1, trial_signal_coefficient=1, degree_norm=1, target_fraction=0),
        dict(trial_thickness_m=1, trial_signal_coefficient=1, degree_norm=1, target_fraction=1.1),
    ],
)
def test_fraction_validation(kwargs):
    with pytest.raises(ValueError):
        thickness_for_degree_fraction(**kwargs)


def test_rms_validation():
    with pytest.raises(ValueError):
        thickness_for_degree_rms(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        thickness_for_degree_rms(1.0, 1.0, 1.0, 0.0)
