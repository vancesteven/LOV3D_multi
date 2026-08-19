import math

import pytest

from pylov3d.mars_gravity_harmonics import (
    MARS_MEAN_RADIUS_M,
    degree_from_wavelength,
    harmonic_gravity_bound,
    radial_gravity_attenuation,
    thin_sheet_surface_gravity,
    wavelength_from_degree,
)


def test_degree_wavelength_roundtrip():
    for degree in (2.0, 10.0, 50.0, 100.0):
        wavelength = wavelength_from_degree(degree)
        assert degree_from_wavelength(wavelength) == pytest.approx(degree)


def test_surface_attenuation_is_unity():
    assert radial_gravity_attenuation(20.0, 0.0) == pytest.approx(1.0)


def test_attenuation_decreases_with_degree_and_altitude():
    a10 = radial_gravity_attenuation(10.0, 300e3)
    a20 = radial_gravity_attenuation(20.0, 300e3)
    a20_hi = radial_gravity_attenuation(20.0, 500e3)
    assert 0.0 < a20 < a10 < 1.0
    assert 0.0 < a20_hi < a20


def test_thin_sheet_anchor_420_kg_m3_per_km():
    g = thin_sheet_surface_gravity(420.0, 1000.0)
    # 1 mGal = 1e-5 m/s^2; previous planar diagnostic gives 17.613 mGal.
    assert g / 1e-5 == pytest.approx(17.613, rel=5e-5)


def test_harmonic_bound_is_surface_scale_times_attenuation():
    surface = thin_sheet_surface_gravity(420.0, 10e3)
    degree = degree_from_wavelength(1000e3)
    attenuation = radial_gravity_attenuation(degree, 300e3)
    assert harmonic_gravity_bound(420.0, 10e3, degree, 300e3) == pytest.approx(
        surface * attenuation
    )


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        degree_from_wavelength(0.0)
    with pytest.raises(ValueError):
        wavelength_from_degree(0.0)
    with pytest.raises(ValueError):
        radial_gravity_attenuation(-1.0, 0.0)
    with pytest.raises(ValueError):
        thin_sheet_surface_gravity(420.0, -1.0)
