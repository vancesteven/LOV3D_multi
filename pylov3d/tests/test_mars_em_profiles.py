import pytest

from pylov3d.mars_em_profiles import (
    archie_connected_pore_conductivity,
    layered_conductance_s,
)


def test_archie_full_porosity_full_saturation_returns_fluid_plus_matrix():
    sigma = archie_connected_pore_conductivity(
        1.0, 1.0, 2.0, matrix_conductivity_s_m=0.1
    )
    assert sigma == pytest.approx(2.1)


def test_archie_zero_saturation_returns_matrix_only():
    sigma = archie_connected_pore_conductivity(
        0.3, 0.0, 1.0, matrix_conductivity_s_m=1e-4
    )
    assert sigma == pytest.approx(1e-4)


def test_archie_standard_phi_squared_scaling():
    sigma = archie_connected_pore_conductivity(
        0.3,
        1.0,
        1.0,
        cementation_exponent=2.0,
        saturation_exponent=2.0,
    )
    assert sigma == pytest.approx(0.09)


def test_archie_monotonic_with_saturation():
    dryish = archie_connected_pore_conductivity(0.2, 0.3, 1.0)
    wet = archie_connected_pore_conductivity(0.2, 0.9, 1.0)
    assert wet > dryish


def test_layered_conductance():
    s = layered_conductance_s([1e-3, 1e-1], [5e3, 10e3])
    assert s == pytest.approx(1005.0)


def test_validation():
    with pytest.raises(ValueError):
        archie_connected_pore_conductivity(-0.1, 1.0, 1.0)
    with pytest.raises(ValueError):
        archie_connected_pore_conductivity(0.1, 1.1, 1.0)
    with pytest.raises(ValueError):
        layered_conductance_s([1.0], [1.0, 2.0])
