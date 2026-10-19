import numpy as np
import pytest

from pylov3d.mars_poroelastic import (
    gassmann_bulk_modulus,
    poroelastic_state,
    saturated_density,
)


def test_saturated_density_volume_average():
    got = saturated_density(3000.0, 1000.0, 0.2)
    assert got == pytest.approx(2600.0)


def test_gassmann_saturation_stiffens_bulk_frame():
    Kd = 15e9
    Ks = 70e9
    Kf = 2.2e9
    Ksat = gassmann_bulk_modulus(Kd, Ks, Kf, 0.2)
    assert Kd < Ksat < Ks


def test_gassmann_preserves_shear_modulus():
    state = poroelastic_state(
        K_solid_pa=70e9,
        mu_solid_pa=30e9,
        rho_solid_kg_m3=2900.0,
        K_dry_pa=20e9,
        mu_dry_pa=12e9,
        porosity=0.15,
        saturated=True,
    )
    assert state.mu_pa == 12e9


def test_saturation_changes_vp_and_density_but_not_effective_mu():
    kwargs = dict(
        K_solid_pa=70e9,
        mu_solid_pa=30e9,
        rho_solid_kg_m3=2900.0,
        K_dry_pa=20e9,
        mu_dry_pa=12e9,
        porosity=0.15,
    )
    dry = poroelastic_state(**kwargs, saturated=False)
    wet = poroelastic_state(**kwargs, saturated=True)
    assert wet.K_pa > dry.K_pa
    assert wet.rho_kg_m3 > dry.rho_kg_m3
    assert wet.mu_pa == dry.mu_pa
    assert np.isfinite(wet.chi2)
    assert np.isfinite(dry.chi2)


def test_invalid_frame_bulk_modulus_rejected():
    with pytest.raises(ValueError):
        gassmann_bulk_modulus(70e9, 70e9, 2.2e9, 0.2)
