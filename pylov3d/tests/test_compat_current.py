# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the current PlanetProfile -> pylov3d compatibility adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from pylov3d.compat import planetstruct_shells, planetstruct_to_interior_model


class MockPlanetStruct:
    def __init__(self, *, reduced=False, fluid_shell=False):
        self.r_m = np.array([1821.6e3, 1791.6e3, 1591.6e3, 965.0e3, 0.0])
        self.rho_kgm3 = np.array([3244.0, 3244.0, 3244.0, 5150.0])
        gs = np.array([65.0, 60.0, 55.0, 0.0])
        if fluid_shell:
            gs[1] = 0.0
        self.Seismic = SimpleNamespace(
            GS_GPa=gs,
            KS_GPa=np.array([130.0, 125.0, 120.0, 200.0]),
            VP_kms=np.array([7.8, 7.6, 7.4, 8.5]),
            VS_kms=np.sqrt(np.maximum(gs, 0.0) * 1e9 / self.rho_kgm3) / 1e3,
        )
        self.eta_Pas = np.array([1e23, 1e21, 1e20, np.nan])
        self.phase = np.array([10, 10, 10, 20])
        self.Reduced = SimpleNamespace(
            r_m=None,
            rho_kgm3=None,
            eta_Pas=None,
            phase=None,
            Seismic=SimpleNamespace(GS_GPa=None, VP_kms=None, VS_kms=None),
        )
        if reduced:
            rho = np.array([3200.0, 3400.0, 5100.0])
            gs_r = np.array([62.0, 58.0, 0.0])
            vp = np.array([7.7, 7.5, 8.2])
            vs = np.sqrt(np.maximum(gs_r, 0.0) * 1e9 / rho) / 1e3
            self.Reduced = SimpleNamespace(
                r_m=np.array([1821.6e3, 1500.0e3, 900.0e3]),
                rho_kgm3=rho,
                eta_Pas=np.array([1e23, 1e20, np.nan]),
                phase=np.array([10, 10, 20]),
                Seismic=SimpleNamespace(GS_GPa=gs_r, VP_kms=vp, VS_kms=vs),
            )


def test_full_shell_extraction_reverses_to_core_outward():
    shells = planetstruct_shells(MockPlanetStruct(), prefer_reduced=False)
    np.testing.assert_allclose(shells.outer_radius_m, [965.0e3, 1591.6e3, 1791.6e3, 1821.6e3])
    np.testing.assert_allclose(shells.rho_kgm3, [5150.0, 3244.0, 3244.0, 3244.0])
    np.testing.assert_allclose(shells.mu_Pa, [0.0, 55e9, 60e9, 65e9])
    np.testing.assert_allclose(shells.K_Pa, [200e9, 120e9, 125e9, 130e9])
    assert shells.source == "full"


def test_interior_model_preserves_moduli_and_order():
    model = planetstruct_to_interior_model(MockPlanetStruct(), prefer_reduced=False)
    assert model.n_layers == 4
    np.testing.assert_allclose(model.R0[:4], [965.0, 1591.6, 1791.6, 1821.6])
    np.testing.assert_allclose(model.rho0[:4], [5150.0, 3244.0, 3244.0, 3244.0])
    np.testing.assert_allclose(model.mu0[:4], [0.0, 55e9, 60e9, 65e9])
    np.testing.assert_allclose(model.Ks0[:4], [200e9, 120e9, 125e9, 130e9])
    np.testing.assert_array_equal(np.asarray(model.ocean[:4]), [0, 0, 0, 0])


def test_noncentral_zero_shear_shell_becomes_ocean():
    model = planetstruct_to_interior_model(MockPlanetStruct(fluid_shell=True), prefer_reduced=False)
    assert int(model.ocean[2]) == 1
    assert float(model.mu0[2]) == 0.0


def test_reduced_profile_bulk_is_reconstructed():
    planet = MockPlanetStruct(reduced=True)
    shells = planetstruct_shells(planet, prefer_reduced=True)
    assert shells.source == "Reduced"
    rho = planet.Reduced.rho_kgm3
    vp = planet.Reduced.Seismic.VP_kms * 1e3
    vs = planet.Reduced.Seismic.VS_kms * 1e3
    expected = rho * (vp**2 - 4.0 * vs**2 / 3.0)
    np.testing.assert_allclose(shells.K_Pa, expected[::-1], rtol=1e-12)


def test_missing_full_bulk_modulus_fails_loudly():
    planet = MockPlanetStruct()
    planet.Seismic.KS_GPa = None
    with pytest.raises(ValueError, match="Seismic.KS_GPa"):
        planetstruct_shells(planet, prefer_reduced=False)
