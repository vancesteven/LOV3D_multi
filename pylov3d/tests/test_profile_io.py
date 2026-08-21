# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from pylov3d.profile_io import (
    load_radial_artifact_shells,
    radial_artifact_to_interior_model,
    radial_shell_bulk_diagnostics,
)


def _write_profile(path: Path):
    # Surface -> inward samples, matching the neutral PlanetProfile/PlanetThrak
    # artifact convention. The deepest sample is a fluid central core shell.
    depth = np.array([0.0, 30e3, 230e3, 2430e3])
    rho_surface_in = np.array([2900.0, 3100.0, 3500.0, 6500.0])
    R = 3390e3
    r = (R - depth)[::-1]
    rho = rho_surface_in[::-1]
    r_in = np.concatenate(([0.0], r[:-1]))
    mass = (4.0 * np.pi / 3.0) * np.sum(rho * (r**3 - r_in**3))
    C = (8.0 * np.pi / 15.0) * np.sum(rho * (r**5 - r_in**5))
    cmr2 = C / (mass * R**2)
    np.savez_compressed(
        path,
        schema_version=np.array(1),
        depth_m=depth,
        pressure_MPa=np.array([0.1, 100.0, 1000.0, 20000.0]),
        temperature_K=np.array([220.0, 300.0, 1000.0, 1800.0]),
        density_kg_m3=rho_surface_in,
        bulk_modulus_Pa=np.array([70e9, 90e9, 130e9, 180e9]),
        shear_modulus_Pa=np.array([30e9, 40e9, 60e9, 0.0]),
        meta_body_radius_m=np.array(R),
        meta_body_mass_kg=np.array(mass),
        meta_cmr2=np.array(cmr2),
        meta_body=np.array("Mars"),
    )


def test_shell_loader_reverses_surface_profile_to_core_outward(tmp_path: Path):
    path = tmp_path / "profile.npz"
    _write_profile(path)
    shells = load_radial_artifact_shells(path)
    np.testing.assert_allclose(shells.outer_radius_m, [960e3, 3160e3, 3360e3, 3390e3])
    np.testing.assert_allclose(shells.rho_kgm3, [6500.0, 3500.0, 3100.0, 2900.0])
    np.testing.assert_allclose(shells.mu_Pa, [0.0, 60e9, 40e9, 30e9])
    assert shells.metadata["body"] == "Mars"


def test_artifact_to_interior_model_preserves_moduli_and_core_convention(tmp_path: Path):
    path = tmp_path / "profile.npz"
    _write_profile(path)
    model = radial_artifact_to_interior_model(path)
    assert model.n_layers == 4
    np.testing.assert_allclose(model.R0[:4], [960.0, 3160.0, 3360.0, 3390.0])
    np.testing.assert_allclose(model.Ks0[:4], [180e9, 130e9, 90e9, 70e9])
    np.testing.assert_allclose(model.mu0[:4], [0.0, 60e9, 40e9, 30e9])
    np.testing.assert_array_equal(np.asarray(model.ocean[:4]), [0, 0, 0, 0])


def test_bulk_diagnostics_close_against_artifact_metadata(tmp_path: Path):
    path = tmp_path / "profile.npz"
    _write_profile(path)
    diag = radial_shell_bulk_diagnostics(load_radial_artifact_shells(path))
    assert diag.mass_kg > 0
    assert 0 < diag.cmr2 < 0.4
    assert abs(diag.mass_relative_error) < 1e-14
    assert abs(diag.cmr2_error) < 1e-14


def test_missing_radius_metadata_fails(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1),
        depth_m=np.array([0.0, 1.0]),
        density_kg_m3=np.array([3000.0, 3100.0]),
        bulk_modulus_Pa=np.array([70e9, 80e9]),
        shear_modulus_Pa=np.array([30e9, 35e9]),
    )
    with pytest.raises(ValueError, match="body_radius"):
        load_radial_artifact_shells(path)
