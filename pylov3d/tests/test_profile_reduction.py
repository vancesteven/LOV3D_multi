# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from pylov3d.constants import MAX_LAYERS
from pylov3d.profile_io import RadialArtifactShells, load_radial_artifact_shells
from pylov3d.profile_reduction import reduce_radial_shells, reduced_shells_to_interior_model


def _uniform_density_shells(n=32):
    r = np.linspace(100e3, 3390e3, n)
    rho = np.full(n, 3500.0)
    K = np.linspace(160e9, 80e9, n)
    mu = np.linspace(70e9, 30e9, n)
    mu[:4] = 0.0
    return RadialArtifactShells(r, rho, K, mu, {"body": "synthetic"})


def test_reduction_preserves_mass_and_moi_for_uniform_density_exactly():
    shells = _uniform_density_shells(32)
    reduced, diag = reduce_radial_shells(shells, target_layers=8)
    assert reduced.outer_radius_m.size == 8
    assert abs(diag.mass_relative_change) < 5e-15
    assert abs(diag.cmr2_change) < 5e-15


def test_reduction_never_blends_fluid_solid_boundary():
    shells = _uniform_density_shells(32)
    reduced, _ = reduce_radial_shells(shells, target_layers=8, fluid_mu_tol_Pa=1.0)
    fluid = reduced.mu_Pa <= 1.0
    assert np.any(fluid)
    assert np.any(~fluid)
    # Because cross-boundary merges are prohibited, fluid shells remain exactly zero.
    assert np.all(reduced.mu_Pa[fluid] == 0.0)
    model = reduced_shells_to_interior_model(reduced)
    assert model.n_layers == 8


def test_high_resolution_artifact_requires_explicit_inspection_flag(tmp_path: Path):
    n = MAX_LAYERS + 4
    radius = 3390e3
    depth = np.linspace(0.0, radius - 100e3, n)
    path = tmp_path / "hires.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(1),
        depth_m=depth,
        density_kg_m3=np.full(n, 3500.0),
        bulk_modulus_Pa=np.full(n, 100e9),
        shear_modulus_Pa=np.full(n, 50e9),
        meta_body_radius_m=np.array(radius),
    )
    with pytest.raises(ValueError, match="MAX_LAYERS"):
        load_radial_artifact_shells(path)
    shells = load_radial_artifact_shells(path, enforce_max_layers=False)
    assert shells.outer_radius_m.size == n
