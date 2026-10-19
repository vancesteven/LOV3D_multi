# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
)
from pylov3d.rheology import get_rheology
from pylov3d.rheology_grid import process_lateral_fractional_grids


def _normalized_io():
    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(10)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)
    return model, numerics


def test_zero_fractional_grid_recovers_uniform_complex_rheology():
    model, numerics = _normalized_io()
    ilayer = IO_ASTHENOSPHERE_LAYER_INDEX
    target = complex(model.muC[ilayer])
    z = np.zeros((16, 32))  # lmax=8 native equiangular grid
    out, lateral = process_lateral_fractional_grids(
        model,
        dmu_grids={ilayer: z},
        deta_grids={ilayer: z},
        rheology_cutoff=numerics.rheology_cutoff,
    )
    np.testing.assert_allclose(complex(out.muC[ilayer]), target, rtol=0, atol=2e-12)
    assert bool(lateral.uniform[ilayer])


def test_degree2_grid_retains_nonuniform_rheology():
    model, numerics = _normalized_io()
    ilayer = IO_ASTHENOSPHERE_LAYER_INDEX
    lmax = 8
    nlat, nlon = 2 * lmax, 4 * lmax
    lat = -90.0 + 180.0 / (4 * lmax) + 180.0 / (2 * lmax) * np.arange(nlat)
    lon = -180.0 + 180.0 / (4 * lmax) + 180.0 / (2 * lmax) * np.arange(nlon)
    theta = np.deg2rad(90.0 - lat)[:, None]
    phi = np.deg2rad(lon)[None, :]
    pattern = 0.04 * (3.0 * np.cos(theta) ** 2 - 1.0) + 0.02 * np.sin(theta) ** 2 * np.cos(2 * phi)
    out, lateral = process_lateral_fractional_grids(
        model,
        dmu_grids={ilayer: pattern},
        deta_grids={ilayer: -2.0 * pattern},
        rheology_cutoff=numerics.rheology_cutoff,
    )
    assert not bool(lateral.uniform[ilayer])
    assert any(int(n) == 2 for n, _m in lateral.variations)
    assert np.isfinite(complex(out.muC[ilayer]).real)
    assert np.isfinite(complex(out.muC[ilayer]).imag)
