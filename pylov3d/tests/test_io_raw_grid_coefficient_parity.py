# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Strict MATLAB anchor for the TASK-046 nonlinear raw-grid transform."""

import numpy as np

from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_grids,
)
from pylov3d.rheology import get_rheology
from pylov3d.rheology_grid import process_lateral_fractional_grids


MATLAB_MU00 = 1.157992033067e-05 + 2.205138890911e-06j
MATLAB_MODES = {
    (2, -2): -9.6073566100e-08 + 5.6553618800e-08j,
    (2, 0): +2.8501436300e-07 - 1.5799859000e-07j,
    (2, 2): -9.6954363400e-08 + 5.5029931100e-08j,
    (4, -2): +9.1094425400e-10 - 1.8874091200e-09j,
    (4, 0): -1.6999805900e-09 + 4.5297687300e-09j,
    (4, 2): +9.4062583800e-10 - 1.8727935400e-09j,
}


def _raw_grid_case():
    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(10)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)
    _lat, _lon, dmu, deta = io_mu_eta_grids(l_max=100)
    ilayer = IO_ASTHENOSPHERE_LAYER_INDEX
    model, lateral = process_lateral_fractional_grids(
        model,
        dmu_grids={ilayer: dmu},
        deta_grids={ilayer: deta},
        rheology_cutoff=numerics.rheology_cutoff,
        minimum_rheology_value=-13.0,
    )
    return model, lateral, ilayer


def test_io_raw_grid_mean_complex_rheology_matches_matlab():
    model, _lateral, ilayer = _raw_grid_case()
    np.testing.assert_allclose(
        complex(model.muC[ilayer]), MATLAB_MU00, rtol=5e-10, atol=1e-15
    )


def test_io_raw_grid_retained_coefficients_match_matlab():
    _model, lateral, ilayer = _raw_grid_case()
    modes = [tuple(map(int, nm)) for nm in np.asarray(lateral.variations)]
    assert set(modes) == set(MATLAB_MODES)
    assert len(modes) == len(MATLAB_MODES)
    for j, nm in enumerate(modes):
        np.testing.assert_allclose(
            complex(lateral.muC_amp[ilayer, j]),
            MATLAB_MODES[nm],
            rtol=5e-9,
            atol=1e-15,
        )
