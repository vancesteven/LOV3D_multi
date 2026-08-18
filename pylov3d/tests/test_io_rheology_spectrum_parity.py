# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""TASK-046 regression for the authoritative Io viscoelastic SH closure.

Native MATLAB, using the original raw lat/lon grid path from
Consistency_test_Energy.m, retains six rheology modes and produces active
solution counts [43, 41, 41] for forcing m=[0,-2,+2].  The earlier 125-mode
coefficient-path result is intentionally *not* a target because it mixed SH
coefficient conventions before the nonlinear Maxwell grid transform.
"""

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_variable,
)
from pylov3d.rheology import get_rheology, process_lateral_variations

from scripts.io_rheology_spectrum_diagnostic import (
    TARGET_ACTIVE_COUNTS,
    TARGET_RETAINED_RHEOLOGY_MODES,
    matlab_work_spectrum,
)


def _counts(variations, forcings, order):
    return [
        len(
            get_couplings(
                variations,
                f.n,
                f.m,
                perturbation_order=order,
            ).n_s
        )
        for f in forcings
    ]


def _case():
    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(10)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)
    mu_variable, eta_variable, _ = io_mu_eta_variable()
    return model, forcings, numerics, mu_variable, eta_variable


def test_matlab_work_grid_reproduces_raw_grid_anchor():
    """Pin the independently established raw-grid MATLAB target."""
    model, forcings, numerics, mu_variable, eta_variable = _case()
    _, kept, variations = matlab_work_spectrum(
        model, mu_variable, eta_variable, cutoff=numerics.rheology_cutoff
    )
    assert len(kept) == TARGET_RETAINED_RHEOLOGY_MODES
    assert _counts(variations, forcings, numerics.perturbation_order) == TARGET_ACTIVE_COUNTS


def test_general_processor_matches_raw_grid_anchor():
    """Promotion gate: the production processor must reproduce MATLAB."""
    model, forcings, numerics, mu_variable, eta_variable = _case()
    _, lateral = process_lateral_variations(
        model,
        forcings,
        mu_variable=mu_variable,
        eta_variable=eta_variable,
        rheology_cutoff=numerics.rheology_cutoff,
    )
    assert len(lateral.variations) == TARGET_RETAINED_RHEOLOGY_MODES
    assert _counts(lateral.variations, forcings, numerics.perturbation_order) == TARGET_ACTIVE_COUNTS
