# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Slow, publication-facing end-to-end TASK-046 raw-grid MATLAB gate."""

import numpy as np
import pytest

from pylov3d.io_lateral import build_io_forcings, build_io_model, io_default_numerics
from scripts.io_energy_gate_bc_multibasis import (
    MATLAB,
    TARGET_MODE_COUNTS,
    forcing_mode_k,
    love_energy_estimate,
    matlab_lateral_k_ref,
    monopole_direct_energy,
    relerr,
    solve_uniform,
)
from scripts.io_energy_gate_c_raw_grid import solve_lateral_raw_grid


@pytest.mark.slow
def test_io_raw_grid_gate_c_matches_native_matlab_end_to_end():
    """Raw maps -> Maxwell SH coefficients -> coupled k -> energy, Nrbase=50."""

    raw_model = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(50)

    model_u, num_u, uniform = solve_uniform(raw_model, forcings, numerics)
    model_l, num_l, lateral_rheology, couplings, lateral = solve_lateral_raw_grid(
        raw_model, forcings, numerics
    )

    k_u = [forcing_mode_k(r[3], f) for r, f in zip(uniform, forcings)]
    k_l = [forcing_mode_k(r[3], f) for r, f in zip(lateral, forcings)]
    mode_counts = [len(c.n_s) for c in couplings]

    uniform_n_s = [np.asarray([f.n], dtype=int) for f in forcings]
    uniform_m_s = [np.asarray([f.m], dtype=int) for f in forcings]
    e_direct_u = monopole_direct_energy(
        uniform, forcings, model_u, num_u, uniform_n_s, uniform_m_s
    )
    e_direct_l = monopole_direct_energy(
        lateral,
        forcings,
        model_l,
        num_l,
        [c.n_s for c in couplings],
        [c.m_s for c in couplings],
        couplings_list=couplings,
        lateral=lateral_rheology,
    )

    prefactor = 5.0 / float(model_u.Gg)
    e_love_u = prefactor * love_energy_estimate([r[3] for r in uniform], forcings)
    e_love_l = prefactor * love_energy_estimate([r[3] for r in lateral], forcings)

    assert mode_counts == TARGET_MODE_COUNTS
    assert all(relerr(x, MATLAB["k_uni"]) < 1e-7 for x in k_u)
    for forcing, value in zip(forcings, k_l):
        assert relerr(value, matlab_lateral_k_ref(forcing.m)) < 1e-7
    assert relerr(e_direct_u, MATLAB["e_direct_uni"]) < 1e-7
    assert relerr(e_direct_l, MATLAB["e_direct_lat"]) < 1e-7
    assert relerr(e_love_u, MATLAB["e_love_uni"]) < 1e-7
    assert relerr(e_love_l, MATLAB["e_love_lat"]) < 1e-7

    # This is a separate inherited numerical/method question, not a port gate.
    mismatch_u = abs(e_direct_u - e_love_u) / abs(e_love_u)
    mismatch_l = abs(e_direct_l - e_love_l) / abs(e_love_l)
    assert 0.015 < mismatch_u < 0.03
    assert 0.015 < mismatch_l < 0.03
