# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Pointwise connectivity mixing: exact reductions and tensor sanity.

PRELIMINARY science scope: this machinery supports the higher-degree
extension of the proposal's Figure-3 study, pending updated calculations
from Berne (2026-09-29 direction from Steve). The exact reductions and
bound orderings tested here are convention/bookkeeping guards, not final
science numbers.
"""

import numpy as np
import pytest

from pylov3d.extended_love import extended_love_tensor
from pylov3d.love import get_love
from pylov3d.mars import MARS_FORCING_TD
from pylov3d.mars_hydration import (
    RATIO_SCENARIOS,
    hydration_lateral_variables,
    mean_softened_crust_moduli,
)
from pylov3d.mars_hydration_connectivity import (
    MIXING_LAWS,
    mix_pointwise,
    pointwise_hydrated_crust,
)
from pylov3d.mars_lateral import CRUST_LAYER_INDEX
from pylov3d.types import make_forcing, make_numerics

CENTRAL = RATIO_SCENARIOS["central"]


class TestExactReductions:

    def test_voigt_mean_matches_mean_only_pipeline(self):
        for f_h in (0.1, 0.3, 0.5):
            p = pointwise_hydrated_crust(f_h, *CENTRAL, "voigt",
                                         lmax_field=2, lmax_out=2)
            mu0, Ks0 = mean_softened_crust_moduli(f_h, *CENTRAL)
            assert p.mu_bar == pytest.approx(mu0, rel=1e-13)
            assert p.Ks_bar == pytest.approx(Ks0, rel=1e-13)
            assert p.clip_fraction == 0.0

    def test_voigt_lateral_matches_linear_pipeline(self):
        p = pointwise_hydrated_crust(0.3, *CENTRAL, "voigt",
                                     lmax_field=2, lmax_out=2)
        mv, Kv = hydration_lateral_variables(0.3, *CENTRAL, lmax=2)
        for mine, ref in ((p.mu_variable, mv), (p.K_variable, Kv)):
            a = dict(((n, m), v) for n, m, v in mine[CRUST_LAYER_INDEX])
            b = dict(((n, m), v) for n, m, v in ref[CRUST_LAYER_INDEX])
            scale = max(abs(v) for v in b.values())
            for k in set(a) | set(b):
                assert abs(a.get(k, 0) - b.get(k, 0)) <= 1e-13 * scale, k

    def test_f_h_zero_is_dry_model(self):
        p = pointwise_hydrated_crust(0.0, *CENTRAL, "reuss")
        assert p.mu_variable == {} and p.K_variable == {}
        assert p.clip_fraction == 0.0

    def test_bound_ordering_reuss_le_hill_le_voigt(self):
        f = np.linspace(0.0, 1.0, 11)
        v = mix_pointwise(f, 30e9, 10e9, "voigt")
        h = mix_pointwise(f, 30e9, 10e9, "hill")
        r = mix_pointwise(f, 30e9, 10e9, "reuss")
        assert np.all(r <= h + 1e-6) and np.all(h <= v + 1e-6)
        for law in MIXING_LAWS:
            p = pointwise_hydrated_crust(0.5, *CENTRAL, law,
                                         lmax_field=2, lmax_out=2)
            assert 0 < p.mu_bar < 30e9

    def test_nonlinear_law_shifts_mean_above_mean_only(self):
        """Jensen: Reuss (1/(linear in f)) and Hill are convex in f, so the
        area mean of the pointwise-mixed field lies ABOVE mixing at the
        mean fraction — mean-only Reuss/Hill slightly overstate the mean
        softening. This is the sense in which the pointwise experiment
        genuinely differs from the Figure-3 mean-only one."""
        f_h = 0.5
        for law in ("hill", "reuss"):
            p = pointwise_hydrated_crust(f_h, *CENTRAL, law,
                                         lmax_field=2, lmax_out=2)
            from scripts.mars_serpentinite_connectivity_sensitivity import mixed_moduli
            mu_mean_only_law, _ = mixed_moduli(f_h, *CENTRAL, law)
            assert p.mu_bar > mu_mean_only_law


@pytest.mark.slow
def test_higher_degree_tensor_sanity():
    """One coupled solve: k_2m diagonals bracket the 1D value and the
    degree-3 response is nonzero for the nonlinear law."""
    p = pointwise_hydrated_crust(0.5, *CENTRAL, "reuss",
                                 lmax_field=2, lmax_out=2)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=30,
                             perturbation_order=2)
    love, _, _ = get_love(p.model, make_forcing(MARS_FORCING_TD, 2, 0, 1.0),
                          numerics)
    k2_1d = complex(love.k[0])
    T = extended_love_tensor(p.model, numerics, MARS_FORCING_TD,
                             mu_variable=p.mu_variable,
                             K_variable=p.K_variable)
    diag = {mp: T.entry(2, mp, 2, mp) for mp in range(-2, 3)}
    for mp, k in diag.items():
        assert abs(k.imag) < 1e-9  # elastic, real structure
        assert abs(k - k2_1d) < 0.05 * abs(k2_1d)  # splitting is a perturbation
    # order-dependent splitting exists and degree-3 modes respond
    split = {abs(mp): abs(diag[mp] - k2_1d) for mp in (0, 1, 2)}
    assert len({round(v, 12) for v in split.values()}) > 1
    l3 = max(abs(T.entry(3, m, 2, 0)) for m in range(-3, 4))
    assert l3 > 0.0
