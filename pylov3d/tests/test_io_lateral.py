# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.io_lateral (TASK-046).

Covers: the MATLAB ``legendre``/``scipy.special.lpmv`` convention
equivalence (the load-bearing check for the whole heating-pattern port);
an independently hand-computed pinned ``z`` value at a specific grid point;
field-normalization sanity (``dmu``/``deta`` vanish where the melt-fraction
anomaly is zero; degree-2 dominance, degree-4 secondary, in the reporting
basis); the basis-convention finding documented in
``pylov3d/io_lateral.py``'s module docstring (``mars_lateral``'s converter
is demonstrably the *wrong* basis for ``process_lateral_variations``'s
viscoelastic branch, and the ``_sh_analysis``/``_sh_synthesis`` pair this
module actually uses is an exact round trip); the Io model-builder
constants against the spec table; and a cheap end-to-end Gate-A smoke
solve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import lpmv

from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    IO_ETA0,
    IO_FORCING_COMPONENTS,
    IO_FORCING_TD,
    IO_KS0,
    IO_MU0,
    IO_OMEGA0,
    IO_R0_KM,
    IO_RHO0,
    _io_dmu_deta,
    _io_z_pattern,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_heating_grid,
    io_matlab_p2_legendre,
    io_mu_eta_variable,
    io_pattern_lateral_fields,
)
from pylov3d.mapping import fully_normalized_legendre
from pylov3d.mars_lateral import _real_sh_to_complex_mu_variable
from pylov3d.rheology import _sh_analysis, _sh_grid, _sh_synthesis


# ---------------------------------------------------------------------------
# Pattern-convention: MATLAB legendre(2, .) == scipy.special.lpmv(., 2, .)
# ---------------------------------------------------------------------------

class TestLegendreConvention:
    """Catches a Condon-Shortley phase mismatch, the single most likely way
    to silently mis-port the MATLAB heating pattern (a sign flip on odd
    orders that a peak-to-peak-only check would miss)."""

    def test_p21_matches_hand_formula(self):
        """P_2^1(0.5) with Condon-Shortley: -3*x*sqrt(1-x^2)."""
        x = 0.5
        expected = -3.0 * x * math.sqrt(1.0 - x ** 2)
        assert expected == pytest.approx(-1.299038105676658, rel=1e-13)
        assert lpmv(1, 2, x) == pytest.approx(expected, rel=1e-13)

    def test_p20_p22_hand_formulas(self):
        x = 0.3
        assert lpmv(0, 2, x) == pytest.approx(0.5 * (3 * x ** 2 - 1))
        assert lpmv(2, 2, x) == pytest.approx(3.0 * (1.0 - x ** 2))

    def test_io_matlab_p2_legendre_matches_lpmv_directly(self):
        colat = np.array([0.1, 0.7, 1.3, 2.5, 3.0])
        P = io_matlab_p2_legendre(colat)
        x = np.cos(colat)
        for m in (0, 1, 2):
            np.testing.assert_allclose(P[m], lpmv(m, 2, x))

    def test_condon_shortley_present_not_absent(self):
        """Distinguishes MATLAB's convention from the no-CS
        fully_normalized_legendre convention this module explicitly does
        NOT use here: at m=1 the two differ by an exact sign flip, which a
        peak-to-peak check would not catch."""
        x = 0.5
        cs = lpmv(1, 2, x)  # Condon-Shortley (MATLAB / io_matlab_p2_legendre)
        no_cs_style = -cs  # what a no-CS convention would give at odd m
        assert cs < 0.0
        assert no_cs_style > 0.0


# ---------------------------------------------------------------------------
# Heating-grid replication + a hand-computed pinned value
# ---------------------------------------------------------------------------

class TestHeatingGrid:

    def test_grid_spacing_matches_l_max_formula(self):
        l_max = 100
        lat, lon, z = io_heating_grid(l_max=l_max)
        delta = 180.0 / (2.0 * (2 * l_max - 1))
        assert lon[1] - lon[0] == pytest.approx(delta, rel=1e-12)
        assert lat[1] - lat[0] == pytest.approx(delta, rel=1e-12)
        assert lon.min() == pytest.approx(-180.0 + delta / 2.0, rel=1e-12)
        assert lon.max() == pytest.approx(180.0 - delta / 2.0, rel=1e-12)
        assert lat.min() == pytest.approx(-90.0 + delta / 2.0, rel=1e-12)
        # MATLAB's latM reuses delta_lon (numerically == delta_lat here).
        assert lat.max() == pytest.approx(90.0 - delta / 2.0, rel=1e-12)

    def test_z_area_weighted_mean_is_one(self):
        """Only zero-mean degree-2 terms enter gv, so the true spherical
        mean of z must equal 1 (a naive unweighted grid average would not,
        since equal-angle cells are not equal-area -- this check uses a
        cos(lat)-weighted average, the correct spherical one)."""
        lat, lon, z = io_heating_grid(l_max=100)
        w = np.cos(np.radians(lat))
        mean = float(np.average(np.average(z, axis=1), weights=w))
        assert mean == pytest.approx(1.0, abs=2e-3)

    def test_z_pinned_value_at_equator_prime_meridian(self):
        """Independent hand computation at colat=90deg (equator), lon=0.

        cos(colat) = 0 => P_2^0(0) = -1/2, P_2^2(0) = 3*(1-0) = 3.
        gv = (-33/7)*(-1/2) + (9/14)*3*cos(0) = 33/14 + 27/14 = 60/14 = 30/7.
        z = (21/5 + 0.5*30/7) / (21/5).
        """
        colat = np.array([math.pi / 2.0])
        lon = np.array([0.0])
        z = _io_z_pattern(colat, lon)
        gv_expected = (-33.0 / 7.0) * (-0.5) + (9.0 / 14.0) * 3.0 * 1.0
        assert gv_expected == pytest.approx(30.0 / 7.0, rel=1e-13)
        z_expected = (21.0 / 5.0 + 0.5 * gv_expected) / (21.0 / 5.0)
        assert z_expected == pytest.approx(1.5102040816326532, rel=1e-13)
        assert z[0] == pytest.approx(z_expected, rel=1e-13)

    def test_z_pinned_value_at_pole(self):
        """At the pole (colat=0): cos=1, P_2^0(1)=1, P_2^2(1)=0 (no lon dep)."""
        colat = np.array([0.0])
        lon = np.array([1.2345])  # irrelevant at the pole (P_2^2=0 there)
        z = _io_z_pattern(colat, lon)
        gv_expected = (-33.0 / 7.0) * 1.0
        z_expected = (21.0 / 5.0 + 0.5 * gv_expected) / (21.0 / 5.0)
        assert z[0] == pytest.approx(z_expected, rel=1e-13)


# ---------------------------------------------------------------------------
# Field normalization
# ---------------------------------------------------------------------------

class TestFieldNormalization:

    def test_fields_zero_at_z_equals_one(self):
        """z=1 => Q_diff=0 => Phi_diff=0 => eta_rel=mu_rel=1 => dmu=deta=0."""
        z = np.array([1.0, 1.0])
        dmu, deta = _io_dmu_deta(z)
        np.testing.assert_allclose(dmu, 0.0, atol=1e-15)
        np.testing.assert_allclose(deta, 0.0, atol=1e-15)

    def test_degree2_dominance_degree4_secondary(self):
        """Reporting-basis (real, 4pi-normalized) degree spectrum: degree 2
        must dominate and degree 4 must be the next-largest (both fields),
        matching the spec's expectation from the nonlinear Q -> Phi ->
        (eta_rel, mu_rel) maps of an underlying pure-degree-2 pattern."""
        fields = io_pattern_lateral_fields(lmax_sh=8, l_max_report=40)
        for spectrum in (fields["mu_degree_spectrum"], fields["eta_degree_spectrum"]):
            assert max(spectrum, key=spectrum.get) == 2
            even = {n: v for n, v in spectrum.items() if n % 2 == 0 and n > 0}
            ranked = sorted(even, key=lambda n: -even[n])
            assert ranked[:2] == [2, 4]
            # Odd degrees are floating-point noise relative to degree 2.
            for n, v in spectrum.items():
                if n % 2 == 1:
                    assert v < 1e-6 * spectrum[2]

    def test_reporting_basis_ratio_matches_solver_basis_ratio(self):
        """Parseval cross-check: the real-4pi-normalized reporting spectrum
        and the sph_harm_y solver-facing entries are different bases for
        the same field, but the degree-4/degree-2 POWER RATIO must agree
        between them (this is what makes the reporting spectrum trustworthy
        even though it is not the basis actually fed to the solver)."""
        fields = io_pattern_lateral_fields(lmax_sh=8, l_max_report=40)
        report_ratio = fields["mu_degree_spectrum"][4] ** 2 / fields["mu_degree_spectrum"][2] ** 2

        entries = fields["mu_variable_entries"]
        power2 = sum(abs(a) ** 2 for n, m, a in entries if n == 2)
        power4 = sum(abs(a) ** 2 for n, m, a in entries if n == 4)
        solver_ratio = power4 / power2
        assert solver_ratio == pytest.approx(report_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# Basis-convention finding (see pylov3d/io_lateral.py module docstring)
# ---------------------------------------------------------------------------

class TestBasisConvention:

    def test_mars_lateral_converter_is_wrong_basis_for_sh_synthesis(self):
        """`_real_sh_to_complex_mu_variable` targets `complex_sh_synthesis`'s
        (fully_normalized_legendre) basis, NOT the sph_harm_y basis
        `process_lateral_variations` actually uses for viscoelastic layers
        (`rheology._sh_synthesis`). Demonstrates the mismatch is a genuine,
        constant, derivable rescaling (1/sqrt(4*pi) at m=0) -- not simply a
        relabelling of the same basis -- so nobody re-introduces this
        converter into the viscoelastic path by analogy with the
        (correct, for elastic layers) Mars/Moon crust code."""
        entries = _real_sh_to_complex_mu_variable({(2, 0): 1.0})
        theta, phi, _w = _sh_grid(4)
        field = _sh_synthesis(entries, theta, phi)
        P = fully_normalized_legendre(2, np.cos(theta))
        target = np.tile(P[2, 0, :, None], (1, len(phi)))
        ratio = (field.real - 1.0) / target
        assert np.allclose(ratio, 1.0 / math.sqrt(4.0 * math.pi), rtol=1e-9)
        assert not np.allclose(ratio, 1.0, rtol=1e-6)

    def test_sh_analysis_synthesis_round_trip_reproduces_mean_removed_field(self):
        """The actual converter this module uses (`_sh_analysis`, the exact
        inverse of `_sh_synthesis`) reproduces the target dmu(theta,phi)
        field -- MINUS its own spherical mean -- to quadrature truncation.

        The mean subtraction is required, not a test-only convenience:
        `process_lateral_variations._unify_modes` unconditionally strips
        any (0,0) entry from mu_variable/eta_variable ("the mean is
        handled separately" -- the layer's own mu0/eta0 IS the mean
        reference), so degree-0 content in the target field can never be
        communicated through this API regardless of what this module
        feeds it. See `test_nonzero_field_mean_is_a_documented_api_gap`
        below for why that residual is small but non-negligible here.
        """
        theta, phi, weights = _sh_grid(6)
        z = _io_z_pattern(theta[:, None], phi[None, :])
        dmu, _deta = _io_dmu_deta(z)
        mu_sh = _sh_analysis(dmu.astype(complex), theta, phi, weights, 6)
        entries = [(n, m, amp) for (n, m), amp in mu_sh.items() if n > 0 and abs(amp) > 0.0]
        resynth = _sh_synthesis(entries, theta, phi) - 1.0
        gl_weights_2d = weights[:, None] * np.ones((1, len(phi)))
        dmu_mean = np.sum(dmu * gl_weights_2d) / np.sum(gl_weights_2d)
        # atol accounts for the expected degree>6 truncation leakage (the
        # field is not exactly band-limited at lmax=6 -- see this module's
        # degree spectrum: degree-8 power is already ~1e-7 relative to
        # degree-2), not loosened to hide a real discrepancy.
        np.testing.assert_allclose(resynth.real, dmu - dmu_mean, atol=1e-6)
        np.testing.assert_allclose(resynth.imag, 0.0, atol=1e-8)

    def test_nonzero_field_mean_is_a_documented_api_gap(self):
        """Quantifies the mean-offset gap the round-trip test above works
        around: because mu_rel/eta_rel are NONLINEAR (rational) functions
        of the underlying zero-mean degree-2 pattern, dmu/deta's own
        spherical means are not exactly zero, even though the pattern
        they are built from is. `process_lateral_variations` cannot
        represent that offset (previous test's docstring), so it is
        silently dropped from the solver-facing field. Pinned here at the
        percent level (of the degree-2 amplitude) so a future change to
        the pattern constants is caught if it makes this gap large enough
        to matter; it is currently small (a few percent of the dominant
        degree-2 term) and reported, not corrected, per the TASK-046 spec's
        instruction not to silently paper over normalization gaps."""
        fields = io_pattern_lateral_fields(lmax_sh=8, l_max_report=100)
        lat_deg = fields["lat_deg"]
        w = np.cos(np.radians(lat_deg))
        mean_dmu = float(np.average(np.average(fields["dmu_grid"], axis=1), weights=w))
        mean_deta = float(np.average(np.average(fields["deta_grid"], axis=1), weights=w))
        deg2_mu = fields["mu_degree_spectrum"][2]
        deg2_eta = fields["eta_degree_spectrum"][2]
        assert 0.0 < abs(mean_dmu) / deg2_mu < 0.05
        assert 0.0 < abs(mean_deta) / deg2_eta < 0.10


# ---------------------------------------------------------------------------
# Io model-builder constants vs the spec table
# ---------------------------------------------------------------------------

class TestIoModelConstants:

    def test_radii_densities_moduli_viscosities(self):
        assert IO_R0_KM == (965.0, 1591.6, 1791.6, 1821.6)
        assert IO_RHO0 == (5150.0, 3244.0, 3244.0, 3244.0)
        assert IO_MU0 == (0.0, 6e10, 7.8e5, 6.5e10)
        # The MATLAB Consistency_test_Energy.m value, NOT the
        # pylov3d/tests/test_energy.py fixture's 200e16 -- see the audit in
        # scripts/io_energy_consistency.py's module docstring.
        assert IO_KS0 == (0.0, 200e12, 200e12, 200e12)
        assert IO_ETA0 == (None, 1e20, 1e11, 1e23)

    def test_forcing_period(self):
        assert IO_FORCING_TD == pytest.approx(2.0 * math.pi / IO_OMEGA0)
        assert IO_OMEGA0 == pytest.approx(4.1086e-05)

    def test_forcing_components(self):
        expected = (
            (2, 0, 3.0 / 4.0 * math.sqrt(1.0 / 5.0)),
            (2, -2, -7.0 / 8.0 * math.sqrt(6.0 / 5.0)),
            (2, 2, 1.0 / 8.0 * math.sqrt(6.0 / 5.0)),
        )
        for (n, m, F), (en, em, eF) in zip(IO_FORCING_COMPONENTS, expected):
            assert (n, m) == (en, em)
            assert F == pytest.approx(eF, rel=1e-14)

    def test_build_io_model_shape_and_values(self):
        model = build_io_model()
        assert model.n_layers == 4
        np.testing.assert_allclose(np.asarray(model.R0[:4]), IO_R0_KM)
        np.testing.assert_allclose(np.asarray(model.rho0[:4]), IO_RHO0)
        np.testing.assert_allclose(np.asarray(model.mu0[:4]), IO_MU0)
        np.testing.assert_allclose(np.asarray(model.Ks0[:4]), IO_KS0)
        # Core Delta_rho0 auto-fill matches the MATLAB single assignment
        # (rheology.normalize only ever reads index 0 -- see build_io_model
        # docstring).
        assert float(model.Delta_rho0[0]) == pytest.approx(IO_RHO0[0] - IO_RHO0[1])

    def test_build_io_forcings_uses_shared_td(self):
        forcings = build_io_forcings()
        assert len(forcings) == 3
        assert [f.n for f in forcings] == [2, 2, 2]
        assert [f.m for f in forcings] == [0, -2, 2]
        for f in forcings:
            assert f.Td == pytest.approx(IO_FORCING_TD)


# ---------------------------------------------------------------------------
# Gate-A smoke test (cheap end-to-end solve at Nrbase=5)
# ---------------------------------------------------------------------------

class TestGateASmoke:
    """Not marked slow (measured ~2-3 s: builds the full lateral field at
    lmax_sh=8 and solves both the uniform and lateral (2,0) forcing at
    Nrbase=5, well under the spec's 30 s threshold). Catches any
    regression that would break the whole Gate-A pipeline end to end
    (finite spectra, non-forcing modes excited, nonzero dissipative
    forcing-mode Im(k)) -- not a numerical-value pin, since Nrbase=5 is
    far from radially converged (see scripts/io_energy_consistency.py's
    Gate B findings)."""

    def test_uniform_and_lateral_solve_finite_and_excite_modes(self):
        from pylov3d.couplings import get_couplings
        from pylov3d.grid import set_boundary_indices
        from pylov3d.love import extract_love_numbers
        from pylov3d.rheology import get_rheology, process_lateral_variations
        from pylov3d.solver import get_solution

        model_raw = build_io_model()
        forcing = build_io_forcings()[0]  # (2, 0) only, for speed
        numerics = io_default_numerics(5)
        mu_variable, eta_variable, _ = io_mu_eta_variable(lmax_sh=8)

        # Uniform
        numerics_u, model_u = set_boundary_indices(numerics, model_raw)
        model_u = get_rheology(model_u, forcing)
        y_uni, r_uni, _Y, Aprop_uni = get_solution(model_u, forcing, numerics_u)
        assert np.all(np.isfinite(y_uni))
        love_uni = extract_love_numbers(y_uni, model_u, forcing)
        k_uni = complex(love_uni.k[0])
        assert math.isfinite(k_uni.real) and math.isfinite(k_uni.imag)
        assert k_uni.imag != 0.0
        assert k_uni.imag < 0.0  # dissipative sign convention (see driver script)

        # Lateral
        numerics_l, model_l = set_boundary_indices(numerics, model_raw)
        model_l = get_rheology(model_l, forcing)
        model_l, lateral = process_lateral_variations(
            model_l, forcing, mu_variable=mu_variable, eta_variable=eta_variable,
            rheology_cutoff=numerics_l.rheology_cutoff,
        )
        couplings = get_couplings(lateral.variations, 2, 0, perturbation_order=2)
        assert len(couplings.n_s) > 1  # non-forcing modes excited (Gate A2)
        y_lat, r_lat, _Y, Aprop_lat = get_solution(
            model_l, forcing, numerics_l, couplings=couplings, lateral=lateral,
        )
        assert np.all(np.isfinite(y_lat))
        love_lat = extract_love_numbers(y_lat, model_l, forcing, couplings=couplings)
        idx = np.where((love_lat.n == 2) & (love_lat.m == 0))[0][0]
        k_lat = complex(love_lat.k[idx])
        assert math.isfinite(k_lat.real) and math.isfinite(k_lat.imag)
        assert k_lat.imag < 0.0
