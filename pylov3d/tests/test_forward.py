# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.forward (TASK-012 body-agnostic MC framework).

(a) A TOY 3-layer body unrelated to Mars proves the generic machinery is
    body-agnostic: build_model round-trip, mass/moi analytics vs hand-coded
    shell-formula values, and likelihood(truth) > likelihood(perturbed).
(b) The real Mars instantiation (pylov3d.mars_mc): log-posterior at the
    TASK-011 point fit is finite and decreases under a perturbation of each
    free parameter individually.
(c) Out-of-bounds theta -> -inf, without ever invoking the (expensive)
    forward model.
(d) A MICRO pocomc smoke run (@pytest.mark.slow): the sampler completes and
    its posterior-median k2 is within 3 sigma of the observed 0.169.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pylov3d.forward import (
    BodyParameterization,
    Constraint,
    Free,
    LayerSpec,
    Scaled,
    analytic_mass_moi,
    build_model,
    compute_observables,
    log_likelihood,
    log_prior,
    make_log_posterior,
)
from pylov3d.mars_mc import (
    MARS_BOUNDS,
    MARS_FREE_PARAMS,
    MARS_PARAMETERIZATION,
    mars_forcing,
    mars_log_posterior,
    mars_numerics,
    mars_point_fit_theta,
)
from pylov3d.types import make_forcing, make_numerics

# ---------------------------------------------------------------------------
# (a) Toy 3-layer body (proves body-agnosticism: no Mars-specific code path)
# ---------------------------------------------------------------------------

_TOY_R_KM = (1000.0, 1800.0, 2000.0)  # core / mantle / crust outer radii
_TOY_MU_MANTLE_BASE = 8.0e10
_TOY_MU_CRUST_BASE = 4.0e10
_TOY_CRUST_RHO = 3000.0

_TOY_LAYERS = (
    LayerSpec(
        name="core", radius_km=_TOY_R_KM[0], rho=Free("rho_core"),
        mu=0.0, Ks=200e9, eta=None,
    ),
    LayerSpec(
        name="mantle", radius_km=_TOY_R_KM[1], rho=Free("rho_mantle"),
        mu=Scaled(_TOY_MU_MANTLE_BASE, "mu_scale"), Ks=150e9, eta=None,
    ),
    LayerSpec(
        name="crust", radius_km=_TOY_R_KM[2], rho=_TOY_CRUST_RHO,
        mu=Scaled(_TOY_MU_CRUST_BASE, "mu_scale"), Ks=100e9, eta=None,
    ),
)

_TOY_FREE_PARAMS = ("rho_core", "rho_mantle", "mu_scale")
_TOY_BOUNDS = {
    "rho_core": (6000.0, 10000.0),
    "rho_mantle": (2500.0, 5000.0),
    "mu_scale": (0.5, 2.0),
}

_TOY_PARAMETERIZATION = BodyParameterization(
    name="toy_body", layers=_TOY_LAYERS, free_params=_TOY_FREE_PARAMS, bounds=_TOY_BOUNDS,
)

_TOY_TRUTH = (8000.0, 3500.0, 1.0)  # rho_core, rho_mantle, mu_scale


def _toy_forcing():
    return make_forcing(Td=100000.0, n=2, m=0, F=1.0)


def _toy_numerics():
    return make_numerics(n_layers=3, method="combination", Nrbase=40)


def _hand_mass_moi(radii_km, rho):
    """Independent (not reusing pylov3d.forward.analytic_mass_moi) shell
    mass/MoI calculation, to catch a bug shared between the implementation
    and a naive re-check."""
    boundaries_m = [0.0] + [r * 1e3 for r in radii_km]
    M = 0.0
    I = 0.0
    for i, rho_i in enumerate(rho):
        r_in, r_out = boundaries_m[i], boundaries_m[i + 1]
        M += (4.0 * math.pi / 3.0) * rho_i * (r_out**3 - r_in**3)
        I += (8.0 * math.pi / 15.0) * rho_i * (r_out**5 - r_in**5)
    return M, I


class TestToyBodyRoundTrip:

    def test_build_model_resolves_fixed_free_and_scaled(self):
        model = build_model(_TOY_PARAMETERIZATION, _TOY_TRUTH)
        n = model.n_layers
        assert n == 3
        R0 = [float(x) for x in np.asarray(model.R0[:n])]
        rho0 = [float(x) for x in np.asarray(model.rho0[:n])]
        mu0 = [float(x) for x in np.asarray(model.mu0[:n])]
        assert R0 == pytest.approx(list(_TOY_R_KM))
        assert rho0 == pytest.approx([8000.0, 3500.0, _TOY_CRUST_RHO])
        # mu_scale=1.0 -> Scaled bases pass through unchanged; core mu fixed 0.
        assert mu0 == pytest.approx([0.0, _TOY_MU_MANTLE_BASE, _TOY_MU_CRUST_BASE])

    def test_scaled_parameter_actually_scales(self):
        theta = (8000.0, 3500.0, 1.5)  # mu_scale = 1.5
        model = build_model(_TOY_PARAMETERIZATION, theta)
        mu0 = [float(x) for x in np.asarray(model.mu0[:3])]
        assert mu0[1] == pytest.approx(1.5 * _TOY_MU_MANTLE_BASE)
        assert mu0[2] == pytest.approx(1.5 * _TOY_MU_CRUST_BASE)

    def test_wrong_theta_length_raises(self):
        with pytest.raises(ValueError):
            build_model(_TOY_PARAMETERIZATION, (8000.0, 3500.0))

    def test_mismatched_free_params_and_bounds_raises(self):
        with pytest.raises(ValueError):
            BodyParameterization(
                name="bad", layers=_TOY_LAYERS,
                free_params=_TOY_FREE_PARAMS, bounds={"rho_core": (0, 1)},
            )

    def test_core_radius_km_observable_is_cheap_and_correct(self):
        """S1 (review): core_radius_km is the generic, body-agnostic
        identifiability observable added for Mars's R_core (see
        pylov3d.mars_mc, "Identifiability") -- must equal model.R0[0]
        exactly and not require a solve (forcing/numerics are unused for
        this ``which``)."""
        model = build_model(_TOY_PARAMETERIZATION, _TOY_TRUTH)
        obs = compute_observables(
            model, _toy_forcing(), _toy_numerics(), which=("core_radius_km",),
        )
        assert obs["core_radius_km"] == pytest.approx(_TOY_R_KM[0])

    def test_mass_moi_match_hand_calculation(self):
        model = build_model(_TOY_PARAMETERIZATION, _TOY_TRUTH)
        M, I = analytic_mass_moi(model)
        rho = [8000.0, 3500.0, _TOY_CRUST_RHO]
        M_hand, I_hand = _hand_mass_moi(_TOY_R_KM, rho)
        assert M == pytest.approx(M_hand, rel=1e-12)
        assert I == pytest.approx(I_hand, rel=1e-12)
        # Sanity: mean density order of magnitude for a rocky/metal body.
        R_surface_m = _TOY_R_KM[-1] * 1e3
        mean_rho = M / (4.0 / 3.0 * math.pi * R_surface_m**3)
        assert 3000.0 < mean_rho < 8000.0


class TestBuildModelRadiusValidation:
    """D3 (review): a free radius must not be allowed to produce a zero- or
    negative-volume shell (e.g. a free core radius exceeding the next fixed
    layer's radius) -- previously this reached the solver silently and
    could score a finite posterior for an unphysical body."""

    _LAYERS = (
        LayerSpec(name="core", radius_km=Free("R_core"), rho=6000.0, mu=0.0, Ks=200e9, eta=None),
        LayerSpec(
            name="mantle", radius_km=1800.0, rho=3500.0,
            mu=Scaled(8.0e10, "mu_scale"), Ks=150e9, eta=None,
        ),
        LayerSpec(
            name="crust", radius_km=2000.0, rho=3000.0,
            mu=Scaled(4.0e10, "mu_scale"), Ks=100e9, eta=None,
        ),
    )
    _PARAMS = BodyParameterization(
        name="mono_radius_test", layers=_LAYERS, free_params=("R_core", "mu_scale"),
        bounds={"R_core": (500.0, 2500.0), "mu_scale": (0.5, 2.0)},
    )

    def test_core_radius_exceeding_next_layer_raises(self):
        # R_core=2500 km > the mantle's fixed 1800 km outer radius: a
        # negative-volume mantle shell. Well inside R_core's own prior box
        # (500, 2500), so only build_model's own validation can catch this.
        with pytest.raises(ValueError, match="strictly increasing"):
            build_model(self._PARAMS, (2500.0, 1.0))

    def test_valid_radii_do_not_raise(self):
        model = build_model(self._PARAMS, (1000.0, 1.0))
        assert list(np.asarray(model.R0[:3])) == pytest.approx([1000.0, 1800.0, 2000.0])


class TestComputeObservablesRejectsFixedMethod:
    """D4 (review): numerics.method="fixed" silently rewrites the solver's
    layer radii (pylov3d.grid.set_boundary_indices), decoupling the
    analytic mass/moi_mean/core_radius_km observables (read from the
    caller's original model.R0) from the Love numbers (computed on the
    solver's rewritten radii) -- compute_observables must refuse it
    outright rather than silently mixing two slightly different bodies."""

    def test_fixed_method_raises_even_for_analytic_only_which(self):
        model = build_model(_TOY_PARAMETERIZATION, _TOY_TRUTH)
        forcing = _toy_forcing()
        bad_numerics = make_numerics(n_layers=3, method="fixed", Nrbase=40)
        with pytest.raises(NotImplementedError, match="fixed"):
            compute_observables(model, forcing, bad_numerics, which=("mass",))

    def test_combination_method_is_unaffected(self):
        model = build_model(_TOY_PARAMETERIZATION, _TOY_TRUTH)
        obs = compute_observables(
            model, _toy_forcing(), _toy_numerics(), which=("mass",),
        )
        assert np.isfinite(obs["mass"])


@pytest.fixture(scope="module")
def toy_truth_observables():
    model = build_model(_TOY_PARAMETERIZATION, _TOY_TRUTH)
    return compute_observables(model, _toy_forcing(), _toy_numerics())


@pytest.fixture(scope="module")
def toy_constraints(toy_truth_observables):
    return (
        Constraint("mass", toy_truth_observables["mass"], 0.001 * toy_truth_observables["mass"]),
        Constraint("moi_mean", toy_truth_observables["moi_mean"], 0.001),
        Constraint("k2", toy_truth_observables["k2"], 0.01),
    )


class TestToyBodyLikelihood:
    """The toy body's own get_love solve defines "truth" constraints, then
    checks the posterior/likelihood machinery prefers truth over a
    perturbation — this exercises Stage 2 (Love numbers) and Stage 3
    (likelihood) together, still with zero Mars-specific code."""

    def test_likelihood_peaks_at_truth(self, toy_constraints):
        log_post = make_log_posterior(
            _TOY_PARAMETERIZATION, toy_constraints, _toy_forcing(), _toy_numerics(),
        )
        ll_truth = log_post(_TOY_TRUTH)
        assert np.isfinite(ll_truth)

        perturbed = (8000.0 + 400.0, 3500.0, 1.0)  # rho_core off by 400 kg/m^3
        ll_perturbed = log_post(perturbed)
        assert ll_perturbed < ll_truth

    def test_out_of_bounds_theta_is_minus_inf(self, toy_constraints):
        log_post = make_log_posterior(
            _TOY_PARAMETERIZATION, toy_constraints, _toy_forcing(), _toy_numerics(),
        )
        assert log_post((100000.0, 3500.0, 1.0)) == -np.inf  # rho_core way OOB

    def test_solver_exception_guard_increments_warn_count_not_raises(self, toy_constraints):
        """Directly exercise the exception guard in make_log_posterior:
        a mu_scale of 0 makes the mantle/crust perfectly fluid on top of an
        already-fluid core, and the analytic fluid-CMB boundary condition
        machinery is documented (pylov3d/mars.py) to assume exactly one
        fluid layer at index 0 -- more than one fluid layer is exactly the
        kind of ill-posed input this guard exists for."""
        bad_params = BodyParameterization(
            name="toy_body_bad", layers=_TOY_LAYERS,
            free_params=_TOY_FREE_PARAMS,
            bounds={"rho_core": (6000.0, 10000.0), "rho_mantle": (2500.0, 5000.0), "mu_scale": (0.0, 2.0)},
        )
        log_post = make_log_posterior(
            bad_params, toy_constraints, _toy_forcing(), _toy_numerics(),
        )
        assert log_post.warn_count["n"] == 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = log_post((8000.0, 3500.0, 0.0))
        assert result == -np.inf
        assert log_post.warn_count["n"] == 1  # the bad theta was actually counted


# ---------------------------------------------------------------------------
# (b) Mars instantiation: point fit is finite and (locally) near-maximal
# ---------------------------------------------------------------------------

class TestMarsPosterior:

    def test_point_fit_is_finite(self):
        log_post = mars_log_posterior(Nrbase=40)
        theta0 = mars_point_fit_theta()
        assert np.isfinite(log_post(theta0))

    @pytest.mark.parametrize(
        "param,delta",
        [("rho_core", 30.0), ("rho_lm", 40.0), ("mu_scale", 0.03), ("R_core", 15.0)],
    )
    def test_point_fit_is_locally_near_maximal(self, param, delta):
        """Perturbing any one free parameter away from the TASK-011 point
        fit, holding the others fixed, must not increase the posterior
        (checked in both directions)."""
        log_post = mars_log_posterior(Nrbase=40)
        theta0 = np.array(mars_point_fit_theta())
        base = log_post(theta0)
        idx = MARS_FREE_PARAMS.index(param)

        for sign in (+1.0, -1.0):
            theta = theta0.copy()
            theta[idx] += sign * delta
            assert log_post(theta) < base

    def test_out_of_bounds_is_minus_inf_without_calling_solver(self, monkeypatch):
        import pylov3d.forward as forward_mod

        def _boom(*args, **kwargs):
            raise AssertionError("compute_observables must not be called for out-of-bounds theta")

        monkeypatch.setattr(forward_mod, "compute_observables", _boom)
        log_post = mars_log_posterior(Nrbase=40)
        theta = np.array(mars_point_fit_theta())
        lo, hi = MARS_BOUNDS["R_core"]
        theta[MARS_FREE_PARAMS.index("R_core")] = hi + 100.0
        assert log_post(theta) == -np.inf


class TestLogPriorAndLikelihood:

    def test_log_prior_zero_inside_box(self):
        theta0 = mars_point_fit_theta()
        assert log_prior(theta0, MARS_PARAMETERIZATION) == 0.0

    def test_log_prior_minus_inf_outside_box(self):
        theta = list(mars_point_fit_theta())
        theta[0] = MARS_BOUNDS["rho_core"][1] + 1.0
        assert log_prior(theta, MARS_PARAMETERIZATION) == -np.inf

    def test_log_likelihood_zero_residual_is_gaussian_peak(self):
        constraints = (Constraint("k2", 0.169, 0.006),)
        ll_peak = log_likelihood({"k2": 0.169}, constraints)
        ll_off = log_likelihood({"k2": 0.169 + 0.006}, constraints)
        assert ll_peak > ll_off
        expected_peak = -math.log(0.006 * math.sqrt(2 * math.pi))
        assert ll_peak == pytest.approx(expected_peak)

    def test_log_likelihood_missing_observable_raises(self):
        constraints = (Constraint("k2", 0.169, 0.006),)
        with pytest.raises(KeyError):
            log_likelihood({"mass": 1.0}, constraints)


# ---------------------------------------------------------------------------
# (d) Micro pocomc smoke run
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPocoMCSmoke:

    def test_micro_run_completes_and_recovers_k2(self):
        pocomc = pytest.importorskip("pocomc")
        from scipy.stats import uniform

        from pylov3d.forward import build_model as _build

        np.random.seed(0)
        Nrbase = 15  # coarse grid: fast, k2 still accurate to ~1e-4 (see
        # pylov3d.mars "Discretization note" -- far inside the 0.006 sigma).
        log_post = mars_log_posterior(Nrbase=Nrbase)

        dists = [
            uniform(loc=MARS_BOUNDS[name][0], scale=MARS_BOUNDS[name][1] - MARS_BOUNDS[name][0])
            for name in MARS_FREE_PARAMS
        ]
        prior = pocomc.Prior(dists)

        sampler = pocomc.Sampler(
            prior, log_post, n_dim=len(MARS_FREE_PARAMS),
            n_active=8, n_effective=8, dynamic=False, random_state=0,
        )
        sampler.run(n_total=32, n_evidence=0, progress=False)

        samples, weights, _logl, _logp = sampler.posterior()
        assert samples.shape[0] > 0
        assert np.isfinite(samples).all()

        median = np.average(samples, axis=0, weights=weights)
        model = _build(MARS_PARAMETERIZATION, median)
        obs = compute_observables(model, mars_forcing(), mars_numerics(Nrbase=Nrbase), which=("k2",))
        assert abs(obs["k2"] - 0.169) < 3 * 0.006
