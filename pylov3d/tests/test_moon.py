# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for pylov3d.moon / pylov3d.moon_mc — TASK-018 Moon reference model
and Monte Carlo parameterization.

Mirrors ``pylov3d/tests/test_mars.py``'s structure (constraint-residual
checks, Love-number sanity, density profile checks) plus Moon-specific
pieces: a byte-for-byte cross-check against the MATLAB-validated ocean
harness (``test_matlab_validation_ocean.py``), the as-built residual table
from ``pylov3d.moon`` (non-zero residuals matching documented values, NOT
an exact fit), the ``pylov3d.moon_mc`` parameterization/identifiability
argument, a D8 framework-fix regression, and a micro pocomc smoke run
(mirroring ``test_forward.py::TestPocoMCSmoke`` for Mars).
"""

from __future__ import annotations

import numpy as np
import pytest

from pylov3d.forward import build_model, compute_observables, log_likelihood, log_prior
from pylov3d.love import get_love
from pylov3d.moon import (
    LAYER_KS,
    LAYER_MU,
    LAYER_NAMES,
    LAYER_OCEAN,
    LAYER_RADII_KM,
    LAYER_RHO,
    MOON,
    WEBER_K2_UNIFORM,
    build_moon_model,
    moon_forcing,
    moon_mass,
    moon_moi_factor,
    moon_numerics,
)
from pylov3d.moon_mc import (
    MOON_BOUNDS,
    MOON_CONSTRAINTS,
    MOON_CORE_LAYER_INDEX,
    MOON_FREE_PARAMS,
    MOON_PARAMETERIZATION,
    moon_log_posterior,
    moon_point_estimate_theta,
)

# Pinned as-built residuals (see pylov3d.moon docstring, "As-built
# residuals" table); the Weber profile is NOT fit to these constraints, so
# these must be non-zero, not exactly the constraint values.
_EXPECTED_MASS_RESID_REL = -0.004424048553358782
_EXPECTED_MOI_RESID = -0.0007388448282794613
_EXPECTED_K2_RESID = -0.0010608577714824388
_EXPECTED_CORE_RADIUS_RESID_KM = -50.0

# Pinned 4-parameter MAP (scipy L-BFGS-B, several starting points, best of;
# see moon_mc module docstring, "Mass-closure parameter"; reproducibility
# regression, not a fit at import time). core_rho_scale rails at its
# physical density floor (0.88) -- NOT zero-residual: moi_mean is off by
# ~-0.95 sigma, core_radius_km by ~-1.33 sigma at this theta.
_MAP_4PARAM_THETA = (
    0.88, 0.9654755646373745, 326.8893505613952, 1.0063816825495973,
)
_MAP_4PARAM_NEG_LOGPOST = 37.4977448076148


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def moon_model():
    return build_moon_model()


@pytest.fixture(scope="module")
def moon_love(moon_model):
    love, y_rad, model = get_love(moon_model, moon_forcing(), moon_numerics())
    return love, y_rad, model


# ---------------------------------------------------------------------------
# 1. Reference model matches the MATLAB-validated ocean harness exactly
# ---------------------------------------------------------------------------

class TestReferenceModelMatchesOceanHarness:
    """pylov3d.moon._load_weber_profile deliberately re-derives (does not
    import) test_matlab_validation_ocean.py's _build_weber_moon_model (see
    pylov3d.moon docstring, "Why re-derive"). This is the executable
    guarantee against the two drifting apart."""

    def test_reference_model_matches_ocean_harness(self, moon_model):
        from pylov3d.tests.test_matlab_validation_ocean import (
            _build_weber_moon_model,
        )

        ref = _build_weber_moon_model()
        n = moon_model.n_layers
        assert n == ref.n_layers == 10
        for name in ("R0", "rho0", "mu0", "Ks0", "ocean"):
            a = np.asarray(getattr(moon_model, name)[:n], dtype=float)
            b = np.asarray(getattr(ref, name)[:n], dtype=float)
            assert np.array_equal(a, b), f"{name} differs from the ocean harness"

    def test_layer_arrays_match_model(self, moon_model):
        """pylov3d.moon's exported LAYER_* tuples (reused by moon_mc) are
        exactly the model's own arrays, not independently re-typed."""
        n = moon_model.n_layers
        assert LAYER_RADII_KM == tuple(float(x) for x in np.asarray(moon_model.R0[:n]))
        assert LAYER_RHO == tuple(float(x) for x in np.asarray(moon_model.rho0[:n]))
        assert LAYER_MU == tuple(float(x) for x in np.asarray(moon_model.mu0[:n]))
        assert LAYER_KS == tuple(float(x) for x in np.asarray(moon_model.Ks0[:n]))
        assert LAYER_OCEAN == tuple(int(x) for x in np.asarray(moon_model.ocean[:n]))
        assert len(LAYER_NAMES) == n

    def test_exactly_one_fluid_layer_at_index_2(self, moon_model):
        n = moon_model.n_layers
        ocean_flags = [int(x) for x in np.asarray(moon_model.ocean[:n])]
        assert sum(ocean_flags) == 1
        assert ocean_flags[2] == 1
        assert float(moon_model.mu0[2]) == 0.0


# ---------------------------------------------------------------------------
# 2. k2 anchor tie-in: uniform k2 matches the MATLAB reference to 1e-9
# ---------------------------------------------------------------------------

class TestK2AnchorTieIn:

    def test_uniform_k2_matches_matlab_reference(self):
        from pylov3d.tests.test_matlab_validation_ocean import _load_qin_reference

        ref = _load_qin_reference("moon_3D_UM_20")
        assert abs(WEBER_K2_UNIFORM - ref["k2_uniform"]) < 1e-9

    def test_uniform_k2_pinned_value(self):
        assert round(WEBER_K2_UNIFORM, 9) == 0.023159142

    def test_uniform_k2_reproduced_live(self, moon_love):
        """WEBER_K2_UNIFORM is hardcoded provenance (see pylov3d.moon
        docstring); this re-runs get_love live and checks it still matches."""
        love, _, _ = moon_love
        k2 = complex(love.k[0])
        assert abs(k2.imag) < 1e-10
        assert k2.real == pytest.approx(WEBER_K2_UNIFORM, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. As-built residuals vs the published constraints (NOT a fit — assert
#    the documented non-zero residuals, per TASK-018 spec)
# ---------------------------------------------------------------------------

class TestAsBuiltResiduals:

    def test_mass_residual(self, moon_model):
        M = moon_mass(moon_model)
        target = MOON["M"]
        resid_rel = (M - target) / target
        assert resid_rel == pytest.approx(_EXPECTED_MASS_RESID_REL, rel=1e-6)
        assert resid_rel != 0.0

    def test_moi_residual(self, moon_model):
        moi = moon_moi_factor(moon_model)
        resid = moi - MOON["MoI_factor"]
        assert resid == pytest.approx(_EXPECTED_MOI_RESID, rel=1e-6)
        assert resid != 0.0

    def test_k2_residual(self, moon_love):
        love, _, _ = moon_love
        k2 = complex(love.k[0]).real
        resid = k2 - MOON["k2"]
        assert resid == pytest.approx(_EXPECTED_K2_RESID, rel=1e-6)
        assert resid != 0.0

    def test_core_radius_residual(self, moon_model):
        core_radius_km = float(np.asarray(moon_model.R0[2]))
        target_km = MOON["fluid_core_radius"] / 1e3
        resid = core_radius_km - target_km
        assert resid == pytest.approx(_EXPECTED_CORE_RADIUS_RESID_KM, rel=1e-6)
        assert resid != 0.0
        # As-built core radius is exactly Weber et al. (2011)'s own seismic
        # value, not an independent number.
        assert core_radius_km == pytest.approx(MOON["fluid_core_radius_weber_seismic"] / 1e3)


# ---------------------------------------------------------------------------
# 4. Love-number sanity + density profile
# ---------------------------------------------------------------------------

class TestLoveNumberSanity:

    def test_k2_is_real(self, moon_love):
        love, _, _ = moon_love
        assert abs(complex(love.k[0]).imag) < 1e-10

    def test_h2_positive_and_greater_than_k2(self, moon_love):
        love, _, _ = moon_love
        h2 = complex(love.h[0]).real
        k2 = complex(love.k[0]).real
        assert h2 > 0
        assert h2 > k2


class TestDensityProfile:

    def test_monotonic_non_increasing_outward(self, moon_model):
        n = moon_model.n_layers
        rho = [float(x) for x in np.asarray(moon_model.rho0[:n])]
        # Not strictly monotonic: the artificial 50 km core and rigidified
        # inner core share rho=8000 (a plateau, not an inversion).
        assert all(rho[i] >= rho[i + 1] for i in range(len(rho) - 1))

    def test_crust_thickness_within_grail_range(self, moon_model):
        n = moon_model.n_layers
        R_surface_km = float(np.asarray(moon_model.R0[n - 1]))
        R_crust_base_km = float(np.asarray(moon_model.R0[n - 2]))
        thickness_km = R_surface_km - R_crust_base_km
        assert MOON["crust_thickness_lo"] / 1e3 <= thickness_km <= MOON["crust_thickness_hi"] / 1e3

    def test_surface_radius_matches_moon_r_within_tenth_percent(self, moon_model):
        """S4: MOON["R"] (1737.15 km, the published reference radius the
        MoI/k2 constraints are pinned to -- see pylov3d.moon's
        "Reference-radius convention") must agree with the profile's own
        surface radius to < 0.1%, guarding against the ~2 sigma silent k2
        shift a mismatched reference radius (e.g. R = 1738 km) would cause
        (see pylov3d.moon's "Tidal k2" section)."""
        n = moon_model.n_layers
        R_surface_km = float(np.asarray(moon_model.R0[n - 1]))
        rel_diff = abs(R_surface_km - MOON["R"] / 1e3) / (MOON["R"] / 1e3)
        assert rel_diff < 0.001


# ---------------------------------------------------------------------------
# 5. pylov3d.moon_mc parameterization + constraints
# ---------------------------------------------------------------------------

class TestConstraints:

    def test_constraint_values_match_moon_dict(self):
        by_name = {c.name: c for c in MOON_CONSTRAINTS}
        assert set(by_name) == {"mass", "moi_mean", "k2", "core_radius_km"}
        assert by_name["mass"].value == MOON["M"]
        assert by_name["moi_mean"].value == MOON["MoI_factor"]
        assert by_name["moi_mean"].sigma == MOON["MoI_factor_sigma"]
        assert by_name["k2"].value == MOON["k2"]
        assert by_name["k2"].sigma == MOON["k2_sigma"]
        assert by_name["core_radius_km"].value == pytest.approx(MOON["fluid_core_radius"] / 1e3)
        assert by_name["core_radius_km"].sigma == pytest.approx(
            MOON["fluid_core_radius_sigma"] / 1e3
        )


class TestParameterization:

    def test_free_params_and_bounds_consistent(self):
        assert set(MOON_PARAMETERIZATION.free_params) == set(MOON_FREE_PARAMS)
        assert set(MOON_BOUNDS.keys()) == set(MOON_FREE_PARAMS)
        assert len(MOON_FREE_PARAMS) == 4
        for lo, hi in MOON_BOUNDS.values():
            assert lo < hi

    def test_core_layer_index_is_2(self):
        """D8 regression: MOON_PARAMETERIZATION must route the
        core_radius_km observable to layer 2 (the physical fluid outer
        core), not the BodyParameterization default of layer 0 (the fixed
        artificial inert stub here) -- see pylov3d.forward.
        BodyParameterization.core_layer_index."""
        assert MOON_PARAMETERIZATION.core_layer_index == 2 == MOON_CORE_LAYER_INDEX

    def test_point_estimate_theta_reproduces_reference_model(self, moon_model):
        theta0 = moon_point_estimate_theta()
        built = build_model(MOON_PARAMETERIZATION, theta0)
        n = moon_model.n_layers
        for name in ("R0", "rho0", "mu0", "Ks0", "ocean"):
            a = np.asarray(getattr(built, name)[:n], dtype=float)
            b = np.asarray(getattr(moon_model, name)[:n], dtype=float)
            assert np.array_equal(a, b), f"{name} differs from build_moon_model()"

    def test_fluid_core_radius_is_layer_index_2(self):
        """R_fluid_core must resolve to layer index 2's radius (the
        physical outer core), not layer 0 (the artificial inert stub)."""
        theta = list(moon_point_estimate_theta())
        theta[MOON_FREE_PARAMS.index("R_fluid_core")] = 400.0
        built = build_model(MOON_PARAMETERIZATION, theta)
        assert float(np.asarray(built.R0[2])) == pytest.approx(400.0)
        assert float(np.asarray(built.R0[0])) == pytest.approx(LAYER_RADII_KM[0])  # unchanged

    def test_mantle_rho_scale_applies_to_mantle_layers_only(self):
        """mantle_rho_scale (S3) must scale layers 3-8 (the six Weber
        mantle shells) and leave the core complex (0-2) and crust (9)
        untouched."""
        theta = list(moon_point_estimate_theta())
        theta[MOON_FREE_PARAMS.index("mantle_rho_scale")] = 1.02
        built = build_model(MOON_PARAMETERIZATION, theta)
        rho = np.asarray(built.rho0[: built.n_layers])
        for i in range(3, 9):
            assert rho[i] == pytest.approx(LAYER_RHO[i] * 1.02)
        for i in (0, 1, 2, 9):
            assert rho[i] == pytest.approx(LAYER_RHO[i])


# ---------------------------------------------------------------------------
# 6. D8 regression: the core_radius_km footgun this framework fix closes
# ---------------------------------------------------------------------------

class TestD8CoreLayerIndexRegression:
    """Pre-D8, core_radius_km was hardcoded to model.R0[0] -- the FIXED
    50 km stub, not the free R_fluid_core (layer 2). At theta=(1,1,380,1),
    R_fluid_core equals the constraint's own center: pre-fix code would
    report a CONSTANT (50-380)/40 = -8.25 sigma core residual always."""

    _THETA_FOOTGUN = (1.0, 1.0, 380.0, 1.0)

    def test_core_radius_km_reads_layer_2_not_layer_0(self):
        model = build_model(MOON_PARAMETERIZATION, self._THETA_FOOTGUN)
        obs = compute_observables(
            model, moon_forcing(), moon_numerics(), which=("core_radius_km",),
            core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
        )
        assert obs["core_radius_km"] == pytest.approx(380.0)

    def test_pre_fix_default_would_have_given_constant_minus_8p25_sigma(self):
        """compute_observables with its DEFAULT core_layer_index=0 (as an
        earlier moon_mc.py draft effectively used) returns the 50 km stub
        -- exactly the -8.25 sigma footgun."""
        model = build_model(MOON_PARAMETERIZATION, self._THETA_FOOTGUN)
        obs_default = compute_observables(
            model, moon_forcing(), moon_numerics(), which=("core_radius_km",),
        )
        assert obs_default["core_radius_km"] == pytest.approx(50.0)
        sigma_resid = (obs_default["core_radius_km"] - 380.0) / 40.0
        assert sigma_resid == pytest.approx(-8.25)

    def test_log_posterior_does_not_carry_constant_core_penalty(self):
        """Fixed moon_log_posterior must NOT reproduce the constant
        -8.25 sigma penalty at theta = (1, 1, 380, 1); the core term
        should be ~0 sigma there, since R_fluid_core IS the center."""
        log_post = moon_log_posterior()
        val = log_post(self._THETA_FOOTGUN)
        assert np.isfinite(val)
        model = build_model(MOON_PARAMETERIZATION, self._THETA_FOOTGUN)
        obs = compute_observables(
            model, moon_forcing(), moon_numerics(),
            which=("mass", "moi_mean", "k2"),
            core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
        )
        bulk_constraints = tuple(c for c in MOON_CONSTRAINTS if c.name != "core_radius_km")
        core_constraint = next(c for c in MOON_CONSTRAINTS if c.name == "core_radius_km")
        expected_core_ll = -0.5 * 0.0**2 - np.log(
            core_constraint.sigma * np.sqrt(2.0 * np.pi)
        )
        expected = log_likelihood(obs, bulk_constraints) + expected_core_ll
        assert val == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Identifiability
# ---------------------------------------------------------------------------

class TestIdentifiability:

    def test_4x4_jacobian_is_full_rank(self):
        """S3: the 4x4 relative-normalized Jacobian of (mass, moi_mean, k2,
        core_radius_km) w.r.t. all 4 free params, at theta0, must be full
        rank -- unlike an earlier 3-param draft's 3x3 Jacobian (mass,
        moi_mean, k2 only), near-degenerate. Condition number checked
        against a 3x band around the measured value (~187)."""
        theta0 = np.array(moon_point_estimate_theta())
        which = ("mass", "moi_mean", "k2", "core_radius_km")

        def f(theta):
            m = build_model(MOON_PARAMETERIZATION, theta)
            o = compute_observables(
                m, moon_forcing(), moon_numerics(), which=which,
                core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
            )
            return np.array([o[k].real if isinstance(o[k], complex) else o[k] for k in which])

        f0 = f(theta0)
        n = len(MOON_FREE_PARAMS)
        J = np.zeros((n, n))
        eps_rel = 1e-4
        for j in range(n):
            dtheta = np.zeros(n)
            step = max(abs(theta0[j]) * eps_rel, 1e-6)
            dtheta[j] = step
            J[:, j] = (f(theta0 + dtheta) - f0) / step

        Jn = J * np.abs(theta0)[None, :] / np.abs(f0)[:, None]
        rank = np.linalg.matrix_rank(Jn)
        assert rank == n

        s = np.linalg.svd(Jn, compute_uv=False)
        condition_number = s.max() / s.min()
        _MEASURED_CONDITION_NUMBER = 187.3
        assert _MEASURED_CONDITION_NUMBER / 3.0 <= condition_number <= _MEASURED_CONDITION_NUMBER * 3.0


# ---------------------------------------------------------------------------
# 8. moon_log_posterior
# ---------------------------------------------------------------------------

class TestLogPosterior:

    def test_finite_at_point_estimate(self):
        log_post = moon_log_posterior()
        val = log_post(moon_point_estimate_theta())
        assert np.isfinite(val)

    def test_minus_inf_outside_bounds(self):
        log_post = moon_log_posterior()
        theta = list(moon_point_estimate_theta())
        theta[MOON_FREE_PARAMS.index("core_rho_scale")] = MOON_BOUNDS["core_rho_scale"][1] + 1.0
        assert log_post(theta) == -np.inf

    def test_log_prior_minus_inf_outside_box(self):
        theta = list(moon_point_estimate_theta())
        theta[0] = MOON_BOUNDS[MOON_FREE_PARAMS[0]][1] + 1.0
        assert log_prior(theta, MOON_PARAMETERIZATION) == -np.inf

    def test_core_radius_km_via_which_now_reads_layer_2(self):
        """Post-D8, core_radius_km CAN be routed through
        compute_observables/make_log_posterior for this parameterization
        (unlike an earlier draft, which raised ValueError here) -- it
        correctly reads layer 2 because MOON_PARAMETERIZATION carries
        core_layer_index=2."""
        log_post = moon_log_posterior(which=("mass", "moi_mean", "k2", "core_radius_km"))
        val = log_post(moon_point_estimate_theta())
        assert np.isfinite(val)

    def test_map_point_improves_on_as_built(self):
        """Pinned 4-param MAP (see moon_mc "Mass-closure parameter") beats
        theta0's log-posterior but is NOT zero-residual: core_rho_scale
        rails at its physical density floor (0.88), so mass/k2 close
        almost exactly while moi_mean/core_radius_km carry real,
        floor-limited residuals."""
        log_post = moon_log_posterior()
        theta0 = moon_point_estimate_theta()
        assert log_post(_MAP_4PARAM_THETA) > log_post(theta0)
        assert log_post(_MAP_4PARAM_THETA) == pytest.approx(-_MAP_4PARAM_NEG_LOGPOST, rel=1e-4)
        assert _MAP_4PARAM_THETA[MOON_FREE_PARAMS.index("core_rho_scale")] == pytest.approx(
            MOON_BOUNDS["core_rho_scale"][0]
        )

        model = build_model(MOON_PARAMETERIZATION, _MAP_4PARAM_THETA)
        obs = compute_observables(
            model, moon_forcing(), moon_numerics(),
            which=("mass", "moi_mean", "k2", "core_radius_km"),
            core_layer_index=MOON_PARAMETERIZATION.core_layer_index,
        )
        assert obs["k2"] == pytest.approx(MOON["k2"], abs=1e-4)
        assert obs["mass"] == pytest.approx(MOON["M"], rel=1e-3)
        # moi_mean and core_radius_km carry real residuals (see docstring
        # above) -- assert the sigma-scale of the residual, not near-zero.
        moi_resid_sigma = (obs["moi_mean"] - MOON["MoI_factor"]) / MOON["MoI_factor_sigma"]
        assert -1.5 < moi_resid_sigma < -0.5
        core_resid_sigma = (obs["core_radius_km"] - MOON["fluid_core_radius"] / 1e3) / (
            MOON["fluid_core_radius_sigma"] / 1e3
        )
        assert -2.0 < core_resid_sigma < -0.5


# ---------------------------------------------------------------------------
# 9. Micro pocomc smoke run (mirrors test_forward.py::TestPocoMCSmoke)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPocoMCSmoke:

    def test_micro_run_completes_and_recovers_k2(self):
        pocomc = pytest.importorskip("pocomc")
        from scipy.stats import uniform

        np.random.seed(0)
        Nrbase = 15  # coarse grid: fast; log_posterior differs from the
        # Nrbase=50-default value by ~1.4e-6 at the as-built theta (spot
        # checked during development), far inside the sampler's noise floor.
        log_post = moon_log_posterior(Nrbase=Nrbase)

        dists = [
            uniform(loc=MOON_BOUNDS[name][0], scale=MOON_BOUNDS[name][1] - MOON_BOUNDS[name][0])
            for name in MOON_FREE_PARAMS
        ]
        prior = pocomc.Prior(dists)

        sampler = pocomc.Sampler(
            prior, log_post, n_dim=len(MOON_FREE_PARAMS),
            n_active=8, n_effective=8, dynamic=False, random_state=0,
        )
        sampler.run(n_total=32, n_evidence=0, progress=False)

        samples, weights, _logl, _logp = sampler.posterior()
        assert samples.shape[0] > 0
        assert np.isfinite(samples).all()

        median = np.average(samples, axis=0, weights=weights)
        model = build_model(MOON_PARAMETERIZATION, median)
        obs = compute_observables(
            model, moon_forcing(), moon_numerics(Nrbase=Nrbase), which=("k2",)
        )
        assert abs(obs["k2"] - MOON["k2"]) < 3 * MOON["k2_sigma"]
