# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Moon instantiation of the body-agnostic forward-model framework (TASK-018).

Wires the 10-layer Weber-structure Moon reference model established in
:mod:`pylov3d.moon` into the generic :mod:`pylov3d.forward` machinery,
mirroring :mod:`pylov3d.mars_mc`'s pattern for Mars. Every numeric constant
is imported from :mod:`pylov3d.moon` -- this module intentionally re-types
nothing.

Unlike Mars (a fresh 4-layer point fit perturbed by 4 free parameters), the
Moon's free parameters perturb the *existing*, MATLAB-validated Weber
structure (see ``pylov3d.moon``'s "Design decision") rather than a new
geometry built from scratch.

Free parameters (4) and constraints (4)
--------------------------------------------------------------------------
====================  ===========================  ===============================================
Name                  Bounds                        Meaning
====================  ===========================  ===============================================
``core_rho_scale``     [0.88, 1.2]                   Shared multiplicative density scale on BOTH the
                                                       physical solid inner core (layer 1, base
                                                       8000 kg/m^3) and the fluid outer core (layer 2,
                                                       base 5100 kg/m^3) -- a single
                                                       :class:`~pylov3d.forward.Scaled` parameter
                                                       applied to two layers, preserving the Weber
                                                       profile's inner/outer core density contrast.
                                                       The lower bound is an absolute-density
                                                       plausibility FLOOR, not a fitted range: it
                                                       keeps the fluid outer core >= 4500 kg/m^3
                                                       (0.88 = 4500/5100, rounded; Weber et al. 2011's
                                                       own 5100 kg/m^3 allows ~-12% before leaving the
                                                       liquid Fe-S plausibility band, ~5-7 g/cm^3 at
                                                       lunar core pressures). The MAP is EXPECTED to
                                                       rail at this floor (see "Exact determination
                                                       and the physical density floor" below) -- the
                                                       round-1 mass artifact wants the core lighter
                                                       than this, and the floor deliberately refuses
                                                       that, documented rather than hidden.
``mu_scale``            [0.3, 3.0]                    Shared shear-modulus scale on the six Weber
                                                       mantle layers (indices 3-8, base values from
                                                       ``pylov3d.moon.LAYER_MU``); same bounds as
                                                       ``pylov3d.mars_mc``'s ``mu_scale`` (same
                                                       physical meaning, reused directly). The
                                                       rigidified inner core (layer 1) and crust
                                                       (layer 9) are excluded, matching the Mars
                                                       module's convention of leaving the crust
                                                       shear modulus outside the tidal-tuning scale.
``R_fluid_core``        [300, 460] km                 Fluid-core (layer 2) outer radius -- widened to
                                                       +/-2 sigma around the Garcia et al. (2011)
                                                       380 +/- 40 km seismic value (not the Weber
                                                       et al. 2011 330 km as-built value; see
                                                       ``pylov3d.moon.MOON["fluid_core_radius"]``), so
                                                       the ``core_radius_km`` Gaussian constraint
                                                       below, not a hard box edge, carries the
                                                       identifying information (see
                                                       "Identifiability" below).
``mantle_rho_scale``    [0.95, 1.05]                  Shared multiplicative density scale on the same
                                                       six Weber mantle layers ``mu_scale`` scales --
                                                       a PRAGMATIC mass-closure parameter, not a claim
                                                       about lunar mantle density; see "Mass-closure
                                                       parameter" below.
====================  ===========================  ===============================================

Constraints: mass, moi_mean, k2, **core_radius_km** -- see "Identifiability"
below for why the 4th constraint is included given ``R_fluid_core`` is free.

All other layer scalars (layer-0 artificial core, layer 1's rigidified
mu/Ks, layer 9's crust mu, every layer's Ks, and every FIXED layer radius)
are FIXED at the values in ``pylov3d.moon.LAYER_RADII_KM`` / ``LAYER_RHO``
/ ``LAYER_MU`` / ``LAYER_KS``.

Mass-closure parameter: why ``mantle_rho_scale`` was added, and why it is
NOT a physical claim
--------------------------------------------------------------------------
A 3-parameter version of this module (``core_rho_scale``, ``mu_scale``,
``R_fluid_core`` only -- an earlier draft) is structurally unable to reach
the mass constraint: the core complex (layers 1-2) is only ~1.3% of the
Moon's total mass in this profile, so even railed at the core_rho_scale
floor (0.88, see "Exact determination and the physical density floor" below) the best 3-parameter fit
found (MAP, L-BFGS-B, several starting points) sits at core_rho_scale =
0.880, R_fluid_core = 375.8 km, and STILL misses the mass constraint by
-4.9 sigma. This is the real content behind :mod:`pylov3d.moon`'s
"As-built residuals" mass explanation (a rounding/digitization artifact in
``Moon_Weber.dat``'s mantle densities, not a genuine data conflict) -- a
mantle-level correction, not a core-level one, is what the mass residual
actually calls for. Adding ``mantle_rho_scale`` (applied to the same six
mantle layers ``mu_scale`` scales) confirms this directly: the best
4-parameter MAP found is core_rho_scale = 0.880 (railed at the floor, see
below), mu_scale = 0.9655, R_fluid_core = 326.9 km, mantle_rho_scale =
1.00638 (+0.64%) -- log-posterior improves from -51.86 (best 3-parameter
MAP) to -37.50 (best 4-parameter MAP, same theta0-independent global
search; see ``test_moon.py::TestLogPosterior::
test_map_point_improves_on_as_built``). mass and k2 residuals both drop
under 0.05 sigma, but moi_mean remains under-satisfied by -0.95 sigma and
core_radius_km by -1.33 sigma (326.9 vs 380 +/- 40 km) -- **not** a
zero-residual fit, because the physical density floor stops the search
before it reaches the fully-relaxed optimum (see "Exact determination and the physical density floor").

This parameter is deliberately framed as a **pragmatic mass-closure knob**,
not physical inference about lunar mantle density: ``Moon_Weber.dat``'s six
mantle layers carry only two distinct rounded density values (3400 and 3220
kg/m^3, see ``pylov3d.moon``'s "As-built residuals"), so a ~+0.6% correction
to all six is absorbing a known rounding/inheritance artifact in the input
data, not resolving a genuine structural question about the Moon's mantle.
The cleaner future fix is a mass-consistent published profile (e.g. Garcia
et al. 2011's VPREMOON, which -- unlike the ``Moon_Weber.dat`` values used
here -- was itself fit to reproduce total mass) in place of this
scale-factor patch.

Exact determination and the physical density floor
--------------------------------------------------------------------------
With 4 free parameters against 4 constraints, this system is formally
**exactly determined**: absent any bound restricting the search, a
zero-residual fit generically exists and by itself carries NO
goodness-of-fit information about the input data -- the round-1 mass
tension (see :mod:`pylov3d.moon`'s "As-built residuals") was *absorbed* by
``mantle_rho_scale``, not independently *resolved*, and a perfect fit
would not have meant the Weber profile's mantle densities were confirmed
correct. In practice, ``core_rho_scale``'s lower bound (0.88) is a genuine
physical floor, not a fitted range: it keeps the fluid outer core's
density >= 4500 kg/m^3 (Weber et al. 2011 itself adopts 5100 kg/m^3 for
this layer; liquid Fe-S at lunar core pressures is plausibly 5-7 g/cm^3,
so the floor allows ~-12% before leaving that band). The round-1 mass
artifact wants the core lighter than this floor permits, so the MAP is
**expected** to rail there (core_rho_scale = 0.880 exactly, both 3- and
4-parameter fits, see "Mass-closure parameter" above) -- this is
documented behavior, not a bug or an unexplored corner: the floor
deliberately prevents the "exactly determined -> zero residual" outcome
from masking an unphysical fit, at the cost of a real, visible residual
(moi_mean -0.95 sigma, core_radius_km -1.33 sigma at the 4-parameter MAP).

Identifiability
--------------------------------------------------------------------------
With 4 free parameters and 4 constraints (mass, moi_mean, k2,
core_radius_km), a finite-difference, relative-normalized Jacobian of those
4 observables with respect to (core_rho_scale, mu_scale, R_fluid_core,
mantle_rho_scale), evaluated at the as-built theta0
(:func:`moon_point_estimate_theta`), is **full rank** (rank 4 of 4;
verified via ``numpy.linalg.matrix_rank``) with singular values
[2.253, 0.999, 0.405, 0.0120] -- condition number ~187. This is a large
improvement over an earlier 3-parameter draft's 3x3 Jacobian of just
(mass, moi_mean, k2) against (core_rho_scale, mu_scale, R_fluid_core),
which had condition number ~6.1e3 (a near-degenerate direction dominated by
core_rho_scale and R_fluid_core moving oppositely): ``mantle_rho_scale``
does not merely add a 4th knob, it gives mass its own largely-independent
handle, which is what breaks the near-degeneracy. The system is not
perfectly conditioned (~187 is still large relative to 1), so
``core_radius_km``'s Gaussian constraint remains meaningful rather than
redundant -- see ``test_moon.py::TestIdentifiability`` for the numerical
check this claim is based on.

A caveat on what the core-radius constraint actually adds: the as-built
profile's own outer-core boundary (Weber et al. 2011, 330 km) and the
constraint center (Garcia et al. 2011, 380 +/- 40 km) are **both seismic**
determinations from different published analyses -- not two independent
data classes (e.g. "seismic vs. LLR/tidal"; see ``pylov3d.moon``'s "Fluid
core radius" section). What the constraint adds is not a new *kind* of
data, but a real piece of information -- Garcia et al. (2011)'s own
seismic solution -- that the bulk tidal/gravimetric observables (mass,
moi_mean, k2) alone only weakly pin down (per the Jacobian above), so it
still substantively shapes the posterior even though it enters through the
same underlying data type (seismology) the reference profile's own
structure came from.

Anelastic bias: where it actually shows up (corrected -- was wrong for the
4-parameter model)
--------------------------------------------------------------------------
This Monte Carlo stage fits a purely **elastic** k2 to the Moon's
*observed* (anelastic) k2 = 0.02422; the as-built elastic k2 (0.023159)
falls short by 4.6% -- see :mod:`pylov3d.moon`'s "As-built residuals" for
the "consistent-with, not diagnostic-of" framing. An earlier draft of this
section claimed the resulting bias shows up as a systematically enlarged
R_fluid_core (citing a single "367-388 km" range folding together 3- and
4-parameter results) -- **wrong for the 4-parameter model**, corrected
here: with ``mantle_rho_scale`` absorbing mass and ``mu_scale`` alone
already closing the k2 gap to <1e-4 sigma (see "Mass-closure parameter"),
the anelastic bias does **not** migrate to R_fluid_core in the
4-parameter fit -- it stays in ``mu_scale`` (and, for the mass residual
specifically, ``mantle_rho_scale``), exactly where each was added to
absorb it. R_fluid_core is a separate story, and now depends on whether
the physical density floor (below) is active:

- **Without the floor** (an intermediate, unphysical [0.75, 1.25]
  core_rho_scale range explored during development): nothing pulls on
  R_fluid_core at all -- its marginal is essentially just the (2 sigma
  -truncated) Garcia prior itself. The MAP sits at R_fluid_core = 380.000
  km, exactly the prior mean; a properly-resolved posterior
  (``n_active=64``, ``n_effective=128``, ``n_total=512``) gives
  363.7 +/- 33.2 km, consistent with the truncated-prior width -- i.e. the
  bulk observables carry ~no information about R_fluid_core once
  core_rho_scale is free to reach an unphysically low value.
- **With the floor** (0.88, the value this module actually ships):
  core_rho_scale can no longer drop low enough to fully absorb the mass
  residual, so a residual MASS/MoI pull reappears -- but pulling
  R_fluid_core DOWN, toward the as-built Weber value (~327 km at the MAP,
  see "Mass-closure parameter"), not up toward Garcia's 380 km. This is a
  mass/MoI mechanism, not an anelastic one (k2's own residual stays ~0
  sigma throughout).

How large should a genuine elastic-to-anelastic k2 bias be, in principle
(context; not what drives R_fluid_core here)? Williams et al. (2014)
Section 6 gives a lunar monthly tidal quality factor Q = 38 +/- 4; with an
Andrade rheology (alpha ~ 0.3-0.35, commonly adopted for silicate mantles),
a Q this low predicts roughly a 4-5% elastic-to-anelastic k2 enhancement at
the monthly period -- close to the 4.6% gap measured here. The broader
literature range is wider, 4-10%: Williams et al. (2014) Table 10's
zero-period (fully relaxed) values imply a 5-9% enhancement; Garcia et al.
(2019) report an elastic k2 = 0.02277 +/- 0.00058 for their own (different)
reference profile, a 6.4% gap to the same 0.02422 target; Nimmo et al.
(2012) estimate a larger ~10% effect. This module's 4.6% sits at the low
end of that range -- consistent with, but on its own not strong evidence
for, any particular anelastic model, especially since ``mu_scale`` closes
it cleanly with no need to invoke R_fluid_core at all.
"""

from __future__ import annotations

from .forward import (
    BodyParameterization,
    Constraint,
    Free,
    LayerSpec,
    Scaled,
    make_log_posterior,
)
from .moon import (
    LAYER_KS,
    LAYER_MU,
    LAYER_NAMES,
    LAYER_RADII_KM,
    LAYER_RHO,
    MOON,
    MOON_NUMERICS_NRBASE,
    moon_forcing,
    moon_numerics,
)

# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

#: Mantle layer indices (the six Weber shells scaled by mu_scale AND
#: mantle_rho_scale); layer 0 (artificial core), 1 (rigidified inner core),
#: 2 (fluid outer core), and 9 (crust) are excluded -- see module docstring.
_MANTLE_IDX = range(3, 9)

_layers: list[LayerSpec] = [
    LayerSpec(
        name=LAYER_NAMES[0],
        radius_km=LAYER_RADII_KM[0],
        rho=LAYER_RHO[0],
        mu=LAYER_MU[0],
        Ks=LAYER_KS[0],
        eta=None,
        ocean=0,
    ),
    LayerSpec(
        name=LAYER_NAMES[1],
        radius_km=LAYER_RADII_KM[1],
        rho=Scaled(LAYER_RHO[1], "core_rho_scale"),
        mu=LAYER_MU[1],  # rigidified inner core; not part of mu_scale
        Ks=LAYER_KS[1],
        eta=None,
        ocean=0,
    ),
    LayerSpec(
        name=LAYER_NAMES[2],
        radius_km=Free("R_fluid_core"),
        rho=Scaled(LAYER_RHO[2], "core_rho_scale"),
        mu=0.0,  # physically fluid (Vs=0); always zero, not scaled
        Ks=LAYER_KS[2],
        eta=None,
        ocean=1,
    ),
]
for _i in _MANTLE_IDX:
    _layers.append(
        LayerSpec(
            name=LAYER_NAMES[_i],
            radius_km=LAYER_RADII_KM[_i],
            rho=Scaled(LAYER_RHO[_i], "mantle_rho_scale"),
            mu=Scaled(LAYER_MU[_i], "mu_scale"),
            Ks=LAYER_KS[_i],
            eta=None,
            ocean=0,
        )
    )
_layers.append(
    LayerSpec(
        name=LAYER_NAMES[9],
        radius_km=LAYER_RADII_KM[9],
        rho=LAYER_RHO[9],
        mu=LAYER_MU[9],  # crust mu fixed, excluded from mu_scale
        Ks=LAYER_KS[9],
        eta=None,
        ocean=0,
    )
)

MOON_LAYERS: tuple[LayerSpec, ...] = tuple(_layers)
del _layers, _i

MOON_FREE_PARAMS: tuple[str, ...] = (
    "core_rho_scale", "mu_scale", "R_fluid_core", "mantle_rho_scale",
)

#: LOV3D layer index of the physically free "core" this parameterization
#: tracks for the core_radius_km observable: layer 2 (the fluid outer
#: core), NOT layer 0 (the fixed artificial numerically-inert stub -- see
#: pylov3d.moon's "Design decision"). Forwarded to
#: BodyParameterization.core_layer_index (see pylov3d.forward), which
#: routes compute_observables()/make_log_posterior() to read
#: model.R0[2] instead of the default model.R0[0].
MOON_CORE_LAYER_INDEX: int = 2

#: Fluid-core radius +/- 2*sigma (Garcia et al. 2011: 380 +/- 40 km),
#: derived from pylov3d.moon.MOON rather than re-typed. Bounded above by
#: layer 3's fixed 480 km radius and below by layer 1's fixed 240 km
#: radius, with headroom to spare (300-460 km strictly inside (240, 480)).
_R_FLUID_CORE_KM = MOON["fluid_core_radius"] / 1e3
_R_FLUID_CORE_SIGMA_KM = MOON["fluid_core_radius_sigma"] / 1e3

#: core_rho_scale's lower bound (0.88) is an absolute-density plausibility
#: FLOOR, not a fitted range: 0.88 * 5100 kg/m^3 = 4488 kg/m^3, just above
#: 4500 kg/m^3 (s_min = 4500/5100 = 0.8824, rounded to 0.88) -- Weber et al.
#: (2011)'s own fluid-outer-core density is 5100 kg/m^3; liquid Fe-S at
#: lunar core pressures is plausibly 5-7 g/cm^3, so this floor allows
#: ~-12% before leaving that band. See module docstring, "Exact
#: determination and the physical density floor" -- the MAP is EXPECTED to
#: rail here (both 3- and 4-parameter fits do).
MOON_BOUNDS: dict[str, tuple[float, float]] = {
    "core_rho_scale": (0.88, 1.2),
    "mu_scale": (0.3, 3.0),
    "R_fluid_core": (
        _R_FLUID_CORE_KM - 2 * _R_FLUID_CORE_SIGMA_KM,
        _R_FLUID_CORE_KM + 2 * _R_FLUID_CORE_SIGMA_KM,
    ),
    "mantle_rho_scale": (0.95, 1.05),
}

MOON_PARAMETERIZATION = BodyParameterization(
    name="moon",
    layers=MOON_LAYERS,
    free_params=MOON_FREE_PARAMS,
    bounds=MOON_BOUNDS,
    core_layer_index=MOON_CORE_LAYER_INDEX,
)

# ---------------------------------------------------------------------------
# Constraints (values/sigmas cited in pylov3d.moon; not restated here)
# ---------------------------------------------------------------------------

MOON_CONSTRAINTS: tuple[Constraint, ...] = (
    # sigma = 0.1%, a conservative round number, NOT a propagated
    # measurement error: GM (Williams, Boggs & Folkner 2013, ~3e-8
    # relative) and G (CODATA 2018, ~2.2e-5 relative) are both known far
    # tighter than this. 0.1% is instead a deliberate MODELLING-ERROR
    # allowance -- this stage-1 model's own structural/rounding
    # uncertainty (see pylov3d.moon's "As-built residuals": Moon_Weber.dat
    # is published to only 2-3 significant figures) swamps the underlying
    # data's actual precision -- so mass barely constrains
    # (core_rho_scale, mantle_rho_scale) relative to the much tighter
    # MoI/k2 constraints -- intentional, matches pylov3d.mars_mc's
    # identical convention.
    Constraint(name="mass", value=MOON["M"], sigma=0.001 * MOON["M"]),
    Constraint(name="moi_mean", value=MOON["MoI_factor"], sigma=MOON["MoI_factor_sigma"]),
    Constraint(name="k2", value=MOON["k2"], sigma=MOON["k2_sigma"]),
    # 4th constraint, required for identifiability with R_fluid_core free --
    # see module docstring, "Identifiability". Correctly reads layer index
    # 2 (not the default layer 0) via MOON_PARAMETERIZATION's
    # core_layer_index -- see pylov3d.forward.BodyParameterization and
    # test_moon.py's D8 regression test.
    Constraint(
        name="core_radius_km", value=_R_FLUID_CORE_KM, sigma=_R_FLUID_CORE_SIGMA_KM
    ),
)


def moon_log_posterior(Nrbase: int = MOON_NUMERICS_NRBASE, which: tuple[str, ...] | None = None):
    """Convenience: :func:`pylov3d.forward.make_log_posterior` wired up for
    the Moon.

    Simply forwards to the generic framework function -- unlike an earlier
    draft of this module, no custom bypass is needed here: ``core_radius_km``
    is correctly computed from layer 2 because
    :data:`MOON_PARAMETERIZATION` carries ``core_layer_index=2``, which
    :func:`~pylov3d.forward.make_log_posterior` reads and forwards to
    :func:`~pylov3d.forward.compute_observables` automatically.

    Parameters
    ----------
    Nrbase : forwarded to :func:`pylov3d.moon.moon_numerics`.
    which : observables to compute; defaults to every constrained observable
        (mass, moi_mean, k2, core_radius_km).
    """
    return make_log_posterior(
        MOON_PARAMETERIZATION,
        MOON_CONSTRAINTS,
        moon_forcing(),
        moon_numerics(Nrbase=Nrbase),
        which=which,
    )


# ---------------------------------------------------------------------------
# As-built Weber profile, expressed as a theta vector (MOON_FREE_PARAMS
# order) -- the natural "all scales = 1, radius = as-built" starting point,
# analogous to pylov3d.mars_mc.mars_point_fit_theta but NOT a fit (see
# pylov3d.moon's "As-built residuals": this theta reproduces the
# non-trivial residuals documented there, it does not resolve them).
# ---------------------------------------------------------------------------


def moon_point_estimate_theta() -> tuple[float, float, float, float]:
    """The as-built Weber profile, as a theta vector: unit scales, the
    profile's own (Weber-seismic, not the Garcia-et-al.-constraint)
    fluid-core radius.

    Feeding this theta through :func:`pylov3d.forward.build_model` with
    :data:`MOON_PARAMETERIZATION` reproduces :func:`pylov3d.moon.
    build_moon_model`'s output exactly (unit scales are no-ops; see
    ``test_moon.py::test_point_estimate_theta_reproduces_reference_model``).
    """
    R_core_km = MOON["fluid_core_radius_weber_seismic"] / 1e3
    return (1.0, 1.0, R_core_km, 1.0)
