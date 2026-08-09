# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Moon 1D radial reference interior model (TASK-018).

Stage-1 elastic reference model for the Moon, wired into the body-agnostic
forward-model framework (:mod:`pylov3d.forward`) the same way
:mod:`pylov3d.mars` wires up Mars. Unlike Mars (a fresh 4-layer point fit,
see ``pylov3d/mars.py``), the Moon's reference model is **not** a new fit:
it is the pre-existing 10-layer Weber et al. (2011) profile already
MATLAB-cross-validated in Milestone 5
(``pylov3d/tests/test_matlab_validation_ocean.py``,
``_build_weber_moon_model`` / ``TestMoonOceanValidation.
test_uniform_k2_matches_matlab``, uniform k2 agreeing with MATLAB's k2_Q to
~2e-9 relative). See "Design decision" below for why this module reuses
that structure rather than deriving a new one, and "As-built residuals"
for how well it lands on the bulk constraints below (not exactly — that
gap motivates the Monte Carlo stage, :mod:`pylov3d.moon_mc`).

**Reference-radius convention.** The MoI and k2 constraints below are each
published at a specific reference radius (noted per-row); this profile's
own surface radius is 1737.1 km (:data:`LAYER_RADII_KM` ``[-1]``). The
adopted k2 (see "Citation notes" below) matches a solution referenced at
R = 1737.151 km specifically -- closest to this surface -- not R = 1738 km
(a second referencing, ~0.06 sigma different, not used).
``test_moon.py`` checks :data:`MOON` ``["R"]`` against the profile's own
surface radius to guard against silent drift.

Published bulk constraints (cited exactly; do not alter without flagging)
---------------------------------------------------------------------------
=====================================  ========================================  =========================================================
Quantity                                Value                                     Source
=====================================  ========================================  =========================================================
GM                                      4902.80007 +/- 0.00014 km^3/s^2           Williams, J. G., Boggs, D. H., & Folkner, W. M. (2013),
                                                                                    "DE430 Lunar Orbit, Physical Librations, and Surface
                                                                                    Coordinates", JPL IOM 335-JW,DB,WF-20130722-016, Table 5
                                                                                    -- the DE430 ephemeris GM, adopted in Williams et al.
                                                                                    (2014) Table 1; see "Citation notes" below.
Mass M                                  GM/G, G = 6.6743e-11 (CODATA 2018)         derived, ~= 7.3458e22 kg -- matches the pre-existing
                                         ~= 7.3458e22 kg                            ``pylov3d.bodies`` catalog entry (id 31, "Moon") to 5
                                                                                    significant figures, an independent cross-check of GM.
Mean radius R                           1737.15 km                                Neumann, G. A. (2013), LRO LOLA topographic data
                                                                                    products, as adopted in Williams et al. (2014) Table 2;
                                                                                    Smith, D. E., et al. (2010), GRL 37, L18204, gives a
                                                                                    closely similar but distinct value, 1737.153 +/- 0.010 km.
Whole-Moon mean MoI factor I/MR^2       0.3931 +/- 0.0002                         Konopliv, A. S., Binder, A. B., Hood, L. L., et al.
(the constraint used)                                                             (1998), "Improved gravity field of the Moon from Lunar
                                                                                    Prospector", Science, 281, 1476-1480 (verbatim: "the
                                                                                    average moment I/MR^2 = 0.3931 +/- 0.0002"); see
                                                                                    "MoI: whole-Moon vs. solid-Moon" below.
Tidal k2 (monthly, degree-2)            0.02422 +/- 0.00022                       Williams, J. G., Konopliv, A. S., Boggs, D. H., et al.
                                                                                    (2014), "Lunar interior properties from the GRAIL
                                                                                    mission", JGR Planets, 119, 1546-1578, Section 5: the
                                                                                    unweighted mean of two GRAIL-only k2 solutions, referenced
                                                                                    to R = 1737.151 km; see "Citation notes" below (NOT a combined
                                                                                    LLR+GRAIL solution -- DE430/LLR holds k2 fixed).
Fluid core radius                       380 +/- 40 km                             Garcia, R. F., Gagnepain-Beyneix, J., Chevrot, S., &
                                                                                    Lognonne, P. (2011), "Very preliminary reference Moon
                                                                                    model", Physics of the Earth and Planetary Interiors,
                                                                                    188, 96-113 (seismic; as attributed by Williams et al.
                                                                                    2014, Section 7) -- see "Citation notes" below.
Crust thickness (global mean)           34-43 km; adopt 40 km                     Wieczorek, M. A., et al. (2013), "The crust of the Moon as
                                                                                    seen by GRAIL", Science 339, 671-675.
=====================================  ========================================  =========================================================

Citation notes (science review round 1; see docs/MOON_MODEL.md for the full
record of what changed)
---------------------------------------------------------------------------
**GM.** 4902.80007 +/- 0.00014 km^3/s^2 is the **DE430** ephemeris GM
(LLR-based): Williams, Boggs & Folkner (2013), JPL IOM
335-JW,DB,WF-20130722-016, Table 5; Williams et al. (2014) Table 1 adopts
this value directly. An earlier draft of this module misattributed it to
Konopliv et al. (2013) GL0660B, which independently gives a close but
distinct GM = 4902.80031 +/- 0.00044 km^3/s^2 -- NOT the source of the
digit string used here. The *value* (4902.80007e9 m^3/s^2) is unchanged and
independently corroborated by the pre-existing ``pylov3d.bodies`` catalog
entry (id 31, "Moon"): ``Mass=7.3458e22`` kg matches ``GM["GM"]/G`` here to
5 significant figures.

**GRAIL solution naming.** Konopliv et al. (2013)'s own JPL solutions are
GL0420A (primary mission) / GL0660B (primary + extended); "GRGM900C" is a
*different*, GSFC-led solution (Lemoine et al. 2014, GRL 41, 3382-3389).
Where this module says "the Konopliv et al. (2013) GRAIL solution" it means
GL0660B specifically (see "Citation notes" below).

**MoI: whole-Moon vs. solid-Moon (NOT a polar-vs-mean distinction).** An
earlier draft framed :data:`MOON` ``["MoI_polar_factor"]`` (0.393112) as a
*polar* moment needing a Mars-style J2 correction -- wrong. 0.393112 is
Williams et al. (2014) Section 5's normalized **solid-Moon** mean moment
(0.392728 +/- 0.000012 as published, re-expressed at R = 1737.151 km): the
fluid core's own moment contribution is handled separately there (LLR alone
does not resolve the fluid-core/whole-Moon polar-moment ratio C_f/C), so it
is **not comparable** to what :func:`pylov3d.forward.analytic_mass_moi`
computes (a plain shell-sum over every layer, fluid core included). Kept as
:data:`MOON` ``["MoI_solid_moon_factor"]`` for context ONLY. The constraint
actually used, I/MR^2 = 0.3931 +/- 0.0002, is instead Konopliv et al.
(1998)'s explicitly whole-Moon "average moment" -- the physically correct,
comparable target.

**Tidal k2.** 0.02422 +/- 0.00022 is Williams et al. (2014) Section 5's
**unweighted mean of two GRAIL-only k2 solutions**, referenced to
R = 1737.151 km -- NOT "combined LLR+GRAIL" as an earlier draft stated
(DE430/LLR holds k2 *fixed*, contributing no independent k2 information).
A second referencing at R = 1738 km gives k2 = 0.02416 +/- 0.00022; not
used here because 1737.151 km is closer to this model's own 1737.1 km
surface (see "Reference-radius convention" above; the two differ by only
~0.06 sigma). Context only, not a constraint: Konopliv et al. (2013)
GL0660B (R = 1738 km) gives k2 = 0.02405 +/- 0.00018 -- see :data:`MOON`
``["k2_grail_only"]``. An earlier draft's "k2_grail_only_sigma = 0.000414"
(attributed to Konopliv et al. 2013) could not be traced to any source and
has been removed.

**Fluid core radius.** 380 +/- 40 km is attributed by Williams et al.
(2014) Section 7 to Garcia, Gagnepain-Beyneix, Chevrot & Lognonne (2011),
PEPI 188, 96-113 (seismic) -- an earlier draft incorrectly attributed it to
Williams et al. (2014)'s own LLR solution. **Both** this value and the
as-built profile's own 330 km outer-core boundary (Weber et al. 2011) are
therefore seismic determinations from different analyses, not independent
data classes -- see :mod:`pylov3d.moon_mc`'s "Identifiability" for what
this means for the constraint's evidentiary weight. Weber et al. (2011)
publishes error bars not carried in ``Moon_Weber.dat`` itself: inner core
240 +/- 10 km, outer (fluid) core 330 +/- 20 km, partial-melt-zone top at
480 +/- 15 km (:data:`LAYER_RADII_KM` ``[3]``, :data:`LAYER_NAMES`
``[3] == "partial_melt_zone"``).

Design decision: Weber-structure reference model, not a fresh fit
---------------------------------------------------------------------------
TASK-018 explicitly directs *against* a fresh 4-layer deterministic fit (the
Mars pattern) for the Moon reference model, in favor of reusing the 10-layer
Weber et al. (2011) profile already built and MATLAB-validated by the
ocean-solver harness (Milestone 5,
``pylov3d/tests/test_matlab_validation_ocean.py:_build_weber_moon_model``).
That profile prepends an artificial 50 km, 8000 kg/m^3 numerically-inert
core (LOV3D's boundary-condition machinery always treats layer 0 as the
core and never integrates through it, exactly as for Mars's liquid core --
see ``pylov3d/boundary_conditions.py``), rigidifies the physical solid inner
core (mu, Ks x1000, per the notebook's own annotation), and tags the fluid
outer core (Vs=0) with ``ocean=1`` at layer index 2. This module's reference
model *is* that profile: :func:`build_moon_model` returns it unmodified.

Why re-derive (not import) the harness construction, and a data dependency
this creates
---------------------------------------------------------------------------
This module reads ``data/tests/moon/Moon_Weber.dat`` (the raw Weber et al.
2011 profile) directly, at import time (:data:`_WEBER` below is built
during module import so :data:`LAYER_RADII_KM` etc. exist as plain module
constants). That is a genuine runtime data dependency on a file under
``data/tests/`` -- a path name suggesting test-only data. It stays there
because the file is the actual published Weber et al. (2011) profile, not
synthetic fixture data, and relocating it is a larger reorganization out of
scope for TASK-018; this is an accepted, if slightly awkward, dependency
for a research package, called out explicitly rather than left implicit.

Separately, TASK-018 asks that this module not duplicate
``_build_weber_moon_model``'s parsing/rigidification logic. Importing that
function directly from ``pylov3d.tests.test_matlab_validation_ocean`` was
considered and rejected: that test module does an unconditional
module-level ``import pytest``, so importing it here would additionally
make **pytest** -- currently only a ``pylov3d[test]`` optional dependency,
not a runtime one -- a hard import-time dependency of this module. Instead,
:func:`_load_weber_profile` below is deliberately byte-for-byte-identical
arithmetic to the harness function (same magic numbers: 50 km / 8000 kg/m^3
artificial core, x1000 inner-core rigidification, ocean flag at index 2),
and ``test_moon.py::test_reference_model_matches_ocean_harness`` imports
the real harness function (pytest is already a test-environment dependency)
and asserts element-by-element equality against :func:`build_moon_model`'s
output -- an executable guarantee against drift, in lieu of a static
import.

As-built residuals (why the Monte Carlo stage exists)
---------------------------------------------------------------------------
The Weber profile, taken as-is, does **not** hit the constraints above
exactly -- it was built (in the MATLAB notebook this repo cross-validates
against) as a seismic velocity/density profile, not as a fit to GM, MoI, or
tidal k2. Computed via :func:`build_moon_model` + :func:`~pylov3d.forward.
analytic_mass_moi` + :func:`~pylov3d.love.get_love` (elastic, as-is; no
free parameters):

=================  ========================  ========================  ===================
Observable          As-built (Weber profile)   Constraint (this module)   Residual
=================  ========================  ========================  ===================
Mass                7.31329e22 kg               7.34579e22 kg              -0.442% (-4.42e-3 rel)
Mean MoI I/MR^2      0.392361                    0.3931 +/- 0.0002          -7.39e-4 (-3.7 sigma)
Tidal k2 (elastic)   0.0231591                   0.02422 +/- 0.00022        -1.061e-3 (-4.8 sigma)
Fluid-core radius    330 km                      380 +/- 40 km              -50 km (-1.25 sigma)
=================  ========================  ========================  ===================

**Mass residual: a digitization artifact, not a data conflict.** The
as-built mean density is 3330.8 kg/m^3 against the observed 3345.3 kg/m^3
(at GM/G, R = 1737.15 km). ``Moon_Weber.dat``'s densities are published to
only 2-3 significant figures, and its six mantle shells (indices 3-8)
carry only two distinct rounded values, 3400 kg/m^3 (four shells) and 3220
kg/m^3 (two shells): Weber et al. (2011) published core densities from
their own seismic inversion, but the mantle values were inherited from
earlier reference models (Lognonne et al. 2003, Gagnepain-Beyneix et al.
2006) never themselves fit to reproduce total mass. The -0.44% residual is
this rounding/inheritance chain, with no published uncertainty budget
covering it (GM itself is known to ~3e-8 relative, see the GM row above).
:mod:`pylov3d.moon_mc`'s ``mantle_rho_scale`` free parameter exists
specifically to absorb this digitization artifact.

**k2 residual: consistent with, but not diagnostic of, anelasticity.** k2 =
0.02422 is the Moon's *observed* value, which -- like Mars's -- includes
the tide's actual anelastic response, while the profile above is elastic;
an elastic model needs a softer mantle to reach the same k2 an anelastic
one reaches with a stiffer one (see :mod:`pylov3d.mars`'s identical
caveat). The as-built gap (4.6% low) is of the right sign/size for that
effect but is not by itself strong evidence of it -- see
:mod:`pylov3d.moon_mc`'s "Anelastic bias" for the fuller picture
(literature-predicted enhancement size, and competing non-rheological
explanations for the same gap).

These non-trivial, multi-sigma residuals are exactly the motivation for
:mod:`pylov3d.moon_mc`: a small free-parameter perturbation of this same
Weber structure (core density scale, mantle density scale, mantle
shear-modulus scale, fluid-core radius), fit by Bayesian inference against
these same four constraints, rather than discarding the
seismically-anchored profile in favor of an unconstrained new one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import G
from .forward import analytic_mass_moi
from .types import InteriorModel, make_forcing, make_interior_model, make_numerics

# ---------------------------------------------------------------------------
# Published bulk constraints (cite exactly; see module docstring table)
# ---------------------------------------------------------------------------

MOON: dict[str, float] = {
    # GM: DE430 ephemeris value, Williams, Boggs & Folkner (2013), JPL IOM
    # 335-JW,DB,WF-20130722-016, Table 5 (4902.80007 +/- 0.00014 km^3/s^2),
    # adopted in Williams et al. (2014) Table 1; see "Citation notes" above.
    # km^3/s^2 -> m^3/s^2.
    "GM": 4902.80007e9,
    "GM_sigma": 0.00014e9,
    # G: CODATA 2018 (matches pylov3d.constants.G).
    "G": G,
    # Mass: derived, M = GM/G ~= 7.3458e22 kg (matches pylov3d.bodies
    # catalog id 31 "Moon" Mass=7.3458e22 to 5 sig figs -- independent
    # cross-check, see module docstring).
    "M": 4902.80007e9 / G,
    # Mean radius: Neumann (2013), LRO LOLA, as adopted in Williams et al.
    # (2014) Table 2. km -> m.
    "R": 1737.15e3,
    # Solid-Moon mean moment-of-inertia factor: Williams et al. (2014)
    # Section 5 (0.392728 +/- 0.000012 as published, re-expressed at
    # R = 1737.151 km). Context/cross-check ONLY -- NOT comparable to a
    # whole-body shell-sum and NOT used in any fit or constraint here; see
    # "MoI: whole-Moon vs. solid-Moon" above. (An earlier draft of this
    # module mislabeled this "MoI_polar_factor" and treated it as a polar
    # moment needing a J2 correction -- both wrong, corrected above.)
    "MoI_solid_moon_factor": 0.393112,
    "MoI_solid_moon_factor_sigma": 0.000012,
    # Whole-Moon mean moment-of-inertia factor I/MR^2 (the actual
    # fit/constraint target): Konopliv et al. (1998), Science 281,
    # 1476-1480 ("the average moment I/MR^2 = 0.3931 +/- 0.0002",
    # verbatim); see "Citation notes" above.
    "MoI_factor": 0.3931,
    "MoI_factor_sigma": 0.0002,
    # Tidal k2 (monthly, degree-2): Williams et al. (2014) Section 5,
    # unweighted mean of two GRAIL-only k2 solutions, referenced to
    # R = 1737.151 km; see "Citation notes" above (NOT combined LLR+GRAIL --
    # DE430/LLR holds k2 fixed).
    "k2": 0.02422,
    "k2_sigma": 0.00022,
    # Same underlying GRAIL k2 solutions, referenced instead to R = 1738 km
    # (Williams et al. 2014 Section 5). NOT used as a constraint (the
    # 1737.151 km referencing above matches this model's own surface
    # radius); context only.
    "k2_R1738": 0.02416,
    "k2_R1738_sigma": 0.00022,
    # GRAIL-primary-mission-only comparison k2: Konopliv et al. (2013)
    # GL0660B, R = 1738 km. Context only, not used as a constraint. (An
    # earlier draft of this module instead listed a
    # "k2_grail_only_sigma = 0.000414" that could not be traced to any
    # source during review; replaced with this traceable value -- see
    # "Tidal k2" above.)
    "k2_grail_only": 0.02405,
    "k2_grail_only_sigma": 0.00018,
    # Fluid core radius: Garcia, Gagnepain-Beyneix, Chevrot & Lognonne
    # (2011), PEPI 188, 96-113 (seismic), as attributed by Williams et al.
    # (2014) Section 7 -- NOT a Williams et al. (2014) LLR result itself
    # (an earlier draft of this module said it was); see "Fluid core
    # radius" above. km -> m.
    "fluid_core_radius": 380e3,
    "fluid_core_radius_sigma": 40e3,
    # Weber et al. (2011) seismic outer-core radius: this is the value
    # literally built into the as-built reference profile's outer-core
    # boundary (see "As-built residuals"). A different published seismic
    # analysis than the Garcia et al. (2011) value above, not an
    # independent data class -- see "Citation notes" above. Context
    # only, not itself used as a Constraint (pylov3d.moon_mc constrains
    # against fluid_core_radius above).
    "fluid_core_radius_weber_seismic": 330e3,
    "fluid_core_radius_weber_seismic_sigma": 20e3,
    # Weber et al. (2011) solid inner-core radius, with its published
    # error bar (not carried in Moon_Weber.dat itself). Context only.
    "inner_core_radius_weber_seismic": 240e3,
    "inner_core_radius_weber_seismic_sigma": 10e3,
    # Weber et al. (2011) partial-melt-zone top, with its published error
    # bar. This is LAYER_RADII_KM[3] (fixed in this profile). Context only.
    "partial_melt_radius_weber_seismic": 480e3,
    "partial_melt_radius_weber_seismic_sigma": 15e3,
    # Crust thickness: Wieczorek et al. (2013), GRAIL. km -> m.
    "crust_thickness_lo": 34e3,
    "crust_thickness_hi": 43e3,
    "crust_thickness_adopted": 40e3,
}

# ---------------------------------------------------------------------------
# Reference model: the Weber et al. (2011) 10-layer profile, as used by the
# MATLAB-validated ocean harness (see "Design decision" / "Why re-derive"
# above). Re-derived here (not imported): both a data-file load (this
# module has a genuine import-time runtime dependency on
# data/tests/moon/Moon_Weber.dat) and a rederivation of the harness's own
# parsing/rigidification arithmetic, byte-identical to it and pinned equal
# to it by pylov3d/tests/test_moon.py::test_reference_model_matches_ocean_harness.
# ---------------------------------------------------------------------------

#: Path to the raw Weber et al. (2011) seismic profile (radius [m], density
#: [kg/m^3], Vp [m/s], Vs [m/s]), shared with the ocean-validation harness.
_MOON_DATA_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "tests" / "moon" / "Moon_Weber.dat"
)


def _load_weber_profile() -> InteriorModel:
    """Build the 10-layer Weber Moon exactly as the MATLAB-validated ocean
    harness does (``pylov3d/tests/test_matlab_validation_ocean.py:
    _build_weber_moon_model``, lines ~107-142).

    ``Test_Moon_MultiLayered_Lateral_Variations.mlx`` does not use the raw
    9-layer Weber profile directly: it prepends an artificial 50 km,
    8000 kg/m^3 core (so the fluid outer core is not directly the LOV3D
    core layer -- unsupported in MATLAB and here) and rigidifies the solid
    inner core (mu, Ks x1000, "let's make the inner core rigid" per the
    notebook). The Vs=0 layer (fluid outer core) is tagged ``ocean=1``; it
    lands at layer index 2.

    ``Moon_Weber.dat`` columns: radius [m], density [kg/m^3], Vp [m/s],
    Vs [m/s]. mu = rho*Vs^2 ; Ks = rho*(Vp^2 - 4/3 Vs^2).

    This function is deliberately byte-for-byte-identical arithmetic to the
    harness function above (see module docstring, "Why re-derive (not
    import) the harness construction", for why this is a re-derivation
    rather than a direct import).
    """
    dat = np.loadtxt(_MOON_DATA_FILE)
    r_m, rho, vp, vs = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3]

    mu = list(rho * vs**2)
    Ks = list(rho * (vp**2 - 4.0 / 3.0 * vs**2))
    mu[0] *= 1000.0  # notebook: "let's make the inner core rigid"
    Ks[0] *= 1000.0

    R0 = [50.0] + list(r_m / 1e3)
    rho0 = [8000.0] + list(rho)
    mu0 = [0.0] + mu
    Ks0 = [0.0] + Ks
    ocean = [1 if m_i == 0.0 else 0 for m_i in mu0]
    ocean[0] = 0  # layer 0 is the LOV3D core, not a subsurface ocean

    return make_interior_model(R0_km=R0, rho0=rho0, mu0=mu0, Ks0=Ks0, ocean=ocean)


def build_moon_model() -> InteriorModel:
    """Return the Moon reference :class:`~pylov3d.types.InteriorModel`.

    The Weber et al. (2011) 10-layer profile, exactly as built by the
    MATLAB-validated ocean harness (see module docstring). No free
    parameters, no fit -- this is intentionally the "as observed" profile;
    see "As-built residuals" above for how it compares against the bulk
    constraints, and :mod:`pylov3d.moon_mc` for the Monte Carlo stage that
    perturbs it to close that gap.
    """
    return _load_weber_profile()


# Layer arrays extracted once from the reference model (cheap: a 9-row
# ``np.loadtxt`` plus list arithmetic, no tidal solve) -- reused by
# pylov3d.moon_mc so that module never re-parses Moon_Weber.dat either.
_WEBER = build_moon_model()
_N_LAYERS = _WEBER.n_layers

#: Layer outer radii, core to surface [km] (fixed model geometry; index 2,
#: the fluid-core outer radius, is free in pylov3d.moon_mc).
LAYER_RADII_KM: tuple[float, ...] = tuple(
    float(x) for x in np.asarray(_WEBER.R0[:_N_LAYERS])
)
#: Layer densities [kg/m^3] as built (index 1 = inner core, index 2 = outer
#: core are free-parameter *base* values in pylov3d.moon_mc).
LAYER_RHO: tuple[float, ...] = tuple(float(x) for x in np.asarray(_WEBER.rho0[:_N_LAYERS]))
#: Layer shear moduli [Pa] as built (indices 3-8, the mantle, are
#: free-parameter *base* values in pylov3d.moon_mc; index 1's rigidified
#: value and index 9's crust value are fixed there).
LAYER_MU: tuple[float, ...] = tuple(float(x) for x in np.asarray(_WEBER.mu0[:_N_LAYERS]))
#: Layer bulk moduli [Pa] as built (fixed everywhere, including in
#: pylov3d.moon_mc -- no free parameter scales Ks).
LAYER_KS: tuple[float, ...] = tuple(float(x) for x in np.asarray(_WEBER.Ks0[:_N_LAYERS]))
#: Ocean (fluid, Vs=0) flags per layer; exactly one 1, at index 2.
LAYER_OCEAN: tuple[int, ...] = tuple(int(x) for x in np.asarray(_WEBER.ocean[:_N_LAYERS]))

#: Descriptive layer names, core to surface, for documentation/debugging
#: only (not used by any solver call). Index 0 = artificial numerically-
#: inert core; index 1 = physical solid inner core (rigidified); index 2 =
#: physical fluid outer core (ocean=1); index 3 = top of Weber et al.
#: (2011)'s partial-melt zone (480 +/- 15 km, see MOON
#: ["partial_melt_radius_weber_seismic"]); indices 4-8 = the remaining five
#: Weber mantle shells; index 9 = crust.
LAYER_NAMES: tuple[str, ...] = (
    "artificial_core",
    "inner_core",
    "outer_core",
    "partial_melt_zone",
    "mantle_2",
    "mantle_3",
    "mantle_4",
    "mantle_5",
    "mantle_6",
    "crust",
)

# Forcing period for the monthly (sidereal-month) tide: Td = 2360591.6 s
# (27.3216615 d -- the sidereal month; matches the digit string in the
# TASK-018 spec, 27.32166 d, to rounding). Purely elastic model (no eta
# anywhere), so -- exactly as for pylov3d.mars -- Td does not actually
# affect k2 (elastic Love numbers are frequency-independent); retained only
# because make_forcing requires a value, and for forward compatibility with
# an eventual anelastic (Andrade/Maxwell) extension where it would matter.
MOON_FORCING_TD = 2360591.6  # s

# Numerics matching the MATLAB-validated harness exactly (see module
# docstring): method="variable" (not "combination", unlike pylov3d.mars),
# Nrbase=50. Not independently re-tuned here; inherited as-is from the
# cross-validated harness configuration.
MOON_NUMERICS_METHOD = "variable"
MOON_NUMERICS_NRBASE = 50

# Uniform (no lateral variation) elastic k2 of the as-built reference
# model, computed via get_love(build_moon_model(), moon_forcing(),
# moon_numerics()) -- pinned here as documented provenance; the identical
# value is independently re-verified live (not just imported) by
# pylov3d/tests/test_moon.py's TestK2AnchorTieIn, and matches
# TestMoonOceanValidation.test_uniform_k2_matches_matlab's own reference
# (0.023159142178491576, MATLAB k2_Q) to ~2e-9 relative.
WEBER_K2_UNIFORM = 0.02315914222851756


def moon_forcing():
    """Forcing used for the reference model's k2 (n=2, m=0; matches the
    ocean-validation harness's n/m; frequency-irrelevant for this purely
    elastic model, see :data:`MOON_FORCING_TD` docstring above)."""
    return make_forcing(Td=MOON_FORCING_TD, n=2, m=0, F=1.0)


def moon_numerics(Nrbase: int = MOON_NUMERICS_NRBASE):
    """Numerics config matching the MATLAB-validated ocean harness
    (``method="variable"``, ``Nrbase=50`` by default)."""
    return make_numerics(
        n_layers=_N_LAYERS, method=MOON_NUMERICS_METHOD, Nrbase=Nrbase,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def moon_moi_factor(model: InteriorModel | None = None) -> float:
    """Return I/(M*R^2) of a Moon interior model (default: the as-built
    reference model, :func:`build_moon_model`).

    Uses :func:`pylov3d.forward.analytic_mass_moi` (shared with
    :mod:`pylov3d.mars` / :mod:`pylov3d.moon_mc`, not re-derived here), and
    the model's own surface radius (not :data:`MOON` ``["R"]``), so this
    diagnostic is self-consistent for any Moon-like profile, not just the
    Weber reference.
    """
    if model is None:
        model = build_moon_model()
    n = model.n_layers
    R_surface_m = float(np.asarray(model.R0[n - 1])) * 1e3
    M, I = analytic_mass_moi(model)
    return I / (M * R_surface_m**2)


def moon_mass(model: InteriorModel | None = None) -> float:
    """Return mass [kg] of a Moon interior model (default: the as-built
    reference model, :func:`build_moon_model`)."""
    if model is None:
        model = build_moon_model()
    M, _I = analytic_mass_moi(model)
    return M
