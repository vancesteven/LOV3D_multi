# Moon Reference Model and Monte Carlo Parameterization (TASK-018)

The Moon instantiation of the body-agnostic forward-model framework
(`pylov3d.forward`), following the Mars pattern (`pylov3d/mars.py` +
`pylov3d/mars_mc.py`, `docs/MARS_MODEL.md`). Implementation:
`pylov3d/moon.py` (reference model) + `pylov3d/moon_mc.py` (Monte Carlo
parameterization). Tests: `pylov3d/tests/test_moon.py`.

Unlike Mars, the Moon's reference model is **not** a fresh deterministic
fit: it reuses the 10-layer Weber et al. (2011) profile already built and
MATLAB-cross-validated by the Milestone 5 ocean-solver harness
(`pylov3d/tests/test_matlab_validation_ocean.py`) -- see "Design decision".

> **Revision notes (2 science-review rounds):** round 1 found citation
> errors, a framework bug (`core_radius_km` reading the wrong layer), and
> an under-constrained MC parameterization. Round 2 found an anelastic-bias
> claim that was wrong for the 4-parameter model and an unphysical
> `core_rho_scale` floor. All corrected below.

## Published bulk constraints

| Quantity | Value | Source |
|---|---|---|
| GM | 4902.80007 ± 0.00014 km^3/s^2 | Williams, J. G., Boggs, D. H., & Folkner, W. M. (2013), "DE430 Lunar Orbit, Physical Librations, and Surface Coordinates," JPL IOM 335-JW,DB,WF-20130722-016, Table 5 — the DE430 ephemeris GM, adopted in Williams et al. (2014) Table 1. |
| Mass M | GM/G with G = 6.6743e-11 (CODATA 2018) ≈ 7.3458e22 kg | derived; matches the pre-existing `pylov3d.bodies` catalog entry (id 31, "Moon") `Mass=7.3458e22` to 5 significant figures — an independent cross-check that predates and shares no code with this module. |
| Mean radius R | 1737.15 km | Neumann, G. A. (2013), LRO LOLA topographic data products, as adopted in Williams et al. (2014) Table 2; Smith et al. (2010), *GRL*, 37, L18204, gives a closely similar but distinct value, 1737.153 ± 0.010 km. |
| Whole-Moon mean MoI factor I/MR^2 (the constraint used) | 0.3931 ± 0.0002 | Konopliv, A. S., Binder, A. B., Hood, L. L., et al. (1998), "Improved gravity field of the Moon from Lunar Prospector," *Science*, 281, 1476-1480 (verbatim: "the average moment I/MR^2 = 0.3931 ± 0.0002"). |
| Tidal k2 (monthly, degree-2) | 0.02422 ± 0.00022 | Williams, J. G., Konopliv, A. S., Boggs, D. H., et al. (2014), "Lunar interior properties from the GRAIL mission," *JGR Planets*, 119, 1546-1578, Section 5: unweighted mean of two GRAIL-only k2 solutions, referenced to R = 1737.151 km (**not** combined LLR+GRAIL — DE430/LLR holds k2 fixed). |
| Fluid core radius | 380 ± 40 km | Garcia, R. F., Gagnepain-Beyneix, J., Chevrot, S., & Lognonné, P. (2011), "Very preliminary reference Moon model," *Physics of the Earth and Planetary Interiors*, 188, 96-113 (seismic), as attributed by Williams et al. (2014) Section 7. |
| Crust thickness (global mean) | 34-43 km; adopt 40 km | Wieczorek, M. A., et al. (2013), "The crust of the Moon as seen by GRAIL," *Science*, 339, 671-675. |

## Citation corrections (science review round 1)

A science review verified every citation in the first draft against the
primary sources. Several were wrong; all are corrected in this document and
in `pylov3d/moon.py`'s docstring/`MOON` dict comments. Recorded here so the
correction (not just the current state) is visible.

**GM.** The first draft attributed 4902.80007 km^3/s^2 to Konopliv et al.
(2013) GL0660B — wrong; that GRAIL-only solution independently gives a
close but distinct GM = 4902.80031 ± 0.00044 km^3/s^2. The digit string
actually used is the **DE430** ephemeris value (LLR-based): Williams,
Boggs & Folkner (2013), JPL IOM 335-JW,DB,WF-20130722-016, Table 5
(4902.80007 ± 0.00014 km^3/s^2), which Williams et al. (2014) Table 1
adopts directly. The task spec's original attribution to "Williams et al.
(2014)" was in fact closer to correct than the first draft's "correction"
to Konopliv et al. (2013) — an error introduced during this
implementation's own citation-verification pass, now reverted.

**GRAIL solution naming.** The first draft called Konopliv et al. (2013)'s
field "GRGM900C" — wrong. GRGM900C is a different, GSFC-led solution:
Lemoine, F. G., et al. (2014), "GRGM900C: A degree 900 lunar gravity model
from GRAIL primary and extended mission data," *GRL*, 41, 3382-3389.
Konopliv et al. (2013)'s own JPL solutions are GL0420A (primary mission)
and GL0660B (primary + extended mission).

**Whole-Moon vs. solid-Moon MoI (the important one).** The first draft
mislabeled 0.393112 as a *polar* moment C/MR^2 needing a Mars-style
J2-based mean-moment correction. That framing was entirely wrong: 0.393112
is not polar at all — it is Williams et al. (2014) Section 5's normalized
**solid-Moon** mean moment of inertia (0.392728 ± 0.000012 as published,
re-expressed at R = 1737.151 km). "Solid-Moon" means the fluid core's own
moment-of-inertia contribution is handled separately in that quantity,
because LLR alone does not resolve the fluid-core/whole-Moon polar-moment
ratio C_f/C — so it is **not directly comparable** to what
`pylov3d.forward.analytic_mass_moi` computes (a plain shell-sum over every
layer, fluid core included). The constraint actually used, I/MR^2 = 0.3931
± 0.0002, is instead Konopliv et al. (1998)'s explicitly whole-Moon
"average moment" — the physically correct target for a shell-sum that
includes the fluid core. This is kept as `pylov3d.moon.MOON
["MoI_solid_moon_factor"]` for context only, not used in any fit.

**Tidal k2.** The first draft called 0.02422 ± 0.00022 a "combined LLR +
GRAIL" solution — wrong. It is Williams et al. (2014) Section 5's
unweighted mean of two **GRAIL-only** k2 solutions, referenced to
R = 1737.151 km (DE430/LLR holds k2 fixed rather than solving for it, so
LLR contributes no independent k2 information). A second referencing of
the same solutions exists at R = 1738 km, k2 = 0.02416 ± 0.00022; not used
here because 1737.151 km is closer to this model's own 1737.1 km surface
(see "Reference-radius convention" below). The first draft's
"k2_grail_only_sigma = 0.000414" (attributed to Konopliv et al. 2013) could
not be traced to any source and has been removed; replaced with a
traceable comparison value, Konopliv et al. (2013) GL0660B k2 = 0.02405 ±
0.00018 (R = 1738 km), not used as a constraint.

**Fluid core radius.** The first draft attributed 380 ± 40 km to Williams
et al. (2014)'s own LLR tidal-dissipation solution — wrong. Williams et al.
(2014) Section 7 itself attributes this value to Garcia et al. (2011)
(seismic). Consequence: **both** 380 ± 40 km (Garcia et al. 2011) and the
as-built profile's own 330 km outer-core boundary (Weber et al. 2011) are
seismic determinations from different published analyses — not two
independent data classes (e.g. "seismic vs. LLR/tidal") — see
"Identifiability" below for what this means for the constraint's
evidentiary weight. Weber et al. (2011) also publishes error bars on its
layer boundaries not carried in `Moon_Weber.dat` itself: inner core
240 ± 10 km, outer (fluid) core 330 ± 20 km, and the top of a partial-melt
zone at 480 ± 15 km (`pylov3d.moon.LAYER_RADII_KM[3]`, now named
`"partial_melt_zone"` in `LAYER_NAMES`).

**Mean radius (nit).** 1737.15 km is cited to Neumann (2013) LRO LOLA
products (as adopted in Williams et al. 2014 Table 2), not Smith et al.
(2010) directly — Smith et al. (2010) gives a closely similar but distinct
1737.153 ± 0.010 km.

## Reference-radius convention

The MoI and k2 constraints above are each published at a specific
reference radius. This model's own surface radius is 1737.1 km
(`pylov3d.moon.LAYER_RADII_KM[-1]`, the Weber profile's own outermost
boundary). The adopted k2 (0.02422 ± 0.00022) is chosen because it is
referenced to R = 1737.151 km specifically — the closer of the two
published referencings to this model's own surface — not R = 1738 km
(k2 = 0.02416 ± 0.00022 there, ~0.06σ different, not used).
`test_moon.py` checks `MOON["R"]` against the profile's own surface radius
to < 0.1% to guard against this silently drifting apart in a future edit.

## Design decision: the Weber-structure reference model

TASK-018 directs against a fresh 4-layer deterministic fit (the Mars
pattern) for the Moon, in favor of reusing the 10-layer Weber et al. (2011)
profile already built and MATLAB-cross-validated by the Milestone 5 ocean
harness (`_build_weber_moon_model` in
`pylov3d/tests/test_matlab_validation_ocean.py`, backing
`TestMoonOceanValidation.test_uniform_k2_matches_matlab`: measured agreement
with MATLAB's `k2_Q` to ~2e-9 relative). That construction:

1. Prepends an **artificial 50 km, 8000 kg/m^3, numerically-inert layer 0**
   (LOV3D's boundary-condition machinery always treats layer 0 as the core
   and never integrates through it — the same mechanism as Mars's liquid
   core, `pylov3d/boundary_conditions.py`). This buffer exists because a
   fluid layer sitting *directly* on LOV3D's core boundary condition is
   unsupported (in MATLAB and here); it decouples the boundary-condition
   machinery from the physically real fluid outer core.
2. Rigidifies the physical **solid inner core** (Weber's own layer 0,
   `r=240` km, published error ±10 km) by ×1000 on both mu and Ks (the
   notebook's own annotation: "let's make the inner core rigid") — now
   LOV3D layer index 1.
3. Places the physical **fluid outer core** (Weber's layer 1, `r=330` km,
   published error ±20 km, Vs=0) at LOV3D layer index 2, tagged `ocean=1`.
4. The remaining six Weber mantle shells (`r=480` km, the published top of
   a partial-melt zone at ±15 km, through `1703.1` km) and one crust shell
   (`r=1737.1` km) follow unmodified, at indices 3-9.

`pylov3d.moon.build_moon_model()` returns exactly this profile — no fit, no
free parameters. The gap between this "as observed" profile and the bulk
constraints above (next section) is real, and is exactly what
`pylov3d.moon_mc` (below) exists to close.

### Why `pylov3d.moon` re-derives the harness construction, and a data dependency this creates

`pylov3d.moon` reads `data/tests/moon/Moon_Weber.dat` (the raw Weber et al.
2011 profile) directly, at import time. That is a genuine runtime data
dependency on a file under `data/tests/` — a path name that suggests
test-only data. It stays there because the file is the actual published
Weber et al. (2011) profile, not synthetic fixture data, and relocating it
is a larger repository reorganization out of scope for TASK-018; this is
called out explicitly as an accepted, if slightly awkward, dependency for a
research package, rather than left implicit.

Separately, TASK-018 asks that `pylov3d.moon` not duplicate
`_build_weber_moon_model`'s parsing logic. Importing that function directly
from `pylov3d.tests.test_matlab_validation_ocean` was considered and
rejected for a narrower reason: that test module does an unconditional
module-level `import pytest`, so importing it from `pylov3d.moon` would
additionally make **pytest** — currently only a `pylov3d[test]` optional
dependency, not a runtime one — a hard import-time dependency of this
module. `pylov3d.moon._load_weber_profile` therefore re-derives the
identical construction (same magic numbers: 50 km / 8000 kg/m^3 artificial
core, ×1000 inner-core rigidification, ocean flag at index 2), and
`test_moon.py::TestReferenceModelMatchesOceanHarness.
test_reference_model_matches_ocean_harness` imports the real harness
function (pytest is already a dependency in the test environment) and
asserts element-by-element equality against `build_moon_model()`'s output —
an executable guarantee against drift, in place of a static import.

## As-built residuals (why the Monte Carlo stage exists)

Computed via `build_moon_model()` + `pylov3d.forward.analytic_mass_moi` +
`pylov3d.love.get_love` (elastic, as-is; no free parameters; numerics
`method="variable"`, `Nrbase=50`, matching the harness):

| Observable | As-built (Weber profile) | Constraint | Residual |
|---|---|---|---|
| Mass | 7.31329e22 kg | 7.34579e22 kg | −0.442% (−4.42e-3 relative) |
| Mean MoI I/MR^2 | 0.392361 | 0.3931 ± 0.0002 | −7.39e-4 (−3.7σ) |
| Tidal k2 (elastic) | 0.0231591 | 0.02422 ± 0.00022 | −1.061e-3 (−4.8σ) |
| Fluid-core radius | 330 km | 380 ± 40 km | −50 km (−1.25σ) |

**Mass residual: a digitization artifact, not a data conflict.** The
as-built model's mean density is 3330.8 kg/m^3 against the observed
3345.3 kg/m^3 (at GM/G and R = 1737.15 km). `Moon_Weber.dat`'s densities
are published to only 2-3 significant figures, and its six mantle shells
carry only two distinct rounded values, 3400 kg/m^3 (four shells) and
3220 kg/m^3 (two shells): Weber et al. (2011) published core densities from
their own seismic inversion, but the mantle values were inherited from
earlier reference models (Lognonné et al. 2003, Gagnepain-Beyneix et al.
2006) that were never themselves fit to reproduce the Moon's total mass.
The −0.44% residual is this rounding/inheritance chain, with no published
uncertainty budget covering it — not evidence the Moon's mass is in doubt
(GM itself is known to ~3e-8 relative). This is why `pylov3d.moon_mc` adds
a `mantle_rho_scale` free parameter (below) rather than leaning harder on
the core.

**k2 residual: consistent with, but not diagnostic of, anelasticity.** The
constraint k2 = 0.02422 is the Moon's *observed* value (anelastic), while
the profile above is evaluated purely elastically — see "Anelastic bias"
below for the fuller picture, including the literature-predicted
enhancement size and the competing (non-rheological) explanations for the
same gap.

These non-trivial, multi-sigma residuals are the motivation for
`pylov3d.moon_mc`: a small free-parameter perturbation of this same Weber
structure, fit by Bayesian inference against these same four constraints,
rather than discarding the seismically-anchored profile for an
unconstrained new one.

The uniform (no lateral variation) elastic k2 of the as-built model,
`WEBER_K2_UNIFORM = 0.02315914222851756`, is pinned in `pylov3d.moon` and
matches `TestMoonOceanValidation.test_uniform_k2_matches_matlab`'s own
MATLAB reference (`k2_Q = 0.023159142178491576`) to ~2e-9 relative — this
module's k2 anchor **is** the ocean-harness's own MATLAB-validated anchor,
not an independently computed value that happens to be close.

## Framework fix: `BodyParameterization.core_layer_index` (D8)

Science review caught a real bug: `pylov3d.forward.compute_observables`'s
`"core_radius_km"` observable was hardcoded to read `model.R0[0]` —
correct for `pylov3d.mars_mc` (the physically free liquid core *is* layer 0
of that 4-layer model) but **wrong** for the Moon: layer 0 here is the
fixed 50 km artificial stub, and the free fluid-core radius is layer index
2. An earlier draft of `pylov3d.moon_mc` worked around this with a
hand-written bypass (a direct Gaussian applied to `theta` rather than going
through `compute_observables`). That workaround is now removed in favor of
a proper framework fix, applied to `pylov3d.forward` itself:

- `BodyParameterization` gained a `core_layer_index: int = 0` field
  (default preserves Mars's existing behavior exactly).
- `compute_observables(..., core_layer_index=0)` reads
  `model.R0[core_layer_index]` for `"core_radius_km"` instead of a
  hardcoded `model.R0[0]`.
- `make_log_posterior` reads `parameterization.core_layer_index` and
  forwards it to `compute_observables` automatically.

`pylov3d.moon_mc.MOON_PARAMETERIZATION` now sets `core_layer_index=2`, and
`moon_log_posterior` is a plain call to `pylov3d.forward.make_log_posterior`
— no custom bypass code remains. **Mars is unaffected**: verified three
ways — (1) `test_mars.py` (39 tests) and `test_forward.py` (3 tests) pass
unchanged; (2) `mars_log_posterior(Nrbase=15)` evaluated at the TASK-011
point fit gives the bit-identical value `-42.55837369785579` before and
after this change (compared via `git stash` of `forward.py` alone); (3) a
dedicated regression test
(`test_moon.py::TestD8CoreLayerIndexRegression`) demonstrates the bug this
fix closes directly: at theta = (1, 1, 380, 1) — `R_fluid_core` set to
exactly the constraint's own center — the pre-fix code (equivalent to
calling `compute_observables` with its default `core_layer_index=0`) would
have returned a *constant* `core_radius_km = 50.0` regardless of
`R_fluid_core`, i.e. a constant `(50 - 380) / 40 = -8.25σ` core-likelihood
penalty baked into every single posterior evaluation. The fixed code
correctly returns `core_radius_km = 380.0` there (a ~0σ residual).

## Monte Carlo parameterization (`pylov3d.moon_mc`)

### Free parameters (4) and constraints (4)

| Name | Bounds | Meaning |
|---|---|---|
| `core_rho_scale` | [0.88, 1.2] | Shared multiplicative density scale on **both** the solid inner core (layer 1, base 8000 kg/m^3) and the fluid outer core (layer 2, base 5100 kg/m^3) — a single `Scaled` parameter across two layers, preserving the Weber profile's inner/outer core density contrast. **The lower bound is a physical density floor, not a fitted range**: 0.88 keeps the fluid outer core ≥ 4488 kg/m^3 (s_min = 4500/5100 = 0.8824, rounded to 0.88; Weber et al. 2011 adopts 5100 kg/m^3 for this layer, and liquid Fe-S at lunar core pressures is plausibly 5-7 g/cm^3, so the floor allows ~-12% before leaving that band). The MAP is **expected** to rail here — see "Mass-closure parameter" below — documented rather than hidden. |
| `mu_scale` | [0.3, 3.0] | Shared shear-modulus scale on the six Weber mantle layers (indices 3-8). Same bounds as `pylov3d.mars_mc`'s `mu_scale`, reused directly. The rigidified inner core (layer 1) and the crust (layer 9) are excluded. |
| `R_fluid_core` | [300, 460] km | Fluid-core (layer 2) outer radius, ±2σ around the Garcia et al. (2011) 380 ± 40 km seismic value (not the Weber et al. 2011 330 km as-built value). Strictly inside the fixed neighboring radii (240 km, 480 km). |
| `mantle_rho_scale` | [0.95, 1.05] | Shared multiplicative density scale on the same six mantle layers `mu_scale` scales — a **pragmatic mass-closure parameter**, see below, NOT a claim about lunar mantle density. |

Constraints: mass, moi_mean, k2, **core_radius_km** (now correctly read
from layer 2 via `core_layer_index`, see "Framework fix" above). All other
layer scalars (layer-0 stub, layer 1's rigidified mu/Ks, crust mu, every
layer's Ks, every fixed radius) are FIXED at the values in
`pylov3d.moon.LAYER_RADII_KM` / `LAYER_RHO` / `LAYER_MU` / `LAYER_KS`.

### Mass-closure parameter: why `mantle_rho_scale` was added, and exact determination (S3, M3)

With 4 free parameters against 4 constraints, this system is formally
**exactly determined**: absent any bound restricting the search, a
zero-residual fit generically exists and by itself carries **no
goodness-of-fit information** about the input data — the round-1 mass
tension (see "As-built residuals" above) is *absorbed* by
`mantle_rho_scale` below, not independently *resolved*, and a perfect fit
would not mean the Weber profile's mantle densities were confirmed
correct. In practice, `core_rho_scale`'s lower bound is a genuine physical
floor (see table above), so the fit shipped here does **not** reach that
unconstrained zero-residual optimum, as the numbers below show.

A 3-parameter version of this module (`core_rho_scale`, `mu_scale`,
`R_fluid_core` only) is structurally unable to reach the mass constraint:
the core complex (layers 1-2) is only ~1.3% of the Moon's total mass in
this profile. The best 3-parameter MAP found (L-BFGS-B, several starting
points; current `core_rho_scale` floor 0.88) sits at `core_rho_scale = 0.880`
(railed at the floor), `mu_scale = 0.968`, `R_fluid_core = 375.8` km,
log-posterior `-51.86`, and **still misses the mass constraint by −4.9σ**
-- a mantle-level correction, not a core-level one, is what the mass
residual calls for.

Adding `mantle_rho_scale` helps substantially but, with the floor in place,
does not fully close the gap. The best 4-parameter MAP found:

```
core_rho_scale = 0.880 (railed at the floor),  mu_scale = 0.9655,
R_fluid_core = 326.9 km,  mantle_rho_scale = 1.00638 (+0.64%)
```

log-posterior improves from **−51.86 → −37.50** (mass and k2 residuals
both drop under 0.05σ), but **moi_mean remains off by ~−0.95σ and
core_radius_km by ~−1.33σ (326.9 vs. 380 ± 40 km) — this is NOT a
zero-residual fit**: the density floor stops the search before it reaches
the fully-relaxed optimum described in "Exact determination" above, on
purpose.

This parameter is deliberately framed as a **pragmatic mass-closure knob**,
not physical inference about lunar mantle density: `Moon_Weber.dat`'s six
mantle layers carry only two distinct rounded density values (3400 and
3220 kg/m^3), so a ~+0.6% correction is absorbing a known
rounding/inheritance artifact in the input data, not resolving a genuine
structural question about the Moon's mantle. **The cleaner future fix** is
a mass-consistent published profile — e.g. Garcia et al. (2011)'s own
VPREMOON, which (unlike the `Moon_Weber.dat` values used here) was itself
fit to reproduce total mass — in place of this scale-factor patch.

### Identifiability

A finite-difference, relative-normalized Jacobian of all 4 observables
(mass, moi_mean, k2, core_radius_km) w.r.t. all 4 free parameters, at the
as-built theta0, is **full rank** (rank 4 of 4), singular values
`[2.253, 0.999, 0.405, 0.0120]`, condition number **≈187** — a large
improvement over an earlier 3-parameter draft's 3×3 Jacobian (mass,
moi_mean, k2 only), condition number ≈6.1e3 (a near-degenerate direction
mixing `core_rho_scale`/`R_fluid_core`): `mantle_rho_scale` gives mass its
own largely-independent handle, breaking the near-degeneracy. Not
perfectly conditioned (≈187 >> 1), so `core_radius_km`'s constraint remains
meaningful — see `test_moon.py::TestIdentifiability` (condition number
guarded within 3× of measured; full rank asserted directly). Caveat: the
as-built profile's own 330 km outer-core boundary (Weber et al. 2011) and
the constraint center (Garcia et al. 2011, 380 ± 40 km) are **both
seismic** determinations from different analyses, not independent data
classes — the constraint adds real information the bulk observables alone
only weakly pin down, not a new *kind* of data.

### Anelastic bias: where it actually shows up (S2, corrected — was wrong for the 4-parameter model)

This Monte Carlo stage fits a purely **elastic** k2 to the Moon's
*observed* (anelastic) k2 = 0.02422; the as-built elastic k2 falls short by
4.6%. **An earlier draft of this document claimed the resulting bias shows
up as a systematically enlarged `R_fluid_core`, quoting a single
"367-388 km" range that folded together 3- and 4-parameter results — this
is wrong for the 4-parameter model and has been corrected.** With
`mantle_rho_scale` absorbing mass and `mu_scale` alone already closing the
k2 gap to <1e-4σ (see "Mass-closure parameter" above), the anelastic bias
does **not** migrate to `R_fluid_core` in the 4-parameter fit — it stays in
`mu_scale` (and, for the mass residual specifically, `mantle_rho_scale`),
exactly where each parameter was added to absorb it. `R_fluid_core`'s
behavior is a separate story, and depends on whether the physical density
floor is active:

- **Without the floor** (an intermediate, unphysical [0.75, 1.25]
  `core_rho_scale` range explored during development, since superseded):
  nothing pulls on `R_fluid_core` at all — its marginal is essentially just
  the (2σ-truncated) Garcia prior itself. The MAP sat at
  `R_fluid_core = 380.000` km, exactly the prior mean; a properly-resolved
  posterior (`n_active=64`, `n_effective=128`, `n_total=512`) gave
  **363.7 ± 33.2 km**, consistent with the truncated-prior width — i.e. the
  bulk observables carried ~no information about `R_fluid_core` once
  `core_rho_scale` was free to reach an unphysically low value.
- **With the floor** (0.88, what this module actually ships): `core_rho_scale`
  can no longer drop low enough to fully absorb the mass residual, so a
  residual mass/MoI pull reappears — but pulling `R_fluid_core` **down**,
  toward the as-built Weber value (326.9 km at the MAP, see "Mass-closure
  parameter"), **not up toward Garcia's 380 km**. This is a **mass/MoI
  mechanism**, a direct consequence of the physical density floor — not an
  anelastic one (k2's own residual stays ~0σ throughout, absorbed cleanly
  by `mu_scale`).

How large should a genuine elastic-to-anelastic k2 bias be, in principle —
context, not what actually drives `R_fluid_core` here? Williams et al. (2014)
Section 6 gives a lunar monthly Q = 38 ± 4; with an Andrade rheology
(α ~ 0.3-0.35, commonly adopted for silicate mantles), a Q this low
predicts roughly a 4-5% elastic-to-anelastic k2 enhancement at the monthly
period — close to the 4.6% gap measured here. The broader literature range
is wider, 4-10%: Williams et al. (2014) Table 10's zero-period values imply
5-9%; Garcia et al. (2019) report elastic k2 = 0.02277 ± 0.00058, a 6.4%
gap to the same target; Nimmo et al. (2012) estimate ~10%. This module's
4.6% sits at the low end — consistent with, but not strong evidence for,
any particular anelastic model.

**This gap is consistent-with, not diagnostic-of, anelasticity**, for two
further reasons: (1) the as-built elastic k2 is specific to the
Weber-family profile — a Garcia-et-al.-(2011)-family profile (the source of
the core-radius constraint itself) gives elastic k2 ~ 0.0223, a different
starting point for the same comparison; (2) purely structural changes with
no rheology involved — a low-velocity zone (LVZ) near the base of the
mantle, or simply a larger fluid-core radius (both discussed by
Williams et al. 2014 Section 7) — can span a comparable fraction of the
same 4.6% gap. Unlike an earlier draft's claim, none of this manifests as
an `R_fluid_core` bias in the shipped (floored) 4-parameter fit — that
parameter's own behavior is governed by the mass/MoI mechanism above, not
by k2 at all.

### Mass-sigma rationale (D9)

The mass constraint's 0.1% sigma is **not** "dominated by G's uncertainty"
as an earlier draft claimed (and as `pylov3d.mars_mc`'s comment also
incorrectly claimed, now fixed identically in both modules): CODATA 2018's
G has relative uncertainty ~2.2e-5, and GM here is known to ~3e-8 relative
— both far tighter than 0.1%. The 0.1% is instead a deliberate **modelling-
error allowance**: this stage-1 model's own structural/rounding
uncertainty swamps the underlying data's actual precision, so mass barely
constrains the density-scale parameters relative to the much tighter
MoI/k2 constraints — intentional, and now stated correctly in both
`pylov3d.moon_mc` and `pylov3d.mars_mc`.

### Posterior smoke run — a smoke-test artifact, not a reference posterior

A micro `pocomc` run (`n_active=8`, `n_effective=8`, `n_total=32`,
`Nrbase=15`, `random_state=0`, 4 free parameters, current `core_rho_scale`
floor [0.88, 1.2]; `pylov3d/tests/test_moon.py::TestPocoMCSmoke`, marked
`slow`, mirroring `test_forward.py::TestPocoMCSmoke` for Mars) recovers a
posterior median

```
core_rho_scale ≈ 0.960,  mu_scale ≈ 0.966,
R_fluid_core ≈ 317.3 km,  mantle_rho_scale ≈ 1.0055
```

whose k2 (0.02410) is within 0.54σ of the target — comfortably inside the
test's own 3σ assertion. **These medians are a smoke-test artifact, not a
reference posterior**: with only 8 active particles and 32 total draws in
4-D this run is badly under-resolved (the tell: its spread is narrower
than the prior, which an under-sampled SMC run systematically produces) —
it exists only to check `moon_log_posterior` + `pocomc.Sampler` wire
together correctly and finish in seconds.

**The reference posterior (converged, under the shipped floored bounds)**
is the TASK-019 production run (`scripts/moon_pocomc.py`: `n_active=64`,
`n_effective=128`, `Nrbase=50`, dynamic termination; 4105 samples, Kish
ESS = 4087.7; chain archived as
`docs/figures/proposal/moon_posterior_chain.npz` with pairplot alongside).
Weighted medians ±1σ:

| parameter | median | 16%/84% |
|---|---|---|
| core_rho_scale | 0.900 | 0.886 / 0.928 (lower tail touches the 0.88 floor) |
| mu_scale | 0.965 | 0.955 / 0.975 |
| **R_fluid_core** | **321 km** | 308 / 340 km |
| mantle_rho_scale | 1.006 | 1.005 / 1.007 |

Median-model observables satisfy all four constraints (mass 7.344e22 kg,
moi_mean 0.3929, core_radius 321.2 km, k2 0.02419). As the module
docstring predicts, `R_fluid_core` lands near the as-built Weber ~327 km
— pulled DOWN from the Garcia prior mean by the residual mass/MoI
mechanism under the density floor (not by anelasticity). The historical
pre-floor run (363.7 ± 33.2 km, unphysical [0.75, 1.25] core-density
range) is superseded and retained above only as review context.

## Relationship to the Milestone 5 ocean-harness validation

This module's reference model **is** the model MATLAB-validated in
Milestone 5: `pylov3d.moon.build_moon_model()` produces the identical
10-layer profile as `_build_weber_moon_model()` in
`test_matlab_validation_ocean.py` (enforced by
`test_moon.py::test_reference_model_matches_ocean_harness`), and
`pylov3d.moon.WEBER_K2_UNIFORM` is the same uniform k2 already cross-checked
against MATLAB's `k2_Q` to ~2e-9 relative in that harness. TASK-018 does not
extend the coupled-ocean (lateral-variation) solve path — `pylov3d.moon_mc`
only perturbs the same 1D radial structure the ocean harness's *uniform*
(no-lateral-variation) case already validates; the harness's own
lateral-variation coupled-solver tests (`TestMoonCoupledOceanValidation`)
are unaffected by, and independent of, this module.

## Tests

`pylov3d/tests/test_moon.py` (31 tests: 30 fast + 1 `@pytest.mark.slow`
pocomc smoke), against the full suite run with
`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q -m ""` (full lane;
the default lane excludes `slow`-marked tests, see `pylov3d/pyproject.toml`
`addopts`):

1. **Reference model matches the ocean harness** element-by-element, and
   `pylov3d.moon`'s exported `LAYER_*` tuples match the built model.
2. **k2 anchor tie-in**: `WEBER_K2_UNIFORM` matches the harness's loaded
   MATLAB reference to < 1e-9 absolute, and is reproduced live.
3. **As-built residuals** pinned to documented values (1e-6 relative),
   explicitly asserted non-zero (this is not a fit).
4. Love-number sanity, crust thickness within 34-43 km, and surface radius
   vs. `MOON["R"]` agreement < 0.1% (S4).
5. **Parameterization**: bounds/free-params (4 params, `core_rho_scale`
   floor 0.88), `core_layer_index == 2`, the as-built theta reproduces
   `build_moon_model()`, `R_fluid_core` resolves to layer 2, and
   `mantle_rho_scale` scales only layers 3-8.
6. **Constraint values** match `pylov3d.moon.MOON` exactly (no re-typing).
7. **D8 regression** (`TestD8CoreLayerIndexRegression`): the pre-fix bug is
   reproduced (constant −8.25σ core penalty at theta=(1,1,380,1)) and shown
   absent in the fixed `moon_log_posterior`.
8. **Identifiability**: the 4×4 Jacobian is full rank, condition number
   guarded within 3× of the measured ≈187.
9. **`moon_log_posterior`**: finite at theta0, `-inf` outside bounds,
   `core_radius_km` usable via `which` (post-D8), and the pinned MAP beats
   theta0's log-posterior with mass/k2 near-exact but moi_mean/
   core_radius_km carrying real, floor-limited residuals (M3).
10. **Micro pocomc smoke** (`@pytest.mark.slow`): completes, posterior
    finite, median k2 within 3σ (explicitly labeled a smoke-test artifact,
    not a reference posterior — see "Posterior smoke run" above).

Also affected by this round: `pylov3d/forward.py` (D8:
`BodyParameterization.core_layer_index`, `compute_observables`'s
`core_layer_index` kwarg, `make_log_posterior` forwarding it — Mars
unaffected, see "Framework fix" above) and `pylov3d/mars_mc.py` (D9: the
mass-sigma rationale comment fix, text-only, no behavior change).

## Anelasticity (TASK-025a)

Stage a of TASK-025: validate pylov3d's viscoelastic solver path and
produce forward anelastic Love-number calculations for the as-built Moon
model. New modules `pylov3d/anelastic.py` (shared machinery + Mars) and
`pylov3d/anelastic_moon.py` (Moon-specific -- split into two files to
keep both under this repo's 500-line-per-file convention); tests
`pylov3d/tests/test_anelastic.py` (30 tests: 19 fast + 11 `slow`,
covering both modules). No `pylov3d` solver module was modified. Stage b
(threading anelasticity into `pylov3d.moon_mc`'s Bayesian fit) is
separate, future work; every function in `pylov3d.anelastic_moon` (and
`pylov3d.anelastic`) takes a plain scalar parameter and returns a
complex Love number, the call shape a stage-b log-likelihood would need.

### What the solver actually supports

`pylov3d` implements **Maxwell** viscoelasticity only, via per-layer
`eta0` → `pylov3d.rheology.compute_complex_rheology` (`muC = mu / (1 -
i/MaxTime)`). A repo-wide check (`grep -rln -i andrade` across both
`pylov3d/` and the vendored MATLAB source under `src/`, now a live guard
in `TestSolverCapabilityAudit`) finds **no Andrade implementation
anywhere in either codebase checked into this repository** — only prose
noting it as future work (this module's own docstring, plus pre-existing
notes in `pylov3d/moon.py` and `pylov3d/moon_mc.py`). Where this
document needs Andrade, the numbers below come from **PyALMA3** (`alma`,
optional dependency), called directly as an external reference — never
routed through `pylov3d.love.get_love`.

Building the validation harness surfaced two more load-bearing facts
about PyALMA3 itself (v1.0.1, as installed): (1) its propagator is
**incompressible** — `alma.build_model` takes no bulk-modulus argument at
all, and neither `direct_matrix` nor `complex_rigidity` in
`alma/__init__.py` reference one; comparing it against pylov3d's
realistically compressible Moon model at face value disagrees by several
percent on Re(k2), a compressibility artifact, not a rheology bug — fixed
by driving pylov3d's own `Ks0` to the incompressible limit
(`1e7 * mu0_surface`) for the comparison only, which recovers < 1e-4
relative agreement (below); (2) it supports a **fluid layer only at the
center** (`complex_rigidity` has no branch for rheology code 0 at all
except the special-cased core boundary condition), so it cannot represent
the Weber profile's internal ocean (the fluid outer core, layer index 2,
sandwiched between the solid inner core and the solid mantle) — exactly
the feature pylov3d's own dedicated ocean/coupled solver (TASK-005/006/007)
was built to add.

### Validation

**Maxwell, real structure vs. simplified structure (internal
cross-check).** Because of finding (2) above, the real 10-layer Weber
structure cannot be given to PyALMA3 directly. `moon_simplified_body()`
merges the three innermost layers (artificial core + solid inner core +
fluid outer core, 0-330 km) into a single fluid layer (mean density —
combined mass over combined volume of the three, 6215.55 kg/m³), keeping
layers 3-9 (partial melt zone through crust) exactly as built — a
fluid-core + viscoelastic-mantle + elastic-crust body PyALMA3 can
represent, at Moon-relevant radii/densities/rigidities and the draconic-
month forcing period (below). `pylov3d`'s own Maxwell solver, run on
this simplified body, matches its gap-closing viscosity and implied Q to
the *real*-structure result (next section) to 5% / 10% respectively
(`TestSimplifiedBodyCrossCheck`) — the simplification does not distort
the rheology being tested.

**Maxwell, pylov3d vs. PyALMA3 (both incompressible-limit).** On the
simplified body, at `Td` = the draconic month (below) and `eta_mantle`
∈ {1e15, 1e17, 1e19} Pa s (`TestMoonMaxwellPyALMA3Validation`):

| eta [Pa s] | pylov3d k2 | PyALMA3 k2 | rel. diff Re | rel. diff Im |
|---|---|---|---|---|
| 1e15 | 0.244279 − 0.428312i | 0.244264 − 0.428300i | 6.1e-5 | 3.0e-5 |
| 1e17 | 0.022946 − 0.005495i | 0.022945 − 0.005495i | 4.4e-5 | 4.4e-5 |
| 1e19 | 0.022908 − 5.497e-5i | 0.022907 − 5.496e-5i | 4.4e-5 | 4.3e-5 |

All within the test's stated 1e-3 (0.1%) tolerance — the same tolerance
the pre-existing toy-body benchmark (`test_benchmark_pyalma3.py`) already
uses — by more than an order of magnitude. This is the deliverable-1
validation: the Maxwell complex-modulus mechanics agree with an
independent reference implementation on a Moon-relevant body at the
Moon-relevant forcing period. (These numbers were recomputed for this
revision after fixing the forcing-period error described below — see
"Forcing-period provenance"; an earlier draft of this table used the
wrong period and is superseded.)

**Andrade: validated only where a comparison is possible at all.**
Since pylov3d has no Andrade path, no pylov3d-vs-PyALMA3 agreement check
exists for it. `TestAndradeExternalSanity` instead anchors PyALMA3's
Andrade branch to the one place a comparison *is* possible — the elastic
limit (`eta → ∞`), where Andrade's complex modulus is real by
construction and must equal the incompressible-elastic k2 pylov3d
computes independently (matches to 1e-3 relative) — plus sign-convention
and forward-response sanity checks. The Andrade numbers below (Q vs.
alpha) are therefore reported as an **external-tool estimate**, not a
pylov3d-validated result, exactly as the task calls for ("do not fake an
Andrade capability").

### Forcing-period provenance

**Correction (this revision): the period below was previously wrong.**
An earlier draft of this section closed the 4.6% elastic k2 gap
(`WEBER_K2_UNIFORM` = 0.023159 vs. observed `MOON["k2"]` = 0.02422 ±
0.00022) at the **anomalistic month** (perigee-to-perigee), 27.55455
days. That is not the period Williams & Boggs (2015) report Q = 38 ± 4
at. The correct period is the **draconic (nodical) month** — the period
of the Moon's argument of latitude F (node-crossing period) — 27.212
days = 2,351,116.8 s (`pylov3d.anelastic_moon.MOON_DRACONIC_MONTH_TD`,
renamed from `MOON_ANOMALISTIC_MONTH_TD`) — still **not** the sidereal
month (27.32166 d) `pylov3d.moon.MOON_FORCING_TD` uses elsewhere in this
repo for the (frequency-irrelevant, purely elastic) k2 anchor. Every
Moon anelastic number in this section was recomputed under the corrected
period; see "Headline" below for the new values.

**Williams, J. G., & Boggs, D. H. (2015), "Tides on the Moon: Theory and
determination of dissipation," *JGR Planets*, 120(4), 689-724,
doi:10.1002/2014JE004755**, report the lunar monthly tidal quality
factor **Q = 38 ± 4** at their reference period Pref = 27.212 days. Per
web search of the primary source in this session (the paper is
paywalled; the following is a paraphrase of what the search returned
from its own text, not a verbatim quotation retrieved directly), Pref is
described there as the period of the largest latitude libration, due to
the tilt of the equator plane to the orbit plane — i.e. the draconic
month, not the anomalistic month. This is corroborated by **Williams,
J. G., Konopliv, A. S., Boggs, D. H., et al. (2014)**, *JGR Planets*,
119, 1546-1578 (the GRAIL interior-properties paper already cited
throughout this document for k2 and the fluid-core radius), which (same
caveat: paraphrased from a web search of the paywalled primary text, not
a direct verbatim retrieval) writes its monthly (k2/Q)_F relation over a
27.212-day period and defines "P_F" in its own notation table as the
27.212-day period of argument of latitude. The 27.55455-day anomalistic
period does appear in this literature, but attached to the
ΔJ2/ΔC22/ΔS22 tide amplitude rather than to the Q = 38 ± 4 value used
here — verified verbatim in **Williams et al. (2014)** ("the 27.555 day
anomalistic period of ΔJ2, ΔC22, and ΔS22"); an earlier draft of this
paragraph attributed that statement to Williams & Boggs (2015) instead,
which the re-verification pass could not confirm. Two further
independent corroborations of the 27.212-day reference period were
retrieved directly: **Briaud, A., et al. (2023)**, "Constraints on the
lunar core viscosity from tidal deformation," *Icarus* (arXiv:2301.04035),
states that per Williams & Boggs (2015) "the major periods of interest
... are F = 27.212 days and l_0 = 365.260 days"; and **Walterová,
Běhounková & Efroimsky (2023)** restate W&B's results as "Q = 38 ± 4 at
the period of 1 month, Q = 41 ± 9 at 1 year, and lower bounds of
Q ≥ 74 at 3 years and Q ≥ 58 at 6 years", matching the period list below.
(A citation initially considered here, **Tan & Harada (2021)**, "Tidal
constraints on the low-viscosity zone of the Moon," *Icarus* 365,
114361, was briefly dropped after an earlier draft cited it under
arXiv:2301.04035 — an id that in fact resolves to the Briaud et al.
paper above. The Tan & Harada paper itself is verified to exist and is
cited here under its journal reference.) Elsewhere, Williams & Boggs (2015) also report a mild period
dependence beyond the monthly value: Q = 41 ± 9 at 1 year, Q ≥ 74 at 3
years, Q ≥ 58 at 6 years (again per web search of the primary source,
not a direct verbatim retrieval).

**27.55455 days itself** (where still legitimately mentioned above as
the anomalistic month, for contrast) is an Astronomical Almanac / NASA
value, not a Williams & Boggs (2015) one — an earlier draft of this
section implied otherwise by using it as if it were the paper's own
reference period.

**Citation correction, and a correction to that correction.** The
pre-existing "Anelastic bias" section above attributes Q = 38 ± 4 to
"Williams et al. (2014) Section 6." An earlier draft of this
Forcing-period-provenance section asserted that Williams et al. (2014)
"does not itself contain this Q determination" as the reason that
attribution was wrong. That stated reason is itself false: per web
search of the primary source in this session, Williams et al. (2014)
*does* report a monthly tidal Q determination — Q = 37.5 ± 4 (in both
the abstract and body), alongside Q_365 = 37 ± 9 and Q_F = 45 ± 5 (their
own notation for the values at the 365-day and F/draconic-month
reference periods respectively). The attribution's actual problem is
narrower: the pre-existing text's *value*, 38 ± 4, is Williams & Boggs
(2015)'s number, not Williams et al. (2014)'s — the two papers report
close but distinct monthly Q values (37.5 ± 4 vs. 38 ± 4) at overlapping
but not identical reference periods, and the pre-existing text combined
Williams et al. (2014)'s citation with Williams & Boggs (2015)'s value.
The *correction's conclusion* (cite Williams & Boggs 2015, a separate,
later paper specifically about lunar tidal dissipation, for 38 ± 4)
still stands; only the stated *reasoning* for it was wrong, and is
corrected here. The pre-existing "Anelastic bias" section's attribution
is left as-is above per this task's append-only scope.

### Headline: Maxwell vs. Andrade gap-closing, and the Q-consistency check

`fit_moon_maxwell_gap()` bisects the uniform mantle Maxwell viscosity
(layers 3-8, the same six layers `pylov3d.moon_mc`'s `mu_scale` scales)
that raises the real 10-layer structure's Re(k2) from 0.023159 up to the
observed 0.02422, using pylov3d's own (validated above) solver directly
— **no PyALMA3 needed for this number**, since the internal-ocean
structure is exactly what pylov3d's own solver already handles natively.

All numbers below were recomputed under the corrected draconic-month
forcing period (see "Forcing-period provenance" above); an earlier draft
of this table used the wrong period (27.55455 d) and is superseded.

| Rheology | eta_mantle [Pa s] | k2 | Q_implied | vs. Williams & Boggs (38 ± 4) |
|---|---|---|---|---|
| Maxwell (real 10-layer structure, native pylov3d) | 1.783e16 | 0.024219 − 0.030633i | **0.79** | 9.3σ away — grossly inconsistent |
| Maxwell (simplified body, native pylov3d, structure control) | 1.798e16 | 0.023956 − 0.030379i | 0.79 | (confirms the row above) |
| Maxwell (simplified body, PyALMA3, independent code path) | 1.797e16 | 0.023957 − 0.030399i | 0.79 | (confirms the two rows above) |
| Andrade, alpha=0.15 (PyALMA3, simplified body) | 9.306e24 | 0.023955 − 0.000251i | 95.3 | 14.3σ away |
| Andrade, alpha=0.20 | 5.486e22 | 0.023956 − 0.000341i | 70.3 | 8.1σ away |
| Andrade, alpha=0.25 | 2.497e21 | 0.023955 − 0.000434i | 55.2 | 4.3σ away |
| Andrade, alpha=0.30 | 3.127e20 | 0.023956 − 0.000536i | 44.7 | 1.7σ away |
| Andrade, alpha=0.35 | 6.994e19 | 0.023956 − 0.000650i | 36.8 | **0.3σ away** |
| Andrade, alpha=0.40 | 2.252e19 | 0.023955 − 0.000785i | 30.5 | 1.9σ away |

**On the second and third rows above.** An earlier draft of this table
had a single "Maxwell (simplified body, PyALMA3, cross-check)" row whose
digits were actually pylov3d's own `moon_simplified_maxwell_k2()` result
(0.023956...) mislabeled as PyALMA3's, and the only test backing it
called that pylov3d function, never `moon_alma_k2()` — so the "fully
independent code path" claim made below was asserted, not demonstrated.
Fixed here by splitting into two rows: the middle row is what the
original row actually was — a pylov3d **structure** control (same
solver, same Maxwell rheology, simplified body instead of the real
10-layer one) — and the third row is a new PyALMA3 result, obtained by
actually calling `moon_alma_k2(rheology="maxwell")` and bisecting for
the same fractional-gap target
(`TestMoonMaxwellPyALMA3Validation::test_maxwell_gap_closing_via_alma_matches_headline_row`).
The "fully independent code path" claim below now refers to this third
row specifically.

(The Andrade rows use the exact elastic-limit baseline, computed with an
actual PyALMA3 ``rheology="elastic"`` run — `moon_simplified_elastic_k2_alma()`,
k2 = 0.022907 (PyALMA3's own incompressible-elastic value; note the
Maxwell rows' simplified-body elastic limit, 0.022908, is pylov3d's own
incompressible-elastic number — the two agree to 4 significant figures
but are not the same computation, an earlier draft of this document
conflated them) — rather than a large-``eta`` Andrade/Maxwell limit. The
latter converges slowly at small alpha, e.g. still ~0.8% off true
elastic at ``eta = 1e30`` for ``alpha = 0.15`` — the residual term
decays as ``(s·eta/mu)^(-alpha)``, so smaller alpha converges more
slowly — and this biased the fractional-gap target enough to matter at
low alpha in an earlier draft of this table; caught and fixed while
building this section.)

(Andrade rows target the same *fractional* gap on the simplified body —
its own elastic k2, 0.022907 (PyALMA3), differs slightly from the real
structure's 0.023159, so the absolute target is rescaled by the same
1.0458 ratio; see `fit_moon_andrade_gap()` docstring.)

**Maxwell alone cannot simultaneously match the observed k2 gap and a
plausible Q.** Closing the gap with Maxwell rheology requires sitting
almost exactly at the Maxwell dissipation peak (omega·tau_M ~ 1 at this
forcing period), which forces catastrophic dissipation — Q ≈ 0.79, off
by more than 9σ from the measured 38 ± 4. This is not a numerical
artifact: both the pylov3d-only simplified-body structure control and
the PyALMA3 (independent code path) cross-check reproduce the same
viscosity (1.798e16 / 1.797e16 vs. 1.783e16 Pa s) and the same Q to 2
significant figures. It is also not a new claim — it is the quantified,
pylov3d-computed version of a documented, narrower result in the
literature: Nimmo, F., Faul, U. H., & Garnero, E. J. (2012), "Dissipation
at tidal and seismic frequencies in a melt-free Moon," *JGR Planets*,
117, E09005, doi:10.1029/2012JE004160 (already cited in this document's
"Anelastic bias" section above) show, using a laboratory-based extended
Burgers model, that the Moon's observed k2 and low tidal Q can be
matched *without* invoking mantle melt — i.e. their conclusion is the
opposite of "melt/an implausible basal low-viscosity zone is required."
What they *do* find, and what is cited for here, is narrower: their
models could not reproduce the Moon's observed (weak) frequency
dependence of Q — a limitation Maxwell shares in the more extreme form
quantified above (Maxwell forces a single relaxation peak, i.e.
effectively alpha=1 in the Andrade sense, and so cannot reproduce a
weak, broad frequency dependence at all). (An earlier draft of this
section cited Nimmo et al. (2012) for the claim that Maxwell "requires
an implausible basal low-viscosity zone or an additional dissipation
mechanism" — that disjunction is actually Walterová, Běhounková &
Efroimsky (2023)'s sentence about their own Model 1 (a homogeneous
Andrade mantle, not Maxwell; see "Literature parameter ranges" below),
citing Khan, A., Connolly, J. A. D., Pommier, A., & Noir, J. (2014), "Geophysical evidence for melt in the deep lunar interior and implications for lunar evolution," *JGR Planets*, 119(10), 2197-2221, for the low-viscosity-zone branch and Nimmo et
al. (2012) only for the "additional dissipation mechanism" branch —
retrieved and corrected in this session by fetching the paper directly.)

**Andrade at literature-favored alpha is consistent with the measured
Q, but alpha=0.35 is not squarely inside this document's own
commonly-adopted range.** This document's own `ANDRADE_ALPHA_COMMON_RANGE`
constant, its headline table (immediately below), and a test comment all
put the commonly-adopted range at 0.2-0.3 (Efroimsky 2012; see
"Literature parameter ranges" below) — alpha=0.35 sits above that range,
in the upper part of the broader 0.2-0.4 laboratory range instead. Both
values are reported in the table above so the sensitivity is visible: at
alpha=0.30 (inside the document's own commonly-adopted range), Q ≈ 44.7,
1.7σ from the measured 38 ± 4; at alpha=0.35 (above that range, inside
the broader laboratory range), Q ≈ 36.8, 0.3σ away. Both are consistent
with the measurement at the few-sigma level; neither is a precise
determination of alpha, and the closer agreement at 0.35 is not evidence
that 0.35 is the "correct" value — it is one gap-closing point per alpha
on a strongly alpha-dependent curve (Q spans 30-95 across the swept
0.15-0.40 range), so finding *some* alpha in the broader laboratory range
that reproduces the known Q is not surprising by itself.

**A fixed-mantle-rigidity caveat, made prominent.** Every row in the
table above holds mantle rigidity at its as-built value and attributes
the entire 4.6% k2 gap to anelastic softening. That attribution is not
unique: this document's own TASK-019 reference posterior (see "Posterior
smoke run" above) finds that the *elastic*, zero-anelasticity 4-parameter
fit already closes almost all of this gap by adjusting `mu_scale` alone
— its shipped median is `mu_scale` = 0.965 with k2 = 0.02419, and an
independent scan run for this revision confirms a purely elastic
`mu_scale` in [0.955, 0.96] reproduces k2 = 0.0242 ± 0.0001, the full
target, with **zero** anelasticity. In other words: the shipped Moon fit
already absorbs this gap into rigidity, not into rheology. This is the
same point the pre-existing "Anelastic bias" section above makes for the
Monte Carlo stage ("this gap is consistent-with, not diagnostic-of,
anelasticity") — the headline table here should be read the same way:
it establishes that Andrade rheology at literature-plausible parameters
*can* explain the gap and the measured Q simultaneously, and that Maxwell
*cannot*, not that anelasticity *is* what closes the as-built gap. A
joint (mu_scale, eta, alpha) fit that could separate the two contributions
is TASK-025b scope, not attempted here.

**Caveats.** (1) The Andrade numbers are computed on the simplified
body, not the real 10-layer structure (see "Validation" above) — the
Maxwell cross-check suggests this does not materially change the
result, but it has not been directly confirmed for Andrade specifically.
(2) Both rheology forms here use a single uniform mantle viscosity
across all six mantle layers; a depth-dependent viscosity (e.g. a basal
low-viscosity zone, as Khan et al. 2014 and Williams et al. 2014
Section 7 both discuss) is not explored. (3) This is a forward
consistency check at one gap-closing point per alpha, not a joint
(eta, alpha) posterior — that is exactly the TASK-025b scope. (4) See
"A fixed-mantle-rigidity caveat" immediately above — this whole table is
computed at fixed as-built mantle rigidity, and the shipped elastic fit
already closes most of the gap without any anelasticity at all.

### Literature parameter ranges for stage b priors

| Parameter | Range | Source |
|---|---|---|
| Andrade alpha (silicate mantles, general) | 0.2 - 0.4 | Walterová, M., Běhounková, M., & Efroimsky, M. (2023), "Is There a Semi-Molten Layer at the Base of the Lunar Mantle?", *JGR Planets*, e2022JE007652 (arXiv:2301.02476v2) -- three authors, an earlier draft of this table omitted Efroimsky. Retrieved and quoted directly (arXiv PDF fetched in this session): alpha "typically lies in the interval 0.2−0.4, although values outside this range have also been observed... Geodetic measurements performed on the Earth favour a narrower interval of 0.14−0.2, and the currently accepted model of tides in the solid Earth, presented in the IERS Conventions on Earth Rotation, employs the value of α = 0.15." (An earlier draft of this table attributed a different quotation to this paper -- "experiments with samples of different minerals most often furnish values... from 0.15 to 0.4, while geodetic measurements give slightly lower values" -- which does not appear in it; that fabricated quotation is removed here.) |
| Andrade alpha (commonly adopted) | 0.2 - 0.3 | Efroimsky, M. (2012), "Bodily tides near spin-orbit resonances," *Celestial Mechanics and Dynamical Astronomy*, 112, 283-330 (arXiv:1105.6086v9, fetched directly in this session). Its own stated range is broader than "0.2-0.3" in isolation -- "for all minerals (including ices) the values of α belong to the interval from 0.14 through 0.4 (more often, through 0.3)" -- but that "more often, through 0.3" framing is the basis for treating 0.2-0.3 as the commonly-adopted narrower range here. (An earlier draft cited Castillo-Rogez et al. (2011) for this range; that citation could not be verified in this session and is dropped.) Dumoulin, C., Tobie, G., Verhoeven, O., Rosenblatt, P., & Rambaux, N. (2017), "Tidal constraints on the interior of Venus," *JGR Planets*, 122(6), 1338-1352, doi:10.1002/2016JE005249: they *adopt* alpha in 0.2-0.3 for their prospective Venus interior models because it "frames the typical value required to explain the Q factor of the Earth's mantle". **Provenance caveat:** the metadata and the substance of this row were confirmed, but an independent re-verification pass could not retrieve the paper's full text (every PDF route blocked), so the quoted wording above is **not** independently confirmed verbatim and should be treated as a paraphrase-grade attribution until someone retrieves the PDF -- i.e. the range is imported from Earth, not derived from or shown to explain any Venus tidal measurement, and the paper's own conclusion is that a future mission (EnVision) is needed before Venus tidal data can constrain mantle rheology at all. (An earlier draft of this table claimed Dumoulin et al. "find" this range "explains Venus's short-period tidal response" -- overstated, corrected here.) |
| Andrade alpha (Moon-specific fits, upper-range value) | 0.35 | This document's own headline table above sweeps this value alongside 0.30 because it gives the closest Q match to the measured 38 ± 4 (0.3σ) -- see "Headline" above for why this is not the same as 0.35 being independently favored by the literature; 0.35 sits above, not inside, the 0.2-0.3 commonly-adopted range in the row above. |
| Andrade alpha (Model 1 in Walterová et al. 2023 -- a homogeneous mantle, no basal layer -- rejected by the authors) | 0.08 (+0.03/-0.02) | Walterová et al. (2023), Section 5.5/7 (retrieved directly in this session): fitting the Moon's selenodetic data with "a model consisting of a fluid core and a viscoelastic mantle governed by the Andrade rheology" (their Model 1 -- homogeneous, no basal low-viscosity layer) gives alpha = 0.08+0.03/-0.02, well below the literature range; the authors use this low value, plus the model's unrealistically low predicted seismic Q, as evidence *against* Model 1, concluding the Moon's tidal response "probably cannot be explained by the Andrade model alone and requires either a basal low-viscosity zone (in line with the conclusion of Khan et al., 2014) or an additional dissipation mechanism in the mantle (similar to Nimmo et al., 2012)." (An earlier draft of this table described alpha=0.08 as coming from "a semi-molten basal-mantle-layer model" that "fits the Moon's data" -- backwards: it is the homogeneous *no*-basal-layer model's result, used to reject that model, not a basal-layer model's fit.) |
| Lunar mantle viscosity (monthly + yearly tide inversion) | not a single number; implies a low-viscosity LOWER mantle | Goossens, S., et al. (2024), "A Low-Viscosity Lower Lunar Mantle Implied by Measured Monthly and Yearly Tides," *AGU Advances*, e2024AV001285 — independent support for depth-dependent (not uniform) mantle viscosity, relevant to caveat (2) above. |
| Lunar monthly tidal Q | 38 ± 4 (at the draconic month, 27.212 d) | Williams & Boggs (2015), *JGR Planets*, 120(4), 689-724 (see "Forcing-period provenance" above; `pylov3d.anelastic_moon.MOON_MONTHLY_Q` / `MOON_MONTHLY_Q_SIGMA`). |
| Lunar yearly/multi-year tidal Q | 41 ± 9 (1 yr), ≥ 74 (3 yr), ≥ 58 (6 yr) | Williams & Boggs (2015), same source — context for any future frequency-dependent (Andrade, not single-frequency Maxwell) stage-b likelihood. |

All values also collected as plain constants: `ANDRADE_ALPHA_RANGE` /
`ANDRADE_ALPHA_COMMON_RANGE` in `pylov3d/anelastic.py` (body-agnostic),
`MOON_MONTHLY_Q` / `MOON_MONTHLY_Q_SIGMA` in `pylov3d/anelastic_moon.py`
-- for stage-b reuse; this table is the citation record of provenance.

## Joint anelastic fit (TASK-025b)

Stage b of TASK-025 threads anelasticity into the Bayesian fit: a joint
(structure, viscosity, alpha) posterior that adds the Williams & Boggs
(2015) monthly **Q = 38 ± 4** as a fifth observable (an imaginary-part
constraint) on top of the four structural observables the TASK-019 elastic
fit used (mass, moi_mean, k2, core_radius_km). The question this stage
exists to answer, posed at the end of the "A fixed-mantle-rigidity caveat"
subsection above: **the elastic TASK-019 fit already closes the 4.6% k2 gap
with `mu_scale` alone (median 0.965) and zero anelasticity — does adding a
real dissipation constraint (Q) break the rigidity/anelasticity degeneracy,
and if so, where does `mu_scale` then land and what alpha does the data
prefer?**

Driver: `scripts/moon_anelastic_pocomc.py` (not a `pylov3d` module — no
solver code was modified for this stage). It builds a **custom 5-or-6
parameter log-posterior** rather than reusing `pylov3d.forward.
make_log_posterior`, for one structural reason: `forward.log_likelihood`
compares only **Re(k2)**, and the Q constraint is a statement about
**Im(k2)** (`implied_Q(k2) = -Re(k2)/Im(k2)`, `pylov3d.anelastic`) — it
cannot be expressed through the real-only likelihood. The custom posterior
gates structural bounds via `forward.log_prior`, computes the structural
observables (mass, moi_mean, core_radius_km — rheology-independent, analytic)
from the real θ-scaled 10-layer body via `compute_observables`, and adds
the rheology-appropriate anelastic k2 (and, when enabled, the Q term from
its imaginary part).

### The 2×2 design and why each cell exists

Four converged runs: {Maxwell, Andrade} × {with-Q, without-Q}. The
`--no-q` runs drop the Q term to isolate what Q *adds*; the two rheologies
answer different questions:

- **Maxwell** runs on the **real 10-layer Weber body** natively (pylov3d's
  own ocean-aware solver — no PyALMA3, no structural simplification), with
  a single uniform mantle viscosity `eta` (layers 3-8, log-uniform prior).
  This is the physically complete body, so its k2/Q are directly comparable
  to the elastic TASK-019 fit.
- **Andrade** runs via PyALMA3 on the **simplified body** (`moon_simplified_body`,
  the 3-innermost-layers-merged construction from TASK-025a's "Validation"
  section) — the only route to an Andrade complex modulus, since neither
  pylov3d nor the vendored MATLAB has an Andrade path (TASK-025a
  "What the solver actually supports"). The simplified body is made
  **θ-aware** (`_simplified_body_from_model` rebuilds the core-merge on the
  scaled model each evaluation) so `mu_scale` flows into k2 and Q — without
  this the `mu_scale`–alpha correlation this stage measures would be forced
  to zero by construction.

Prior on viscosity: log-uniform over `MARS_MANTLE_ETA_ANDRADE_RANGE` =
1e19–1e22 Pa s. **This range is imported from the Mars silicate-mantle
Andrade work as a Moon analogue, not a Moon-specific citation** — TASK-025a
collected no cited Moon mantle viscosity *range* (Goossens et al. 2024
argues for a low-viscosity *lower* mantle but publishes no single uniform
number; see the "Literature parameter ranges" table above). The prior on
alpha is uniform over `ANDRADE_ALPHA_RANGE` = 0.2–0.4 (the broad silicate
laboratory range). Both are documented as analogue/laboratory priors, not
Moon determinations.

### Converged posteriors

All four at the TASK-019 production standard (`n_active=64`,
`n_effective=128`, `Nrbase=50`, dynamic termination; Andrade at
`ndigits=48`, `order=8`). Kish ESS > 4000 and **zero solver failures** in
every run. Weighted medians ±1σ:

| Run | `mu_scale` | log10(eta) | alpha | median-model k2 | median-model Q | corr(mu,eta) | corr(mu,alpha) |
|---|---|---|---|---|---|---|---|
| Maxwell **+Q** | 1.78 (railed) | 19.0 (railed to floor) | — | 0.01411 ✗ | 232.9 ✗ | +0.01 | — |
| Maxwell **−Q** | **0.965** | 20.5 ± 1.0 | — | 0.02419 ✓ | 15251 (negligible) | −0.02 | — |
| Andrade **+Q** | 1.012 ± 0.018 | 20.3 ± 0.5 | 0.286 ± 0.06 | 0.02422 ✓ | 36.6 ✓ | +0.51 | **−0.70** |
| Andrade **−Q** | 1.007 ± 0.05 | 20.4 ± 1.0 | 0.296 ± 0.06 | 0.02412 ✓ | 42.0 | −0.67 | −0.67 |

Chains and pairplots archived as
`docs/figures/proposal/moon_anelastic_chain_{maxwell,maxwell_noq,andrade,andrade_noq}.npz`
with matching `_pairplot_*.png` (copied from the gitignored
`scripts/output/`, per the TASK-021b artifact-commit precedent).

### The three questions, answered in numbers

**1. Is the degeneracy broken? No — Q tightens the anelastic parameters but
does not separate rigidity from anelasticity.** Comparing Andrade −Q vs +Q:
adding Q roughly **halves** the viscosity width (log10_eta 20.4 ± 1.0 →
20.3 ± 0.5) and sharpens alpha (0.296 → 0.286), but leaves `mu_scale`
essentially unmoved (1.007 → 1.012, well inside 1σ) and leaves the
`mu_scale`–alpha anticorrelation strong (−0.673 → −0.697). Q selects *among*
Andrade dissipation models (which viscosity/alpha combination gives the
observed loss) but does **not** distinguish "softer mantle, less
anelasticity" from "as-built-rigidity mantle, more anelasticity" — that
trade-off, visible as the persistent −0.70 `mu_scale`–alpha correlation,
survives the Q constraint intact. This is the quantified confirmation of
the "A fixed-mantle-rigidity caveat" subsection's claim that the gap is
"consistent-with, not diagnostic-of" anelasticity: even *with* a direct
dissipation measurement, the data cannot uniquely partition the k2 gap
between rigidity and rheology.

**2. Where `mu_scale` lands vs. the elastic fit: it moves UP, from 0.965 to
1.012.** The elastic TASK-019 reference median is `mu_scale` = 0.965 (a 3.5%
mantle softening that supplies the entire k2 enhancement elastically). In
the Andrade +Q fit, `mu_scale` = 1.012 (+0.018/−0.016) — **+4.9%**, back to
approximately as-built rigidity. The mechanism is exactly the expected
partition: once Andrade anelasticity is present to enhance k2, the fit no
longer needs to soften the mantle to reach the observed k2, so `mu_scale`
relaxes back toward 1.0. **Structural caveat:** this comparison crosses
bodies — the elastic 0.965 is on the real 10-layer structure, the Andrade
1.012 on the PyALMA3 simplified body. TASK-025a's Maxwell control measured
the real-vs-simplified elastic-k2 structural offset at ~+1.1% (0.023159 real
vs 0.022907 simplified), so a small part (roughly a fifth) of the +4.9%
`mu_scale` shift is this structural offset rather than the anelastic
partition; the offset cannot be removed because a real-body Andrade
calculation is not possible (no Andrade path in pylov3d). The **direction
and bulk** of the shift are the anelastic partition, not the structural
offset.

**3. What alpha the data prefers: 0.286 (+0.066/−0.052), the top of
Efroimsky's commonly-adopted 0.2–0.3 band.** The Andrade +Q posterior
median alpha is 0.286, with ±1σ spanning ~0.23–0.35 — sitting at the upper
edge of the `ANDRADE_ALPHA_COMMON_RANGE` (0.2–0.3, Efroimsky 2012) and
overlapping the broader laboratory range (0.2–0.4). This is **consistent
with, but not a sharp determination of, alpha**: the ±1σ width is ~0.06 on
a [0.2, 0.4] prior (the posterior is informative — narrower than the prior —
but not tight), and the strong `mu_scale`–alpha anticorrelation (question 1)
means the preferred alpha is coupled to how much mantle rigidity the fit
assigns. The value is close to the TASK-025a forward-consistency headline
(where alpha ≈ 0.30 gave Q ≈ 44.7, 1.7σ from measured, and alpha ≈ 0.35
gave the closest 0.3σ match), and the joint fit's slight preference for
0.286 over 0.30 reflects that it also fits mass/MoI/core-radius/k2
simultaneously, not Q alone.

### The Maxwell negative control

The two Maxwell rows make the case that the Andrade result is not an
artifact of the fitting machinery:

- **Maxwell +Q is infeasible against all five observables at once.** The
  sampler drives `mu_scale` to 1.78, `R_fluid_core` to the 459 km bound
  ceiling, and `eta` to the 1e19 Pa s prior floor — and *still* lands at
  k2 = 0.0141 (vs 0.0242 observed) and Q = 233 (vs 38 ± 4). There is no
  point in the 5-D box where Maxwell rheology reproduces both the observed
  k2 and a plausible Q; the posterior rails on multiple parameters trying,
  and fails on both observables. (`corr(mu,eta)` ≈ 0 here because the
  parameters are pinned at bounds, not trading off.) This is the joint-fit
  confirmation of TASK-025a's headline finding that "Maxwell alone cannot
  simultaneously match the observed k2 gap and a plausible Q" — there
  established at a single gap-closing viscosity (Q ≈ 0.79), here established
  as a full-posterior infeasibility across the whole parameter box.
- **Maxwell −Q cleanly recovers the elastic TASK-019 answer.** Dropping Q,
  the Maxwell fit reproduces `mu_scale` = 0.965 (bit-matching the elastic
  reference median), k2 = 0.0242, with eta drifting to a high, unconstrained
  value (log10_eta = 20.5 ± 1.0, Q ≈ 15000 → negligible dissipation). With
  no dissipation constraint, the viscosity is unidentified and the fit
  falls back to the elastic solution — a consistency check that the
  real-body Maxwell path and the elastic TASK-019 path agree when Q is
  removed.

### Caveats

1. **Andrade is on the simplified body, Maxwell on the real body.** The
   Andrade posterior (the scientifically interesting one) inherits the
   TASK-025a simplified-body limitation: PyALMA3 cannot represent the
   internal fluid ocean, so layers 0-2 are merged. The Maxwell −Q row is
   the control that the simplification does not distort the elastic anchor
   (`mu_scale` = 0.965 recovered on the real body); the ~+1.1% structural
   k2 offset is quantified in question 2 above. A real-body Andrade result
   would require an Andrade implementation in pylov3d (out of scope).
2. **Uniform mantle viscosity.** Both rheologies use a single `eta` across
   all six mantle layers. A depth-dependent viscosity (Goossens et al. 2024;
   Khan et al. 2014's basal low-viscosity zone) is not explored and would
   add parameters this 5 observables cannot constrain.
3. **The eta and alpha priors are analogue/laboratory, not Moon
   determinations** (Mars silicate range for eta; broad laboratory range
   for alpha) — see the prior discussion above. The posterior on alpha is
   informative relative to these priors, but its absolute location should be
   read against that prior provenance.
4. **The degeneracy is not broken (question 1).** The headline scientific
   result is a *negative* one about identifiability: adding a direct
   dissipation constraint sharpens the anelastic parameters but does not let
   the data uniquely separate mantle rigidity from mantle anelasticity. Any
   `mu_scale` value the reader takes from this stage is conditional on the
   assumed rheology (elastic → 0.965; Andrade → 1.012), not a
   rheology-independent determination.

## Moon lateral crust stage (TASK-031)

TASK-031 applies the validated coupled-ocean machinery to the committed
GRAIL/LOLA fields for the first Moon-specific lateral forward model. The
reproducible path is `pylov3d/moon_lateral.py`; the production driver is
`scripts/moon_lateral_spectrum.py`, with arrays and figure archived as
`docs/figures/proposal/moon_lateral_spectrum.{npz,png}`.

### Field-to-rigidity construction

The construction follows the Mars Airy stage in its physical sequence, but
the constants and low-degree choices are Moon-specific:

1. Truncate GRGM900C and MoonTopo719 to `lmax=4`. Convert normalized gravity
   coefficients to first-order equipotential heights with `N_lm = r0 C_lm`
   (and the sine analogue), then subtract them from LOLA shape.
2. Remove degree 1 from both fields. MoonTopo719 is in a principal-axis,
   center-of-mass frame; its degree-1 field reaches +/-1.935 km (3.869 km
   peak-to-peak) and represents the center-of-figure translation, not a
   physical crustal load. Retaining it would turn a coordinate-origin offset
   into a +/-12.9 km (25.8 km peak-to-peak) Airy root.
3. Remove C20 from both shape and equipotential height, while retaining C21,
   S21, C22, and S22. This treats the zonal hydrostatic/tidal figure as part
   of the reference shape rather than a crustal load. This is not inherited
   silently from Mars: the alternative is evaluated explicitly below.
4. Apply the as-built density contrast, `rho_c=2800` and `rho_m=3220 kg/m3`,
   giving the Airy factor `rho_c/(rho_m-rho_c)=6.6667`.
5. Linearize thickness into the surface crust layer's rigidity using the
   independently adopted 40 km mean crust thickness and the Weber crust/
   adjacent-mantle shear moduli (28.672/63.478 GPa). The numerical Weber
   surface shell is 34 km thick; using it as the denominator would conflate
   the profile discretization with the independently adopted mean crust and
   would push the field outside the linear model's domain. The 40-vs-34 km
   mismatch remains a structural approximation, stated rather than hidden.

The default field has per-degree coefficient RMS thicknesses of 5.738,
5.247, and 3.952 km at degrees 2, 3, and 4. On a 180x360 grid,
`max|delta_t|=32.628 km` (0.816 of the 40 km reference) and
`max|delta_mu/mu_bar|=0.9902`. Thus the full-amplitude result is valid but
only narrowly inside the positive-rigidity boundary; amplitude or density
sweeps cannot be interpreted linearly much beyond this reference case.

### Why C20 is excluded

Retaining the areoid-referenced C20 residual raises the degree-2 thickness
RMS from 5.738 to 6.671 km, `max|delta_t|` to 35.527 km, and
`max|delta_mu/mu_bar|` to **1.0782**. The last value implies negative local
shear modulus in the linearized surface layer. The public API exposes
`include_c20=True` for this diagnostic, but refuses to send that full-
amplitude field to the solver. This makes the degree-2 choice a physical
domain decision, not bookkeeping. A nonlinear moving-boundary or positive-
definite mixing formulation is required before the C20-retaining case can
be compared fairly.

### Coupled spectrum

The production solve uses the as-built ten-layer Weber model, including its
fluid outer core, the validated `method="variable"`, `Nrbase=30`,
`perturbation_order=2`, and a unit `(2,0)` monthly forcing. It activates 115
coupled modes and completed in 207.5 s on the producing machine.

| Quantity | TASK-031 result |
|---|---:|
| uniform Weber `k2` | 0.02315914223 |
| lateral forcing-mode `k20` | 0.02316054935 |
| `Delta k20` | +1.40712e-6 |
| `|Delta k20| / sigma_k2` | 0.6396% |
| largest off-forcing pair | `(2,+/-2)`, `|k|=3.13471e-6` |
| next pair | `(2,+/-1)`, `|k|=2.76868e-6` |
| next pair | `(3,+/-3)`, `|k|=2.01884e-6` |

The diagonal perturbation is negligible against the measured
`sigma_k2=2.2e-4`; as in the Mars stage, the spatial information resides in
the off-forcing spectrum rather than the bulk k2 shift. No detectability
claim is made here because converting these Moon coefficients to a mission
measurement requirement needs a separate observable/noise analysis.

Radial convergence is strong: at the same `lmax=4`, reducing Nrbase from 30
to 15 changes `Delta k20` from 1.407122e-6 to 1.407061e-6, a relative change
of 4.3e-5. Angular convergence is not established: the inexpensive
`lmax=2` comparison gives `Delta k20=2.37290e-7`, so lmax=4 is 5.93 times
larger. The archived spectrum is therefore the first fixed-cutoff Moon
lateral prediction, not a truncation-converged endpoint; a Moon analogue of
TASK-027's lmax ladder is the next numerical check.

### Truncation convergence (TASK-031b)

The Moon analogue of TASK-027's lmax ladder was run with the driver
`scripts/moon_lateral_convergence.py` (consumer only; no solver module was
modified). It climbs the angular truncation at fixed `Nrbase=30`, tracking
the `(2,0)` forcing-mode `Delta k20` and the three named off-forcing pairs,
and independently re-verifies the radial (`Nrbase`) convergence rather than
importing it from Mars, because the Weber Moon carries a fluid outer core the
Mars model lacks. Artifacts:
`docs/figures/proposal/moon_lateral_convergence.{npz,png}`.

**The finding is that the linearized Airy pipeline cannot be carried past
`lmax=4`.** The positivity margin of the linearized crust rigidity,
`max|delta mu / mu_bar|`, is a monotone function of the truncation and crosses
the physical unit bound between `lmax=4` and `lmax=5`:

| lmax | N modes | max\|dmu/mu_bar\| | `Delta k20` | step-to-step |
|---:|---:|---:|---:|---:|
| 2 | 43 | 0.3471 | 2.372903e-7 | -- |
| 3 | 75 | 0.6752 | 4.521374e-7 | +90.5% |
| 4 | 115 | 0.9902 | 1.407122e-6 | +211.2% |
| 5 | -- | 1.1531 | blocked | linearization non-positive |
| 6 | -- | 1.2897 | blocked | linearization non-positive |

At `lmax=5` and `lmax=6` `mu_variable_from_topography` raises by design (the
linearized crust rigidity would go non-positive; at `lmax=6` the Airy
thickness variation also exceeds the 40 km reference crust). These are not
solver failures — the field is genuinely non-physical under the linearization
and the guard correctly refuses it. The achievable physical ladder is
therefore `lmax = 2, 3, 4` only.

**Consequence for the TASK-031 result.** `Delta k20` is still climbing steeply
where the linearization runs out (+90.5% from lmax=2->3, +211.2% from
lmax=3->4), so angular convergence is *not* demonstrated and cannot be
demonstrated by climbing lmax with this pipeline — the field goes non-physical
before the sequence flattens. The TASK-031 value `Delta k20 = 1.407e-6` should
therefore be reported as **the highest-lmax value the Airy linearization
admits (lmax=4), not a truncation-converged endpoint.** Per the task spec,
this is the finding; no extrapolation to a converged value is offered. Note in
contrast that the off-forcing pairs are far better behaved across the same
ladder — `(2,+/-2)` moves only -0.11% then -5.68%, `(3,+/-3)` is flat to
+0.00% from lmax=3->4 — so the spatial spectrum that carries the lateral
information is closer to settled than the bulk `Delta k20` is.

**Radial convergence (re-verified for the Moon, not imported).** Holding
`lmax=4` fixed and varying `Nrbase`:

| Nrbase | `Delta k20` | rel. to Nrbase=50 |
|---:|---:|---:|
| 15 | 1.407061e-6 | 4.42e-5 |
| 30 | 1.407122e-6 | 1.25e-6 |
| 50 | 1.407124e-6 | 0 (ref) |

Radial convergence is strong and the production `Nrbase=30` is effectively
converged (1.25e-6 relative to `Nrbase=50`), confirming for the fluid-core
Weber model what TASK-031 assumed. The angular truncation, not the radial
grid, is the binding limitation.

**(4,0) rheology channel is first order (generalizes TASK-028).** Isolating
the `(4,0)` harmonic of the default (C20-dropped) Moon field and scaling its
amplitude by `eps in {1e-3, 1e-2}` gives a `|Delta k20|` log-log slope of
**1.0001** — first order. Because C20 is dropped, the Moon field has no `(2,0)`
self-coupling channel, so `(4,0)` is the sole first-order driver of the `(2,0)`
forcing mode. This confirms that the first-order `(even,0) -> (2,0)` coupling
established for Mars in TASK-028 is a general feature of the coupled operator,
not a Mars-specific artifact.

**Memory.** Peak RSS scaled steeply with mode count: 1.3 GB (lmax=2, 43
modes), 4.1 GB (lmax=3, 75 modes), 7.4 GB (lmax=4, 115 modes at Nrbase 15-30),
and 10.8 GB at `lmax=4, Nrbase=50` — the heaviest physical solve. The blocked
`lmax=5,6` rungs cost nothing (they fail at the cheap diagnostics stage before
any coupled solve). This corroborates the TASK-021b memory caution: the
ten-layer Moon is expensive, and `lmax=6/Nrbase=50` — had it been physical —
would have been the same >15 GB regime seen there.

### Amplitude wall T-sweep (TASK-036b)

The TASK-031b section above establishes that the amplitude wall is not a solver
limit but a geometric constraint of the Voigt volume-fraction mixing rule: the
coefficient `d(δμ/μ̄)/d(dt) = (μ_c − μ_m)/(T·μ_c)` exceeds unity at
`|dt| = T/contrast = 32.95 km`, which is less than the 40 km reference shell
(contrast = 1.2139 for the Weber model). TASK-036 (design note) proposed testing
whether thickening the reference shell T — which scales the coefficient as 1/T
— unblocks angular convergence without changing the underlying physics.

**Driver:** `scripts/moon_lateral_t_sweep.py`. Nrbase = 30 throughout.
Artifacts: `docs/figures/proposal/moon_lateral_t_sweep.{npz,png}`.

#### T-sweep (lmax = 4, fixed contrast)

The T-sweep asks: as T grows, does `|Δk20|` settle to a stable value?

| T [km] | max\|δμ/μ̄\| | N | \|Δk20\| | step |
|---:|---:|---:|---:|---:|
| 40 | 0.9902 | 115 | 1.407e-6 | — |
| 55 | 0.7202 | 115 | 9.031e-7 | −35.8% |
| 70 | 0.5658 | 115 | 6.573e-7 | −27.2% |
| 85 | 0.4660 | 115 | 5.137e-7 | −21.9% |

Off-modes `(2,±2)`, `(2,±1)`, `(3,±3)` scale in lock-step with `|Δk20|` across
all T, with step sizes of −27%, −21%, −18% respectively — the whole lateral
spectrum shrinks proportionally as T grows.

**The spectrum does not converge in T.** The last step is −22%, far above the
5% convergence threshold, and the trend shows no sign of flattening. This is the
expected behaviour: thickening T dilutes the fixed crustal-thickness variation
`dt` into a larger reference volume, reducing the Voigt coefficient and therefore
the amplitude of every coupled mode proportionally. `|Δk20|` is chasing zero,
not a stable value. Fix (A) — thicken T — does not produce a convergent
spectrum; it trades amplitude wall for amplitude loss. This is a physical
consequence, not a pipeline artifact, and it rules out T-thickening as a
resolution-only change that leaves published numbers intact.

#### lmax ladder at T = 85 km

At T = 85 km the positivity margins are well below 1.0 for all rungs through
lmax = 6 (max\|δμ/μ̄\| = 0.607 at lmax = 6), so the full angular ladder is
physically accessible. The question is whether the spectrum converges in lmax at
this larger T.

| lmax | max\|δμ/μ̄\| | N | \|Δk20\| | step |
|---:|---:|---:|---:|---:|
| 2 | 0.163 | 43 | 5.104e-8 | — |
| 3 | 0.318 | 75 | 9.875e-8 | +94% |
| 4 | 0.466 | 115 | 5.137e-7 | +420% |
| 5 | 0.543 | 163 | 5.212e-7 | +1.5% |
| 6 | 0.607 | 219 | 5.257e-7 | +0.9% |

**Angular convergence is confirmed at T = 85 km.** The lmax = 4→5 and 5→6
steps are +1.5% and +0.9%, well inside the 5% threshold. Peak RSS at lmax = 6
is 22.4 GB (Nrbase = 30); total wall time for the full run was 1953 s.

For comparison, the corresponding steps at T = 40 km (TASK-031b) were +90.5%
and blocked — the positivity wall prevented any solve at lmax ≥ 5. At T = 85 km
those same rungs are accessible and the spectrum is converged.

#### Interpretation

These two results together say: thickening T does unlock angular convergence, but
at the cost of a T-dependent amplitude. The converged value at T = 85 km,
`|Δk20| ≈ 5.26e-7`, is 2.68× smaller than the T = 40 km result (1.407e-6), and
the ratio equals the coefficient ratio `(μ_c − μ_m)/(85·μ_c)` /
`(μ_c − μ_m)/(40·μ_c)` = 40/85 = 0.471 — exactly as the Voigt formula predicts.
There is no physical preference between T = 40 km and T = 85 km on this evidence
alone. The reference shell thickness is a modelling choice, not a measurable
quantity, and changing it moves the published `Δk20` by a factor of T_new/T_old.

**Conclusion for TASK-036b.** Fix (A) unblocks the angular convergence test but
does not provide T-convergence. The follow-on question — what reference shell T
best represents the Moon's crust geometry — is a physics argument, not a
numerical one, and belongs in a separate ticket (per TASK-036 design note). The
deliverable here is the convergence diagnosis: the spectrum *is* angularly
convergent once T is large enough to carry the full field, and the angular
converged value scales exactly as 1/T.

### Correction: the response does not scale as 1/T (A, 2026-08-13)

The T-sweep entry above states that `|Delta k20|` scales exactly as `1/T`.
It does not, and neither does the analytic argument that was offered in
support of it. Both are corrected here from the sweep's own committed
artifact.

What *is* exactly `1/T` is the **perturbation amplitude**:
`max|dmu/mu_bar|` measures 0.9902 / 0.7202 / 0.5658 / 0.4660 at
T = 40 / 55 / 70 / 85 km, matching `1/T` to four decimals. That part is
algebraic, since `dmu/mu_bar = dt K / T`.

The **response** does not inherit that scaling. `|Delta k20|` measures
1.40712 / 0.903108 / 0.657276 / 0.513658 e-6 across the same rungs, a
`T^-1.338` power law — falling *faster* than `1/T`, and departing from a
`1/T` law by 22.4% across the sweep.

The reason is visible in the modes. The off-forcing pairs scale as
`T^-0.996`, `T^-0.996` and `T^-0.995` — pure first-order coupling
responses, exactly linear in the perturbation, exactly `1/T`. The
forcing-mode shift is not pure first order, so it does not.

Fitting `Delta k20 = a/T + b/T^2` (first order plus second order)
reproduces all four rungs to better than 0.2%:

| T [km] | measured [e-6] | fitted [e-6] | residual |
|---:|---:|---:|---:|
| 40 | 1.40712 | 1.40671 | -0.03% |
| 55 | 0.903108 | 0.904293 | +0.13% |
| 70 | 0.657276 | 0.657190 | -0.01% |
| 85 | 0.513658 | 0.512799 | -0.17% |

with `a = 3.232e-5` and `b = 9.581e-4`. That decomposes the shipped
result: at T = 40 km the Moon's forcing-mode shift is **57.4% first
order and 42.6% second order**, the first-order part being the `(4,0)`
channel alone, since this field removes `C20`.

Two consequences. The sweep was not a wasted negative result — it
measures the order decomposition, which no single solve reveals. And the
excursion rule of TASK-037 does *not* pin the amplitude after all: fixing
`|dmu/mu_bar| = K/2` fixes the perturbation, but the response still
depends on T through both terms, so whether it converges under the rule
remains an open question rather than a foregone one.
