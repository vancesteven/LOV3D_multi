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
together correctly and finish in seconds. **The reference posterior for
`R_fluid_core`** is instead the reviewer-verified, properly-resolved run
(`n_active=64`, `n_effective=128`, `n_total=512`, pre-M3 unphysical
[0.75, 1.25] `core_rho_scale` range): **363.7 ± 33.2 km** (quoted above,
"Anelastic bias"). A properly-resolved run under the *current* (floored)
bounds has not been performed in this round — given the MAP shift to
~327 km once the floor is active, that resolved number should **not** be
assumed to still apply; a fresh production run under the physically-floored
parameterization is future work (mirrors Mars's TASK-015).

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
