# Mars Reference Model and 3D Tidal Response

*(Retitled 2026-08-16, TASK-041. This document was called "Mars 1D Radial
Reference Model" long after half of it stopped being 1D — the title
actively hid the 3D capability, and the PI read it as "3D Mars isn't
implemented." It is; the map below says where.)*

## What exists — capability map

| Capability | Section (task) | Code |
|---|---|---|
| 1-D elastic reference model, fit + anchors | TASK-011 (below) | `pylov3d/mars.py` |
| Monte Carlo / posterior over interior parameters | TASK-012 | `pylov3d/mars_mc.py` |
| **3D lateral crust → coupled Love-number spectrum** (N=115, MATLAB-anchored) | TASK-016 | `pylov3d/mars_lateral.py` |
| Non-Airy crustal model substitution (InSight Moho) | TASK-028 | `pylov3d/mars_crust_models.py` |
| Hydration-front tidal signature | TASK-021 | `pylov3d/mars_hydration.py` |
| Off-(2,0) detectability vs mission precision | TASK-026 | `pylov3d/mars_detectability.py`, `mars_detectability_k2m.py` |
| Anelasticity audit (Maxwell-only; Andrade via PyALMA3) | TASK-025a | `pylov3d/anelastic.py` |
| Fixed-shell amplitude bound (closed form) | TASK-036a | tests + `docs/tasks/TASK-036*` |
| **3D spatial response maps** (gravity/displacement over the surface) | TASK-041 | `scripts/mars_response_maps.py` |

"3D" here is the LOV3D sense: lateral variations of interior properties
solved by spectral mode coupling (Rovira-Navarro et al. 2024), not a
finite-element volume mesh. The lateral stage currently varies one layer
(crustal rigidity); a laterally varying mantle is designed, not built —
`docs/tasks/TASK-042-design-mars-mantle-3d.md`.

## 1-D radial reference model (TASK-011)

A stage-1, purely elastic, 4-layer radial interior model for Mars, fit to
published bulk geophysical constraints and solved through the existing
pylov3d radial tidal solver (`pylov3d.love.get_love`). Implementation:
`pylov3d/mars.py`. Tests: `pylov3d/tests/test_mars.py`. Catalog entry:
`pylov3d/bodies.py`, id 40 (second digit 0 = the planet itself, orbiting the
Sun; 41 is reserved for Phobos under the existing catalog numbering).

> **Revision note:** an earlier draft of this document/model contained
> several citation errors and one science-affecting bug, found in scientific
> review. All are corrected below; see "Corrections applied" at the end for
> a record of what changed and why.

## Published bulk constraints

| Quantity | Value | Source |
|---|---|---|
| GM | 42828.375 km^3/s^2 | Konopliv, A. S., Park, R. S., & Folkner, W. M. (2016), "An improved JPL Mars gravity field and orientation from Mars orbiter and lander tracking data," *Icarus*, 274, 253-260. (MRO120D gravity field.) |
| Mass M | GM/G with G = 6.6743e-11 m^3 kg^-1 s^-2 (CODATA 2018) ≈ 6.4169e23 kg | derived |
| Mean radius R | 3389.5 km | Seidelmann, P. K., et al. (2007), "Report of the IAU/IAG Working Group on cartographic coordinates and rotational elements: 2006," *Celestial Mechanics and Dynamical Astronomy*, 98, 155-180 (IAU report / MOLA-derived mean radius). |
| Polar moment-of-inertia factor C/MR^2 | 0.3644 ± 0.0005 | Konopliv, A. S., et al. (2011), "Mars high resolution gravity fields from MRO, Mars seasonal gravity, and other dynamical parameters," *Icarus*, 211, 401-428. |
| J2 (dynamical form factor) | 1.9555e-3 | Konopliv, Park & Folkner (2016), *Icarus*, 274 (MRO120D). |
| **Mean** moment-of-inertia factor I/MR^2 (the fit target — see below) | 0.36310 ± 0.0005 | Derived: I/MR^2 = C/MR^2 − (2/3)·J2 = 0.3644 − (2/3)(1.9555e-3) = 0.3630963... → 0.36310. |
| Tidal k2 (degree-2, solar semidiurnal) | 0.169 ± 0.006 | Konopliv, Park & Folkner (2016), *Icarus*, 274 (MRO120D); consistent with k2 = 0.174 ± 0.008 from Chandler-wobble / MRO120F gravity, Konopliv, A. S., et al. (2020), "Detection of the Chandler wobble of Mars from orbiting spacecraft," *Geophysical Research Letters*, 47, e2020GL090568. |
| Core radius | 1830 ± 40 km, liquid | Stähler, S. C., et al. (2021), "Seismic detection of the martian core," *Science*, 373, 443-448 (InSight seismology); Le Maistre, S., et al. (2023), "Spin state and deep interior structure of Mars from InSight radio tracking," *Nature*, 619, 733-737, gives 1835 ± 55 km from the spin-state (RISE) solution. |
| Crustal thickness (global mean) | ~24-72 km; adopt 50 km | Knapmeyer-Endrun, B., et al. (2021), "Thickness and structure of the martian crust from InSight seismic data," *Science*, 373, 438-443. |
| Crust density | 2900 kg/m^3 | Within the Knapmeyer-Endrun et al. (2021) range. |
| Mantle seismic properties | Vs ~4.4-5.0 km/s, rho ~3400-4000 kg/m^3 | Khan, A., et al. (2021), "Upper mantle structure of Mars from InSight seismic data," *Science*, 373, 434-438; Stähler et al. (2021), *Science*, 373. |

### Note on the RISE citation

RISE (the InSight lander's Rotation and Interior Structure Experiment) is a
radio-tracking spin-state experiment; a stationary lander cannot itself
measure the degree-2 tidal Love number k2 (that requires orbital/gravity
tracking over many tidal cycles). The 1835 ± 55 km core-radius estimate
attributed to Le Maistre et al. (2023) above is a legitimate RISE spin-state
result. The k2 = 0.174 ± 0.008 value is **not** a RISE/lander measurement —
it comes from Konopliv et al. (2020)'s orbital Chandler-wobble gravity
analysis (MRO120F). An earlier draft of this document incorrectly attached
both numbers to "InSight RISE" and to Le Maistre et al.; that has been
corrected.

## Moment-of-inertia derivation (mean vs. polar)

**This is the one correction that changes the fitted numbers**, not just a
citation. Konopliv et al. (2011)'s 0.3644 is the normalized **polar**
moment of inertia, C/MR^2 — the moment about Mars's rotation axis, which is
larger than the mean moment because Mars is rotationally/tidally flattened.
A spherically symmetric 1D radial profile (as built here) has no way to
distinguish polar from equatorial axes, so it can only be fit to the
**mean** moment of inertia, I/MR^2, not C/MR^2 directly. The two are related
through the dynamical form factor J2 by:

```
I / (M R^2) = C / (M R^2) − (2/3) * J2
            = 0.3644 − (2/3)(1.9555e-3)
            = 0.3630963...  →  retarget to 0.36310
```

(J2 = 1.9555e-3 from Konopliv, Park & Folkner 2016, MRO120D.) `pylov3d.mars`
uses the rounded value **0.36310 ± 0.0005** (sigma carried over from the
polar-moment uncertainty; J2's own uncertainty is negligible in comparison)
as the actual `_solve_densities` target — fitting the unrounded 0.3644
polar value directly, as an earlier draft did, biases the fitted mantle/core
densities.

## Elastic-vs-anelastic caveat (important)

The observed k2 = 0.169 (Konopliv, Park & Folkner 2016) is measured from
Mars's actual tidal response, which includes an **anelastic** (dissipative)
contribution from mantle viscoelastic relaxation at the solar semidiurnal
tidal frequency. This stage-1 model is **purely elastic** (no viscosity in
any layer). Fitting the elastic model's k2 to the observed value therefore
slightly **underestimates the true mantle rigidity**: an elastic model
needs a somewhat softer mantle than an anelastic one would, in order to
reach the same k2, because anelastic relaxation itself increases the
effective tidal compliance beyond the purely elastic value. This is an
accepted approximation for a first reference model. Incorporating
anelasticity (e.g., an Andrade or Maxwell rheology in the mantle layers,
which pylov3d already supports via `eta0`) is left as future work.

## Caveat: Stähler-family core vs. the 2023 basal-melt reinterpretation

Khan, A., et al. (2023), "Evidence for a liquid silicate layer atop the
Martian core," *Nature*, 622, 718-723, doi:10.1038/s41586-023-06586-4,
and Samuel, H., et al. (2023),
"Geophysical evidence for an enriched molten silicate layer above Mars's
core," *Nature*, 622, 712-717, reinterpret the seismic CMB reflector
originally identified by Stähler et al. (2021) as instead marking the top
of a ~150 km thick molten (silicate-melt) basal mantle layer. Under that
reinterpretation, the metallic core itself is smaller (~1650-1675 km
radius) and denser (~6.65 g/cm^3) than the Stähler-family solution used in
this document. This stage-1 model deliberately adopts the Stähler et al.
(2021) / RISE 1830 km-radius core parameterization; the fitted core density
below (~6.13 g/cm^3) should be read as the **large-core, lower-density
branch** value corresponding to that parameterization, not as evidence
against the Khan/Samuel (2023) basal-melt branch. Revisiting the model
against that branch (smaller core, denser core, molten basal mantle layer)
is future work.

## Model structure

Four homogeneous layers, core (layer 0) to surface (layer 3). The liquid
outer core is represented natively by LOV3D's fluid core-mantle-boundary
(CMB) condition — layer 0 has `mu0 = 0` and no `ocean` flag is needed or
used; `pylov3d/boundary_conditions.py`'s `assemble_bc_no_ocean` always
treats layer 0 as the core, so a zero shear modulus alone reproduces a
liquid core. **The core's `Ks0` (bulk modulus) is numerically inert**: the
solver integrates layers 1..n_layers-1 only and applies an analytic
fluid-core boundary condition at the CMB (`pylov3d/solver.py`, integration
loop over `range(1, n_layers)`), so layer 0's `Ks0` (155e9 Pa, retained here
for documentation/physical-plausibility purposes only) never enters the
radial integration.

| Layer | Radii [km] | Density [kg/m^3] | mu0 [Pa] | Ks0 [Pa] | Notes |
|---|---|---|---|---|---|
| L0 core (liquid) | 0 - 1830 | 6128.1 (**fitted**) | 0 | 155e9 (inert, see above) | Density fitted; Ks nominally from Stähler et al. (2021) liquid-core Vp ≈ 5 km/s |
| L1 lower mantle | 1830 - 2340 | 4136.5 (**fitted**) | 100e9 × s = 9.648e10 (**fitted** s) | 160e9 | s = MARS_MU_SCALE |
| L2 upper mantle | 2340 - 3339.5 | 3400 (fixed) | 70e9 × s = 6.754e10 (**fitted** s) | 115e9 | Density from Khan et al. (2021) |
| L3 crust | 3339.5 - 3389.5 | 2900 (fixed) | 30e9 (fixed, Vs ≈ 3.2 km/s) | 70e9 | 50 km adopted thickness |

The L1/L2 (lower mantle / upper mantle) boundary is at **2340 km**
(~1050 km depth), the olivine → wadsleyite mantle phase transition inferred
from InSight seismic data by Khan et al. (2021), *Science*, 373, and
Stähler et al. (2021), *Science*, 373; the direct seismic detection of
this discontinuity is Huang, Q., et al. (2022), "Seismic detection of a
deep mantle discontinuity within Mars by InSight," *PNAS*, 119,
e2204474119. (An earlier draft placed this boundary at an arbitrary
2550 km with no cited basis; that has been corrected, and doing so also
changes the fitted densities and mu_scale — see "Corrections applied"
below.)

**Interpretation caveat on the fitted L1 density:** the fitted lower-mantle
density (4136.5 kg/m^3) exceeds the Khan et al. (2021) mantle range quoted
in the constraint table (~3400-4000 kg/m^3), and the resulting L1/L2
density step (21.6%) is far larger than the ~6-8% jump of the real
olivine → wadsleyite transition. This is a consequence of holding the
1000 km-thick L2 shell at a shallow-mantle density (3400), which forces L1
to absorb the remaining mass and moment. L1 should therefore be read as a
**mass/moment-balancing shell** whose boundary is *placed at* the phase
transition depth, not as a literal wadsleyite-layer density estimate.
A finer radial density parameterization would relax this.

## Fit method (deterministic — no black-box optimizer)

**Step 1 — densities, exact 2x2 linear solve.** With rho_um = 3400 and
rho_crust = 2900 fixed, (rho_core, rho_lm) are the unique solution of:

```
M = sum_i (4*pi/3) * rho_i * (R_i^3 - R_{i-1}^3)              = GM / G
I = sum_i (8*pi/15) * rho_i * (R_i^5 - R_{i-1}^5)             = MoI_factor * M * R^2
```

where `MoI_factor` is the **mean** moment 0.36310 (see derivation above),
not the published polar moment 0.3644.

The `8*pi/15` coefficient for a uniform shell's moment of inertia was
verified directly: a solid, uniform-density sphere has
`I = (2/5) M R^2 = (2/5) (4/3 pi rho R^3) R^2 = (8*pi/15) rho R^5`; a
shell's moment of inertia is the difference between the moments of inertia
of the two solid spheres bounding it (both at the shell's own density), so
the same `8*pi/15` prefactor applies to `rho_i * (R_i^5 - R_{i-1}^5)`.

This is a 2-equation, 2-unknown linear system in (rho_core, rho_lm), solved
directly with `numpy.linalg.solve` (`pylov3d.mars._solve_densities`) — not
an iterative fit. `_solve_densities` is **lazily evaluated** — it is a
plain (`functools.lru_cache`-memoized) function, not run at import time —
so that a bad constant/target raises only when actually called, which is
what makes the guard rails testable in isolation. A sanity check requires
`rho_core` in `[5700, 6300] kg/m^3` (mean core density 5.7-6.3 g/cm^3,
Stähler et al. 2021) and `rho_lm > rho_um`; violations raise `ValueError`.

**Step 2 — mantle shear-modulus scale, bisection on k2.** A single scale
factor `s` (`mu_lm = 100e9*s`, `mu_um = 70e9*s`) is found by bisecting
`s ∈ [0.3, 3.0]` so that k2 computed via `get_love` (forcing: `n=2, m=0,
F=1.0`, `Td = 44387.62 s` — half a sol (one sol = 88775.244 s), the
semidiurnal period; note some references (e.g. Pou et al. 2022, §3.1)
characterize the main Mars solar tide by the diurnal 24.6 h period —
because the model is purely elastic the choice does not affect the
result at all: elastic Love numbers are frequency-independent, Td is
only supplied because `make_forcing` requires one) equals the target
k2 = 0.169. Numerics:
`make_numerics(n_layers=4, method="combination", Nrbase=100)`.
k2(s) is monotonically decreasing over this range (stiffer mantle → smaller
tidal response), so plain bisection is sufficient and deterministic
(`pylov3d.mars.fit_mu_scale`); by default it converges to
`|k2 - 0.169| < 1e-4` and raises `RuntimeError` if `max_iter` is exhausted
first, or if the target isn't bracketed by `k2(s_lo)`/`k2(s_hi)`.

The converged value is hardcoded as `pylov3d.mars.MARS_MU_SCALE`, computed
by `fit_mu_scale(tol=1e-12)` (tighter than the default `tol=1e-4`, so that
the hardcoded constant is essentially exact), so that importing the module
never re-runs the tidal solver. `fit_mu_scale()` (default `tol=1e-4`) is
kept as the reproducibility path and is regression-tested to reproduce the
hardcoded constant to within 1e-3; `fit_mu_scale(tol=1e-12)` is
regression-tested to reproduce it *exactly*.

**Discretization note:** `MARS_MU_SCALE` is tied to
`method="combination"`. Re-solving with `method="fixed"` (same `Nrbase`)
shifts k2 by about −1.3e-5 (measured directly) — negligible relative to
both the 1e-4 fit tolerance and the 0.006 observational uncertainty, but
noted here because the two grid methods are not bit-identical.

## Achieved values vs. targets

| Quantity | Target | Achieved | Residual |
|---|---|---|---|
| Mass M | 6.416908889e23 kg (GM/G) | 6.416908889e23 kg | ~0 (exact by construction, 2x2 linear solve) |
| MoI factor I/MR^2 (mean, not polar) | 0.36310 ± 0.0005 | 0.36310 | ~0 (exact by construction, 2x2 linear solve) |
| k2 (n=2, m=0 solar semidiurnal) | 0.169 ± 0.006 | 0.169, residual under 1e-12 (s = MARS_MU_SCALE = 0.964824766102174, solved via `fit_mu_scale(tol=1e-12)`; the residual's trailing digits are environment-dependent) | well inside both the 1e-4 fit tolerance and the 0.006 observational uncertainty |

Fitted values: `rho_core = 6128.076 kg/m^3`, `rho_lm = 4136.504 kg/m^3`,
`MARS_MU_SCALE = 0.964824766102174` (giving `mu_lm ≈ 9.648e10 Pa`,
`mu_um ≈ 6.754e10 Pa`). At this solution, `h2 = 0.31563220568...` and
`l2 = 0.05159595220...` (both real to machine precision, as expected for a
purely elastic model; `h2 > k2` as generically expected for a
solid/liquid-core body). `rho_core` sits comfortably inside the
`[5700, 6300] kg/m^3` guard (about 172 kg/m^3 below the upper bound, 428
above the lower bound) — not edge-of-range, but on the dense side of the
guard, consistent with the "large-core/lower-density branch" caveat above.

**MATLAB cross-validation (TASK-014 part 1).** The same 4-layer model, run
through the native MATLAB LOV3D 1D solver (`scripts/mars_1d_cross_check.m`,
identical radii/densities/mu/Ks, purely elastic), reproduces all three Love
numbers to ~1e-12 relative: k2 = 0.169000000000 (2.0e-12), h2 =
0.315632205682 (1.2e-12), l2 = 0.051595952202 (8.6e-13). This gives Mars the
same independent MATLAB anchor the Moon has. In particular the elevated
**h2/k2 = 1.8676** — above the ~1.6-1.7 typical of published Mars models —
is reproduced bit-for-bit by MATLAB, confirming it as a feature of
this coarse 4-layer / purely-elastic parameterization rather than a Python
port artifact. (Porting note: MATLAB's `get_rheology` treats a layer as
elastic only when `eta0` is *empty/absent*; `eta0 = NaN` — pylov3d's elastic
sentinel — would instead enter the viscoelastic branch and yield a singular
BC matrix, so the MATLAB driver omits `eta0` for elastic layers.) The driver's
full console output and a small `.mat` of the computed Love numbers are
committed at `data/tests/mars/mars_1d_cross_check.{log,mat}` (MATLAB
R2025b, 25.2.0.3150157 Update 4) so the numbers above are verifiable without
re-running MATLAB.

## Tests

`pylov3d/tests/test_mars.py` checks (against the full suite, run with
`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`):

1. Mass of the built model within 0.1% of GM/G.
2. Mean MoI factor within 0.36310 ± 0.0005, and that the fit target is
   demonstrably the *mean* moment (differs from the published *polar*
   moment by exactly (2/3)·J2) — guards against silently re-fitting to the
   polar value.
3. k2 through `get_love` within the observational error bar (0.169 ± 0.006)
   and within 1e-3 of the fit target (regression pin on the hardcoded
   scale).
4. k2 real to < 1e-10 (elastic); h2 positive and h2 > k2; regression pins
   on h2 and l2 themselves (abs tolerance 1e-4).
5. Density profile monotonically non-increasing outward; the fitted
   densities themselves pinned to ~1e-6 relative (catches a wrong shell
   coefficient that would otherwise cancel in the mass/MoI ratio checks);
   rho_core within [5700, 6300] kg/m^3; and a dedicated test that
   `_solve_densities` (lazily evaluated, not run at import time) raises
   `ValueError` when called with a deliberately bad MoI target.
6. Fitted mantle mu×s values within [50e9, 200e9] Pa (physically
   plausible).
7. `fit_mu_scale()` (default tol) reproduces the hardcoded `MARS_MU_SCALE`
   within 1e-3, and `fit_mu_scale(tol=1e-12)` reproduces it exactly (marked
   slow-ok: runs several `get_love` solves during bisection; the `slow`
   marker is registered in `pylov3d/pyproject.toml`).

## Corrections applied (scientific review, post-initial-implementation)

The following errors were found in the original constraint table/spec and
corrected here (not introduced by this implementation, but fixed in it):

1. GM and k2 = 0.169's gravity field is **MRO120D**, not "GMM-3"; correct
   citation is Konopliv, Park & Folkner (2016), *Icarus* 274, with the
   title "An improved JPL Mars gravity field and orientation from Mars
   orbiter and lander tracking data" (the original spec used a different
   paper's title, Konopliv et al. 2006).
2. k2 = 0.174 ± 0.008 is from Konopliv et al. (2020), *GRL* 47,
   e2020GL090568 (Chandler wobble / MRO120F gravity) — **not** an "InSight
   RISE k2" (a lander cannot measure k2) and **not** Le Maistre et al.
   (2023).
3. Le Maistre et al. (2023) is *Nature* 619, pages **733-737** (not
   743-747), and is correctly cited only for the 1835 ± 55 km core-radius
   spin-state result.
4. The polar MoI 0.3644 ± 0.0005 is from Konopliv et al. (2011), *Icarus*
   211, 401-428 — not Folkner et al. (1997) (which remains a broadly
   consistent, independent older estimate, but is not the source of this
   specific number).
5. **Science-affecting:** 0.3644 is the *polar* moment C/MR^2, not the mean
   moment I/MR^2 that a spherically symmetric radial profile must be fit
   to. Retargeted to I/MR^2 = 0.36310 via C/MR^2 − (2/3)·J2 (J2 = 1.9555e-3,
   Konopliv et al. 2016 MRO120D) — this changes the fitted densities.
6. The core-density guard's cited range [5500, 7500] kg/m^3 misattributed a
   generic range to Stähler et al. (2021); narrowed to the paper's actual
   mean core density range, [5700, 6300] kg/m^3 (5.7-6.3 g/cm^3).
7. The lower-mantle/upper-mantle boundary (2550 km) had no cited physical
   basis; moved to 2340 km (~1050 km depth), the olivine → wadsleyite
   transition (Khan et al. 2021 / Stähler et al. 2021) — this also changes
   the fitted densities and requires re-solving `MARS_MU_SCALE`.
8. `MARS_FORCING_TD` corrected from 44339.8 s to 44387.62 s (half of one
   sol = 88775.244 s).
9. Added the Khan et al. (2023) / Samuel et al. (2023) basal-melt-layer
   caveat (see above), which was previously undocumented.

Re-fitting after corrections 5 and 7 changed `rho_core` (6090.2 → 6128.1
kg/m^3), `rho_lm` (3893.2 → 4136.5 kg/m^3), and `MARS_MU_SCALE`
(0.9191861808300019 → 0.964824766102174, and correspondingly h2/l2). Mass
and the (now mean, not polar) MoI factor remain exact by construction; k2
remains fit to 0.169 to a similarly tight tolerance.

## Monte Carlo framework (TASK-012)

TASK-011's fit above is a deterministic point estimate (exact 2x2 linear
solve for the densities, bisection for the shear-modulus scale). TASK-012
adds a body-agnostic Bayesian layer on top of the same physics, instantiated
here for Mars, so the same machinery also serves the Moon and (behind a
different Stage-2 solver) other bodies later.

**Three stages**, implemented in `pylov3d/forward.py`:

1. **Parameters -> model.** `LayerSpec`/`BodyParameterization` describe an
   ordered layer stack whose scalars are each FIXED (a float) or FREE
   (resolved from a named entry of a flat `theta` vector, optionally as a
   `Scaled` multiplicative factor shared across several layers).
   `build_model(parameterization, theta)` resolves this into an
   `InteriorModel` via `make_interior_model` — no body-specific code.
2. **Model -> observables.** `compute_observables(model, forcing, numerics,
   which=...)` returns a dict of `{"mass", "moi_mean", "core_radius_km",
   "k2", "h2", "l2"}`: mass/moi_mean/core_radius_km analytically from the
   shell density profile (mass/moi_mean via the same 4pi/3, 8pi/15 shell
   formulas as `pylov3d.mars._mass_and_moi`; core_radius_km is simply
   `model.R0[0]` — cheap, no tidal solve for any of the three); k2/h2/l2 via
   `get_love`. Accepts *any* `InteriorModel`, including ones built by
   `pylov3d.compat.planetstruct_to_interior_model` from a
   PlanetProfile/Perple_X output — this is the seam future petrological
   models plug into. Caveat: this stage assumes LOV3D's solid-body,
   fluid-core formulation (`boundary_conditions.py` always treats layer 0 as
   a fluid core); a fluid-dominated body (gas giants) will need a different
   Stage-2 implementation behind the same dict-returning interface.
   `numerics.method == "fixed"` is rejected outright
   (`NotImplementedError`): that grid method rewrites the solver's layer
   radii internally, which would silently decouple the analytic observables
   (read from the caller's original `model.R0`) from the Love numbers
   (computed on the solver's rewritten radii) — two slightly different
   bodies sharing one `theta` (found in review: ~2.2 sigma mass
   inconsistency). Use `"combination"` or `"variable"`.
3. **Likelihood/posterior.** `Constraint(name, value, sigma)` +
   `log_likelihood` (Gaussian, compares complex/viscoelastic observables by
   real part only) + `log_prior` (uniform-box bounds gate) +
   `make_log_posterior(...)` build a `callable(theta) -> float` that returns
   `-inf` outside the prior box or on any forward-model exception (guarded
   with a warning counter) — usable directly as `pocomc`'s `likelihood`
   argument (paired with a separate `pocomc.Prior` built from the same
   bounds for the properly normalized density and SMC proposal).
   `build_model` also rejects (`ValueError`) any `theta` that produces
   non-strictly-increasing layer radii (e.g. a free core radius exceeding
   the next fixed layer) — a zero/negative-volume shell that would
   otherwise silently reach the solver and could score a finite posterior.

**Mars instantiation** (`pylov3d/mars_mc.py`) reuses `pylov3d.mars`'s four
layers and every constant verbatim (no re-typed numbers) with 4 free
parameters and 4 constraints:

| Parameter | Bounds | Basis |
|---|---|---|
| `rho_core` | [5700, 6300] kg/m^3 | `pylov3d.mars.MARS_CORE_DENSITY_BOUNDS` — Stähler et al. (2021) core-density range, single source shared with `pylov3d.mars._solve_densities`'s own guard rail |
| `rho_lm` | [3400, 4400] kg/m^3 | around the TASK-011 fitted value |
| `mu_scale` | [0.3, 3.0] | shared `Scaled` factor on both mantle shear moduli |
| `R_core` | [1750, 1910] km | core_radius ± 2·sigma (Stähler et al. 2021, 1830 ± 40 km) |

Constraints: mass (`GM/G`, sigma = 0.1% — a deliberate modelling-error allowance (G and GM are both known far more precisely), far
looser than the MoI/k2 constraints below), `moi_mean` (0.36310 ± 0.0005),
`k2` (0.169 ± 0.006), and **`core_radius_km`** (1830 ± 40 km) — same
citations as the table at the top of this document. `mars_log_posterior()`
wires these together; `mars_point_fit_theta()` returns the TASK-011 point
fit as a `theta` vector for comparison/initialization.

**Identifiability: why `core_radius_km` is a 4th constraint, not just a free
parameter.** With 4 free parameters and only the 3 TASK-011 constraints
(mass, moi_mean, k2), the Jacobian of those 3 observables with respect to
the 4 parameters generically has a 1-dimensional null space: a combined
shift of all four parameters along that direction leaves mass, moi_mean,
and k2 exactly unchanged to first order, so the posterior is flat along that
ridge — an exact, not merely wide, non-identifiability (found in review:
"zero likelihood spread along the ridge"). This is *not* visible by probing
a single free parameter's axis alone while holding the other three fixed
(e.g. perturbing only `R_core` from the point fit *does* decrease the
posterior, because that axis-aligned direction is generically not aligned
with the flat ridge) — the degeneracy only shows up in the full 4-D
structure a sampler actually explores. The fix is a 4th observable/
constraint whose Jacobian row is independent of the other three's span:
`core_radius_km = model.R0[0]`, constrained at 1830 ± 40 km (Stähler et al.
2021) — this is exactly the seismological information that motivated
`R_core`'s bounds in the first place, previously left as a prior-box edge
only and never fed to the likelihood. `R_core`'s box was correspondingly
widened from ±1σ to ±2σ so this Gaussian constraint — not a hard box edge —
is what actually carries that information into the posterior.

**Sampler driver**: `scripts/mars_pocomc.py` runs `pocomc` (Preconditioned
Monte Carlo) over the 4 free parameters, saves the chain (`.npz`) and a
hand-rolled pairplot (no `corner` dependency), and prints posterior medians
± 1 sigma (3 significant figures) plus the effective sample size (ESS) next
to the point fit. `--quick` (n_active=16, coarse solver grid Nrbase=15,
non-dynamic termination) is a smoke run — it prints an explicit "NOT
CONVERGED — demo only" banner, since its marginal widths run measurably too
narrow (about 2x) relative to a converged/reference posterior (e.g.
propagating just the mass+moi_mean constraints analytically already implies
sigma(rho_core) ≈ 90 kg/m^3, larger than `--quick`'s own marginal). A full
run (`n_active=256` default, dynamic SMC annealing to beta=1) takes far
longer since the annealing schedule always runs to completion regardless of
particle count (measured: ~25 ms/forward-eval at Nrbase=15 warm, ~150
ms/eval at the fit's `Nrbase=100`; the TASK-015 production run used 33,280
likelihood calls including evidence estimation).

**TASK-015 production posterior.** The canonical proposal driver
`scripts/proposal_figures/fig2_mars_posterior.py` now defaults to the same
`Nrbase=100` radial grid as the TASK-011 deterministic fit. With
`n_active=256`, `n_effective=512`, dynamic SMC, seed 0, and pocomc's default
`n_total=4096`, the run completed in 5371.7 s (89.5 min). The archived chain
contains 4288 weighted samples with Kish ESS=4092.5; all samples, weights,
log likelihoods, and log priors are finite and every sample lies within the
declared bounds. The reproducible chain and publication corner plot are
`docs/figures/proposal/mars_posterior_chain.npz` and
`fig2_mars_posterior.{pdf,png}`.

| Parameter | posterior median | 15.87--84.13% interval | TASK-011 point fit |
|---|---:|---:|---:|
| `rho_core` [kg/m^3] | 6113.72 | 6019.97--6203.19 | 6128.08 |
| `rho_lm` [kg/m^3] | 4135.46 | 4027.10--4238.95 | 4136.50 |
| `mu_scale` | 0.97702 | 0.91782--1.04389 | 0.96482 |
| `R_core` [km] | 1835.85 | 1800.57--1871.11 | 1830.00 |

Thus every deterministic point-fit parameter lies inside its corresponding
central 68.26% marginal interval. Re-evaluating the model at the vector of
posterior marginal medians on `Nrbase=100` gives mass
`6.417822e23 kg`, mean MoI `0.3631182`, `k2=0.1684807`, and core radius
`1835.855 km`. This median vector is a summary of four marginals rather than
a separately optimized joint fit, but it remains close to all four targets.

**Surface map**: `pylov3d/mapping.py` provides `sh_to_latlon`, backed by
`fully_normalized_legendre` — a direct port of the MATLAB reference
recursion `src/SPH_Tools/Legendre.m` (Rapp 1982), replacing an earlier
`norm(n,m) * scipy.special.lpmv(m,n,t)` implementation that (found in
review) carried `lpmv`'s Condon-Shortley `(-1)^m` phase (sign-flipping every
odd-m map — a 180° rotation — relative to the documented no-CS convention)
and underflowed its `norm(n,m)` factor to exactly 0.0 beyond `n+m ≈ 170`
(`0 * inf = NaN`, poisoning maps at real gravity/shape data's degree, 120
and 719) — both invisible to a peak-to-peak-only check, which is how they
survived the original MATLAB-validated harness; see
`pylov3d/tests/test_mapping.py::TestMarsTopoHellasIntegration` for an
end-to-end regression against real MarsTopo719 data that catches this bug
class directly (recovers Hellas as the global minimum and Olympus Mons's
real elevation, not just a peak-to-peak number). Plus `plot_map` for quick
Agg-safe rendering. `scripts/mars_fit_map.py` renders the fitted model's
degree-2, zonal (n=2, m=0) tidal pattern in two normalized-unit panels (h2-
and k2-scaled, 4pi-normalized `P̄_2^0(sin(lat))` — polar value ≈ h2·√5 /
k2·√5, *not* h2/k2 themselves; see that script's docstring for the exact
scaling convention, its m=0-vs-Mars's-actual-m=2-tide caveat, and its
caveat vs. an absolute physical amplitude), annotated with the fitted
parameters and each constraint's target/achieved/residual — output PNG at
`scripts/output/mars_fit_map.png` by default.

Tests: `pylov3d/tests/test_forward.py` (a body-agnostic toy 3-layer body
exercising all three stages independently of Mars, including the
radius-monotonicity and `method="fixed"` guard rails; the Mars posterior's
finiteness and local near-maximality at the point fit; out-of-bounds ->
`-inf`; a `@pytest.mark.slow` micro `pocomc` run) and
`pylov3d/tests/test_mapping.py` (`sh_to_latlon` reproduces the original
`_delta_unit_map` values exactly; degree-2 zonal pattern symmetry checks;
no-Condon-Shortley-convention and high-degree-stability regressions; the
MarsTopo719 Hellas/Olympus Mons integration check).

## Lateral variations (TASK-016)

Turns the committed MarsTopo719 shape model into laterally varying crust
rigidity for the TASK-011 4-layer model above, and runs it through the
coupled (mode-coupling) solver already cross-validated against MATLAB on
the Weber Moon (`pylov3d/tests/test_matlab_validation_ocean.py`, fig5) to
produce the Mars Love-number *spectrum* -- response modes beyond the (2,0)
tide that lateral crustal structure excites. Implementation:
`pylov3d/mars_lateral.py` (full derivation in its module docstring; this
section states the approved design, the exact formulas, and the measured
numbers). Tests: `pylov3d/tests/test_mars_lateral.py`. Figure:
`scripts/proposal_figures/fig6_mars_lateral_spectrum.py` ->
`docs/figures/proposal/fig6_mars_lateral_spectrum.{pdf,png}`. Approved
design: `docs/tasks/TASK-016-design.md` (Airy compensation, `n_lv <= 4`,
fixed-amplitude forward runs -- no free lateral-amplitude MC parameter
yet).

### 1. Crustal thickness variation (Airy, areoid-referenced)

`crustal_thickness_variation(lmax=4)` loads MarsTopo719
(`pylov3d.sh_data.load_shape`), truncates to degree 4, drops C00 (mean
radius) and C20 (dominant rotational flattening -- an equilibrium-figure
effect, not a crustal load; same areoid-proxy precedent as fig3/the Hellas
integration test).

**Revision D2 (post-review): reference to the low-degree areoid, not the
bare sphere.** Airy compensation is properly a statement about topography
relative to a *level* (equipotential) surface, not a sphere with only the
rotational term subtracted. The first implementation skipped this: it
compensated the raw (C00/C20-dropped) MOLA shape directly. The correction
computes the low-degree areoid height from the committed GMM-3 gravity
field (`pylov3d.sh_data.load_shadr`, `data/mars/gmm3_120_sha.tab`, r0 =
3396 km) via the first-order Bruns relation: at the reference sphere r0,
the disturbing potential is `T(r0) = (GM/r0) * sum [Clm cos + Slm sin]
Pbar_nm`, and normal gravity there is (spherical approximation) `gamma0 =
GM/r0^2`, so the geoid undulation is `N = T/gamma0 = r0 * sum [Clm cos +
Slm sin] Pbar_nm` -- i.e. the SH *coefficient* of N is simply `r0 * C_lm`
(`r0 * S_lm` for the sine term). C00/C20 are dropped from the gravity field
too (matching the topography side), and `N` is subtracted from the shape
residual *before* the Airy factor:

```
dt = (h - N) * rho_c / (rho_m - rho_c),  rho_c = 2900, rho_m = 3400 kg/m^3
   = (h - N) * 5.8
```

**Measured** (`crustal_thickness_diagnostics(lmax=4)`): max|dt| = 34.2 km
(34.225 km at 180x360 grid resolution, converges to 34.230 km at 360x720)
-- down from 40.65 km pre-correction, and now much closer to the design
doc's ~20-30 km rule-of-thumb. Comfortably inside the `|dt| < 50 km`
shell-thickness bound (ratio 0.684, checked by `crustal_thickness_diagnostics`,
which raises `ValueError` -- not a bare `assert`, so it survives
`python -O` -- if this is ever violated). The peak sits at lat -8.4, lon
-106.4: **Tharsis**, precisely where the linearized-Airy approximation is
weakest in reality (Tharsis is known to be substantially flexurally/
dynamically supported, not purely Airy-compensated) -- so this single
largest excursion in the field is also the location the model's own
approximation is least trustworthy; reported, not hidden. Per-degree RMS
amplitude (post-D2): degree 1 = 11.06 km (unchanged -- GMM-3's C10/S10 are
identically zero by the center-of-mass coordinate convention, so the
areoid correction has no degree-1 term), degree 2 = 5.83 km, degree 3 =
5.69 km, degree 4 = 3.25 km. Degree 1 is now more clearly the largest
single degree; degrees 2-3 shrank relative to the pre-D2 numbers (7.70,
6.43 km) because part of what was being attributed to topographic load at
those degrees was actually the (now-subtracted) geoid signal. Two
compounding caveats on this dichotomy, both a direct consequence of what
this stage does and does not model: (a) it is somewhat *under-predicted*,
because a real crustal-density-contrast contribution to the low-degree
gravity field (not modeled here -- this stage uses a single fixed
`rho_c`) would, if included, partially cancel against the geoid
subtraction and restore some of degrees 2-3's amplitude; (b) Tharsis
itself (degrees 2-3-heavy) is *over-predicted* by the linearized-Airy
treatment for the reason above. Both push in the same direction: this
dichotomy measurement is not a precision result, only a rough,
order-of-magnitude "degree 1 comparable to 2-3" check, as the design doc
asked for.

One further dropped signal, stated explicitly: this analysis discards the
*entire* C20 term as rotational flattening. Mars's observed C20 is not
purely hydrostatic -- roughly 7% of it is a non-hydrostatic (real
mass-anomaly) contribution, equivalent to about a 2.4 km crustal root,
which this stage therefore also discards along with the hydrostatic 93%.
Not corrected in this pass; a literal (not proportional) hydrostatic-C20
model is the natural follow-up.

### 2. Crust-layer rigidity variation

The reference model's crust is a *fixed* 50 km shell (3339.5-3389.5 km,
`LAYER_MU_CRUST` = 30 GPa). `mu_variable_from_topography(lmax=4)`
linearizes the shell's crust fraction `f = t/50 km` in `dt`, giving the
fractional rigidity perturbation relative to the crust layer's own
reference modulus:

```
d(mu)/mu_bar = (mu_crust - mu_um_eff(mu_scale)) * dt / (50 km * mu_bar)
mu_bar = mu_crust = LAYER_MU_CRUST = 30 GPa
mu_um_eff(mu_scale) = _MU_UM_BASE * mu_scale = 70 GPa * mu_scale
```

`mu_bar = LAYER_MU_CRUST` because the crust *is* the model's surface layer
(index 3 of 4): `process_lateral_variations`'s elastic branch sets
`muC_amp[ilayer] = model.mu[ilayer] * amplitude`, and `model.mu` is
normalized to the surface layer's own `mu0` -- so `model.mu[3] == 1.0`
exactly and the supplied amplitude passes through unchanged. Sign: thicker
crust displaces stiffer effective-upper-mantle material out of the shell,
so mu *decreases* (dt > 0 -> d(mu) < 0, since mu_crust < mu_um_eff); this
is the literal design formula, applied with **no clipping** of `f` to
`[0, 1]` (a deliberate stage-1 choice, not an oversight).

**Revision D3 (post-review): `mu_scale` threading.** `mu_um_eff` depends
on the mantle shear-modulus scale factor `mu_scale` (`pylov3d.mars_mc`
samples this in `[0.3, 3.0]`); the first implementation froze it at the
fitted default `MARS_MU_SCALE` = 0.9648 at import time. Found in review:
this silently mismatches model and lateral amplitude whenever a caller
passes a non-default `mu_scale` to `build_mars_model` without *also*
threading it into `mu_variable_from_topography` -- a 7.5x error in the
lateral amplitude at `mu_scale=0.5` (since `mu_crust - mu_um_eff` changes
sign of magnitude sharply as `mu_um_eff` approaches `mu_crust`).
`_mu_um_eff`/`_dmu_ddt_coeff` are now plain functions of `mu_scale`
(default `MARS_MU_SCALE`, matching the previous frozen behavior exactly
when unspecified), and `mu_scale` is a parameter on
`dmu_over_mu_real`/`mu_variable_from_topography`/`mars_lateral_love_spectrum`/
`export_mu_variable_lateral`, forwarded consistently to `build_mars_model`.
This stage-1 harness itself only ever calls these at the fitted default;
the threading exists for `pylov3d.mars_mc` callers and any future
lateral-amplitude MC extension (design doc "Open decisions", item 3).

**Measured** (post-D2 areoid correction, at the fitted `MARS_MU_SCALE`):
peak |d(mu)/mu_bar)| = 0.857 at the same Tharsis grid point -- **down from
1.017 pre-D2**, so the elastic-positivity violation (the linearized
`mu_eff` implying a slightly negative crust shear modulus) is gone. This
was the D2 correction's primary motivation, not a side effect: the
original 40.65 km/1.017 result mixed real topographic load with
uncompensated long-wavelength gravity signal, exaggerating the peak
perturbation past a physically meaningful bound. Still reported, not
hard-guaranteed for all future inputs -- `crustal_thickness_diagnostics`
computes and returns `max_abs_dmu_over_mubar` every call (pinned `< 1.0`
by `test_mars_lateral.py::TestAiryNumbers::test_dmu_over_mu_below_elastic_positivity_bound`),
but does not itself clip -- a physically-clipped `f in [0,1]` version
remains a natural stage-2 follow-up (design doc "Open decisions").

### 3. Real -> complex spherical harmonics (the delicate part)

The shape file gives real, 4pi-normalized `(C_nm, S_nm)`
(`pylov3d.mapping` convention, no Condon-Shortley phase). The coupled
solver's `mu_variable` wants complex-SH amplitudes in the convention
*implicitly* defined by MATLAB `get_rheology.m`'s peak-to-peak conversion
(lines ~517-589), generalized here from a single-mode percent input to
arbitrary `(C_nm, S_nm)` pairs:

```
amp(n, 0)  = C_n0
amp(n, +m) = (C_nm - i*S_nm) / sqrt(2)          (m > 0)
amp(n, -m) = (-1)**m * (C_nm + i*S_nm) / sqrt(2)
```

With `S_nm = 0` this is *exactly*
`test_matlab_validation_ocean.py::_p2p_to_mu_variable` -- the formula
already end-to-end validated against native MATLAB Love numbers on the
Weber Moon (~2e-6 relative, 5 published cases), whose model is purely
elastic, the same `process_lateral_variations` code path (direct
`muC_amp = model.mu[ilayer] * amplitude` passthrough, no grid synthesis)
the Mars crust layer uses.

**Round-trip validation** (`pylov3d/tests/test_mars_lateral.py::TestCSHRoundTrip`,
the load-bearing test): the complex `Y_n^m` this amplitude convention
implicitly assumes was *derived*, not guessed, by requiring
self-consistency between the `S_nm = 0` case above and its `C_nm = 0`
sibling (from `get_rheology.m`'s `m < 0` branch) against the known real
fields `C_nm * Pbar_n^m * cos(m*phi)` and `S_nm * Pbar_n^m * sin(m*phi)`:

```
Y_n^0    =        Pbar_n^0(sin lat)
Y_n^{+m} =        Pbar_n^m(sin lat) * exp(+i*m*lon) / sqrt(2)   (m > 0)
Y_n^{-m} = (-1)^m Pbar_n^m(sin lat) * exp(-i*m*lon) / sqrt(2)
```

using the same no-Condon-Shortley, 4pi-normalized `Pbar_n^m` as
`pylov3d.mapping.fully_normalized_legendre` (`complex_sh_synthesis`).
Synthesizing a random degree-4 test field (and, separately, the actual
Mars `dt` field) this way and comparing pointwise against the direct
real-SH synthesis via `pylov3d.mapping.sh_to_latlon` on an independent
lat/lon grid: **max relative error ~6.6e-16** (random field) / machine
precision (Mars field) -- both far inside the required < 1e-10. This is
explicitly a *different* convention from the `scipy.special.sph_harm_y`
orthonormal convention used by `pylov3d.rheology._sh_synthesis` (that
helper is exercised only by `process_lateral_variations`'s *viscoelastic*
nonlinear-Maxwell grid path, never touched by the purely-elastic Mars
crust layer): an early probe that instead synthesized against
`_sh_synthesis` directly (the wrong convention for this, purely-elastic
code path) failed to reconstruct the real field, which is exactly the
failure mode this derived-not-assumed round trip exists to catch -- do not
conflate the two conventions.

### 4. Coupled solve

`mars_lateral_love_spectrum(lmax=4, forcing=(2,0), perturbation_order=2)`
builds `build_mars_model()`, applies `mu_variable_from_topography`, and
solves with `get_love(..., mu_variable=...)` (NumPy coupled path).

**Measured**: real MarsTopo719 at degree 4 has 23 nonzero rheology `(n,
m)` modes (cosine *and* sine are generically both nonzero, unlike a
synthetic single-mode test case) -- substantially more than the design
doc's assumption of a handful of dominant modes. This activates **N = 115**
coupled solution modes at `perturbation_order=2` (42 at
`perturbation_order=1`) -- well above the design's `N~15-30` rule-of-thumb,
reported rather than forced to fit; the D2 areoid correction changes *which*
real-valued amplitudes feed those 23 modes, not which `(n, m)` are
nonzero, so N is unchanged (115) pre- and post-correction. `get_couplings`
itself is cheap (~6.5 s); the radial solve dominates. `Nrbase=30` (the
production default) is converged to **1.4e-11 relative** in k2 against
`Nrbase=15` (re-measured after the D2 correction; corrects an earlier,
looser ~3e-9 figure from a different run). Wall time is machine-dependent
-- measured 91-181 s for Nrbase=15/30 respectively on one run of the
development machine, ~70-140 s on another -- comfortably inside the
design's 5-minute runtime guard either way (`Nrbase=100`, the 1D fit's own
resolution, is unnecessary at N=115 and was not used as the default for
that reason).

Forcing-mode (2,0) perturbation (post-D2): k2 shifts from the uniform
0.169000... to **0.1690552...** -- a shift of **~5.52e-5**, under 1% of
the observational uncertainty (0.006) and far inside the "<<0.006"
requirement: the lateral rigidity variation is a perturbation on the
fixed-forcing response, not a re-fit of it. See section 5 below for what
actually drives this shift.

**MATLAB cross-validation (TASK-014 part 2).** The same lateral model --
the exact 4-layer Mars body plus the crust-layer complex-SH `mu_variable`
field committed in `data/mars/mars_mu_variable_lateral.npz` -- run through
the native MATLAB LOV3D *coupled* solver
(`scripts/mars_lateral_cross_check.m`, `method='variable'`, `Nrbase=30`,
`perturbation_order=2`) reproduces the Python spectrum essentially
bit-for-bit: **N = 115 coupled modes (exact match)**; k2_uniform =
0.169000000002 (identical to 12 digits); forcing-mode k2 =
0.169055174106 (**rel err 2.95e-13**); k2 lateral shift = 5.517410e-5 vs
Python 5.517410435e-5 (the 7.9e-8 residual on the *shift* is pure
float64 cancellation from differencing two near-equal ~0.169 values -- the
raw k2 agrees to 2.95e-13). The non-forcing response spectrum matches mode
for mode -- (3,0) largest at -7.29e-5, then the (2,+/-2) conjugate pair at
+3.22e-5 +/- 2.03e-5i, the (3,+/-1) and (3,+/-3) pairs, etc. This gives the
Mars *lateral* model the same independent native-MATLAB anchor the 1D
model has (part 1). The driver reads the committed npz directly (a minimal
in-script `.npy` parser -- no Python bridge) and feeds the complex-SH
amplitudes to MATLAB's `mu_variable` path directly, bypassing the
peak-to-peak percent conversion; `eta0` is omitted on all four layers
(elastic -- the part-1 NaN-poisons-the-solve gotcha). The driver's full
console output and a small `.mat` of the computed coupled spectrum (all 115
modes: `n`, `m`, complex `k`, plus `k2_uniform`/`k2_forcing`/`k2_shift`) are
committed at `data/tests/mars/mars_lateral_cross_check.{log,mat}` (MATLAB
R2025b, 25.2.0.3150157 Update 4) so the numbers above are verifiable without
re-running MATLAB.

### 5. Linearity check, its subtlety, and the forcing-mode's first-order term

Scaling the Airy `mu_variable` amplitude by `eps in {1e-3, 1e-2}`, the
*dominant* (largest-amplitude) order-1 coupled modes scale linearly in
`eps` to < 0.3% (ratio ~10.00-10.03 vs. the exact-linear 10.0). A wider
probe across all 38 order-1-*discoverable* modes with resolvable amplitude
found 30 cleanly linear, 4 scaling ~quadratically (ratio ~100), and 4
ambiguous/near the float64 noise floor.

**Why "order-1-discoverable" is only an upper bound on the true leading
order, and what actually causes it (D4, corrected from an earlier, partly
wrong explanation).** `get_couplings.get_active_modes` discovers a mode's
*reachability* order by tracking `(n, m, ST)` triples (`ST` = spheroidal/
toroidal) through `next_coupling`, then collapses to `(n, m)` keeping the
*minimum* order and discarding `ST` (`pylov3d/couplings.py:143-147`). A
mode reachable as toroidal at order 1 (or with an order-1 coupling
coefficient that is zero to numerical precision) but spheroidal only at
order 2 is therefore labeled
"order 1" even though its *visible* response -- `k`, which by construction
comes only from the potential/spheroidal branch, since toroidal
deformation carries no gravitational potential perturbation -- is actually
order 2. Two distinct mechanisms produce this in the Mars rheology set,
verified directly against `coupling_coefficients` (not just inferred from
the response curve):
- **(3, +/-2)**: its only order-1 *spheroidal* channel (rheology mode
  `(3, +/-2)` coupling with the forcing mode) has a coupling coefficient
  of **~7.6e-17** -- a true, isolated selection-rule zero (not a
  parity/toroidal artifact; its other order-1 channels, via rheology
  `(2, +/-2)` and `(4, +/-2)`, *are* toroidal, but this one is spheroidal
  and simply numerically vanishes).
- **(5, +/-4)**: its only order-1 channel (rheology mode `(4, +/-4)`) has
  a *significant* coupling coefficient, **0.632** -- not small at all --
  but forcing degree 2 + rheology degree 4 + response degree 5 has odd
  parity (2+4+5=11), which `next_coupling` assigns to the toroidal branch.
  So `k` is zero at that order regardless of the coefficient's size, for a
  completely different reason than (3, +/-2).

The test (`test_mars_lateral.py::TestLinearity`) restricts its strict
linearity assertion to the *dominant* modes, where this ambiguity does not
arise (measured: the top ~20 by amplitude all agree with linear scaling to
<0.3%).

**D5: the forcing mode's own k2 shift is first order, not second, and is
dominated by one harmonic.** Naively, the forcing mode's own response
"should" be second order (it must return to `(2,0)` via two coupling
steps, since a single rheology coupling generically changes `(n,m)`). This
holds for most harmonics -- e.g. isolating rheology mode `(3,0)` alone and
fitting the shift's `eps`-scaling exponent between `eps=1e-3` and
`eps=1e-2` gives **2.002**, second order, as expected. But isolating
`(4,0)` alone gives exponent **1.000**: *first* order. The mechanism is
even parity self-coupling -- forcing degree 2 + rheology degree 4 +
response degree 2 has even parity (2+2+4=8, spheroidal), so `(4,0)`
rheology couples the forcing mode directly back to itself at order 1. **On
the Airy field this is the only such channel that contributes, because the
Airy path drops C20 by construction -- but (4,0) is not the only rheology
degree for which the mechanism is possible.** `(2,0)` rheology couples the
forcing mode to itself at order 1 by the identical even-parity argument
(2+2+2=6); the Airy path never exercises it only because it structurally
never retains a C20 term to couple with. Checked directly against
`coupling_coefficients`: max|C| = 0.6389 for (2,0), 0.8571 for (4,0),
identically zero for (1,0), (3,0), (5,0), (6,0) -- see "Non-Airy crustal
model substitution (TASK-028)" below, section 5, where a C20-retaining
field (unlike this one) makes both channels active simultaneously and they
substantially cancel. Measured absolute contribution on *this* (Airy,
C20-free) field: the `(4,0)`-alone shift, extrapolated linearly to the
harmonic's full physical amplitude, is **~1.75e-5** -- about a third of the
total measured shift (~5.52e-5, all 23 harmonics together) -- confirming it
is the dominant single contributor on this field, not a curiosity.
Consequences worth stating plainly for the proposal: (1) the forcing-mode
shift scales ~1:1 (not quadratically) with the Airy calibration for this
component, so it is comparatively sensitive to the Airy factor / crust
density assumptions; (2) Mars's observed k2 itself therefore carries a
real, if small (~5.5e-5 out of k2=0.169, i.e. ~3e-4 relative), first-order
signature of degree-4 zonal crustal structure *under the Airy assumption
specifically* -- a novel, proposal-relevant point distinct from the
non-forcing-mode spectrum shown in fig6, but one that does not generalize
unchanged to a C20-retaining crustal field (TASK-028 below). Pinned by
`test_mars_lateral.py::TestLinearity::test_forcing_mode_scaling_exponents`
(exponent bounds [0.9,1.1] for `(4,0)`, [1.8,2.2] for `(3,0)`).

### 6. Robustness: truncation convergence and Airy-calibration sensitivity (TASK-027)

The design doc asked for cutoff sensitivity to be *reported*, not assumed.
`(4,0)` -- the harmonic identified in section 5 as driving the
forcing-mode k2 shift to first order -- sits right at the `n_lv<=4`
truncation edge, making the forcing-mode shift the sharpest available
probe of whether that cutoff is adequate. An earlier revision of this
section reported a single lmax=4->5 spot check (8.3% step) and concluded
`n_lv<=4` was adequate. TASK-027 continued that sequence and separately
quantified the crustal-model dependence, using the driver
`scripts/mars_lateral_robustness.py` (driver-only; no solver module
modified, per the TASK-021b precedent). Artifacts:
`docs/figures/proposal/mars_lateral_robustness.{npz,png}`.

**Truncation ladder (fixed Nrbase=30).** Full coupled spectrum at
lmax=4/5/6, tracking the (2,0) forcing-mode shift and the off-(2,0)
response modes assessed for detectability in TASK-026 (section "Off-(2,0)
detectability"). |k| is the mode amplitude; the step columns are relative
to the previous lmax.

| lmax | N | (2,0) shift | Δ vs prev | \|k(3,0)\| | Δ | \|k(2,±2)\| | Δ | \|k(3,±1)\| | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 115 | 5.5174e-5 | — | 7.2893e-5 | — | 3.8071e-5 | — | 2.3521e-5 | — |
| 5 | 163 | 5.9732e-5 | +8.26% | 8.4130e-5 | +15.41% | 3.8053e-5 | −0.05% | 1.5699e-5 | −33.26% |
| 6 | 219 | 6.2311e-5 | +4.32% | 8.5118e-5 | +1.17% | 3.8577e-5 | +1.38% | 1.5442e-5 | −1.64% |

Reading: the forcing-mode shift is a *converging* sequence -- successive
steps shrink (8.26% -> 4.32%) and the sign is monotone upward -- but it is
not flat by lmax=6: the value is still moving ~4% per degree, so the
absolute shift is converged to roughly the ~4-6% level at lmax=6, not
better. lmax=4 and lmax=5 reproduce the earlier spot-check numbers
(5.517e-5, 5.973e-5) to the figures previously quoted; those two points
carry an independent native-MATLAB anchor at lmax=4 (below). Among the
off-modes, (3,0) and (2,±2) settle to ≤1.4% steps by lmax=6, but **(3,±1)
is misestimated at lmax=4**: it drops 33% from lmax=4 to lmax=5 before
settling (−1.6% at lmax=6), so lmax=4 is not adequate for the (3,±1)
tesseral amplitude specifically -- lmax≥5 is required for that mode.

**Radial (Nrbase) independence check (fixed lmax=4).** Truncation is an
angular question and should be radial-grid-independent (TASK-021b's
argument); this is verified rather than assumed. At Nrbase=15/30/50 the
(2,0) shift and every tracked off-mode are identical to <0.005% (the
lmax-ladder rows above are therefore legitimately run at Nrbase=30 despite
TASK-021b having hit >15 GB at lmax=6/Nrbase=50; this run's peak was ~11 GB
at lmax=6/Nrbase=30). This confirms the lmax ladder is a pure angular
convergence study, uncontaminated by radial resolution.

**Airy-calibration sensitivity (lmax=4, Nrbase=30).** Section 5 flagged the
crust/mantle-density (Airy factor) assumption as the weakest input,
because the (4,0)-driven forcing-mode shift scales ~1:1 with it. The
lateral rigidity field is *exactly linear* in
`AIRY_FACTOR = rho_crust/(rho_mantle − rho_crust)` (see
`crustal_thickness_variation`: `dt = clm·AIRY_FACTOR`, and the `dμ/dt`
coefficient is density-independent), and this factor is the only place
crust/mantle density enters the lateral field -- so the sweep is a linear
rescale of the baseline `mu_variable` followed by a full coupled
re-solve (perturbation_order=2 retains the quadratic self-terms of section
5, so outputs are re-solved, not scaled). Sweeping a wide defensible
bracket rho_crust ∈ {2700, 2900, 3100}, rho_mantle ∈ {3400, 3500} kg/m³
(baseline 2900/3400 → AIRY_FACTOR=5.8):

| quantity | min | max | spread (% of mean) |
|---|---|---|---|
| (2,0) shift | 2.2263e-5 (AF=3.375) | 1.6936e-4 (AF=10.333) | 216% |
| \|k(3,0)\| | 4.14e-5 | 1.46e-4 | 133% |
| \|k(2,±2)\| | 1.99e-5 | 9.05e-5 | 164% |
| \|k(3,±1)\| | 1.25e-5 | 5.20e-5 | 152% |

The forcing-mode shift varies by a factor ~7.6 across this bracket, and
its response is *super-linear* at the high-AIRY_FACTOR end (a ×1.78 factor
change produces a ×3.07 shift change), because the order-2 self-coupling
term (section 5) grows quadratically while the (4,0) first-order term grows
linearly. **The crustal-model uncertainty therefore dominates the
truncation uncertainty by more than an order of magnitude** (~factor-7
bracket vs. ~4-6% per-degree truncation step). This is the sensitivity §5
anticipated, now quantified. The wide bracket is deliberately an upper
bound; the InSight-calibrated non-Airy crustal-thickness substitution
(TASK-027 Part 2 second pass) needs a data fetch and is deferred to
Machine A, so this spread should be read as "how much the headline could
move under crustal-model choice", not as a posterior uncertainty.

**Native-MATLAB anchor for the diagonal (2,1)/(2,2) forcing modes
(TASK-027 Part 3).** The diagonal order-splitting Love numbers used in the
detectability analysis (`MARS_K21_FORCING`, `MARS_K22_FORCING`) previously
had a Python-only provenance. `scripts/mars_lateral_cross_check.m` now runs
the native LOV3D coupled solver on the committed model + `mu_variable`
field for all three forcing orders m=0/1/2 (artifacts:
`data/tests/mars/mars_lateral_cross_check.mat`,
`mars_lateral_cross_check_k2m.log`):

| forcing | MATLAB diagonal k | Python reference | rel. err |
|---|---|---|---|
| (2,0) | 0.169055174106 | 0.1690552 (7-fig) | 1.5e-7 |
| (2,1) | 0.169020913466 | 0.16902091346458947 | 7.1e-12 |
| (2,2) | 0.169033995416 | 0.16903399541441133 | 7.0e-12 |

The (2,1)/(2,2) values match to ~7e-12 (the (2,0) figure is limited only by
the 7-digit rounded Python reference; its underlying agreement is at the
same machine-precision level as the m=0 spectrum's 2.95e-13). This is a
lmax=4 anchor: it validates the diagonal values at the production cutoff,
but does not by itself resolve the separate lmax-convergence of the m=1,2
*ordering* flagged in the TASK-026 detectability section (that would need
the m=1,2 forcing re-run at lmax=5/6, which this task did not perform).

**m-ordering convergence (TASK-030).** That re-run has now been done.
`scripts/mars_m_ordering.py` runs the diagonal (2,0)/(2,1)/(2,2) forcing
solves at lmax=4/5/6 at the identical config (Nrbase=30,
`method='combination'`, perturbation_order=2, via the validated
`mars_lateral_love_spectrum` path), and reports the diagonal shift
`dk2m = Re[k2m(lmax)] - k2_uniform` (artifacts:
`docs/figures/proposal/mars_m_ordering.{npz,png}`,
`data/tests/mars/mars_m_ordering.log`; the lmax=4 row reproduces
`MARS_K21_FORCING`/`MARS_K22_FORCING` to ≤0.01%):

| lmax | dk2(2,0) | dk2(2,1) | dk2(2,2) | ordering (\|dk\| desc) |
|---|---|---|---|---|
| 4 | 5.5174e-5 | 2.0913e-5 | 3.3995e-5 | m=0 > m=2 > m=1 |
| 5 | 5.9732e-5 | 2.4731e-5 | 3.6225e-5 | m=0 > m=2 > m=1 |
| 6 | 6.2311e-5 | 2.5169e-5 | 3.7962e-5 | m=0 > m=2 > m=1 |

**The ordering m=0 > m=2 > m=1 is stable across the whole ladder**, and each
`dk2m` grows monotonically with lmax. The closest pair, m=2 vs m=1, stays
well separated: its gap is 1.31e-5 (lmax=4), 1.15e-5 (lmax=5), 1.28e-5
(lmax=6) — it dips at lmax=5 but never approaches the crossing that would
reorder them (the gap stays ~50% of the m=1 shift itself throughout). The
proposal's
"the ordering should not be relied upon" caveat is therefore retired: the
m=1,2 solves are now carried to the same lmax=6 as m=0, and the ordering is
unchanged. This concerns the *ordering* only; the absolute amplitudes still
carry the ~4-6% lmax-6 truncation residual and the larger Airy/crustal-model
uncertainty stated above and in the TASK-027 ladder.

**Do the section-5 conclusions survive?** The qualitative (4,0)-driven
first-order forcing-mode shift survives: the shift is real, grows (does not
shrink) with lmax, and is present in native MATLAB -- it is not a
truncation artifact, and lmax=4 if anything *under*-states it. What does
*not* survive as a precise number is any claim of a converged absolute
amplitude: the shift carries a ~4-6% residual truncation uncertainty at
lmax=6 and a much larger (factor-several) crustal-model uncertainty. The
detectability-relevant off-mode amplitudes (3,0) and (2,±2) are trustworthy
at lmax≥4, but (3,±1) requires lmax≥5. All off-mode amplitudes are subject
to the same order-of-magnitude Airy-calibration spread. The
`test_mars_lateral.py::TestTruncationSensitivity` 20%-per-step bound
(marked `@pytest.mark.slow`) still holds for the forcing mode.

### 7. Export for MATLAB cross-check

`export_mu_variable_lateral()` writes
`data/mars/mars_mu_variable_lateral.npz`: the crust-layer `mu_variable`
entries (layer/n/m/complex amplitude), the real (areoid-corrected) `dt` SH
coefficients, provenance constants (including the `mu_scale` actually
used -- D3 threading, section 2), and an embedded README string --
including the eta0 empty-vs-NaN convention warning (`docs/HANDOFF.md`,
TASK-014 part 1: Python uses `eta0 = NaN` for elastic, MATLAB requires
`eta0` *empty*, not NaN) and a 1-based-vs-0-based layer-index clause
(`crust_layer_index` is 0-based Python numbering, 3; the same layer is
MATLAB `Interior_Model(4)`, 1-based) -- for machine B's native-MATLAB
coupled cross-check (TASK-014 part 2).

## 3D spatial response maps (TASK-041)

The lateral stage's deliverables had all been spectral. This section is the
first spatial synthesis: the time-varying tidal gravity and radial
displacement anomaly over Mars's surface that the lateral spectrum implies.
Drivers: `scripts/mars_lateral_spectrum.py` (part 0),
`scripts/mars_response_maps.py` (part 1),
`scripts/proposal_figures/fig10_mars_response_maps.py` (part 2). Artifacts:
`docs/figures/proposal/mars_lateral_spectrum.npz`,
`mars_response_maps.npz`, `fig10_mars_response_maps.{png,pdf}`.

**Part 0 closed an artifact gap:** Mars had no committed full-spectrum
archive (the Moon has one; Mars's record was the k-only MATLAB anchor
`.mat`). One coupled solve (shipped config, 150.6 s) produced n/m/k/h/l +
uniform references; gated on per-mode agreement with the `.mat` before
use — 115/115 modes, median rel err 4.61e-12, worst 1.55e-10 at (8,±3).

**Scaling convention** (reused from TASK-026, not re-derived):
`S = solar_tide_amplitude_parameter × peak_legendre_factor(2,0)` =
2.278e-9, the (2,0) solar tide's dimensionless potential amplitude at the
obliquity-constrained sub-solar peak. TASK-026's scope caveat carries
over verbatim: the spectrum is the response to unit (2,0) forcing; the
real solar tide has m=0/1/2 components at distinct frequencies, so these
maps are the spatial pattern of the off-(2,0) response as a class.
Mapped is the **lateral part only** (off-forcing modes plus the
forcing-mode shift Δk20/Δh20); the uniform tide is subtracted.

**Results:**

| Observable | Peak | Location | Tharsis | Dichotomy (repr.) | Hellas | InSight |
|---|---:|---|---:|---:|---:|---:|
| gravity anomaly | 0.0011 µGal | 80.6°S, 247.5°E | 0.0008 | 0.0004 | 0.0004 | 0.0001 |
| radial displacement | 0.00355 mm | 57.7°N, 329.5°E | 0.00236 | 0.00169 | 0.00167 | 0.00070 |

**The two observables carry different physics.** The gravity map is
dominated by degree 3 (64.6% of variance; degree 2 32.7%) — a Tharsis
lobe plus a strong southern zonal band. The radial-displacement map is
dominated by **degree 1 (70.0% of variance)**: the crustal dichotomy
couples the (2,0) tide into a north–south asymmetric radial component,
with h(1,0) = 1.59e-4 — the largest lateral radial-displacement mode —
while k(1,m) ≈ 1e-18, i.e. zero at machine precision. That zero is a
physics check, not an omission: the external degree-1 potential must
vanish in the center-of-mass frame, and the solver delivers exactly that.

The degree-1 displacement interpretation needs the tangential Love
number too. In the standard spheroidal Love-number convention a rigid
translation has h(1,m) = l(1,m), whereas the committed zonal response has
h(1,0) = 1.58741e-4 and l(1,0) = 3.40875e-4; the nonzero
h(1,0)-l(1,0) = -1.82135e-4 shows that the response is not a pure rigid
translation. Nevertheless, h(1,m) alone is frame-dependent. Predicting a
differential lander-network observable requires synthesizing the full
vector displacement (radial h plus tangential l) in a stated ephemeris /
center-of-mass frame. The radial map by itself establishes the spatial
component and its scale, not network observability.

**Caveats.**

1. **The h spectrum has no MATLAB anchor.** The `.mat` cross-check
   carries k only, so the displacement map rests on Python-only h values
   (the uniform h2 is validated through the 1-D anchors, but the coupled
   lateral h modes are not). Extending the MATLAB cross-check to h/l is
   a natural ticket if displacement numbers become load-bearing.
2. Amplitudes are small in absolute terms (sub-µGal, µm-scale). The
   detection pathway remains the spectral/Stokes route of TASK-026 —
   these maps say *where* the signal lives and its pattern, not that an
   instantaneous surface measurement resolves it.
3. The reference-shell qualification (§4.4 of the proposal; TASK-036/037)
   applies to every amplitude here: the absolute scale is conditional on
   the 50 km Voigt shell.

## Non-Airy crustal model substitution (TASK-028)

TASK-027 quantified the Airy crust/mantle-density calibration bracket at a
factor ~7.6 in the (2,0) forcing-mode shift, against a truncation-ladder
uncertainty of only ~1.13x (4-6% per lmax step, section 6 above) --
identifying the crustal-model assumption, not the truncation cutoff, as the
dominant remaining error term. That sweep varied the Airy *parameters*
(crust/mantle density) but kept the Airy *assumption itself* -- topography
locally, isostatically compensated by crustal thickening -- fixed. This
section replaces the Airy-derived crustal-thickness field with five
InSight-calibrated crust-mantle interface (Moho) models
(`data/mars/insight_moho/`; provenance, format, and selection criterion in
`data/mars/insight_moho/SOURCES.md` -- full author list not independently
retrieved this session, so not quoted here) and re-runs the identical
downstream lateral-rigidity/coupled-solve machinery, to ask directly: does
the Airy assumption, not just its density parameters, bias the lateral
spectrum? Implementation: `pylov3d/mars_crust_models.py`; `pylov3d.mars_lateral`
is imported, not modified. Tests: `pylov3d/tests/test_mars_crust_models.py`.

The result is not the simple "Airy validated, use a single corrected
sensitivity number" story an earlier pass through this analysis reported.
Retaining C20 (the deliberate, C20-decision departure from the Airy path --
section 1 below) turns out to activate a *second* first-order coupling
channel into the forcing mode that the Airy field structurally cannot
exercise, and that channel nearly cancels the one already documented in
section 5 above. Sections 5-6 below are the load-bearing result of this
task; sections 1-4 are the supporting field construction and validation.

### 1. The crustal-thickness field: direct geometric difference, not compensation

`data/mars/insight_moho/` holds five spherical-harmonic Moho-radius models
(lmax=90, same real 4pi-normalized, no-Condon-Shortley convention as
MarsTopo719, `pylov3d.sh_data.load_shape` reads them unmodified), spanning
distinct interior models (DWAK, DWThotCrust1, EH45Tcold, EH45TcoldCrust1r,
Khan2022), all at the project's own crust density (2900 kg/m^3) and within
1 km of the 50 km reference mean thickness (SOURCES.md's selection
criterion), so a substitution changes the lateral *pattern*, not the mean
thickness or the density assumption. `moho_thickness_variation(model, lmax)`
computes `dt = R_topo - R_moho` as a direct coefficient-wise SH subtraction
(exact, since both fields share the same expansion convention) and removes
only the (0,0) mean.

**The C20 decision.** Unlike the Airy path
(`crustal_thickness_variation`, section 1 above), which drops C20 from
*topography* because it is dominated (~93%, section 1 above) by rotational
flattening -- an equilibrium-figure effect that bulges the whole body, not a
crustal-density load -- this module **retains C20**. The Moho fields are not
derived by applying an isostatic-compensation assumption to topography; they
are independently inverted (gravity + seismic) radius fields, and `dt` is
their literal geometric difference from the topographic surface. Zeroing
C20 would only be correct if the Moho flattened in lockstep with the
topographic surface, making the difference's C20 a redundant restatement of
the same rotational bulge. It measurably does not:

| model | Moho C20 [m] | (R_moho/R_topo)^2-scaled prediction [m] | ratio |
|---|---|---|---|
| DWAK | -2057.6 | -5792.1 | 0.36 |
| DWThotCrust1 | -2247.7 | -5792.9 | 0.39 |
| EH45Tcold | -2114.8 | -5790.3 | 0.37 |
| EH45TcoldCrust1r | -2251.0 | -5789.8 | 0.39 |
| Khan2022 | -2745.1 | -5789.9 | 0.47 |

(topo C20 = -5966.2 m, R_moho/R_topo = 0.9853.) Every model's Moho is
flattened only 0.36-0.47x as much as the scaled prediction, consistently,
not as an outlier of one model.

*Comparator choice, addressed directly because a review raised it:*
`(R_moho/R_topo)^2` is not the general hydrostatic law -- Clairaut's theorem
gives flattening proportional to `r` for a homogeneous body and to `r^3` in
the centrally-condensed limit; `r^2` is intermediate, a plausible but not
uniquely-justified interpolation. It is used anyway because R_moho/R_topo =
0.9853 is close enough to 1 that the exponent barely moves the prediction:
DWAK's ratio is 0.350/0.355/0.361 for exponent 1/2/3, and the full
five-model x three-exponent grid spans 0.350-0.481. The "measurably
decoupled" conclusion is insensitive to which power is chosen. As an
independent cross-check with different physics (Bouguer root of the
non-hydrostatic residual rather than a pure level-surface scaling): the
already-recorded fact (section 1 above) that ~7% of Mars's observed C20 is a
non-hydrostatic mass anomaly equivalent to roughly a 2.4 km crustal root is
consistent, in sign and rough magnitude, with the 3-4 km gap between the
scaled prediction (~-5790 m) and the models' actual C20 (-2058 to -2745 m);
this cross-check was not independently re-derived to the same numerical
precision this session and is reported as a qualitative consistency check,
not a pinned figure.

**Sign, stated plainly because an earlier revision of this analysis had it
backwards:** retaining the Moho's actual (small-magnitude) C20 instead of
the larger passive-geometry value makes the crust **thinner** at the poles,
not thicker. `dt`'s C20 term alone (DWAK, C20 = -3908.6 m) evaluates to
-8740 m at either pole, against only -389 m from the hypothetical
passive-geometry ((R_moho/R_topo)^2-scaled) value -- about 8.4 km of
*additional* thinning at the poles beyond what passive geometry alone would
produce (equivalently, thickening around the equator, consistent with the
equatorial Tharsis root). The zonal-mean of the *full* `dt` field confirms
this independently and by a larger margin (all other retained degrees add
further structure at high latitude): -24.2 km at 84.5N vs. +12.6 km at
85.5S.

A second, independent argument for retaining C20: the 1D reference model
this feeds (`pylov3d.mars.build_mars_model`) has a perfectly spherical
crust-mantle boundary at every layer, so there is no separately-flattened,
load-free reference to avoid double-subtracting against -- unlike the Airy
path, which needs to remove C20 specifically because it is testing
topography *against* an isostatic-load assumption that the rotational bulge
violates. Consequently, no areoid (Bruns-relation) referencing is applied
here either, for the same reason: that correction exists in the Airy path
to convert bare topography into height-above-the-level-surface before
invoking isostatic support, and there is no such support assumption to
correct for when `dt` is already a directly observed thickness field.

### 2. Diagnostics

`moho_thickness_diagnostics` mirrors `crustal_thickness_diagnostics`'s
contract (shell-residency bound, elastic-positivity bound, degree RMS) plus
named-location geographic sanity. Measured, lmax=4, all five models:

| model | peak \|dt\| [km] | \|dt\|/50km | max \|dmu/mu_bar\| | dt(Tharsis) [km] | dt(Hellas) [km] | dt(Utopia) [km] |
|---|---|---|---|---|---|---|
| DWAK | 38.72 | 0.774 | 0.9689 | +37.94 | -11.61 | -35.31 |
| DWThotCrust1 | 33.00 | 0.660 | 0.826 | +32.48 | -9.89 | -29.69 |
| EH45Tcold | 34.57 | 0.691 | 0.865 | +34.00 | -10.37 | -31.20 |
| EH45TcoldCrust1r | 34.41 | 0.688 | 0.861 | +33.83 | -10.32 | -31.12 |
| Khan2022 | 37.89 | 0.758 | 0.948 | +37.00 | -11.35 | -34.91 |
| *Airy (reference)* | *34.2* | *0.684* | *0.857* | | | |

All five stay inside the `|dt| < 50 km` shell-residency bound and the
elastic-positivity bound (`|dmu/mu_bar| < 1`), though DWAK sits close to the
positivity bound (0.9689 against a bound of 1.0) -- close enough that this
linearized rigidity map cannot explore the excluded crust/mantle-density
axis without risking that bound on at least one of the InSight fields (see
section 6 below). Geographic sanity holds for all five: the peak positive
excursion is at Tharsis (matching the Airy field's own peak location,
lat -8.4/lon -106.4), and the two named low-degree crustal-thinning
provinces, Hellas and Utopia, are both negative, with Utopia's excursion
somewhat larger in magnitude than Hellas's at this truncation.

### 3. Comparison: Airy baseline vs. the five InSight models

`compare_crustal_models` at the project's validated `lmax=4`, `Nrbase=30`,
`perturbation_order=2` -- the as-shipped configuration, C20 retained on the
InSight side, never present on the Airy side:

| model | (2,0) shift | vs. Airy | \|k(3,0)\| | vs. Airy | \|k(2,±2)\| | vs. Airy | \|k(3,±1)\| | vs. Airy |
|---|---|---|---|---|---|---|---|---|
| *Airy* | *5.5174e-5* | *--* | *7.2893e-5* | *--* | *3.8071e-5* | *--* | *2.3521e-5* | *--* |
| DWAK | 4.7689e-5 | 0.864x | 8.3806e-5 | +15.0% | 4.2888e-5 | +12.7% | 2.9423e-5 | +25.1% |
| DWThotCrust1 | 3.1962e-5 | 0.579x | 6.9600e-5 | -4.5% | 3.4584e-5 | -9.2% | 2.3622e-5 | +0.4% |
| EH45Tcold | 3.5495e-5 | 0.643x | 7.3135e-5 | +0.3% | 3.6761e-5 | -3.4% | 2.5035e-5 | +6.4% |
| EH45TcoldCrust1r | 3.5849e-5 | 0.650x | 7.3291e-5 | +0.5% | 3.6573e-5 | -3.9% | 2.5039e-5 | +6.5% |
| Khan2022 | 4.9560e-5 | 0.898x | 8.4691e-5 | +16.2% | 4.1730e-5 | +9.6% | 2.9379e-5 | +24.9% |

The three off-diagonal modes are the ones TASK-027's truncation/Airy-
sensitivity tables track. Their spread across the five models: (3,0)
0.9548-1.1619x Airy (spread x1.217), (2,±2) 0.9084-1.1265x (spread x1.240),
(3,±1) 1.0043-1.2509x (spread x1.246) -- note (3,±1)'s bound is **one-sided**
(+0.4% to +25.1%; the whole InSight range sits *above* Airy, not
bracketing it), and (3,±1) is separately known unconverged at lmax=4
(section 6 above: -33% from lmax 4->5), so its InSight-vs-Airy comparison
at lmax=4 specifically should not be over-read. The (2,0) forcing-mode
shift behaves qualitatively differently from the three off-modes -- see
section 5-6 below for why, and section 6 for the crustal-model-sensitivity
statistics this table implies.

### 4. Pattern validation against the Airy field

Correlating each InSight `dt` field against the Airy field's, area-weighted
by `cos(lat)` over a 1x1-degree grid, **on the field as it actually ships
(C20 retained on the InSight side)**:

| model | correlation | rms ratio (model/Airy) |
|---|---|---|
| DWAK | 0.958 | 1.206 |
| DWThotCrust1 | 0.952 | 1.019 |
| EH45Tcold | 0.953 | 1.069 |
| EH45TcoldCrust1r | 0.955 | 1.066 |
| Khan2022 | 0.966 | 1.192 |

Range: correlation 0.952-0.966, rms ratio 1.019-1.206 (up to +21% high). An
earlier pass through this analysis reported correlation 0.984-0.986 and rms
within +/-17% -- those figures are real but were computed on a **C20-excluded**
field (C20 stripped from both the InSight and Airy sides before comparing),
which is not what ships or what feeds the solver:

| model | correlation (C20 excluded both sides) | rms ratio (C20 excluded both sides) |
|---|---|---|
| DWAK | 0.984 | 1.174 |
| DWThotCrust1 | 0.986 | 0.984 |
| EH45Tcold | 0.985 | 1.033 |
| EH45TcoldCrust1r | 0.985 | 1.033 |
| Khan2022 | 0.984 | 1.170 |

Both tables are legitimate measurements of different things: the first is
the pattern agreement of the field that actually drives the coupled solve;
the second isolates how well the two paths agree on everything *except* the
one harmonic (C20) where they structurally disagree by construction (the
Airy path always drops it; this path always keeps it for the InSight
models). Either way, part of the agreement is built in rather than fully
independent: the InSight inversions use the same MOLA topography as the
Airy path's input, and a Mars gravity field evaluated at the same crustal
density (2900 kg/m^3) the Airy path assumes -- so this is not a fully
independent validation of the Airy pattern, only a partially independent
one.

### 5. The headline: (2,0)/(4,0) sign cancellation

Section 5 above (TASK-016) established that (4,0) rheology couples the
(2,0) forcing mode back to itself at first order, via even-parity
self-coupling (2+2+4=8), and called it "the one rheology degree in the
`n_lv<=4` set for which this is possible for a degree-2 zonal tide." That
claim is corrected here: **it is two degrees, not one.** Checked directly
against `pylov3d.couplings.coupling_coefficients(n=2, m=0, na=2, ma=0,
nb=nb, mb=0)` for every even zonal rheology degree in the truncated set:

| rheology degree (n,0) | max\|C\| |
|---|---|
| (1,0) | 0.0 |
| (2,0) | 0.6389 |
| (3,0) | 0.0 |
| (4,0) | 0.8571 |
| (5,0) | 0.0 |
| (6,0) | 0.0 |

Both (2,0) and (4,0) have nonzero self-coupling back to the (2,0) forcing
mode; the odd degrees are identically zero. The rule is: for a degree-2
zonal tide, even zonal rheology degrees (2,0) and (4,0) both couple the
forcing mode to itself at first order -- parity requires an even rheology
degree, the triangle inequality caps it at 4 (since |n_forcing - n_rheology|
<= n_response <= n_forcing + n_rheology with n_forcing = n_response = 2
requires n_rheology <= 4), and order conservation restricts it to m=0. The
Airy path never exercises the (2,0) channel because it drops C20 by
construction (section 1 above); a measured Moho field does not have to.

The Airy field's own (4,0)-alone shift (section 5 above, ~1.75e-5,
extrapolated to full amplitude) is not affected by this correction -- it
remains a real, isolated measurement of that one field. What changes is the
picture on a C20-retaining field. On the DWAK field, isolating each channel
in turn (rheology entries for (2,0) and (4,0) only, scaled by
`eps in {1e-3, 1e-2}` and linearly extrapolated to full physical amplitude,
`perturbation_order=2`, confirmed non-artifact across `eps` in [1e-4, 1e-1]
and `perturbation_order` in {1, 2, 3}; at `perturbation_order=1` the (2,0)
shift is fully present while (3,0) is at float noise, consistent with (2,0)
being truly first order and (3,0) truly second):

| channel | signed shift (extrapolated to full amplitude) |
|---|---|
| (2,0) alone | -1.5901e-5 |
| (4,0) alone | +1.4528e-5 |
| both together | -1.3738e-6 |

The two channels sum to -1.3726e-6, matching the jointly-solved value to
0.1% -- exactly what linear superposition of two first-order terms
requires. But the *net* is a **91.4% cancellation** relative to the larger
single-channel contribution (|-1.3738e-6| / |-1.5901e-5| = 0.086). The
first-order mechanism itself is robust and mode-selection-rule-derived, not
a numerical artifact. Its **net amplitude is not** a fixed physical
constant -- it is the small difference of two comparable, opposite-sign
terms, so it is sensitive to exactly which crustal field supplies the
(2,0)/(4,0) amplitude ratio. The Airy path's +1.75e-5 (4,0)-alone figure is
a property of the Airy field specifically (which has no (2,0) channel to
cancel against); it does not generalize to "the" degree-4 zonal
first-order contribution, because on a C20-retaining field it is
substantially cancelled by the (2,0) channel.

### 6. The (2,0) forcing-shift statistics need three numbers, not one

Because of the cancellation in section 5, the (2,0) forcing-mode shift
(unlike the three off-diagonal modes in section 3) is sensitive to the C20
convention as well as to the crustal pattern, so a single "x-factor" spread
statistic is not sufficient -- three numbers are needed, each describing a
different comparison:

1. **x1.551**, the spread of the (2,0) shift across the five InSight models
   *as shipped* (min 3.1962e-5 DWThotCrust1, max 4.9560e-5 Khan2022) -- the
   number that best describes "how much does the forcing-mode shift move if
   you swap which real crustal model you use, exactly as the pipeline
   actually runs it."
2. **x1.726**, the same spread *including* the Airy baseline (min still
   DWThotCrust1, max still Khan2022 -- Airy's 5.5174e-5 sits inside the
   InSight range, not outside it) -- the number that describes "how much
   does the forcing-mode shift move across everything on the table,
   including the Airy assumption itself."
3. **x1.379**, the spread with **C20 suppressed on both sides** (the InSight
   models' C20 zeroed to match the Airy path's structural omission) -- the
   like-for-like *pattern-only* sensitivity, isolating the crustal-model
   effect from the C20-convention effect. Under this comparison the five
   models bracket Airy **symmetrically**, from -15.1% (DWThotCrust1) to
   +17.1% (DWAK), rather than sitting wholly below it as the as-shipped
   numbers in section 3 do.

An earlier pass through this analysis reported "crustal-model sensitivity is
x1.22-1.25" as if it were a statement about the (2,0) forcing mode. It is
not -- that range is the **off-diagonal mode spread** from section 3 above
((3,0) x1.217, (2,±2) x1.240, (3,±1) x1.246), a different set of modes
entirely, and that claim must not stand as a (2,0) statistic anywhere in
this document.

**None of x1.551, x1.726, or x1.379 replaces TASK-027's x7.6 Airy-parameter
sweep (section 6 above).** The two measurements are not interchangeable:
the density sweep varies the *amplitude calibration* (`AIRY_FACTOR =
rho_crust/(rho_mantle - rho_crust)`, which linearly rescales the whole
field), while these five InSight models vary only the *pattern*, at a mean
thickness (49.6-50.5 km) the selection criterion holds fixed to within ~2%
of the source paper's own published 30-72 km mean-thickness range for the
underlying interior models. A structurally analogous point: Moho relief
itself scales as `1/(rho_mantle - rho_crust)` under an Airy-type
derivation -- the same structural sensitivity `AIRY_FACTOR` carries -- so
the crust/mantle-density axis this task's model selection deliberately held
fixed (all five at 2900 kg/m^3) would, if varied, itself plausibly
contribute a spread of order 3-5x on top of the pattern-only x1.379. The
linearized rigidity map cannot explore that axis without care regardless:
the InSight fields already reach `|dmu/mu_bar| = 0.9689` (DWAK, section 2
above) against the elastic-positivity bound of 1.0, so any density
adjustment that increases the implied thickness variation risks leaving the
regime this linearization is valid in.

### 7. Provenance gaps and unquantified uncertainty

Two provenance caveats, stated explicitly rather than left implicit:

**The source paper's degree-2 treatment.** Wieczorek et al.'s original
paper describing these Moho models is paywalled and was not independently
retrieved this session, so exactly how the paper itself treats degree-2
(whether it discusses the same C20 decoupling documented in section 1
above, or applies its own convention) could not be verified here. The
SOURCES.md provenance record documents the data archive and selection
criterion, not the paper's own degree-2 discussion.

**An independent, order-of-magnitude gravity-based check.** A naive
full-gravity Bouguer inversion at degree 2 -- treating the entire retained
non-hydrostatic gravity signal as if it were a single mass sheet at the
Moho depth -- predicts a Moho C20 of -3363 m (crust/mantle density contrast
700 kg/m^3) to -4708 m (contrast 500 kg/m^3), against the five models'
actual retained C20 of -2058 to -2745 m. That 1.4-2.7 km gap is 35-70% of
the retained C20 -- the exact term that activates the first-order channel
in section 5 -- so it is unquantified uncertainty sitting directly on the
headline result, not a side issue. This estimate was not cross-checked
against the source paper's own methodology (previous caveat), so it should
be read as an independent order-of-magnitude sanity check, not a competing
authoritative value.

### 8. What this means for the proposal

Three points, in order of how directly they touch the text already in the
proposal:

1. **The x7.6 Airy-parameter sensitivity (TASK-027) is not superseded.** It
   remains the largest quantified error term on the lateral spectrum. The
   x1.22-1.25 off-diagonal spread and the x1.379-1.726 (2,0)-specific
   spreads (section 6) are smaller, differently-scoped measurements of
   crustal-*pattern* sensitivity at fixed density, not replacements for it.
2. **The (4,0)-driven first-order forcing-mode mechanism (section 5 above,
   TASK-016) survives, but its previously-reported magnitude does not
   generalize.** (2,0) is an equally first-order channel that was invisible
   to the Airy path by construction, and on a C20-retaining field it
   substantially cancels (4,0)'s contribution (91.4% on DWAK). Any proposal
   text that cites the (4,0)-alone figure as *the* first-order degree-4-
   zonal-crustal-structure signature should note it is an Airy-path-specific
   number, not a field-independent constant.
3. **The pattern substitution itself is a validation, with the two caveats
   above attached.** The InSight-calibrated fields agree with the Airy
   field's off-diagonal spectrum to correlation 0.952-0.966 and an rms
   ratio of 1.019-1.206 (up to +21% high, section 4), and reproduce its
   qualitative geography (Tharsis thick, Hellas/Utopia thin, section 2) --
   support for the Airy pattern's structure, tempered by the shared-input
   caveat in section 4 and the unresolved degree-2 provenance gap in
   section 7.

### 9. Native-MATLAB anchor for the two first-order channels (TASK-029)

The (2,0)/(4,0) sign-cancellation result (section 5) changed a headline
claim in the proposal, and until now rested entirely on the Python port plus
a coupling-coefficient inspection -- unlike every other link in the Mars
chain, which has a native-MATLAB anchor. `scripts/mars_first_order_channels.m`
closes that gap: it runs the native LOV3D solver on the committed 4-layer
model, hands it the *identical* DWAK complex crust `mu_variable` Python used
(exported by `scripts/export_mars_dwak_mu_variable.py` to
`data/mars/mars_dwak_mu_variable.npz`, so no spherical-harmonic bookkeeping
is re-derived in MATLAB), and reproduces all three deliverables at
`method='combination'`, `Nrbase=30`, `perturbation_order=2` -- the same
numerics as the Python reference. Artifacts:
`data/tests/mars/mars_first_order_channels.{log,mat}` (MATLAB R2025b).

**The result reproduces, including the signs.**

| quantity | Python | native MATLAB |
|---|---|---|
| scaling exponent (2,0) | ~1.00 (1st order) | **0.9988** |
| scaling exponent (4,0) | ~1.00 (1st order) | **1.0001** |
| scaling exponent (3,0) | ~2.00 (2nd order) | **2.0005** |
| (2,0)-alone shift (full amp.) | -1.5901e-5 | **-1.5858e-5** |
| (4,0)-alone shift (full amp.) | +1.4528e-5 | **+1.4531e-5** |
| both together | -1.3738e-6 | **-1.3398e-6** |
| cancellation | 91.4% | **91.6%** |
| max\|C\| coupling (2,0) | 0.6389 | **0.6389** |
| max\|C\| coupling (4,0) | 0.8571 | **0.8571** |
| max\|C\| coupling (1,0),(3,0),(5,0),(6,0) | 0.0 | **0.0** |

Three things worth stating plainly. First, the **signs are identical** --
(2,0) negative, (4,0) positive, net negative -- so the opposite-sign
cancellation that underlies the proposal's corrected claim is confirmed in
an independent solver, not an artifact of the Python port. Second, the
MATLAB (2,0)+(4,0) linear sum matches the jointly-solved value to 0.9%,
independently confirming the first-order superposition. Third, the coupling
coefficients themselves match to four decimals -- the selection rule (even
zonal rheology degree, triangle inequality capping at 4, `m=0` order
conservation) is anchored directly, degree by degree, not merely inferred
from the response. The residual ~0.3% differences in the isolated-channel
magnitudes and ~2.5% in the (near-zero) net are consistent with ordinary
solver-level numerical differences at this cancellation depth; they do not
touch the sign or the ~91% cancellation. **No proposal text needs to be
pulled back.**

## Hydration-front tidal signature (TASK-021)

Quantifies the proposal's core hypothesis directly: a downward-propagating
crustal hydration (serpentinization) front produces laterally varying
crust rigidity with a tidal signature, at the scale a real gravity/tidal
measurement could in principle constrain. Built entirely on the
already-validated TASK-016 lateral machinery above and the TASK-011 1D
model -- no `pylov3d` module is modified to build this. Implementation:
`pylov3d/mars_hydration.py` (full derivation in its module docstring; this
section states the numbers, the web-verified serpentinite property
sources, and the deviations). Tests: `pylov3d/tests/test_mars_hydration.py`.
Figure: `scripts/proposal_figures/fig7_hydration_signature.py` ->
`docs/figures/proposal/fig7_hydration_signature.{pdf,png}`.

### 1. Serpentinite elastic properties (web-verified, not from memory)

`mu_serp/mu_crust` and `K_serp/K_crust` ratios against this model's crust
reference (`mu_crust` = `LAYER_MU_CRUST` = 30 GPa, `K_crust` =
`LAYER_KS[3]` = 70 GPa), from published Vp/Vs/density via
`mu = rho*Vs^2`, `K = rho*(Vp^2 - 4/3*Vs^2)`:

| Property | Source | Vp, Vs [km/s] | rho [kg/m^3] | Value | Ratio |
|---|---|---|---|---|---|
| mu central | Christensen, N.I. (1966), "Elasticity of ultrabasic rocks," *J. Geophys. Res.*, 71(24), 5921-5931 (serpentine monomineralic-aggregate) | Vs=2.35 | 2600 (a) | 14.36 GPa | 0.48 |
| mu low | Falcon-Suarez, I., Bayrakci, G., Minshull, T.A., North, L.J., Best, A.I., Roumejon, S., & IODP Exp. 357 Science Party (2017), "Elastic and electrical properties and permeability of serpentinites from Atlantis Massif, Mid-Atlantic Ridge," *Geophys. J. Int.*, 211(2), 686-699 (most-fractured core sample S2) | Vs=1.8 (f) | 2374 (b) | 7.69 GPa | 0.26 |
| mu high | Christensen, N.I. (2004), "Serpentinites, peridotites, and seismology," *Int. Geol. Rev.*, 46(9), 795-816 | Vs=3.0 (c) | 2700 | 24.3 GPa | 0.81 |
| K central | Christensen (1966), same sample as mu central | Vp=5.10, Vs=2.35 | 2600 (a) | 48.48 GPa | 0.69 |
| K low | Falcon-Suarez et al. (2017), same sample as mu low | Vp~3.5 (d), Vs=1.8 (f) | 2374 (b) | 18.83 GPa | 0.27 |
| K high | Christensen (2004), antigorite endmember, computed from measured rho/Vp/Vs | Vp=6.52, Vs=3.57 | n/a (e) | 71.2 GPa | 1.02 |

(a) Christensen (1966) does not give a density for that sample in the
passages found; 2600 kg/m^3 is a representative mid-range serpentinite
bulk density, bracketed by Falcon-Suarez et al.'s (2017) measured
2374-2786 kg/m^3 and the serpentinization density decrease from ~3300
(fresh peridotite) to ~2500-2600 kg/m^3 (fully serpentinized) reported
across the literature. (b) Falcon-Suarez et al. (2017), Table 1, bulk
density of sample S2 (most fractured/altered of their four Atlantis
Massif cores). (c) Christensen (2004) reports serpentinite Vs spanning
roughly 2.3-3.6 km/s depending on serpentine polymorph (lizardite/
chrysotile at the low end, antigorite at the high end); 3.0 km/s is
adopted as a representative *upper* value for lizardite/chrysotile
serpentinite, deliberately short of the antigorite endmember. (d) Not
separately reported for sample S2; taken from Falcon-Suarez et al.'s
(2017) stated Vp at low confining pressure (~3.5 km/s at 45 MPa) to pair
with their low Vs. (e) Christensen (2004) computes mu and K for
antigorite from measured rho/Vp/Vs (mu=34.7, K=71.2 GPa) rather than
reporting them as independently measured quantities -- Vp, Vs, and
density are the directly measured quantities; mu and K are derived from
them via the standard isotropic-elastic relations, here and throughout
this table. No paired density was found in the passages consulted. (f)
The Vs=1.8 km/s value for sample S2 was read off a figure in Falcon-
Suarez et al. (2017), not taken from a tabulated number -- an
approximate, not exact, input.

**Adopted:** `MU_SERP_RATIO_CENTRAL` = 0.48, `MU_SERP_RATIO_BRACKET` =
[0.26, 0.81]; `K_SERP_RATIO_CENTRAL` = 0.69, `K_SERP_RATIO_BRACKET` =
[0.27, 1.02]. K is far less reduced than mu -- physically expected:
serpentinization is well known seismologically for depressed Vs and
*elevated* Vp/Vs (Poisson's ratio), i.e. mu drops faster than K
(Christensen 2004 is the classic reference for using Vp/Vs to detect
serpentinite bodies on exactly this basis). The K-bracket high endpoint
(1.02) is at/above 1.0, i.e. essentially no K softening in this corner of
the bracket, for the same underlying reason as the antigorite caveat
below (that source's antigorite K value *is* the adopted K-high
endpoint, unlike mu).

**Antigorite caveat, stated prominently.** Christensen (2004)'s
antigorite bulk-rock values, computed from measured rho/Vp/Vs (mu=34.7,
K=71.2 GPa), give `mu_serp/mu_crust` = 1.16 -- *stiffer* than this
model's 30 GPa crust reference, essentially no softening signal in mu.
Antigorite is the higher-temperature (~300-500 C) serpentine polymorph;
if a real Mars hydration front is antigorite- rather than
lizardite/chrysotile-dominated (the mineralogy most of the cited data
describes, more plausible at shallow/cooler depths), the true detectable
mu signal could be much weaker than modeled here, down to essentially
zero. Not folded into the mu bracket above; recorded as a stated failure
mode of the mineralogy assumption, independent of everything below.

**Reference-crust caveat (added in review, section 5a below).** The
30 GPa `LAYER_MU_CRUST` denominator itself -- a Christensen-Mooney
average-basalt value, not specific to Mars -- turns out to drive the
result at least as much as the serpentinite bracket does; see section 5a
for a dedicated sensitivity table against InSight's actual in-situ
crustal modulus and against unaltered gabbro/peridotite protoliths.

### 2. Hydration geometry and the mean/lateral split

Hydrated thickness within the fixed 50 km crust shell:
`t_h(theta,phi) = f_h * t_crust(theta,phi)`, `f_h` in [0,1] the global
hydrated-fraction parameter, `t_crust = 50 km + dt` reusing TASK-016's
`crustal_thickness_variation` (Airy, areoid-referenced) as-is. Rationale:
thicker crust hosts proportionally more hydratable volume -- a
deliberately simple stage-1 coupling; a water-table-controlled hydration
depth is the stated stage-2 alternative.

Because `dt` has zero spatial mean by construction (its (0,0) term is
dropped), the effective-rigidity formula splits cleanly into a mean
(degree-0) and a lateral (degree>=1) term:

```
delta_mu/mu_crust(theta,phi) = (mu_serp-mu_crust)/mu_crust * t_h/50km
                              = f_h*(mu_serp-mu_crust)/mu_crust                          [mean]
                              + f_h*(mu_serp-mu_crust)/(mu_crust*50km) * dt(theta,phi)   [lateral]
```

(same form for K). The **mean term** -- first-order important, not
dropped -- becomes a modified crust `mu0`/`Ks0` for a fresh 1D solve:
`mu0_soft = (1-f_h)*mu_crust + f_h*mu_serp` (`build_hydrated_mars_model`,
crust layer only, verified in `test_mars_hydration.py::TestMeanShiftConsistency`
against an independently rebuilt model). The **lateral term** feeds
`get_love`'s `mu_variable`/`K_variable` coupled path, converted to the
solver's complex-SH convention by reusing
`mars_lateral._real_sh_to_complex_mu_variable` unchanged. Its
normalization is *not* the same as the human-readable/plotting form
above: `process_lateral_variations`'s elastic branch injects
`muC_amp = mu_i * mu_map[nm]` with `mu_i` the layer's own normalized
modulus (=1.0 exactly for the crust, the model's surface layer), so the
fractional amplitude actually passed to the solver is normalized by the
*softened* `mu0_soft`/`Ks0_soft`, not the original 30/70 GPa
(`hydration_lateral_variables`; `hydration_dmu_over_mu_bar_real` is the
separate, mu_crust-normalized version used only for fig7's map).

**Clip vs. spectral formulation (deviation, documented).** The spec asks
both for clipping `t_crust` to [0, 50 km] (shell residency) and for
evaluating the perturbation "spectrally (linear in t_crust's SH
coefficients)" -- in tension, since clipping is nonlinear/pointwise and
not expressible as a rescaling of `dt`'s existing SH coefficients. Given
`dt`'s zero mean, clipping `t_crust=50+dt` at 50 km would bind over
roughly *half* the sphere (everywhere `dt>0`) for any `f_h > 0` -- not a
rare edge case. Adopted interpretation: use the linear/spectral form
exactly as specified for the actual mu/K SH coefficients, and enforce the
clip as a **checked bound** instead (`hydration_geometry_diagnostics`,
same pattern as `crustal_thickness_diagnostics`'s existing `|dt|<50km`
check). Verified, not assumed: over the swept range used here (`f_h` <=
0.5, measured max|dt| = 34.2 km, TASK-016), the bound is never violated
(max `t_h_linear` = 0.5*(50+34.2) = 42.1 km < 50 km) -- confirmed
numerically by `hydration_geometry_diagnostics` and pinned by
`test_mars_hydration.py::TestGeometryDiagnostics`. It would start to bind
above `f_h` ~0.59 at the measured peak |dt|; not swept here.

### 3. K_variable gap in every rheology branch, its row-0 normalization, and a silent-collapse bug (all found in review)

**The gap is not elastic-specific (corrected from an earlier draft).**
`pylov3d.rheology.process_lateral_variations` hardcodes `K_amp = 0`
regardless of the `K_variable` argument in *both* its elastic branch
(`rheology.py` ~line 424) *and* its viscoelastic branch (~line 480, same
pattern) -- `LateralRheology.K_amp`'s own docstring: "currently set to 0;
placeholder for future use"; confirmed by
`test_lateral_rheology.py::test_elastic_K_amp_zero`. All four Mars layers
are elastic, so this module only exercises the elastic branch, but the
gap itself is not elastic-specific. `get_love(..., K_variable=...)` would
silently have no effect here. The low-level `K_amp` channel *is* wired
through the NumPy coupled propagator/solver (`solver.py`; validated by
`test_jax_coupled_scan.py::test_nonzero_K_amp_selection`, which injects
`K_amp` post hoc via `LateralRheology._replace`). `get_love_hydrated`
(`mars_hydration.py`) reproduces `get_love`'s pipeline and adds the
missing step by hand. No `pylov3d` module is modified.

**Row-0 normalization convention (review finding, resolved by reading
`pylov3d.propagator` source directly, not assumed).** `_a1a2_geometric`'s
own docstring states: "Row 0 uses `(3*lambda+2*mu)=1` -> multiply by
actual `(3*lambda+2*mu)` for diagonal, or `K_nm` for coupling." Its
partner, `_coupling_A1_A2`, applies `K_nm` to row 0 with **no** extra
factor (`A1c[0,:] += K_nm * Cp[0] * A1g[0,:]`) -- contrast rows 1-5,
where the code itself multiplies `mu_nm` by 2 to match the diagonal's
`2*mu` scaling. For the row-0 coupling amplitude to represent the same
normalized quantity as the diagonal's `(3*lambda+2*mu) = 3*K` (since
`lambda = K - 2*mu/3`), `K_nm` must equal `delta(3*lambda+2*mu) =
3*delta_K`, not plain `delta_K`. The originally shipped code used plain
`delta_K` (`K_amp = Ks_i * K_map[nm]`); corrected to
`K_amp = 3 * Ks_i * K_map[nm]` (`mars_hydration.py`'s `K_ROW0_FACTOR`
constant). No validated reference pins this convention -- every
`pylov3d` rheology branch zeroes `K_amp` unconditionally, so nothing
upstream ever exercised this row for a nonzero K perturbation -- but the
3x reading is the one internally consistent with the diagonal case.
**Impact band, stated for the record:** at `f_h=0.3`, `lmax=2`, central
ratio, four defensible readings of the row-0 K convention span only
3.8x (no `K_variable` at all: +5.47e-7; plain `delta_K`, the original
code: +3.73e-7; `3*delta_K`, adopted: +1.45e-7; an alternative
un-rescaled-fractional reading: +4.68e-7). Because the lateral
contribution is already a small fraction of the total `Delta k2` at
every sampled point (section 5), none of these four readings changes any
conclusion in this document.

**Silent-collapse bug (review finding, fixed).** The original
`get_love_hydrated` passed only `mu_variable` into
`process_lateral_variations`, never `K_variable`. Whenever `mu_variable`
was empty but `K_variable` was not (reachable via `mu_ratio=1.0` with
`K_ratio != 1.0`: `mu_serp == mu_crust` makes the mu coefficient exactly
0.0, filtering all mu entries away, while the K coefficient stays
nonzero), `process_lateral_variations`'s `(n,m)` union saw no variation
in *any* property and silently returned the trivial, uncoupled
`LateralRheology` -- discarding the entire K perturbation without a
warning or error. Fixed by passing `K_variable` into
`process_lateral_variations` as well (its own K_amp output for that call
is still exactly 0, both branches hardcode that regardless -- the
row-0-corrected overwrite immediately after is what does the real work).
Regression-guarded:
`test_mars_hydration.py::TestGetLoveHydratedRegression` (mu-only matches
plain `get_love` bit-for-bit; K_variable demonstrably changes the
solution; the `mu_ratio=1.0` edge case no longer collapses to N=1 mode).

### 4. Runtime, grid reduction (spec explicitly permits this)

The "validated config" (`perturbation_order=2`, NumPy path, `lmax=4`,
`Nrbase=30`) costs ~130-140 s per coupled solve (TASK-016); a full `f_h` x
ratio-bracket sweep at that cost runs tens of minutes, over the ~5 min
guard. Measured on the development machine: `lmax=2, Nrbase=30` -> 10.0
s/solve (N=43 coupled modes); `lmax=4, Nrbase=30` (full validated
config) -> ~139 s/solve (N=115 modes). `mars_hydration.py`'s default
sweep keeps `Nrbase=30` (the validated value) and reduces `lmax` 4 -> 2
(degree-1 dominant in `dt`'s RMS per TASK-016's own degree dichotomy, so
degrees 1-2 retain most of the lateral structure); the full 5-nonzero-`f_h`
x 3-scenario coupled sweep (`DEFAULT_F_H_GRID`, `RATIO_SCENARIOS`) costs
~153 s at this default, well inside the guard
(`test_mars_hydration.py::TestRuntimeGuard` asserts < 480 s -- raised
from an earlier 300 s cap that left only 15% headroom on a reviewer's
slower machine, measured 254 s there).

**lmax=2 vs. lmax=4 spot check (D1, review -- BLOCKING finding: the
originally reported "mean dominates lateral by 2.5-3 orders of
magnitude" was an lmax=2 truncation artifact, not a converged result).**
Three coupled solves at the full `lmax=4, Nrbase=30` config, central
ratio, all K_ROW0_FACTOR-corrected (section 3):

| f_h | lateral, lmax=2 | lateral, lmax=4 | mean (lmax-independent) | mean:lateral, lmax=4 |
|---|---|---|---|---|
| 0.1 | 1.35e-08 | 9.89e-07 | 6.336e-05 | 64:1 |
| 0.3 | 1.45e-07 | 3.14e-06 | 1.911e-04 | 61:1 |
| 0.5 | 5.00e-07 | 5.62e-06 | 3.203e-04 | 57:1 |

`lmax=2` underestimates the lateral contribution by **11x to 73x**
(largest underestimate at the smallest `f_h`, where the missing
linear-in-amplitude channel matters relatively most) -- because it drops
the (4,0) crustal harmonic. On this Airy-derived field specifically (C20
always dropped by construction, section 1 above), (4,0) is the only
first-order (linear-in-amplitude) self-coupling channel back to the forced
(2,0) mode that is actually present to be dropped by the `lmax=2`
truncation; every other harmonic present here couples back only at second
order. This is Airy-path-specific, not a general statement about the
(2,0)/(4,0) mechanism -- on a C20-retaining crustal field, (2,0) rheology is
an equally first-order channel that the Airy field never has in the first
place (section 5 above; "Non-Airy crustal model substitution (TASK-028)"
below, section 5). At the
validated `lmax=4`, the mean term dominates the
lateral term by **~57-64:1** (roughly flat across the sampled `f_h`, not
growing) -- one to two orders of magnitude smaller than the originally
reported (and wrong) ~500:1. This still does not change the qualitative
conclusion: even the largest lateral value found (5.62e-6 at `f_h=0.5`)
is under 0.1% of `sigma_k2`=0.006, so the total (mean+lateral) barely
moves between `lmax=2` and `lmax=4` -- e.g. at `f_h=0.5`, total/sigma_k2
goes from 5.35% (`lmax=2`) to 5.43% (`lmax=4`) (section 5). The
corrected ~60:1 dominance ratio is itself the basis for section 5's
"k2 is blind to WHERE the hydration is" statement: the front's lateral
signature is a small, roughly constant fraction of the total response,
not something that grows into detectability as `f_h` increases.

### 5. Forward sweep and detectability

`hydration_forward_sweep()` at the reduced-`lmax`=2 default: `f_h` in
{0, 0.1, 0.2, 0.3, 0.4, 0.5} x {low, central, high} ratio scenarios (18
points, 15 coupled solves, 153.4 s measured wall time). Central-ratio
`Delta k2` (relative to the exact `f_h=0` baseline k2=0.169000000000):

| f_h | mean contribution | lateral contribution (lmax=2) | total Delta k2 | total / sigma_k2 |
|---|---|---|---|---|
| 0.1 | 6.336e-05 | 1.35e-08 | 6.337e-05 | 1.06% |
| 0.2 | 1.270e-04 | 5.89e-08 | 1.271e-04 | 2.12% |
| 0.3 | 1.911e-04 | 1.45e-07 | 1.912e-04 | 3.19% |
| 0.4 | 2.555e-04 | 2.86e-07 | 2.558e-04 | 4.26% |
| 0.5 | 3.203e-04 | 5.00e-07 | 3.208e-04 | 5.35% |

The **mean term dominates the lateral term by ~57-64:1 at the validated
lmax=4** (section 4; the `lmax=2` column above underestimates the
lateral term 70-100x, a truncation artifact, not a physical result --
do not read a "500:1" or "orders of magnitude" dominance ratio from the
table above). Bracket range at `f_h=0.5` (total, `lmax=2`): low-ratio
(softer, `mu_serp/mu_crust=0.26`) gives 5.10e-4 (8.49% of `sigma_k2`);
high-ratio (stiffer, 0.81) gives 9.32e-5 (1.55%) -- both still well
under `sigma_k2`, and at `lmax=4` the total barely shifts either way
(section 4).

**Plainly stated detectability conclusion.** Across the entire explored range
(`f_h` <= 0.5, full ratio bracket), `Delta k2` **never approaches**
`sigma_k2` = 0.006 -- the largest value found (low-ratio bracket,
`f_h=0.5`) is 5.10e-4, ~8.5% of `sigma_k2`. The central-ratio curve is
very close to linear in `f_h` over the sampled range, so linearly
extrapolating to the full physical range `f_h=1` gives `Delta k2`
~6.4e-4, still only ~11% of `sigma_k2`; even the most favorable corner
of the whole parameter space explored here (low-ratio bracket
extrapolated to `f_h=1`) gives ~1.0e-3, ~17% of `sigma_k2` -- **current
MRO120D/MRO120F k2 precision (sigma_k2=0.006) cannot detect this
hydration-front signal, as parameterized here, at any physically allowed
hydrated fraction, and this conclusion is not sensitive to the lmax=2
truncation** (section 4). No mission or future-precision claim is made
here -- only the number that *would* resolve a specific, modest
hydration scenario: the k2 precision needed to resolve `f_h=0.1` at
1-sigma (central ratio) is **6.34e-5** -- about 95x tighter than the
current 0.006 uncertainty.

**k2 measures the bulk hydrated fraction, not WHERE the hydration is
(S2, review).** The ~57-64:1 mean:lateral dominance at the validated
`lmax=4` (section 4) means k2 is set almost entirely by the
*globally-averaged* hydrated fraction; a laterally uniform hydration of
the same total water content and a front-shaped one (this module's
actual geometry) would produce nearly indistinguishable k2 shifts (to
within that ~1-2%). k2 alone therefore cannot confirm the "front" part
of the hydration-front hypothesis, only its bulk magnitude. The
front's actual lateral signature -- WHERE the hydration is, not just how
much -- lives almost entirely in the off-(2,0) Love spectrum (the
TASK-016 fig6 machinery: mode amplitudes at `(n,m) != (2,0)`), which
this document does not attempt to assess for detectability; that is
explicitly future work, not claimed here.

### 5a. Reference-crust sensitivity (S1, review)

The 30 GPa `LAYER_MU_CRUST` denominator (section 1) is itself a
Christensen-Mooney *average basalt* value -- not a Mars-specific,
InSight-measured number. `crust_reference_sensitivity()` holds the
absolute serpentinite modulus fixed at 14.4 GPa (the central-ratio
value) and varies the crust *reference* instead, computing `Delta k2` as
**self-referenced per row** -- `k2(mu0_soft) - k2(mu_crust_ref,
unsoftened)` -- *not* relative to the single global fitted k2=0.169
(which is tied to the 30 GPa reference specifically; sharing that
baseline would conflate the hydration signal with the separate,
physically real baseline-k2 drift from swapping crust references alone):

| Crust reference | mu_crust_ref | k2 baseline (unsoftened) | k2 softened (f_h=0.5) | Delta k2 | Delta k2 / shipped-30GPa Delta k2 |
|---|---|---|---|---|---|
| InSight in-situ crust (Knapmeyer-Endrun et al. 2021, Science 373) | 17 GPa | 0.169482 | 0.169537 | 5.53e-05 | x0.20 |
| Shipped default (Christensen-Mooney average basalt) | 30 GPa | 0.169000 | 0.169275 | 2.75e-04 | x1.00 |
| Oceanic gabbro (unaltered protolith) | 45.3 GPa | 0.168545 | 0.169005 | 4.60e-04 | x1.67 |
| Unaltered peridotite (unaltered protolith) | 68 GPa | 0.167998 | 0.168659 | 6.60e-04 | x2.40 |

The shipped 30 GPa reference is **too stiff relative to InSight's actual
in-situ crust** (the real crust is softer, closer to the serpentinite
endmember already, so a given hydrated fraction produces a *smaller*
relative contrast and a smaller signal -- x0.20) and **too soft relative
to an unaltered igneous protolith** (gabbro/peridotite start stiffer, so
the same hydration is a *larger* relative softening -- x1.67/x2.40).
This denominator choice spans a wider range (x0.20-x2.40, a 12x span)
than the serpentinite mu-property bracket itself does: computing the
same mu-only, `f_h=0.5` comparison for section 1's low/central/high
`mu_serp/mu_crust` ratios (0.26/0.48/0.81) at the shipped 30 GPa
reference gives x1.47/x1.00/x0.35 of the central value -- a ~4.2x span
(0.35 to 1.47), about a third as wide as the reference-crust span above
-- **the reference-crust modulus drives the result at least as much as,
and arguably more than, the serpentinite bracket.**

**The null-detectability conclusion (section 5) survives the full 12x
span.** Even the best case for detectability -- unaltered peridotite,
the stiffest reference, giving the largest signal -- reaches only 6.60e-4
at `f_h=0.5` (11% of `sigma_k2`) and, linearly extrapolated, ~1.32e-3 at
`f_h=1` (22% of `sigma_k2`); still well short of 1-sigma.

**mu_crust is not freely adjustable, stated plainly.** This table is a
sensitivity study, not a proposal to swap the shipped reference:
`MARS_MU_SCALE` (`pylov3d.mars`) was fit so that the *elastic* k2 of the
whole 4-layer model matches the observed k2=0.169 at the shipped 30 GPa
crust; changing the crust reference alone (without refitting the mantle
shear-modulus scale to compensate) shifts the *baseline* k2 away from
0.169 -- visibly, by row, in the table above (0.1695/0.1690/0.1685/0.1680
across the four references) -- which is exactly why this table reports
self-referenced deltas rather than deltas against the single global
0.169 constant.

## Off-(2,0) detectability (TASK-026)

Closes the loop TASK-021 left open in writing: k2 constrains the
globally-averaged hydrated fraction but is blind to WHERE a hydration
front sits; the front's lateral signature lives almost entirely in the
off-(2,0) tidal Love-number spectrum TASK-016 computed and
MATLAB-cross-validated. Nobody had asked whether that spectrum is
*measurable*. Implementation: `pylov3d/mars_detectability.py` (the
amplitude-to-observable relation, its degree-2 hand check, and the
off-(2,0) required-precision table) and `pylov3d/mars_detectability_k2m.py`
(a related but distinct diagonal observable, the k2m order-splitting
benchmarked against GRAIL/MaQuIs -- see "Two distinct observables" below
for why these are separate modules, not a duplication). Tests:
`pylov3d/tests/test_mars_detectability.py`,
`pylov3d/tests/test_mars_detectability_k2m.py`. Figure:
`scripts/proposal_figures/fig8_off20_detectability.py` ->
`docs/figures/proposal/fig8_off20_detectability.{pdf,png}`.

**Scope note on this section's benchmark.** The task's default scoping
was (a) current Mars-orbiter tracking and (b) a "GRAIL-class dedicated
mission" as a generic placeholder. Mid-task, this was sharpened to a
named mission concept, MaQuIs (Wörner et al. 2023, below), whose own text
identifies the k2m order-splitting (not a generic Love-number precision
level) as a measurement it intends to enable, and states that measuring
it has "proved unsuccessful for Mars" to date. That became this section's
lead result (tier 1); the higher-degree off-(2,0) coupled spectrum
(tier 2, TASK-016's original N=115 modes) remains the second, still novel
tier, since neither GRAIL nor MaQuIs discusses it directly.

### 1. The observable: from |k_nm| to a time-varying Stokes coefficient

Full derivation in `pylov3d/mars_detectability.py`'s module docstring;
summarized here. A nonzero tidal Love number means the tide-raising
potential induces an *additional* gravitational potential at the body's
surface equal to k_n times the tide-raising potential itself. Expanding
the external (solar) tide-raising potential in the response body's own
real, 4pi-fully-normalized, no-Condon-Shortley spherical harmonics (the
`pylov3d.sh_data`/`pylov3d.mapping` convention throughout this project)
via the classical Legendre addition theorem, written in normalized form
and derived (not assumed) from Ferrers/geodesy-normalized `P_nm` and the
standard unnormalized addition theorem, gives the classical same-degree
relation

```
Delta C_nm = -k_n * (GM_ext/GM_body) * (R/d)^(n+1) * (1/(2n+1))
             * Pbar_nm(sin phi') cos(m lam')
```

(the leading minus carries through from the tide-raising potential's own
sign convention; every quantity this document reports is `|Delta C_nm|`,
so the sign is immaterial to every number here and is restored only for
internal algebraic consistency -- it had been dropped silently in an
earlier version of this derivation).

(`phi'`, `lam'`: sub-solar point). **Cross-checked against a published
formula, retrieved this session**: Genova, A., Goossens, S., Lemoine, F.
G., Mazarico, E., Neumann, G. A., Smith, D. E., & Zuber, M. T. (2016),
"Seasonal and static gravity field of Mars from MGS, Mars Odyssey and MRO
radio science," *Icarus*, 272, 228-245 (open access, CC-BY 4.0; fetched
directly from Zenodo record 894840), eq. (5), gives the degree-2 tidal
potential felt by an orbiting spacecraft as `U = k2 (GM_p/R)(R^6/(r^3
r_p^3))[3/2(rhat.rphat)^2 - 1/2]` -- algebraically identical in form to
this derivation's own degree-2 response potential (R^5 = R^6/R matches
exactly), and that paper's own eq. (1)-(2) define `Pbar_lm` with the same
4pi-full, no-Condon-Shortley normalization used throughout this project.

**Generalizing to the off-diagonal (coupled) case.**
`pylov3d.love.extract_love_numbers` defines, for a coupled solve forced
at `(n_f, m_f)` with unit amplitude (`F=1.0`, this project's universal
convention): the forced mode's Love number is `Phi_surf - 1`; every
*other* coupled mode's is `Phi_surf` directly, read from the *same*,
degree-independent potential normalization throughout the solve. Because
the whole boundary-value problem is linear in the forcing amplitude, the
physical response at any coupled mode `(n, m)` is `k_(n,m)[code] *`
(the actual physical tide-raising potential amplitude at the *forcing*
degree/order) -- giving the generalized relation this module implements,
with the **same** `(GM_ext/GM_body)(R/d)^(n_f+1)/(2n_f+1)` prefactor for
*every* response mode `n` (set by the forcing degree, not the response
degree):

```
Delta C_nm = -k_(n,m) * (GM_ext/GM_body) * (R/d)^(n_f+1) * (1/(2n_f+1))
             * Pbar_(n_f,m_f)(sin phi') cos(m_f lam')          (*, uncorrected)
```

Setting `n=n_f, m=m_f` collapses (*) to the diagonal formula exactly --
the hand-checkable degree-2 case: `k2=0.169`, forcing `(2,0)`, mean
Mars-Sun distance, sub-solar point at the equinox (`Pbar_20(0) =
sqrt(5)/2`, the peak value over Mars's obliquity range), gives
`Delta C_20 = 3.850e-10` -- pinned by
`test_mars_detectability.py::TestDegree2HandCheck` two independent ways
(via the module's own functions, and by re-deriving the number from the
raw formula text with no shared helper code). Because this collapse
happens by construction whenever `m = m_f`, the hand check (and a
published-formula cross-check against Genova et al. 2016 eq. (5), which
additionally only validates the `GM`/radial scaling since it uses the
*unnormalized* `P2(cos psi)`) validate only the diagonal case -- neither
exercises the basis-normalization correction below, which is identically
absent (factor 1) whenever `m = m_f`.

**Basis-normalization correction (review-round fix).** The relation above
is missing a factor. `pylov3d`'s solver basis is not uniformly normalized
across `m`: `Y_n^0 = Pbar_n^0` (norm 1) but `Y_n^{+/-m} = (...)
Pbar_n^m exp(+/-i m lam) / sqrt(2)` for `m != 0` (norm `1/sqrt(2)`) --
stated explicitly in both `src/get_map.m` (lines ~196-201, the routine
that synthesizes the forcing field and the solution potential alike) and
its Python port, `pylov3d.mars_lateral.complex_sh_synthesis`. Since
`k_(n,m) = Phi_surf(n,m) / F(n_f,m_f)` is a ratio of coefficients read
directly from that basis, it is a ratio of *differently normed* basis
elements whenever the response order and the forcing order differ in
whether they are zero -- which is every off-diagonal mode of the shipped
(2,0)-forced spectrum. The corrected relation carries a
`c_{n_f,m_f}/c_{n,m}` factor (`c_{n,0}=1`, `c_{n,m}=1/sqrt(2)` for
`m != 0`), which evaluates to `sqrt(2)` for every `m != 0` response mode
here and exactly 1 for `m=0` -- so the `(3,0)` headline mode is
unaffected, but every other tabulated mode's required precision (and
detectability ratio) is `sqrt(2)` tighter than an uncorrected calculation
reports. Verified two independent ways (`pylov3d/mars_detectability.py`
module docstring, "Basis normalization"): (a) synthesizing a conjugate
pair with `complex_sh_synthesis` and projecting onto the real,
4pi-normalized `Pbar_nm` basis by direct numerical quadrature gives
`sqrt(C_nm^2+S_nm^2)/|k| = 1.41421356` for `(2,2)`, `(3,1)`, `(3,3)`,
`(4,2)` and exactly `1.0` for `(3,0)`; (b) an independent `(2,2)`-forced
coupled solve gives `k[(2,0)<-(2,2)]=3.806694e-5`, matching the shipped
spectrum's `k[(2,+2)<-(2,0)]=3.807096e-5` to `1e-4` relative -- consistent
with a `sqrt(2)`-corrected, self-consistent real-basis admittance of
`~5.384e-5` rather than the raw, uncorrected `~3.807e-5`. The 41 tests
that existed before this fix all passed, because the only response mode
any of them pinned was `(3,0)` -- the one mode this error cannot touch;
`test_mars_detectability.py::TestBasisNormalizationCorrection` now pins
an `m != 0` mode directly.

**Constants used** (retrieved this session): the IAU (2015) Resolution B3
nominal solar mass parameter, `GM_sun = 1.3271244e20 m^3/s^2` (exact by
definition; Prsa, A., et al. (2016), "Nominal values for selected solar
and planetary quantities: IAU 2015 Resolution B3," *AJ*, 152, 41;
arXiv:1510.07674, fetched directly), and the IAU (2012) Resolution B2
astronomical unit, `1 AU = 149,597,870,700 m` (exact by definition, same
document). Mars's orbital semi-major axis (227,939,366 km = 1.52368055
AU) and eccentricity (0.0934) are widely tabulated planetary constants
(NASA NSSDC Mars Fact Sheet; direct fetch of `nssdc.gsfc.nasa.gov`
redirected to a generic landing page in this session, so these are
cross-checked only via Wikipedia's Mars infobox, itself citing the NASA
Fact Sheet and Allen (2000), *Astrophysical Quantities* -- not
independently re-verified beyond that aggregator). Mars's obliquity
(25.19 deg) DOES enter a computation, contrary to an earlier version of
this document: `peak_legendre_factor` searches for the peak `|Pbar_nm|`
only over sub-solar latitudes actually reachable (`|phi'| <=
MARS_OBLIQUITY_DEG`), not the full `[-90, 90] deg` range (whose true
global max for `m=0`, 2.236 at the poles, is unreachable by any sub-solar
point). For the `(2,0)`/`(2,2)` pairs this document uses, the
obliquity-constrained peak still falls at the equinox, verified
numerically rather than assumed.

### 2. Two distinct observables, and why they must not be conflated

This section's headline number, and the subject of the mid-task redirect
above, is the **diagonal k2m order-splitting**: does the ordinary
degree-2 Love number itself differ across azimuthal order m=0,1,2 (forced
at `(2,m)`, measured at the *same* `(2,m)`), because a laterally
heterogeneous Mars is not perfectly spherically symmetric? This is a
diagonal entry of the generalized Love-number tensor. The TASK-016
N=115 spectrum's off-(2,0) modes -- e.g. `(2,+/-2)` at `|k|=3.81e-5` --
are **off-diagonal** entries: the response at `(2,+/-2)` when Mars is
forced at `(2,0)`, a coupling/cross-talk effect, not the diagonal
`(2,2)-(2,2)` admittance. These are computed, and reported, separately;
an early draft of this section's framing treated them as interchangeable,
which a review pass caught and corrected -- flagged here because
conflating a diagonal admittance correction with an off-diagonal coupling
term is exactly the kind of normalization error this project's citation
and derivation standards exist to prevent.

### 3. Tier 1: the diagonal k2m order-splitting (MaQuIs/GRAIL benchmark)

Wörner, L., Root, B. C., Bouyer, P., Braxmaier, C., Dirkx, D., Encarnação,
J., Hauber, E., Hussmann, H., Karatekin, Ö., Koch, A., Kumanchik, L.,
Migliaccio, F., Reguzzoni, M., Ritter, B., Schilling, M., Schubert, C.,
Thieulot, C., v. Klitzing, W., & Witasse, O. (2023), "MaQuIs -- Concept
for a Mars Quantum Gravity Mission," *Planetary and Space Science*, 239,
105800, doi:10.1016/j.pss.2023.105800 (open access; full text fetched
directly this session from the TU Delft repository,
`repository.tudelft.nl`, record uuid `5261722a-a969-4aff-a728-
a8317c76ccbf`), state, quoted verbatim (p. 4; an earlier version of this
document located this on p. 5, second column -- wording unchanged, only
the locator was wrong): "For the Moon, separate
values of k20, k21 and k22 have been determined using GRAIL data,
providing further interior constraints (Williams et al., 2014). Although
these values of k2m at different orders m proved to be almost equal to
one another, their small differences may be relevant in processing of
high-accuracy data proposed here. Past attempts to measure these separate
coefficients using Doppler tracking proved unsuccessful for Mars."

**Predicted splitting** (`pylov3d.mars_detectability_k2m.mars_diagonal_k2m_table`).
`m=0`'s value is the already MATLAB-cross-validated TASK-016 forcing-mode
shift (section 5 above, `k2_shift=5.517e-5`). `m=1` and `m=2` were
computed this session by rerunning `mars_lateral_love_spectrum` at
`forcing=(2,1)` and `forcing=(2,2)`, at the *identical* validated
numerical configuration (`lmax=4, Nrbase=30, perturbation_order=2,
method="combination"`) as the MATLAB-cross-validated `(2,0)` run -- not
themselves independently MATLAB-checked, but inheriting the same
validated code path and grid. Both came out real to ~1e-15 (elastic
model, as expected):

| m | k_2m (predicted) | Delta = k_2m - 0.169 | Source |
|---|---|---|---|
| 0 | 0.16905517 | +5.517e-05 | TASK-016, MATLAB-validated |
| 1 | 0.16902091 | +2.091e-05 | this session, Python pipeline |
| 2 | 0.16903400 | +3.400e-05 | this session, Python pipeline |

**Truncation sensitivity -- not yet converged.** The three splittings
above are all at `lmax=4`. Rerunning the identical pipeline at `lmax=2,
Nrbase=30` gives `Delta k20=3.079e-05`, `Delta k21=2.755e-05`, `Delta
k22=1.948e-05` -- changes of 79%, 24%, and 74% moving from `lmax=2` to
`lmax=4`, and the ordering among `m` **reverses** (`lmax=2`: m=1 splits
more than m=2; `lmax=4`: the reverse). This is stated plainly rather than
left for a reader to discover: the ordering among `m` should not be relied
upon as a converged result.

**TASK-027 update.** TASK-027 Part 3 gave the lmax=4 diagonal (2,1)/(2,2)
values a native-MATLAB anchor (match to ~7e-12; see section "Lateral
variations", subsection 6), confirming the numbers above are correct *at
lmax=4*. It did **not** re-run the m=1,2 forcing cases at lmax=5/6, so the
lmax convergence of the m-ordering itself remains open: the forcing-mode
(2,0) shift was still moving ~4% per degree at lmax=6 (subsection 6
ladder), and the m=1,2 diagonals should be expected to move comparably.
The ordering among `m` still should not be relied upon as converged.

**Achieved precision.** Konopliv, A. S., Park, R. S., Yuan, D.-N., et al.
(2013), "The JPL lunar gravity field to spherical harmonic degree 660
from the GRAIL Primary Mission," *J. Geophys. Res. Planets*, 118,
1415-1434 (PDF fetched directly this session), Table 4 (individually
constrained GRAIL Primary Mission solution, "GL0660B"): k20 = 0.02408
+/- 0.00045, k21 = 0.02414 +/- 0.00025, k22 = 0.02394 +/- 0.00028.

**Williams, J. G., Konopliv, A. S., Boggs, D. H., et al. (2014), "Lunar
interior properties from the GRAIL mission," *J. Geophys. Res. Planets*,
119, 1546-1578, WAS retrieved and its Table 4 read directly this session**
(an open-mirror PDF, not the paywalled Wiley `agupubs` page that returns
HTTP 402 on every attempt) -- an earlier version of this document claimed
it could not be retrieved; this project's own `docs/MOON_MODEL.md`
already cites Williams et al. (2014) at section/table granularity in
several places, which should have been the signal that it was
accessible. Its Table 4 gives, for the *independent* GSFC analysis of the
same GRAIL Primary Mission data ("GRGM660PRIM," Lemoine et al. 2013):
k20 = 0.024165 +/- 0.00228, k21 = 0.023915 +/- 0.00033, k22 = 0.024852
+/- 0.00042 -- Table 4's text itself notes the GL0660B column's three
individual k2m values (used above) are "from a separate related
solution," not the officially published GL0660B fit. **No comparable
achieved number exists for Mars at all** -- per the verbatim quote above.

| m | \|Delta k_2m\| (Mars, predicted) | GRAIL sigma(k_2m), JPL (Moon) | ratio, JPL | GRAIL sigma(k_2m), GSFC (Moon) | ratio, GSFC |
|---|---|---|---|---|---|
| 0 | 5.517e-05 | 4.5e-04 | 8.2x | 2.28e-03 | 41.3x |
| 1 | 2.091e-05 | 2.5e-04 | 12.0x | 3.3e-04 | 15.8x |
| 2 | 3.400e-05 | 2.8e-04 | 8.2x | 4.2e-04 | 12.4x |

Even GRAIL's own best demonstrated individual-order Love-number precision
-- achieved at the Moon, a smaller body under a dedicated inter-satellite-
ranging mission architecture, an easier target than Mars in every respect
-- is **8-12x (JPL) or 12-41x (GSFC) too coarse** to resolve the splitting
this project's lateral model predicts for Mars. Reporting both matters:
the two independent analyses of the *same* GRAIL data disagree with each
other on k22 by 9.1e-04 (27x the Mars m=2 splitting being predicted) and
on k21 by 2.25e-04 (11x the Mars m=1 splitting) -- larger than either
paper's own formal per-order uncertainty for m=2, and close to it for
m=1. Demonstrated reproducibility on individual-order k2m is therefore
nearer 1e-3 than the formal 2.5-4.5e-4 either analysis quotes alone
(`pylov3d.mars_detectability_k2m.grail_k2m_cross_analysis_disagreement`).
Current Mars Doppler tracking has never
even produced a number to compare (the MaQuIs paper's own words: "proved
unsuccessful"). MaQuIs's own targets, for context (not combined
algebraically with the ratio above -- see caveat below): current Mars
gravity resolution is "up to the order of 90-100 degree and order,"
targeted to improve "above the spherical harmonic (SH) degree 90...
up to 360 d/o" (Wörner et al. 2023, Sec. 2.5); the CO2-plus-Phobos/Deimos-
tide seasonal polar signal is "in the order of 230 microGal" at 150-200
km altitude (Sec. 3.1, a different forcing body than the solar tide this
document otherwise concerns, noted for scale only); MaQuIs's stated design
target is "to observe 0.01 microGal per year global changes" (Sec. 2.5,
"Scientific requirements" bullet list -- an earlier version of this
document located this quote in Sec. 2.3, "Temporal gravity changes and
seasonal behaviour of CO2 ice," the wrong section). No microGal-space
conversion of Delta k_2m was attempted (a further,
non-trivial Stokes-coefficient-to-orbit-altitude-gravity-anomaly step,
a different calculation than this derivation); the ratio table above
stays entirely in Love-number space, where GRAIL's numbers are already
directly comparable with no unit conversion (this comparison IS
diagonal-vs-diagonal -- both sides are same-order k2m admittances -- so,
unlike tier 2 below, no category error is involved here).

**How much the Love-number-space ratio understates the instrumental gap.**
A Love number is a dimensionless admittance; reaching a given
`sigma(k_2m)` requires a gravity-field precision that scales with the
tide-raising potential itself. The Earth-on-Moon degree-2 tide parameter,
`xi_2 = (GM_ext/GM_body)(R/d)^3/5`, is `1.5007e-6`; the Sun-on-Mars one is
`2.0378e-9` -- a ratio of **736**. So at equal Stokes-coefficient
precision, achieving GRAIL's Love-number precision at Mars requires a
gravity measurement 736x more precise than GRAIL's own, and the
improvement needed in the quantity an instrument actually measures
(`sigma(Delta C_2m)`, not `sigma(k_2m)`) is **~6.0e3, 8.8e3, 6.1e3**
(m=0,1,2; JPL benchmark) -- roughly 3.8 orders of magnitude, not the "one
order of magnitude" the Love-number ratio alone suggests. The comparison
also spans different mission architectures: GRAIL flew dual spacecraft
with Ka-band inter-satellite ranging at 23-55 km altitude, while Mars
k_2m determinations to date rest on Earth-based Doppler tracking alone.
The defensible statement is that the diagonal splitting is the tighter of
the two channels by a wide margin and is a quantified, mode-specific
measurement requirement -- not that it is nearly within reach.

### 4. Tier 2: the higher-degree off-(2,0) coupled spectrum

The off-diagonal spectrum's largest mode overall remains `(3,0)` at
`|k|=7.29e-5` (TASK-016), followed by `(2,+/-2)` at `3.81e-5`, `(3,+/-1)`
at `2.35e-5`, and 18 further modes above `|k|=1e-6` (21 of 114 non-forcing
modes total). Required precision, via eq. (*'), the basis-normalization-
corrected relation (section 1 above), for the top modes
(`pylov3d.mars_detectability.mars_off20_detectability_table`; "optimistic"
bound = perihelion distance + sectoral (m_f=2) Legendre factor, the
largest, most detection-favorable real signal; "conservative" = mean
distance + zonal (m_f=0), matching the given spectrum's own actual
forcing order exactly -- the *self-consistent* bound, about 2.3x
tighter). `(3,0)` is unaffected by the basis-normalization correction
(it is an `m=0` response mode); every other row below is `sqrt(2)`
tighter than an uncorrected calculation would report:

| mode | \|k\| | required \|Delta C\| (optimistic) | required \|Delta C\| (conservative) | ratio, orbiter (opt.) | ratio, orbiter (cons.) |
|---|---|---|---|---|---|
| (3,0) | 7.29e-05 | 3.86e-13 | 1.66e-13 | 28.5x | 66.2x |
| (2,+/-2) | 3.81e-05 | 2.85e-13 | 1.23e-13 | 38.6x | 89.7x |
| (3,+/-1) | 2.35e-05 | 1.76e-13 | 7.58e-14 | 62.4x | 145.1x |
| (3,+/-3) | 1.02e-05 | 7.65e-14 | 3.29e-14 | 143.7x | 334.1x |
| (4,+/-2) | 7.48e-06 | 5.60e-14 | 2.41e-14 | 196.4x | 456.5x |

There is no GRAIL-based column in this table (see below, "the removed
GRAIL comparison"). Achieved precision, current Mars-orbiter tracking (closest real analogue
-- the recovered *seasonal* CO2-mass-exchange low-degree gravity signal,
the only real, time-varying, low-degree Mars gravity anyone has
measured): Genova, A., Goossens, S., Lemoine, F. G., Mazarico, E.,
Neumann, G. A., Smith, D. E., & Zuber, M. T. (2016), "Seasonal and static
gravity field of Mars from MGS, Mars Odyssey and MRO radio science,"
*Icarus*, 272, 228-245, Table 3: formal 1-sigma uncertainty per fitted
annual/semi-annual/tri-annual amplitude term, `sigma(C20)=1.6e-11`,
`sigma(C30)=1.1e-11` (used above). **This project's task spec named
Konopliv et al. (2016, *Icarus* 274) and Konopliv et al. (2020) for this
number; both were tried this session (ScienceDirect/Wiley abstract pages,
ADS) and returned only paywalled or empty responses -- neither's seasonal
C20/C30 uncertainty table was retrieved.** Genova et al. (2016) is the
same generation of MGS/Odyssey/MRO tracking data (it is, in fact, the
source of `data/mars/gmm3_120_sha.tab`, GMM-3, already used elsewhere in
this document for the TASK-016 areoid correction) and is the paper
actually retrieved and used for this number; the substitution is recorded
here, not left silent.

**The removed GRAIL comparison.** An earlier version of this document
additionally compared this tier's off-diagonal `|k_(n,m)|` directly
against GRAIL's degree-3 Love number uncertainty, `sigma(k3) = 0.0021`
(Konopliv et al. 2013, Table 4), asserting "no unit conversion needed."
That is a category error: `sigma(k3)` is a **diagonal** admittance
uncertainty (driven by the Moon's own degree-3 tide), while this tier's
`|k_(n,m)|` values are **off-diagonal** (the response at `(n,m)` when
Mars is forced at `(2,0)`) -- the same diagonal/off-diagonal conflation
section 2 above warns against, just committed in the opposite direction.
Converted properly to Stokes-coefficient space (an order-of-magnitude
check, not a value this module computes or tests), GRAIL's `k3`
precision implies `sigma(Delta C_3m)` of order `~2e-11` at the Moon (a
back-of-envelope reproduction here lands at 2.0-2.7e-11 using the global,
not libration-constrained, Legendre peak; a properly libration-bounded
calculation gives the tighter 1.65-2.13e-11), giving a required
improvement of roughly 40-55x for `(3,0)` rather than the 28.8x an
earlier version of this document reported; that earlier version's claim
that "both benchmarks land within 1% of each other" was an artifact of
the category error (arithmetically the two uncorrected numbers happened
to be 1.09% apart) and has been retracted. Rather than ship an ad hoc
Stokes-space conversion, the GRAIL column has been dropped from
`mars_off20_detectability_table` entirely; the table above uses only the
Mars-orbiter (Stokes-coefficient-space) comparison, at both bounds.
Separately: Williams et al. (2014) Table 4 (now retrieved, see section 3
above) gives k3=0.0089+/-0.0021 (GL0660B, the *same* figure as Konopliv
et al. 2013) and k30=0.00734+/-0.00375 (GRGM660PRIM) -- there is no
`k3~0.0163+/-0.0007` anywhere in that table; an earlier version of this
document attributed that unverified figure to Williams et al. (2014) on
the strength of web-search summaries alone, which this project's citation
rule forbids, and the attribution was wrong regardless. That sentence has
been deleted.

Even at the most detection-favorable (optimistic) bound, current
Mars-orbiter precision is **28-388x too coarse** for the tier-2 spectrum's
tabulated modes (top mode `(3,0)`: 28.5x; smallest tabulated, `(2,-1)`:
387.7x, corrected from an earlier, uncorrected 548x) -- **between 28x and
66x** depending on whether the optimistic or the self-consistent
conservative bound is used for the `(3,0)` headline specifically. No
tuning toward a detectable answer was applied; these are the ratios the
machinery returns.

### 5. Frequency separation

The tidal signal is periodic at the solar semidiurnal period,
`pylov3d.mars.MARS_FORCING_TD = 44,387.62 s` (verified TASK-025a). The
CO2 seasonal signal (section 4 above) is periodic at the Mars orbital
period, `T=686.98 days` (Genova et al. 2016, eq. 3, retrieved this
session) = 59,355,072 s. Their ratio is **~1337x** (1337.20;
`pylov3d.mars_detectability.frequency_separation_factor`) -- the two
signals sit in well-separated Fourier bins over a multi-year tracking
baseline, with no aliasing concern (spacecraft orbital periods, ~2 h, are
far shorter than the 44,387.62 s tidal period). This quantifies *only*
that the achieved seasonal-band precision is not itself degraded by
confusion with the tidal signal, or vice versa -- it is **not**, and
cannot be read as, a statement about achieved precision *at* the
semidiurnal frequency for degree n>=3: no published degree>=3 Mars
gravity recovery at that period was found in this session. That gap is
stated explicitly, not filled by assumption.

**This separation argument holds only for the m=2 (sectoral, semidiurnal)
component.** The `(2,0)` (zonal) component of the *real* solar tide is
not semidiurnal -- it varies on the annual/semi-annual timescale set by
the sub-solar latitude's yearly excursion through Mars's obliquity (the
same band Genova et al. 2016 fit and attribute to CO2 mass exchange).
Under the self-consistent reading that matches the shipped spectrum's own
forcing order (`m_f=0`, the "conservative" bound above), the signal this
document's required-precision numbers describe is *degenerate* with the
seasonal benchmark's own signal in frequency space, not separated from it
by ~1337x or any factor -- that separation factor is a property of the
real m=2 tide only, not of the `(2,0)`-forced spectrum tier 2 actually
uses.

### 6. Forcing-order scope caveat

The TASK-016 N=115 spectrum (tier 2, section 4) was computed with a unit
`(2,0)` forcing -- a documented convenience in this project, because this
purely elastic Mars model's *diagonal* k2 does not depend on which m is
forced. The *coupled, off-diagonal* spectrum does not share that
invariance: `pylov3d.couplings.next_coupling` sets `m_new = m0 + m1` (an
additive selection rule on the spatially fixed real MarsTopo719
heterogeneity pattern), so forcing at `(2,0)` vs `(2,2)` excites a
**different set** of `(n,m)` response modes at different amplitudes.
Mars's real semidiurnal tide is dominated by the `(2,2)` sectoral
component, not `(2,0)`. A reduced-grid spot check this session
(`lmax=2, Nrbase=30`, not MATLAB-cross-validated,
`pylov3d.mars_detectability.forcing_order_robustness_check`) found the
`(2,2)`-forced spectrum's largest mode, `(3,+2)` at `|k|=5.41e-5`, within
30% of the `(2,0)`-forced spectrum's largest mode, `(3,0)` at
`|k|=7.23e-5` (both at this reduced truncation) -- comparable overall
scale, different `(n,m)` identities. Tier 2's required-precision numbers
should therefore be read as an order-of-magnitude measurement requirement
for the class of off-forcing modes, not a mode-by-mode-exact prediction
of the true semidiurnal-frequency response. Tier 1 (section 3) is not
subject to this particular caveat in the same way, since its `(2,1)`/
`(2,2)` diagonal numbers were computed with the physically correct
forcing order directly.

### 7. Verdict

**Neither tier is detectable with current technology, and the
Love-number-space ratios above understate the instrumental gap
substantially.** Tier 1 -- the observable MaQuIs itself identifies as a
goal -- is 8-12x beyond GRAIL's own best demonstrated precision against
the JPL analysis, and 12-41x against the independent GSFC analysis of the
same data (section 3); once the Moon's 736x-stronger tide (`xi_2`) is
accounted for, the instrumental requirement in the quantity a gravity
mission actually delivers is ~3.8 orders of magnitude (~6.0-8.8e3), not
the "one order of magnitude" the Love-number ratio alone suggests. Tier 2's
higher-degree coupled modes are 28-388x beyond current Mars-orbiter
tracking for the tabulated modes, and 28-66x specifically for the top
`(3,0)` mode depending on which bound is used; the GRAIL-class
comparison an earlier version of this section also quoted has been
removed as a category error (section 4). This is a negative result,
stated plainly, consistent with TASK-021's precedent (that section's 95x
measurement-requirement gap for the hydration front's bulk k2 signature)
and not tuned toward a positive answer.

Two further limits, stated rather than deferred: (1) the diagonal
splitting rests on an `lmax=4` truncation whose convergence is not
established -- moving from `lmax=2` to `lmax=4` changes the three
splittings by 24-79% and reverses which order splits most, and `lmax=5`
has not been run (section 3); the ordering among `m` should not be relied
upon until TASK-027 completes. (2) The `~1337x` frequency-separation
argument (section 5) holds only for the real tide's m=2 component; under
the self-consistent `m_f=0` reading that matches the shipped spectrum,
the signal is degenerate with the seasonal benchmark's own signal band,
not separated from it.

What this *does* establish, for the proposal: a quantified
measurement requirement (Delta k_2m ~ 2-6e-5 for tier 1; Delta C_nm ~
5.6e-14 to 3.86e-13, optimistic bound, for tier 2's top five modes) that a future
mission concept -- MaQuIs or otherwise -- would need to reach, expressed
in the same units (Love-number space for tier 1, Stokes-coefficient space
for tier 2) as the closest real, achieved, or demonstrated analogues
found and retrieved in this session, plus the architecture gap the
Love-number comparison alone hides (GRAIL: dual spacecraft, Ka-band
inter-satellite ranging, 23-55 km altitude; Mars k_2m to date: Earth-based
Doppler tracking only). The defensible statement is that the diagonal
splitting is the tighter of the two channels by a wide margin. What this
analysis cannot say: how
close any specific proposed instrument (including MaQuIs's actual
projected performance, which this document did not attempt to extract or
model) would come to those requirements -- that is future work, not
claimed here.

## Anelasticity (TASK-025a)

Stage a of TASK-025: validate pylov3d's viscoelastic solver path and
produce forward anelastic Love-number calculations for the as-built Mars
model. New modules `pylov3d/anelastic.py` (shared machinery + all Mars
functions) and `pylov3d/anelastic_moon.py` (Moon-specific -- split into
two files to keep both under this repo's 500-line-per-file convention);
tests `pylov3d/tests/test_anelastic.py` (30 tests: 19 fast + 11 `slow`,
Mars+Moon combined). No `pylov3d` solver module was modified. Stage b
(threading anelasticity into `pylov3d.mars_mc`'s Bayesian fit) is
separate, future work -- every function in `pylov3d.anelastic` takes a
plain scalar parameter and returns a complex Love number, the call shape
a stage-b log-likelihood would need. The Moon-side headline result
(Q-consistency check against Williams & Boggs 2015) lives in
`docs/MOON_MODEL.md`, "Anelasticity (TASK-025a)" -- read that section
for the full solver-capability audit and PyALMA3-validation methodology,
summarized only briefly here; this section covers the Mars-specific
forward numbers and forcing-period provenance.

### Solver capability (summary; full audit in MOON_MODEL.md)

`pylov3d` implements **Maxwell** viscoelasticity only (per-layer `eta0`
-> `pylov3d.rheology.compute_complex_rheology`). No Andrade
implementation exists anywhere in this Python port or the vendored
MATLAB source under `src/` (`TestSolverCapabilityAudit`, a live grep
guard). Where Andrade is used below it is **PyALMA3** (`alma`), called
directly as an external reference -- never through `pylov3d.love.get_love`.

**Mars needs no structural simplification for the PyALMA3 comparison.**
Unlike the Moon (whose Weber profile has an internal ocean PyALMA3
cannot represent), Mars's core is already a simple fluid layer at the
center -- exactly the one case PyALMA3 supports. The validation below
therefore runs on the *exact* 4-layer `build_mars_model()` structure, no
simplified analogue needed.

**PyALMA3 is incompressible** (`alma.build_model` takes no bulk-modulus
argument; see MOON_MODEL.md for the full finding). Comparing it against
pylov3d's realistically compressible Mars model (`Ks0` values 70-160 GPa,
comparable in magnitude to mu) at face value disagrees by ~4-6% on
Re(k2) -- a compressibility artifact, confirmed by re-running with
pylov3d's own `Ks0` driven to the incompressible limit
(`1e7 * mu0_surface`, `mars_maxwell_incompressible_model()`), which
recovers agreement at the 1e-3 (0.1%) tolerance level (below). The
Maxwell complex-modulus computation itself does not depend on `Ks0` --
only the Lame parameter `lam = Ks - 2/3*muC` does -- so the
incompressible-limit comparison isolates exactly the rheology mechanics,
while the compressible elastic terms are independently validated to
~1e-12 against native MATLAB (TASK-014/TASK-020, `data/tests/mars/`).

### Validation: Maxwell, pylov3d vs. PyALMA3 (incompressible limit)

As-built 4-layer Mars structure, solar-semidiurnal forcing period (below),
`eta_mantle` applied uniformly to both mantle layers (lower + upper),
`eta_mantle` in {1e15, 1e16, 1e17} Pa s (`TestMarsMaxwellPyALMA3Validation`):

| eta [Pa s] | pylov3d k2 | PyALMA3 k2 | rel. diff Re | rel. diff Im |
|---|---|---|---|---|
| 1e15 | 0.167347 - 0.075932i | 0.167340 - 0.075930i | 3.9e-5 | 3.1e-5 |
| 1e16 | 0.159739 - 0.007719i | 0.159733 - 0.007719i | 3.8e-5 | 3.1e-5 |
| 1e17 | 0.159659 - 0.000772i | 0.159653 - 0.000772i | 3.8e-5 | 3.1e-5 |

All within the test's stated 1e-3 tolerance (same tolerance as the
pre-existing toy-body benchmark) by more than an order of magnitude.
Andrade is validated only at the elastic limit (the one point where a
comparison is possible at all -- see MOON_MODEL.md "Validation" for why);
the Mars Andrade forward numbers below are therefore, like the Moon's, an
**external-tool estimate** via PyALMA3, not a pylov3d-validated result.

### Forcing-period provenance

Konopliv, Park & Folkner (2016), Icarus 274, measure k2 = 0.169 ± 0.006
from Mars's response to the **solar** tide -- per web search of the
primary literature (the paper's own framing). The relevant forcing
period is the **solar semidiurnal** tide -- half a Mars solar day (sol)
-- already used throughout this repository as
`pylov3d.mars.MARS_FORCING_TD` = 44387.622 s (= 88775.244 s / 2, an
earlier draft of this document rounded this to 44387.62 s). This is the
period used for every Mars number below.

**Correction: the Konopliv et al. (2020) parenthetical below was wrong
in two ways.** An earlier draft of this section additionally cited
Konopliv, A. S., Park, R. S., Rivoldini, A., et al. (2020), "Detection
of the Chandler Wobble of Mars From Orbiting Spacecraft," *GRL*, 47,
e2020GL090568, as using the notation "k2^s" for the solar-tide Love
number and as reporting a Mars tidal bulk Q. Both claims are wrong,
confirmed by fetching the paper's full text directly in this session:
(1) the "k2^s" notation is not used anywhere in it (that notation
belongs to Pou, L., Nimmo, F., Rivoldini, A., Khan, A., Bagheri, A., et
al. (2022), "Tidal Constraints on the Martian Interior," *JGR Planets*,
127, e2022JE007291 -- fetched directly in this session, which also
confirmed the first-author initial: an earlier draft of this document
elsewhere cited this paper as "Pou, S." -- whose degree-2-order-2 Phobos
tide Love number is written k2200, e.g. "for a mean value of k2200 =
0.174, this gives us an estimation of Q2200 = 93.0 ± 8.40," quoted
directly, a Phobos-tide-frequency Q independently close to the Bills et
al. (2005) / Efroimsky & Lainey (2007) value below); (2) it does not
report a single tidal bulk Q at all. What it does report, quoted directly from the fetched
text: a frequency-dependent shear-dissipation model fit to the Chandler
wobble period, giving "α can vary from 0.07 to 0.25 and Qo from 78 to
90" (hot mantle end-member, core radius 1790 km) and a Chandler-wobble
decay quality factor "QCW is between 98 and 322" (same end-member) with
an overall composition/thermal-model range of "40–350" for Q_CW. It
also reports its own atmospheric-tide-adjusted k2: the raw solar-tide
k2 = 0.169 ± 0.006 is corrected "for atmospheric tide (and not for
anelastic softening)," quoted directly: "resulting in k2 = 0.174 ±
0.008" -- a value an earlier draft of this document omitted entirely.
None of Qo, Q_CW, or this atmospheric-adjusted k2 is the Phobos-tide Q
discussed below; they are kept here only as directly-sourced context.

**This is a different tidal frequency than Mars's other published
anelastic constraint.** Bills, B. G., Neumann, G. A., Smith, D. E., &
Zuber, M. T. (2005), "Improved estimate of tidal dissipation within Mars
from MOLA observations of the shadow of Phobos," *JGR Planets*, 110,
E07004, doi:10.1029/2004JE002376, infer a Mars tidal quality factor
**Q = 85.58 ± 0.37** from Phobos's secular orbital acceleration -- but
that measurement is at the **Phobos-forced degree-2 tide**, period
~19,991 s (~5.55 hr; half the synodic Phobos-Mars angular rate,
(1/2)*(1/T_Phobos - 1/T_Mars_rotation)^-1 with T_Phobos = 7.65384 hr
sidereal and Mars's sidereal rotation = 24.6229 hr -- an earlier draft
of this document wrote the prefactor as "2*(...)^-1" instead of the
correct "(1/2)*(...)^-1," a factor-of-4 error in the formula as written,
and used the less precise T_Phobos = 7.653 hr; both fixed here,
verified by direct recomputation this session: 19,990.9 s, rounds to
19,991 s, vs. the earlier draft's stated ~19,980 s), **not** the
~44,388 s solar-semidiurnal period k2 = 0.169 (or 0.174, atmospheric-
adjusted) is measured at.

**Correction: the Q value itself, and its attribution, were wrong.** An
earlier draft of this document gave this figure as "Q ~ 80 ± 1" and
claimed it was "corroborated ... consistent across every secondary
source found." Both are wrong. The primary source's actual value,
**Bills et al. (2005), Q = 85.58 ± 0.37**, is quoted directly (retrieved
in this session by fetching Efroimsky, M., & Lainey, V. (2007), "Physics
of bodily tides in terrestrial planets, and the appropriate scales of
dynamical evolution," arXiv:0709.1995, which quotes it verbatim):
"Bills et al. (2005) [arrived] at a reasonable value of the Martian
quality factor, 85.58 ± 0.37. (A more recent study by Lainey et al.
(2007) has given a comparable value of 79.91 ± 0.69.)" The ~80 figure in
the earlier draft is Lainey, V., Dehant, V., & Pätzold, M. (2007),
"First numerical ephemerides of the Martian moons," *Astronomy &
Astrophysics*, 465, 1075-1084, Q = 79.91 ± 0.69 -- a different paper's
number, misattributed to Bills et al. (2005). Bagheri, A., et al.
(2022), "Tidal insights into rocky and icy bodies: an introduction and
overview," *Advances in Geophysics*, 63, 231-320 (fetched directly in
this session), restates the Bills et al. (2005) value to the precision
it gives it at -- note this is **not** an independent corroboration, as
Efroimsky is a co-author of both it and Efroimsky & Lainey (2007), the
other source used here for the same number: "Bills et al. (2005) ... deduced the
tidal quality factor to be Q = 85 ± 0.37," and gives the broader range
it reports across the follow-up studies it cites (Jacobson 2010;
Lainey et al. 2007; Lainey et al. 2020; Rainey & Aharonson 2006) as
"78 ≲ Q ≲ 105".
This document now uses **Q = 85.58 ± 0.37 (Bills et al. 2005, as quoted
verbatim by Efroimsky & Lainey 2007, and rounded to Q = 85 ± 0.37 by
Bagheri et al. 2022)**, distinct from Lainey et al. (2007)'s own
Q = 79.91 ± 0.69, and no longer claims a corroboration this document
could not actually perform (the earlier draft's "corroboration" reported
the wrong paper's number in the first place).

Because these are two different forcing frequencies on a body whose
anelastic response is explicitly frequency-dependent (that is the entire
point of Andrade/Maxwell rheology), **the two published Mars anelastic
numbers are not directly comparable without a frequency-dependence
model** -- this document does not attempt to combine them, and the
Andrade sweep below is reported purely as a function of the
solar-semidiurnal-period k2, with the Bills et al. Q kept as literature
context only.

### Forward anelastic k2: literature-consistent mantle viscosities

`mars_maxwell_k2()` (native pylov3d, compressible as-built model) at the
Andrade-appropriate mantle viscosity range from the literature (see
"Literature parameter ranges" below, 1e19-1e22 Pa s):

| eta_mantle [Pa s] | k2 (Maxwell, pylov3d) | Q_implied |
|---|---|---|
| 1e19 | 0.169000000 - 7.80e-6i | 2.17e4 |
| 1e20 | 0.169000000 - 7.80e-7i | 2.17e5 |
| 1e21 | 0.169000000 - 7.80e-8i | 2.17e6 |
| 1e22 | 0.169000000 - 7.80e-9i | 2.17e7 |

**A literature-consistent Andrade viscosity predicts essentially no
Maxwell dissipation at Mars's semidiurnal period.** At these
viscosities the Maxwell relaxation time (tau_M = eta/mu ~ 1e19 Pa s /
1e11 Pa ~ 1e8 s for the lower-mantle mu) is many orders of magnitude
longer than the 44,388 s forcing period, so omega*tau_M >> 1 and the
mantle sits deep in the Maxwell elastic limit -- Q in the tens of
thousands to tens of millions, i.e. no measurable dissipation. Maxwell
only produces significant dissipation near its relaxation peak
(omega*tau_M ~ 1), which for this forcing period and mu requires
eta ~ mu*Td ~ 4-5e15 Pa s -- five to six orders of magnitude below the
literature Andrade range:

| eta_mantle [Pa s] | k2 (Maxwell, pylov3d) | Q_implied |
|---|---|---|
| 5e15 (near Maxwell peak) | 0.169373 - 0.015589i | 10.9 |

This is the same qualitative mismatch quantified for the Moon in
`docs/MOON_MODEL.md` ("Headline: Maxwell vs. Andrade gap-closing"): a
Maxwell rheology forced to use a physically plausible planetary mantle
viscosity cannot reproduce meaningful tidal dissipation at semidiurnal-
to-monthly forcing periods; only a rheology with a broader relaxation
spectrum (Andrade, Burgers) does. This module does not attempt a
Mars-side Q-consistency fit analogous to the Moon's (Mars's own
anelastic-Q measurement is at a different forcing frequency than k2, see
above, so there is no single-frequency target to close a gap against);
the PyALMA3 Andrade sweep below is reported as forward context, not a
gap-closing/consistency claim.

**PyALMA3 Andrade (external reference), same eta range, alpha = 0.25:**

| eta_mantle [Pa s] | k2 (Andrade, PyALMA3) | Q_implied |
|---|---|---|
| 1e19 | 0.169325 - 0.003961i | 42.8 |
| 1e20 | 0.165114 - 0.002246i | 73.5 |
| 1e21 | 0.162731 - 0.001270i | 128.1 |
| 1e22 | 0.161386 - 0.000716i | 225.3 |

**Note: these Re(k2) values are incompressible-limit, and are not
directly comparable in absolute terms to the measured k2 or to the
compressible pylov3d table above.** The eta=1e19 row's Re(k2) = 0.169325
sits directly under the compressible-pylov3d Forward-sweep table above
(0.169000000), which invites reading it as agreement with the measured
k2 = 0.169 -- it is not a meaningful comparison. PyALMA3's own
incompressible-elastic baseline here is 0.159658
(`mars_maxwell_incompressible_model()`'s elastic k2, verified directly
in this session; also exercised by `TestAndradeExternalSanity`), about
6.05% *above* which this Andrade row's Re(k2) sits, and that
incompressible-elastic baseline is itself about 5.53% *below* pylov3d's
own compressible elastic k2 (0.169, `mars_mod.MARS["k2"]`) -- i.e. the
apparent near-match between this row and the measured value is two ~6%
errors of opposite sign (incompressible-vs-compressible, and
Andrade-vs-elastic) approximately cancelling, not a validated prediction
of the measured k2.

At the *same*, literature-sourced mantle viscosities, Andrade rheology
predicts order-tens-to-hundreds Q -- a plausible, non-negligible
dissipation level, in contrast to Maxwell's effectively-zero dissipation
at the same eta. This is offered as context (a demonstration that the
forward machinery works and that Andrade vs. Maxwell matters
quantitatively for Mars too), not compared numerically against Bills et
al.'s Q = 85.58 ± 0.37 given the frequency mismatch documented above.

### Literature parameter ranges for stage b priors

| Parameter | Range | Source |
|---|---|---|
| Mars mantle viscosity (Andrade, tidal-frequency-relevant) | 1e19 - 1e22 Pa s | Bagheri, A., Khan, A., Al-Attar, D., Crawford, O., & Giardini, D. (2019), "Tidal Response of Mars Constrained From Laboratory-Based Viscoelastic Dissipation Models and Geophysical Data," *JGR Planets*, 124(11), 2703-2727, doi:10.1029/2019JE006015, reporting the Castillo-Rogez & Banerdt (2012) estimate. |
| Mars mantle viscosity (Burgers, single relaxation time) | 1e13 - 1e15 Pa s | Sohl, F., & Spohn, T. (1997), "The interior structure of Mars: Implications from SNC meteorites," *JGR*, 102, as quoted/discussed by Bagheri et al. (2019) (fetched directly in this session; an earlier draft of this row attributed the range directly to Bagheri et al. 2019 without the underlying Sohl & Spohn source): "Relying on a Burgers model with a single relaxation time ... Sohl and Spohn (1997) obtained effective mantle viscosities in the range 10^13-10^15 Pa·s that are similar to those of Bills et al. (2005), and therefore imply inadequate treatment of the transient regime in such a model." Context, not recommended for a tidal-frequency stage-b prior. |
| Mars deep-mantle viscosity (present-day, long-timescale) | 2 - 6e22 Pa s, depths > 500 km | Broquet, A., Plesa, A.-C., Klemann, V., et al. (2025), "Glacial isostatic adjustment reveals Mars's interior viscosity structure," *Nature*, 639, 109-113 -- a **different physical regime** (Myr-timescale glacial isostatic adjustment, diffusion-creep viscosity) than the tidal-frequency Andrade transient response above; not directly interchangeable, included only for contrast. |
| Andrade alpha (silicate mantles, general) | 0.2 - 0.4 | Walterová, M., Běhounková, M., & Efroimsky, M. (2023), *JGR Planets*, e2022JE007652 (arXiv:2301.02476v2, three authors -- an earlier draft omitted Efroimsky); see `docs/MOON_MODEL.md` "Literature parameter ranges" for the full citation trail, including the directly-retrieved quotation (same range applies to any silicate mantle, not Moon-specific). |
| Andrade alpha (commonly adopted) | 0.2 - 0.3 | Efroimsky (2012); Dumoulin et al. (2017) (Venus, adopted from Earth, not derived from Venus data -- see `docs/MOON_MODEL.md` "Literature parameter ranges" for the full citation trail). An earlier draft of this row also cited Castillo-Rogez et al. (2011); that citation could not be verified in this session and is dropped. |
| Mars tidal Q (Phobos-tide frequency, ~19,991 s -- NOT the k2=0.169 frequency) | 85.58 ± 0.37 | Bills et al. (2005), *JGR Planets*, 110, E07004, as quoted directly by Efroimsky & Lainey (2007), arXiv:0709.1995, and (to two significant figures, Q = 85 ± 0.37) by Bagheri et al. (2022),
*Advances in Geophysics*, 63, 231-320; see "Forcing-period provenance" above for the frequency caveat and the correction to an earlier draft's misattributed "~80 ± 1" figure (actually Lainey et al. 2007's Q = 79.91 ± 0.69). |

All Andrade-alpha and viscosity constants also collected in
`pylov3d/anelastic.py` (`MARS_MANTLE_ETA_ANDRADE_RANGE`,
`ANDRADE_ALPHA_RANGE`, `ANDRADE_ALPHA_COMMON_RANGE`) for stage-b reuse;
this table is the citation record of provenance.

## Fixed-shell lateral-amplitude bound (TASK-036a)

The amplitude wall described earlier as a limit of a "linearized rigidity
map" is instead fixed-shell geometry. For crust shear modulus `mu_c`,
displaced mantle shear modulus `mu_m`, shell thickness `T`, and crustal-
thickness excursion `dt`, the implemented relation

```
delta_mu / mu_c = dt * (mu_c - mu_m) / (T * mu_c)
```

is the exact Voigt volume-fraction average within that fixed shell. Its
linearity comes from the material fraction, not a solver expansion, so
raising the coupled solver's perturbation order cannot move this bound.

For the committed Mars constants, `mu_c=30.0 GPa`,
`mu_m=70.0 GPa * 0.9648247661 = 67.5377336 GPa`, and `T=50 km`. Therefore

- `d(mu/mu_c)/d(dt) = -2.5025156e-5 m^-1`;
- `|delta_mu/mu_c|=1` at `|dt|=39.9598 km`;
- the lmax=4 DWAK field's `max|dt|=38.7171 km` gives the reported margin
  `max|delta_mu/mu_c|=0.968903`; and
- holding that field and material contrast fixed, `T=60.5564 km` would
  reduce the maximum rigidity fraction to 0.8.

Mars and the Moon are limited by the **same factor**. Mars reaches unity at
only 79.92% of its 50 km shell, while the Moon reaches it at 82.38% of its
40 km shell. In both cases the crust-to-mantle rigidity contrast amplifies a
sub-full thickness fraction to unity before shell fullness is reached. For
Mars the contrast factor is 1.25126, slightly larger than the Moon's
1.21393. Thus DWAK's proximity to the bound is not evidence that its 50 km
shell is nearly full; it is the fixed-shell Voigt geometry acting through
the rigidity contrast. No shipped shell thickness or lateral result is
changed by this diagnostic.
