# Mars 1D Radial Reference Model (TASK-011)

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

Constraints: mass (`GM/G`, sigma = 0.1% — dominated by G's uncertainty, far
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
ridge — a genuine, not merely wide, non-identifiability (found in review:
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
ms/eval at the fit's `Nrbase=100`; a full run is O(10^3-10^4) evaluations).

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
