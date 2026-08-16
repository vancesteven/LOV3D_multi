# TASK-042 design: a second laterally varying layer for Mars

## Scope and recommendation

This is a design record. No tidal solve was run and no solver code or shipped
model was changed.

Implement the **upper-mantle thermal template** first: project a published
Tharsis-centered temperature field to low degree, convert it to fractional
shear-modulus perturbations with an explicit temperature derivative, and fit
one amplitude before freeing individual spherical-harmonic coefficients. It is
the most direct connection to the physical target, adds only one inference
dimension in its first form, and tests radial distinguishability without
letting a flexible map absorb crustal-model error. The agnostic coefficient
model should be the second implementation and the elastic-thickness model the
third.

The committed repository does not contain the auxiliary propagator or an
archived stress/strain profile needed to calculate a numerical radial
sensitivity kernel. Consequently, the statement below that the upper mantle
should be a larger lever than the crust is a prediction, not a result from an
existing profile. Capturing that profile is an implementation gate before any
mantle response is interpreted.

## Fixed reference model

The design retains the four-layer elastic reference model in `pylov3d/mars.py`:

| Layer | Radius (km) | Thickness (km) | Reference shear modulus |
|---|---:|---:|---:|
| Liquid core | 0--1830 | 1830 | 0 |
| Lower mantle | 1830--2340 | 510 | 96.4825 GPa |
| Upper mantle | 2340--3339.5 | 999.5 | 67.5377 GPa |
| Crust | 3339.5--3389.5 | 50 | 30 GPa |

The mantle values include the committed scale factor
`MARS_MU_SCALE = 0.964824766102174`. The first mantle experiment belongs in
layer 2, the upper mantle. Its nearly 1000 km thickness includes the depth
range in which the candidate thermal models place their largest present-day
lateral temperature differences. The mean (`n=0`) component must remain zero,
so this stage does not silently refit the one-dimensional value of `k2`.

## Pluggable parameterizations

All three options produce the same solver-facing object: real spherical-
harmonic coefficients of

\[
f(\theta,\phi)=\frac{\delta\mu}{\bar\mu},\qquad
\mu(\theta,\phi)=\bar\mu[1+f(\theta,\phi)],
\]

converted to the repository's conjugate-paired complex convention. Keeping
that boundary common makes a later Perple_X or PlanetProfile constitutive map
a data-layer replacement rather than a solver rewrite.

### A. Tharsis thermal template (implement first)

**Data source.** Use the present-day three-dimensional Mars mantle-temperature
field and supplementary data from [Plesa et al. (2018)](https://doi.org/10.1029/2018GL080728),
archived by [TU Berlin DepositOnce](https://depositonce.tu-berlin.de/items/8fe0416f-1ad5-4953-8027-d0374e07e42d).
That study's reference calculations contain stable Tharsis/Elysium upwellings
and place the largest lateral temperature differences in the uppermost roughly
500 km. It also reports appreciable Tharsis power above degree 4, so an
`lmax <= 4` pilot is deliberately a long-wavelength test, not a resolved plume
model.

Take a horizontal slice at a registered depth `z*` (initially 400 km), remove
its area-weighted mean, project it into the repository's real, 4-pi-normalized
basis, and truncate to degrees 1--4. Define a unit-RMS template `T_hat` and

\[
\delta T(\theta,\phi)=A_T\widehat T(\theta,\phi),\qquad
f=\frac{1}{\bar\mu}\frac{d\mu}{dT}\delta T.
\]

In the four-layer pilot this horizontal template is constant through layer 2;
`z*` chooses the map, not a thin radial sheet. Subdividing layer 2 to follow a
three-dimensional plume profile is justified only after the archived radial
kernel identifies useful depth resolution.

[Kumazawa and Anderson (1969)](https://doi.org/10.1029/JB074i025p05961)
report an olivine aggregate shear-modulus temperature derivative of about
`-0.013 GPa K^-1`. Applied to the model upper-mantle modulus, the initial
coefficient is `beta_mu = -1.92485e-4 K^-1`; a 300 K RMS anomaly therefore
corresponds to 5.77% RMS softening. This ambient-pressure, single-mineral
derivative is a constitutive prior, not a Mars-depth calibration. The first
fit should therefore expose `A_T` and either fix `beta_mu` with a documented
factor-of-two sensitivity test or infer their product only. They are perfectly
degenerate in a linear template model.

Initial free parameters: one signed amplitude `A_T`; optional discrete depth
`z*` in {300, 400, 500} km; and, only in a sensitivity run, the magnitude of
`dmu/dT`. The map orientation, phase, and harmonic ratios remain fixed by the
source model.

### B. Elastic-lithosphere thickness map

**Data source.** Candidate priors are the global elastic-thickness calculations
of [Grott and Breuer (2010)](https://doi.org/10.1029/2009JE003456), which place
a present-day low-`Te` region near Tharsis, and the independent local volcanic
load estimates of [Broquet and Wieczorek (2019)](https://doi.org/10.1029/2019JE005959).
The former supplies a global template; the latter is a validation set, not a
global gridding source.

`Te` is an integrated flexural parameter and is not a point measurement of
shear modulus. A fixed numerical lid of thickness `H` can mimic its flexural
rigidity only after declaring an equivalence. With fixed Poisson ratio,

\[
D=\frac{E T_e^3}{12(1-\nu^2)},\qquad
\frac{\mu_{\rm eff}}{\mu_{\rm ref}}=
\left(\frac{T_e}{T_{e,\rm ref}}\right)^3
\]

matches bending rigidity in a lid of fixed thickness. This cubic mapping can
turn a moderate `Te` range into a very broad modulus range and does not make
the three-dimensional tidal constitutive structure unique. The alternative,
moving the base of a constant-modulus elastic lid, changes radial layering and
is a separate model migration.

Initial free parameters would be a global `Te_ref`, one contrast multiplier,
and a fixed low-degree template. This option should not be implemented until
the thermal pilot establishes which upper-mantle radial interval the tide can
distinguish; otherwise its effective modulus has no defensible radial support.

### C. Agnostic low-degree coefficients (pocoMC-ready)

This option has no external spatial data source: the coefficients and their
degree-wise shrinkage prior define the model, while the positivity condition
is its physical admissibility filter.

Represent the upper-mantle field directly as

\[
f=\sum_{n=1}^{L}\left[C_{n0}Y_{n0}+
\sum_{m=1}^{n}(C_{nm}Y^c_{nm}+S_{nm}Y^s_{nm})\right].
\]

There are 8, 15, or 24 real free coefficients for `L=2, 3, 4`, respectively.
Start at `L=2`; retain degree 1 because a degree-1 *material field* is physical,
while excluding degree 0 to preserve the radial fit. Use zero-centered,
rotation-invariant shrinkage priors on each degree and reject samples that
fail the positivity test below. The parameter vector maps directly to pocoMC,
but 24 unconstrained coefficients are unlikely to be identifiable from the
small tidal observable vector without strong shrinkage or multiple forcing
orders.

This option is valuable as a residual test after the one-amplitude thermal
template. It should not lead, because it can fit errors in the Airy crust map,
harmonic truncation, or the elastic reference rheology without identifying a
mantle process.

## Positivity and fixed-shell arithmetic

The exact condition for every direct fractional-modulus parameterization is

\[
\min_{\theta,\phi}[1+f(\theta,\phi)]>0.
\]

Each proposal must pass both (1) synthesis on an oversampled grid, including
the extrema, and (2) a conservative coefficient guard such as
`sum_i |a_i| max|Y_i| < 1-epsilon`. The grid test is less conservative; the
coefficient guard prevents a coarse grid from accepting a narrow negative
excursion. Use a nonzero numerical margin `epsilon` and record its value.

For the thermal mapping, the initial absolute derivative and upper-mantle
modulus give `f = -1.92485e-4 deltaT`. A positive temperature anomaly would
reach zero modulus only at about 5195 K. Thus plausible few-hundred-kelvin
thermal priors are constrained by mantle physics well before mathematical
positivity.

For comparison with the fixed-shell Voigt construction used by the crust
stage, move a boundary between the committed upper- and lower-mantle moduli.
For a shell of thickness `T`,

\[
f=\frac{\mu_A-\mu_B}{\mu_B}\frac{\delta t}{T}.
\]

Using the 999.5 km upper-mantle layer as the reference shell gives a contrast
`|96.4825-67.5377|/67.5377 = 0.428571`; the formal unity excursion is
2332.17 km. The physical residence bound `|delta t| <= 999.5 km` binds first
and limits `|f|` to 0.428571. Using the 510 km lower-mantle layer gives contrast
0.3, a unity excursion of 1700 km, and a residence-limited `|f|` of 0.3.
These numbers show that a mantle boundary-exchange model has ample positivity
margin. They do **not** turn `Te` into a material boundary or justify using the
Voigt map for a thermal anomaly.

## Radial sensitivity: prediction and required evidence

For a small elastic shear-modulus perturbation, the first-order change in a
Love number is controlled by a radial Frechet kernel whose material part is
proportional to the background deviatoric strain-energy density. In schematic
form,

\[
\delta k_2 \simeq -\int K_\mu(r) f(r)\,dr,
\]

with the sign convention consistent with the committed one-dimensional fit:
increasing the uniform mantle scale lowers `k2`. A layer comparison must use
integrals of the *same* kernel over the 50 km crust and the candidate mantle
interval, normalized to the same RMS `f`; layer thickness alone is not a
sensitivity kernel.

The provisional prediction is that an upper-mantle anomaly will move `k2` more
than a crust anomaly of the same RMS fractional modulus. It occupies nearly
1000 km rather than 50 km, degree-2 tidal deformation penetrates the mantle,
and the reference `k2` was calibrated by changing mantle rigidity. This does
not establish the ratio or even guarantee the ordering if the relevant shear
strain is strongly surface-weighted.

The numerical check requested by this task cannot be reconstructed from the
committed artifacts. `get_solution` computes `Aprop_aux`, which
`compute_stress_strain` requires, but `get_love` returns a `RadialSolution`
without that array. The Mars cross-check artifact stores Love numbers and
layer properties, not radial stress or strain. Before the first coupled mantle
solve, archive the uniform degree-2 radius grid, deviatoric strain-energy
profile, and layer-integrated kernel fractions. The prediction is falsified if
the upper-mantle integral per unit RMS `f` is no larger than the crust integral.

## Mode closure and resource estimate

`get_active_modes` deduplicates the union of rheology `(n,m)` pairs before
forming the coupled state. Therefore two layers with the **same harmonic
support do not double `N`**. A no-solve evaluation of the committed mode-
selection algorithm gives:

| Mantle/crust union cutoff | Unique real-field pairs | Coupled `N` | Baseline wall estimate | Planning peak RSS |
|---:|---:|---:|---:|---:|
| 2 | 8 | 43 | 11 s | 1.3 GB |
| 3 | 15 | 75 | 48 s | 4.1 GB |
| 4 | 24 | 115 | 153 s | 7.4 GB |

The same `N` values result from the committed Mars crust support, which omits
one zero coefficient at each cutoff. New mantle harmonics beyond the crustal
cutoff increase closure according to the **union**, not the number of layers.

The wall estimates use a log-log fit, `wall = 4.334e-4 N^2.6916 s`, to the
committed Mars `lmax=4/5/6` measurements (150.28/403.23/848.54 s at
`N=115/163/219`, `Nrbase=30`). The RSS column uses the committed same-`N`,
`Nrbase<=30` Moon measurements (1.3/4.1/7.4 GB) as a conservative planning
proxy; Mars has four radial layers rather than ten, and its committed
`N=219` run peaked near 11 GB. Hardware and JAX allocation can move both
figures. Reserve up to twice the baseline wall and roughly 2/6/10 GB for the
three pilot cutoffs until a two-layer dry benchmark is archived.

Layer count still adds coefficient conversion and radial assembly work, so
unchanged `N` is not a promise of identical wall time. It does rule out a
state-dimension doubling when the supports overlap.

## Implementation gates and falsification test

The implementation should stop after the smallest thermal pilot can answer
the distinguishability question:

1. Archive the one-dimensional shear-strain kernel and compare upper-mantle
   and crust integrals at equal RMS `f`.
2. Import and register the Plesa temperature field; preserve its source grid,
   units, longitude convention, normalization, and checksum alongside the
   projected coefficients.
3. Verify zero amplitude reproduces the uniform solution, conjugate pairs
   synthesize a real field, positivity passes with margin, and identical
   two-layer harmonic support reproduces the tabled `N`.
4. At `L=2`, compute response Jacobians for one thermal-template amplitude and
   one crust-amplitude nuisance parameter, using the same observable vector
   and measurement covariance.
5. Whiten the two Jacobians. The thermal option is not useful for this data set
   if its component orthogonal to the crust Jacobian is below one standard
   deviation even at the largest physically admitted thermal amplitude. A
   whitened correlation of at least 0.95 is an additional degeneracy warning,
   but correlation alone is not the decision rule.
6. Only if the `L=2` template passes, compare `L=3` and `L=4`. Failure to gain
   distinguishable response as the known degree-4-and-higher Tharsis content
   enters falsifies the low-degree template, rather than motivating an
   unconstrained production inversion.

If the thermal template fails, the next action is the agnostic `L=2` residual
test—not an immediate high-dimensional `L=4` pocoMC run. If the residual test
also projects onto the crust response, the data do not distinguish a second
three-dimensional layer under this elastic model, and the mantle stage should
stop.
