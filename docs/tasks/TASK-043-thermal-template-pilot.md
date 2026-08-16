# TASK-043: Mars thermal-template pilot

## Acceptance contract

Implement gates 1--5 of the verified TASK-042 design at the smallest useful
cutoff (`L=2`):

1. Archive a converged one-dimensional degree-2 shear-energy profile and
   corroborating finite-difference `dk2/dln(mu)` values for the upper mantle
   and crust at equal fractional-modulus amplitude.
2. Register Plesa et al. (2018) Data Set S1 with source coordinates, units,
   file size, checksums, and explicit raw-data handling; project its 150 km
   temperature field into the repository's real, 4-pi-normalized basis.
3. Prove zero-amplitude reduction, real conjugate synthesis, mathematical
   positivity with a nonzero margin, and the `L=2`, `N=43` two-layer closure.
4. Compute central-difference response Jacobians for signed thermal-template
   amplitude and crust-amplitude nuisance parameters on one shared observable
   vector, with step-halving evidence.
5. Whiten those Jacobians with an explicitly registered covariance and report
   their correlation and the thermal component orthogonal to the crust.

The raw source field is at 150 km, not the provisional 400 km depth in
TASK-042. This task uses the source value and does not invent an unarchived
slice. The low-degree horizontal template is applied through the upper-mantle
layer only as a pilot hypothesis.

## Non-goals and stop conditions

- No `L=3` or `L=4` solve, pocoMC inversion, new free spherical-harmonic
  parameterization, elastic-thickness stage, or solver-core refactor.
- The archive's open-access record does not establish permissive downstream
  redistribution. Do not commit the raw table; use the verified fetch path.
- The repository has no mission covariance for the complete signed C/S
  response vector. Any diagonal seasonal-gravity analogue is a scenario
  benchmark, not a mission-detectability claim.
- `A_T` and `dmu/dT` are exactly degenerate; the pilot fixes
  `beta_mu = -1.92485e-4 K^-1` and varies only `A_T`.
- Stop if the upper-mantle radial response is not larger than the crust at
  equal RMS fractional modulus, if positivity fails, or if the largest
  explicitly admitted `A_T` produces less than one whitened standard
  deviation orthogonal to the crust nuisance.

## Result

TASK-043 reaches the designed stop condition at `L=2`; it does not proceed to
`L=3` or `L=4`.

Gate 1 passes. The archived degree-2 radial evidence at `Nrbase=50,100,200`
converges below 1%: the finest upper-mantle/crust deviatoric strain-energy
integral ratio is 36.8231 and the independent finite-difference
`|dk2/dln(mu)|` ratio is 64.6215. The energy profile is explicitly a radial
sensitivity proxy rather than an absolutely normalized Frechet kernel; the
finite-difference result supplies the independent ordering check. Artifact:
`docs/figures/proposal/mars_1d_shear_kernel.npz`.

Gate 2 passes. The verified raw table is 64,800 by 8 on a one-degree
0--359 degrees-east, 90--(-89) degree grid. Its area-weighted T150 mean is
898.253534 K and centered RMS is 129.821070 K. The real-4pi degree 1--4 RMS
values are 88.633995, 61.717618, 40.900913, and 25.001897 K. The raw table is
not redistributed. Derived artifact:
`data/mars/plesa2018/plesa2018_t150_l4_template.npz`.

Gate 3 passes. The separately normalized `L=2` coefficients have unit
spherical RMS and no degree-zero term. At the declared scenario bound
`|A_T|=300 K`, `min(1+delta_mu/mu)=0.875521` and the conservative coefficient
margin is 0.735325. Complex synthesis is real to numerical precision,
zero thermal amplitude reproduces the one-dimensional `(k,h,l)` solution
exactly when no crust field is supplied, and shared mantle/crust support gives
`N=43` rather than doubling the state.

Gates 4--5 produce a negative distinguishability result. Eight bounded
central-difference solves at `L=2`, `Nrbase=30`, use signed
`[C20,C30,C31,S31,C32,S32,C33,S33]`. The diagonal covariance scenario uses
the repository's seasonal analogues `sigma(C20)=1.6e-11` and
`sigma(degree-3 C/S)=1.1e-11`; it is not a mission covariance or a mission
detectability claim. The half-step errors are 3.12e-6 for the thermal
Jacobian and 1.30e-4 for the crust Jacobian. Their whitened correlation is
0.972940, above the 0.95 warning threshold, and the thermal component
orthogonal to the crust reaches only 0.050187 sigma at 300 K, far below the
one-sigma gate. Artifact:
`docs/figures/proposal/mars_mantle_thermal_l2_pilot.npz` (SHA-256
`09022def88d241318c63271ded6eee5ecddf8c5ca164d1de345b337fd63f65ee`).

Therefore this data/covariance scenario cannot distinguish the one-amplitude
thermal template from the crust nuisance. Per TASK-042, the next scientific
action is the agnostic `L=2` residual test, not a higher-degree thermal run.
