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
