# TASK-038 (Machine B): re-anchor the Moon lateral spectrum with the dichotomy retained

## Why

The PI resolved the TASK-031 plan/implementation divergence (2026-08-14):
**degree-1 is retained** as the nearside–farside crustal dichotomy, per the
original plan. `pylov3d.moon_lateral` now defaults to
`include_degree1=True`; the pre-decision field is reproducible via
`include_degree1=False`.

That changes the shipped Moon field — it gains `dt(1,1) = −6.83` km,
`dt(1,−1) = −2.81` km, `dt(1,0) = +0.92` km — so the TASK-035 MATLAB anchor
now pins the *superseded* field. The anchor method and its conclusions
remain valid for what they anchored (both codes consumed the identical
field, and the port-fidelity conclusion carries), but the field the project
now ships is unanchored. Close that the same way TASK-035 did.

Two facts worth knowing before you start:

- **Degree-1 partially cancels the high-degree extremes.** New positivity
  margins are 0.9898 / 1.0505 / 1.0811 at lmax = 4/5/6 (old:
  0.9902 / 1.1531 / 1.2897). At lmax=6 the field now fits inside the 40 km
  shell (max|dt|/T = 0.89), so the rigidity guard binds there, not the
  thickness guard. lmax=4 remains the only admissible truncation.
- **The mode count does NOT grow.** N = 115 exactly as before — the mode
  set is the lmax=4 coupling closure, which is independent of which field
  coefficients happen to be nonzero. (This spec originally predicted N
  would exceed 115; the rerun refuted that.) Do not assume the
  *composition* of the 115 modes carried over — derive any per-order
  breakdown from your own run's output, not from TASK-035's 1/42/73. The
  committed rerun is `docs/figures/proposal/moon_lateral_spectrum.npz`.

## Method

Same as TASK-035 — your own method, unchanged:

1. Re-run `scripts/export_moon_mu_variable.py` (it follows the module
   default, so it now exports the dichotomy field — 23 amplitudes, adding
   the three degree-1 entries) and confirm the export matches the committed
   `data/moon/moon_mu_variable_lateral.npz` regenerated on machine A.
2. Re-run `scripts/moon_lateral_cross_check.m`. Update its hardcoded
   Python reference constants from the regenerated spectrum npz, **at full
   precision this time** — the verification of TASK-035 (ledger,
   2026-08-14) showed the 6-significant-figure constants made the anchor
   look ~60× looser than it is. Compare per mode against the npz values,
   not rounded copies.
3. Report per-mode agreement with the same structure as TASK-035, plus the
   perturbation-order breakdown of the new N.

## Constraints

- Do not modify `pylov3d/moon_lateral.py` or any solver module.
- Overwrite the TASK-035 artifacts in place
  (`data/tests/moon/moon_lateral_cross_check.{log,mat}`); git history keeps
  the old ones at `47b5377`/`dd86cdb`. Note the supersession in
  `docs/MOON_MODEL.md`.
- Record the MATLAB version. Prose standard: never "genuine" or "honest".
- Suite green. Commit and push as you go (standing authorization).

## Done criteria

Per-mode agreement at full precision; artifacts replaced; `docs/MOON_MODEL.md`
updated to say which field each anchor pins; ledger to DONE awaiting
VERIFIED (A verifies — B implements, so it cannot self-clear).
