# TASK-044: the agnostic L=2 residual test — can ANY mantle pattern be seen?

## Why

TASK-043's thermal template failed distinguishability at L=2 (0.05σ
orthogonal to the crust nuisance at the 300 K bound; whitened correlation
0.973). Per the verified TASK-042 design, the prescribed next action is the
**agnostic L=2 residual test**: drop the thermal prior entirely and ask
whether *any* laterally varying upper-mantle rigidity pattern at L=2
produces observable response orthogonal to the crust nuisance, at
positivity-admissible amplitude. If none does, "the data do not distinguish
a second three-dimensional layer under this elastic model, and the mantle
stage should stop" — the design's own words, and a clean, reportable
negative that sharpens the proposal's focus on crustal science.

## Method

1. **Jacobian basis.** Central-difference response Jacobians for each of
   the five real L=2 upper-mantle coefficients (C20, C21, S21, C22, S22 in
   the repository's real 4π basis), same machinery, observable vector
   [C20,C30,C31,S31,C32,S32,C33,S33], covariance, Nrbase=30, and
   step-halving evidence as TASK-043 (reuse `scripts/` pilot
   infrastructure; 10 + 10 solves at L=2, N=43 — cheap). The crust
   nuisance Jacobian can be reused from the committed pilot npz (state
   its SHA-256) unless step-size hygiene requires recomputation — say
   which.
2. **Subspace analysis.** Whiten all Jacobians with the same registered
   covariance. Project the 5-column mantle Jacobian orthogonal to the
   crust Jacobian; SVD the projected matrix. The top singular direction is
   the most-distinguishable mantle pattern; report all five singular
   values.
3. **Positivity-bounded amplitude.** For the top singular direction
   (unit-RMS normalized pattern), compute the maximum admissible amplitude
   from `min(1+f) > 0` on an oversampled grid AND the conservative
   coefficient guard, exactly as TASK-043 gate 3. Both numbers reported.
4. **Decision metric.** Orthogonal whitened response at the
   positivity-bounded amplitude of the best direction. Gate: ≥1σ to
   proceed; below 1σ the mantle stage stops. Also report the same metric
   at a physically flavored 10% RMS δµ/µ̄ amplitude for context (state
   that it is context, not the gate).
5. **Honesty constraints carried from 043:** the diagonal seasonal-gravity
   covariance is a scenario benchmark, not a mission covariance — the
   conclusion is conditional on it; positivity margins recorded with their
   ε; no L=3/4 anything.

## Deliverables

Driver script (extend the pilot infrastructure; no solver-module change
beyond what `pylov3d/mars_mantle.py` already provides — if a new entry
point is needed there, keep it additive and tested), committed npz artifact
with all Jacobians/singular values/bounds + SHA-256 in the report, a
`## Result` section appended to this file, ledger DONE → different-driver
verification. Suite green. Prose: never "genuine" or "honest".
