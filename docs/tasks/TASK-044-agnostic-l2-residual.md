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

## Result

TASK-044 reaches the designed stop condition at `L=2`: no single-degree-of-freedom
combination of the five real L=2 upper-mantle basis coefficients clears the
1-sigma gate at positivity-admissible amplitude, though the best direction comes
close.

The committed TASK-043 pilot artifact
(`docs/figures/proposal/mars_mantle_thermal_l2_pilot.npz`) is verified byte-for-byte
before use: SHA-256 `09022def88d241318c63271ded6eee5ecddf8c5ca164d1de345b337fd63f65ee`,
matching both the spec and the archived `[A][2026-08-16]` verification. Its crust
nuisance Jacobian and diagonal seasonal-gravity covariance scenario
(`sigma(C20)=1.6e-11`, `sigma(degree-3 C/S)=1.1e-11`) are reused unchanged; the
crust Jacobian's own step-halving evidence from TASK-043
(`crust_step_error=1.30e-4`) is not superseded because the newly computed mantle
basis Jacobians converge comfortably inside the same 1% tolerance (see below), so
mixing artifacts of the same `Nrbase=30` solve introduces no step-policy
inconsistency.

Central-difference Jacobians for the five real L=2 upper-mantle basis coefficients
(`C20, C21, S21, C22, S22`; single unit coefficient per pattern, unit spherical
RMS by construction) use the same observable vector
`[C20,C30,C31,S31,C32,S32,C33,S33]`, `Nrbase=30`, and a fractional
`delta_mu/mu` step of `1.92485e-3` (matching TASK-043's thermal step magnitude,
`|beta_mu|*10 K`, to stay in the same demonstrated near-linear regime). Half-step
relative errors per column are `C20=1.61e-6`, `C21=6.32e-6`, `S21=7.54e-6`,
`C22=6.16e-6`, `S22=6.19e-6`, all far below the 1% convergence tolerance.
Real-conjugate synthesis holds to `1.44e-18`.

Whitening all six Jacobians (five mantle plus the reused crust) by the registered
sigma, projecting the mantle Jacobian orthogonal to the whitened crust direction,
and taking the SVD of the projected `8x5` matrix gives singular values
`[2.239854, 0.011957, 0.011779, 0.003979, 0.001862]`. The top singular direction
is almost pure `C20`:
`{C20: -0.999942, C21: 0.001027, S21: -0.000385, C22: -0.008430, S22: 0.006612}`
— the only basis pattern that couples strongly to the `(2,0)` tidal forcing at
this order; the other four directions carry two orders of magnitude less
orthogonal-to-crust signal.

For the top direction, the positivity-bounded amplitude is
`0.439999` for both signs, with the sign-symmetric conservative coefficient guard
(`sum_i |a_i| * sqrt(2n_i+1)`, TASK-043 gate 3's method, `epsilon=1e-6`) binding
in both directions ahead of the oversampled-grid bound (`nlat=361, nlon=720`); the
coefficient guard is reported alongside the grid bound in the artifact and is used
because it is the more conservative of the two. At that amplitude the whitened
orthogonal response is **0.985534 sigma** — below the 1-sigma gate, so the mantle
stage stops. The top direction also has the best bound-times-singular-value
product across all five directions (`0.985534` vs. `0.003443, 0.003607, 0.001172,
0.000588` for the others); no lower-ranked direction wins on the product, so the
top direction's own metric is the decision metric.

For context only, not the gate: at a physically flavored 10% RMS
`delta_mu/mu` amplitude, the same top-direction metric is **0.223985 sigma**, well
below both the positivity-bounded number and the gate.

Artifact: `docs/figures/proposal/mars_mantle_agnostic_l2.npz` (SHA-256
`4b3afc8d71ec44fa230528eed3af347249bcc0567e85dc83774d805472eb2607`), containing
all five mantle Jacobians and their half-step counterparts, the reused crust
Jacobian and its source SHA-256, sigma, covariance, channel labels, all five
singular values and singular directions, per-direction positivity bounds, and
per-direction decision/context metrics, plus a full `metadata_json` settings
record.

The diagonal seasonal-gravity covariance reused here is a scenario benchmark, not
a mission covariance, and every number above is conditional on it: under this
scenario covariance and this positivity-bounded-amplitude convention, no L=2
upper-mantle rigidity pattern — thermally templated or otherwise unconstrained
in shape — produces a response distinguishable from the crust nuisance at the
1-sigma level; the closest candidate, an almost-pure `C20` pattern, falls short
at 0.985534 sigma. Per TASK-042's stop rule, the mantle stage stops here; no
`L=3` or `L=4` solve was run.

## Verification amendment (A, TASK-045, 2026-08-16)

The paragraph above is correct at its stated convention but **the verdict is
convention-sensitive, and the original artifact could not show it**: the
driver stored the guard-clamped amplitude (0.439999) in all three positivity
slots, so the raw grid bounds were unrecoverable. Recomputed at 721x1440:

| amplitude convention | bound | top-direction metric |
|---|---:|---:|
| coefficient guard, sign-symmetric (committed gate) | 0.439999 | 0.985534 sigma |
| raw grid bound, sign-symmetric | 0.447241 | **1.001754 sigma** |
| raw grid bound, best sign (largest admitted amplitude) | 0.878182 | **1.966996 sigma** |

Under TASK-042's own stop-rule language ("the largest physically admitted
amplitude"), the best-sign grid bound is the applicable reading, and the
C20 pattern **clears the gate at ~2 sigma — at an amplitude where the mantle
rigidity locally approaches zero** (the positivity boundary itself). The
right statement is therefore a threshold, not a binary: **1-sigma
distinguishability requires RMS `delta_mu/mu >= 0.4465`** (= 1/2.239854),
whereas the thermal ceiling at 300 K is 0.0577 RMS — a factor 7.7 short —
and the 10% RMS context amplitude reaches 0.224 sigma. **The mantle stage
still stops, but on physical-amplitude grounds rather than on the
mathematical positivity bound as originally written**, and remains
conditional on the scenario covariance throughout.

The step-policy mixing concern is closed: recomputing the crust Jacobian at
the mantle step policy (0.002, half-step residual 2.6e-7) changes it by
1.9e-4 relative and moves the decision metric by 2.5e-5 sigma — five orders
below the near-miss margin.

The near-pure-C20 dominance follows from the selection rules: with a (2,0)
forcing, a zonal `(2,0)` rheology pattern self-couples the forcing mode
directly (the parity-even `2+2+2` channel), feeding the C20 observable,
while the `m != 0` L=2 patterns reach the observable vector only through
off-forcing modes whose sensitivities are two orders smaller under this
covariance.

The driver now archives the raw grid bounds per sign, the per-convention
metrics, and the 1-sigma threshold amplitudes; the artifact was
regenerated (same solve configuration, SHA-256
`e53ef68330e1488f6ab62f468c737a3894e145ec9e077d66287226f636cd0a27`),
reproducing every committed number and adding the convention-resolved
metrics: guard 0.985534, grid-symmetric 1.001750, grid best-sign
1.966998, threshold 0.446458.
