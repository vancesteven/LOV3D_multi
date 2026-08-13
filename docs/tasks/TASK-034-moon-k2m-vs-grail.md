# TASK-034 (Machine B): predicted Moon k2m splitting against GRAIL's measurement

## Why this is the strongest comparison available anywhere in the project

The Mars detectability argument (TASK-026) has one unavoidable weakness: to
benchmark a Mars prediction we had to borrow GRAIL's *lunar* precision and
correct across bodies by the tide-raising ratio of 736. That correction is
defensible but it is the step a reviewer attacks.

At the Moon there is no such step. **GRAIL measured k20, k21 and k22
separately at the Moon** (Konopliv et al. 2013, Table 4 — already committed
in `pylov3d/mars_detectability_k2m.py` as `GRAIL_K2M` / `GRAIL_K2M_SIGMA`),
and TASK-031 built a Moon lateral model that can predict the splitting those
same coefficients should show. Same body, same coefficients, real published
measurement, no extrapolation.

That makes this the cleanest test of the project's central claim — that
lateral crustal structure imprints on the order-dependence of k2m — and it
can be stated without a single cross-body assumption.

## What to compute

The Moon analogue of TASK-030, using the committed pipeline:

1. **Diagonal k2m solves** at forcing (2,0), (2,1), (2,2) on the Weber model
   with the TASK-031 lateral field, at `lmax=4` — which TASK-031b established
   is the highest cutoff the linearization admits (`max|dmu/mu_bar|` = 0.9902
   at lmax=4, crossing unity at lmax=5), so **do not attempt a higher
   ladder**; that question is closed and the answer is that it cannot be run.
   Report each |Delta k2m| against the uniform Weber k2 = 0.02315914223.

2. **The comparison.** Against GRAIL's measured values and uncertainties:
   k20 = 0.02408 +/- 0.00045, k21 = 0.02414 +/- 0.00025,
   k22 = 0.02394 +/- 0.00028. Report, explicitly:
   - the predicted splitting for each order;
   - the ratio of predicted splitting to each measurement's uncertainty;
   - the *observed* order-to-order spread (max minus min of the three GRAIL
     values) and how it compares both to the individual uncertainties and to
     the predicted splitting.

3. **Say plainly what it means.** Note before you start, so it does not
   surprise you into massaging anything: the observed GRAIL spread across the
   three orders is about 2.0e-4, which is *smaller than the individual
   uncertainties* (2.5-4.5e-4). So the measured order-to-order differences
   are very likely not statistically significant, and the model's predicted
   splitting is expected to be smaller still by a wide margin. **A result of
   the form "the prediction sits two orders of magnitude below both the
   precision and the (insignificant) observed scatter" is the expected and
   perfectly publishable outcome.** Do not tune toward a detection.

## Second part, if budget allows

TASK-025a/b established that the Weber Moon needs anelasticity to reach the
observed k2 at all — the elastic model undershoots GRAIL's k2 = 0.02422 by
~4.6%. The lateral stage is purely elastic. Worth one sentence of analysis,
not a run: does the elastic/anelastic gap dominate the lateral splitting by
so much that the lateral signal is unobservable in principle at the Moon,
independent of measurement precision? Quantify the two against each other.

## Inputs, all committed

- `pylov3d/moon_lateral.py` (pipeline), `scripts/moon_lateral_spectrum.py`
  (driver precedent), `docs/figures/proposal/moon_lateral_spectrum.npz`.
- `docs/MOON_MODEL.md`, "Moon lateral crust stage (TASK-031)" and the
  TASK-031b convergence subsection — read both; 031b's conclusion bounds
  this task.
- GRAIL constants in `pylov3d/mars_detectability_k2m.py` (`GRAIL_K2M`,
  `GRAIL_K2M_SIGMA`). **Reuse those rather than retyping them** — they are
  the values an adversarial review already verified against Konopliv et al.
  (2013) Table 4.

## Constraints

- Driver script only; do not modify `pylov3d/moon_lateral.py` or any solver
  module.
- Archive `.npz` + figure under `docs/figures/proposal/`, and append a
  results subsection to `docs/MOON_MODEL.md`.
- Citation rule: no verbatim quotation unless retrieved in that session. The
  GRAIL numbers are already verified in-repo; cite them as such.
- Prose standard: never "genuine" or "honest".
- Suite green.

## Done criteria

The three predicted splittings; their ratio to GRAIL's per-order
uncertainties; the observed-spread comparison with an explicit statement of
whether that spread is significant; and a plain conclusion about what the
Moon can and cannot test. Ledger updated to DONE awaiting VERIFIED.
