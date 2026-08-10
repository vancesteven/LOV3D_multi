# TASK-027 (Machine B): Mars lateral spectrum — truncation convergence and Airy sensitivity

## Why

Two stated weaknesses in the Mars lateral stage are currently
acknowledged-but-unquantified, and both are load-bearing for the proposal
because the (4,0) first-order k2 result depends on them.

**1. Truncation is not shown to be converged.** `docs/MARS_MODEL.md` §6
reports one spot check: the (2,0) k2 shift moves 5.517e-5 → 5.973e-5 going
from lmax=4 to lmax=5, an **8.3% change**. That passes the 20% adequacy
bound the test asserts, but 8.3% is not convergence — it is one step of a
sequence nobody continued. TASK-021b already showed (for the *hydration*
lateral term) that lmax=6 shifts the lateral contribution by only
0.33–2.05%, suggesting the series does converge; the pure Airy lateral
spectrum has not had the same treatment.

**2. Airy is the only crustal model, and §5 flags this as the weak point.**
The document states the (4,0)-driven forcing-mode shift "scales ~1:1 (not
quadratically) with the Airy calibration for this component, so it is
comparatively sensitive to the Airy factor / crust density assumptions" —
an approximation "already flagged as weakest specifically at Tharsis, and
not independently re-verified against a non-Airy crustal-thickness model."
The PI's original direction was that Airy makes the fewest assumptions and
that an interpreted seismic result could be imposed later. Later is now:
the proposal is stronger if the headline result survives a different
crustal model.

## Owner and routing

**Machine B.** Compute-heavy, token-light: both parts are sweeps over the
existing, validated pipeline with no new physics. Claim in the ledger the
usual way before starting.

## Part 1 — truncation convergence

Extend `mars_lateral_love_spectrum` past the lmax=5 spot check:

- Run lmax = 4, 5, 6 (and 7 if tractable) at fixed Nrbase, tracking the
  (2,0) k2 shift *and* the top few off-(2,0) mode amplitudes (the (3,0),
  (2,±2), (3,±1) modes that TASK-026 is assessing for detectability — their
  convergence matters as much as the forcing mode's).
- Report the step-to-step relative change so the reader can see whether the
  series converges and at what rate. State the mode count N at each lmax.
- **Watch memory.** TASK-021b hit >15 GB at lmax=6/Nrbase=50 and had to drop
  to Nrbase=30. Truncation is an angular question and is radial-independent
  to good approximation (021b's own argument), so running the lmax sweep at
  a modest Nrbase is legitimate — but verify that claim here by holding lmax
  fixed and varying Nrbase at least once, rather than assuming it.
- If the series does *not* converge by lmax=6–7, that is the finding, and it
  materially qualifies the (4,0) result. Report it as such.

## Part 2 — Airy sensitivity

Quantify how much the headline numbers depend on the crustal model:

- **Cheap first pass (do this regardless):** the Airy calculation has
  explicit crust-density and mantle-density inputs and an implied
  compensation factor. Sweep them across their defensible published ranges
  and report the induced spread in (a) the (2,0) k2 shift and (b) the top
  off-(2,0) amplitudes. This is the direct analogue of TASK-021's
  reference-crust sensitivity table, which found the *denominator* choice
  mattered more than the property bracket — expect something similar here.
- **Second pass if tractable:** substitute a published, independently
  derived Mars crustal-thickness model for the Airy-derived one and rerun.
  Candidates are the InSight-calibrated crustal thickness models
  (Wieczorek et al. / Knapmeyer-Endrun et al. family). **This needs a data
  fetch, which B may not be able to do** — if no suitable model is
  available offline, do Part 2's first pass only and say so; A will fetch
  the model and ticket the substitution separately rather than have you
  block on it.

## Constraints

- Do not modify `pylov3d/mars_lateral.py` or any solver module — driver
  scripts only, per the TASK-021b precedent (`scripts/mars_hydration_sweep.py`).
- Archive results as an `.npz` + figure under `docs/figures/proposal/`,
  matching the 021b artifact convention.
- Citation rule: no verbatim quotation unless retrieved in-session.
- Prose standard: never "genuine" or "honest".
- Fast lane stays green.

## Done criteria

A `docs/MARS_MODEL.md` section reporting: the convergence sequence with
step-to-step changes for the forcing mode and the top off-modes; the
Nrbase-independence check; the crust-parameter sensitivity spread; and an
explicit statement of whether the (4,0) first-order result and the
detectability-relevant amplitudes survive both. Ledger updated, awaiting
VERIFIED by a different driver.
