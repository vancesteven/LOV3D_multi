# TASK-026: Detectability of the off-(2,0) tidal Love-number spectrum

## Why this is the critical task

This closes the one loop the proposal's tidal argument currently leaves
open, and it is deferred *in writing* twice in the traceability document:

- `Methods_Models.tex` §4.7: "the front-diagnostic observable ... lives
  almost entirely in the off-$(2,0)$ Love spectrum ... whose detectability
  this document does not attempt to assess; that is explicitly future work."
- §5: "detectability by future gravity investigations is the natural next
  question for the proposal's Task 1."

The chain the proposal makes is: hydration front → lateral crustal rigidity
→ tidal signature → *detectable* signature in gravity (RQ1). TASK-016
established the spectrum exists and MATLAB-validated it. TASK-021
established that k2 itself is blind to front *location*, and that the
location information lives in the off-(2,0) modes. **Nobody has asked
whether those modes are measurable.** Until that is answered, the tidal
channel is a validated capability, not yet a science case — and a reviewer
will ask exactly this question.

Note the answer may well be negative. A negative answer, quantified, is a
perfectly good proposal result (it defines a measurement requirement, as
TASK-021's null result did). Do not tune the analysis toward a positive.

## The numbers you are starting from

From the MATLAB-validated N=115 spectrum (`data/tests/mars/mars_lateral_cross_check.mat`,
Airy-derived crustal rigidity at n_lv≤4, elastic, forcing mode (2,0)):

| mode | \|k\| | vs k2=0.169 |
|---|---|---|
| (3, 0) | 7.29e-5 | 4.3e-4 |
| (2, ±2) | 3.81e-5 | 2.3e-4 |
| (3, ±1) | 2.35e-5 | 1.4e-4 |
| (3, ±3) | 1.02e-5 | 6.0e-5 |
| (4, ±2) | 7.48e-6 | 4.4e-5 |

21 of 114 non-forcing modes exceed \|k\| = 1e-6. For scale, the k2 lateral
shift is 5.52e-5 and current σ_k2 = 0.006.

## What to determine

The core question: **what gravity-field measurement precision is required
to detect a degree-n tidal Love number at these amplitudes, and how does
that compare to what exists and what is plausibly achievable?**

This is a different question from "compare to σ_k2" — σ_k2 is the
uncertainty on a *degree-2* Love number from a specific tracking analysis.
A degree-3 tidal response is a distinct observable with its own recovery
characteristics. Do not simply divide by 0.006 and call it done; that
conflates two different measurements and the review will catch it.

Work out, with sources:

1. **The observable.** What does a nonzero k_{3,0} tidal Love number
   actually produce in the measured gravity field — a time-varying degree-3
   potential coefficient at the tidal frequency? Write the relation between
   \|k_{nm}\| and the amplitude of the periodic ΔC_nm / ΔS_nm variation,
   using the same 4π-normalized convention as `pylov3d.sh_data` (state the
   convention explicitly; a normalization slip here changes the answer by
   order-unity factors).
2. **What has been achieved.** What are the published uncertainties on
   time-varying gravity coefficients for Mars (MRO/MGS/Odyssey tracking —
   Konopliv et al. 2016/2020 report seasonal ΔC20/ΔC30 from CO2 mass
   exchange, which is the closest existing analogue: a real, recovered,
   time-varying low-degree signal). Get the actual quoted uncertainties.
   For a dedicated-mission comparison, GRAIL's lunar degree-2/3 tidal
   recovery is the natural benchmark.
3. **The gap.** Ratio of required to achieved precision for the top modes,
   stated the way TASK-021 stated its 95× figure.
4. **Frequency separation.** The tidal signal is periodic at the solar
   semidiurnal period (44,387.62 s — see `pylov3d.anelastic`, verified in
   TASK-025a). Seasonal CO2 mass exchange is annual. Being at a very
   different frequency from the dominant time-varying signal helps
   separability — say whether and how much, or state plainly if you cannot
   quantify it.

## Constraints and honesty requirements

- **Citations:** the standing project rule. No verbatim quotation unless you
  retrieved the source in this session; otherwise paraphrase and say so. A
  prior task in this family shipped a fabricated quote — that is why this
  is stated in every spec now.
- **Do not overstate.** If the required precision is orders of magnitude
  beyond anything achieved or planned, say so in those words. The proposal
  is better served by a defensible measurement requirement than by a
  strained detectability claim.
- Do not modify `pylov3d/mars_lateral.py` or any solver module. New analysis
  goes in a new module (suggest `pylov3d/mars_detectability.py`) plus tests.
- Prose standard: never the words "genuine" or "honest"; state facts
  directly.
- Fast lane must stay green.

## Deliverables

1. `pylov3d/mars_detectability.py` — the amplitude→observable conversion and
   the required-precision calculation, parameterized so it can be re-run for
   any mode list (the Moon will need the same machinery).
2. Tests pinning the conversion against a hand-checkable case.
3. A `docs/MARS_MODEL.md` section "Off-(2,0) detectability (TASK-026)" with
   the required-precision table, the comparison to achieved precision, all
   sources, and the caveats.
4. A figure (fig8) in the proposal set: spectrum amplitudes against
   the achieved/required precision lines, in the style of fig7.

## Open scoping question — resolve with the PI before finalizing

Which measurement scenarios to benchmark against. Default, if no direction
is given: (a) current Mars-orbiter tracking as the "achieved today" line,
and (b) a GRAIL-class dedicated gravity mission as the "plausible future"
line. The PI is writing the proposal and may want a specific mission
concept used instead — flag this in the report rather than silently
committing to the default.
