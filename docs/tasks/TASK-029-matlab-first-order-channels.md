# TASK-029 (Machine B): native-MATLAB anchor for the two first-order zonal channels

## Why this is the top B task

TASK-028 established a new physical result that is now cited in the NASA
proposal: **two** rheology harmonics couple the degree-2 zonal tide back to
itself at first order — (2,0) and (4,0), not (4,0) alone as this repository
and the proposal previously stated — and they enter with **opposite signs**,
cancelling to ~91%.

That result has **no independent anchor**. Everything else in the Mars chain
does: the 1D model, the lateral spectrum, and the m=0/1/2 diagonal solves are
all MATLAB-verified (7e-12 or better). This one rests entirely on the Python
port plus a coupling-coefficient inspection, and it changed a headline claim.
If it is wrong, the proposal is wrong.

B is the only machine that can close this, because it has MATLAB.

## What to verify

Using the native MATLAB LOV3D solver on the committed 4-layer Mars model
(`scripts/mars_lateral_cross_check.m` is the working driver to extend —
it already loads the model and the committed `mu_variable` field):

1. **The scaling exponents.** Isolate single zonal rheology harmonics and fit
   the forcing-mode k2 shift's scaling in the perturbation amplitude, at
   forcing (2,0). Python gives:
   - (2,0) → **1.003** (first order — the new claim)
   - (4,0) → 1.001 (control; the repo documents 1.000)
   - (3,0) → 2.003 (control; the repo documents 2.002)
   Reproducing the two controls is what makes the (2,0) result credible, so
   run all three even though only (2,0) is new.

2. **The signed contributions and the cancellation.** On the DWAK
   InSight-derived field, extrapolated to full physical amplitude, Python
   gives (2,0) alone = **−1.5901e-5**, (4,0) alone = **+1.4528e-5**, both
   together = **−1.3738e-6** — additive to 0.1%, as first order requires,
   and a 91.4% cancellation. The *signs* matter more than the magnitudes
   here: if MATLAB disagrees on sign, the proposal's framing is wrong.

3. **The coupling coefficients themselves**, if MATLAB exposes them
   conveniently. Python's `coupling_coefficients` gives max|C| = 0.6389 for
   (2,0) and 0.8571 for (4,0), identically zero for (1,0), (3,0), (5,0) and
   (6,0). Confirming the two nonzero values and the four zeros would anchor
   the selection rule directly rather than through the response.

## Inputs, all committed

- `pylov3d/mars_crust_models.py` — the Python side, with the full argument in
  its module docstring.
- `data/mars/insight_moho/` — the five Moho models (`SOURCES.md` explains the
  format; DWAK is the one used above).
- `docs/MARS_MODEL.md`, "Non-Airy crustal model substitution (TASK-028)" —
  the numbers and their derivation.
- `docs/figures/proposal/mars_crust_models.npz` — the comparison artifact.

## Constraints

- MATLAB-side work plus a driver script; do not modify any `pylov3d` module.
- Save console output and a small `.mat` to `data/tests/mars/`, matching the
  TASK-020/TASK-027 artifact practice, so a MATLAB-less reader can verify.
- Record the MATLAB version.
- Citation rule: no verbatim quotation unless retrieved in that session.
- Prose standard: never "genuine" or "honest".

## Done criteria

A `docs/MARS_MODEL.md` note reporting the MATLAB exponents and signed
contributions against the Python values, committed artifacts, and a plain
statement of whether the cancellation reproduces. **If it does not reproduce,
say so prominently — that is the more important outcome, and the proposal
text would need to be pulled back the same day.**
