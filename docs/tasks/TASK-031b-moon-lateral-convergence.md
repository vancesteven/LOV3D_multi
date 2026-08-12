# TASK-031b (Machine B): Moon lateral truncation convergence

## Handoff state — read this first

The Moon lateral **pipeline** is built, tested and committed (`67dd097`,
Codex). What it does not have is a converged harmonic cutoff, and that is
this task.

Everything you need is now on the remote. Note that until `67dd097` these
files were *untracked on machine A* and therefore invisible to your clone —
if you pulled earlier and saw nothing, that is why.

- `pylov3d/moon_lateral.py` — the pipeline (LOLA shape − low-degree GRAIL
  equipotential → Airy → crustal thickness → lateral rigidity on the Weber
  crust layer).
- `pylov3d/tests/test_moon_lateral.py` — 9 tests, both convention decisions
  pinned.
- `scripts/moon_lateral_spectrum.py` — the production driver.
- `docs/figures/proposal/moon_lateral_spectrum.{npz,png}` — the lmax=4 result.
- `docs/MOON_MODEL.md`, "Moon lateral crust stage (TASK-031)" — read the whole
  section; its last paragraph states exactly the gap you are closing.

## The gap

Radial convergence is strong: Nrbase 30 → 15 moves `Delta k20` by 4.3e-5
relative. **Angular convergence is not established.** The cheap `lmax=2`
comparison gives `Delta k20 = 2.373e-7` against `lmax=4`'s `1.407e-6` — a
factor of **5.93**. So the archived spectrum is the first fixed-cutoff Moon
prediction, not a converged endpoint.

## What to run

The Moon analogue of TASK-027 part 1, which is the pattern to follow:

1. **The ladder.** `lmax = 4, 5, 6` (and 7 if it fits in memory) at fixed
   Nrbase, tracking the `(2,0)` forcing-mode `Delta k20` *and* the top
   off-forcing modes the TASK-031 table already names — `(2,±2)` at
   3.13471e-6, `(2,±1)` at 2.76868e-6, `(3,±3)` at 2.01884e-6. Report
   step-to-step relative change so a reader can see whether it converges and
   how fast, and give the mode count N at each rung.

2. **Verify, do not assume, the Nrbase independence.** TASK-027 established
   for Mars that the angular ladder can run at modest radial resolution
   because truncation is an angular question — but it *checked* that by
   holding lmax fixed and varying Nrbase. Do the same here rather than
   importing the Mars conclusion. The Moon model has a fluid outer core the
   Mars model does not, so the radial behaviour is not automatically the same.

3. **Watch memory.** TASK-021b hit >15 GB at lmax=6/Nrbase=50 for Mars. The
   Weber Moon has ten layers against Mars's four, so budget accordingly and
   say what you observed.

4. **If it does not converge by lmax=6–7, that is the finding** and it
   qualifies the TASK-031 result. Report it plainly; do not extrapolate to a
   converged value.

## Second, cheaper question, if the ladder leaves budget

TASK-028 found for Mars that the (2,0) and (4,0) rheology harmonics are
*both* first-order channels into the (2,0) forcing mode and cancel to ~91%.
The Moon pipeline removes C20 by default (retaining it breaks the
linearization at |δμ/μ̄| ≈ 1.08), so the Moon field has **no (2,0) channel at
all** — only (4,0). Worth confirming directly that the Moon's (4,0) channel
behaves as first order here too, by the same exponent-fitting method
(`scripts/mars_first_order_channels.m` is the Mars precedent, and the Python
equivalent is straightforward). It is a cheap check and it tells us whether
the Mars cancellation is a Mars peculiarity or a general feature.

## Constraints

- Driver scripts only; do not modify `pylov3d/moon_lateral.py` or any solver
  module. If you find a bug in the pipeline, report it rather than patching
  around it.
- Archive as `.npz` + figure under `docs/figures/proposal/`, matching the
  TASK-027 artifact convention, and append a results subsection to
  `docs/MOON_MODEL.md`.
- Citation rule: no verbatim quotation unless retrieved in that session.
- Prose standard: never "genuine" or "honest".
- Suite must stay green.

## Done criteria

Convergence sequence with step-to-step changes for the forcing mode and the
three named off-modes; the Nrbase-independence check actually performed;
memory observations recorded; an explicit statement of whether the TASK-031
`Delta k20 = 1.407e-6` survives as a converged number or must be restated;
and the ledger updated to DONE awaiting VERIFIED.
