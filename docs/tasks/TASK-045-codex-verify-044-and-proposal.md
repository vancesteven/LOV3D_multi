# TASK-045 (Codex): verify TASK-044, then write the mantle arc into the proposal

## Part A — verify TASK-044 (different driver: a Sonnet agent under A implemented it)

Read `docs/tasks/TASK-044-agnostic-l2-residual.md` (spec + Result),
`scripts/mars_mantle_agnostic_l2.py`, and the artifact
`docs/figures/proposal/mars_mantle_agnostic_l2.npz` (SHA-256
`4b3afc8d71ec44fa230528eed3af347249bcc0567e85dc83774d805472eb2607`).
A's spot-checks already reproduce the SVD, top direction, and decision
metrics from the artifact — do not stop at repeating those. The two
attacks that matter:

1. **The verdict is convention-sensitive and the artifact cannot settle
   it.** The decision metric is 0.985534σ against a 1σ gate — short by
   1.4%. The amplitude bound used is the conservative coefficient guard
   (0.439999); the Result claims the guard binds "ahead of the
   oversampled-grid bound," but the artifact stores the clamped value in
   all three positivity slots, so the **raw grid bound is not
   recoverable from the artifact**. Recompute the raw oversampled-grid
   positivity bound for the top direction (both signs, ≥361×720 grid,
   state ε) and report the decision metric under BOTH conventions. If
   the grid-bound metric crosses 1σ, the correct statement is not "the
   gate passes" but "the verdict depends on the bound convention" — and
   the Result section must be amended to say so, with both numbers.
   Whichever way it lands, the stored-bounds defect (three slots, one
   value) should be fixed in the driver and the artifact regenerated so
   the raw grid bound is archived.
2. **Step-policy mixing.** TASK-044 reuses TASK-043's crust Jacobian
   (half-step error 1.30e-4) alongside fresh mantle Jacobians
   (half-step errors ~6e-6, a different step in δµ/µ terms). The Result
   argues this is consistent because both converge inside 1%. Check the
   argument, not just the numbers: recompute the crust Jacobian at the
   mantle step policy (two solves) and confirm the decision metric moves
   by less than the near-miss margin. If it moves the metric across 1σ,
   that is a finding.

Also standard: rerun the driver's cheap paths, confirm the appended
Result matches the artifact, suite green, and check the near-pure-C20
top direction against the coupling structure (why do the other four
patterns carry ~200× less orthogonal signal? — a one-paragraph
explanation grounded in the selection rules, or a statement that it
does not follow trivially).

## Part B — consolidated proposal paragraph (after Part A settles the verdict)

`~/SSS_2025_Mars/Methods_Models.tex`. Add one subsection after §4.11
covering the mantle arc (TASK-042 design → 043 thermal pilot → 044
agnostic test): a second laterally varying layer was designed, gated,
and tested; the thermal template reached 0.05σ, the best unconstrained
L=2 pattern [state the settled verdict from Part A, with both bound
conventions if they disagree]; conclusion conditional on the scenario
covariance; the stop rule fired rather than escalating to
higher-dimensional inversions. Frame it as strength: the pipeline
distinguishes what the data can and cannot constrain, and the proposal's
crustal focus is the product of that test, not an assumption. Add a
corrections-table row only if Part A amends a committed number.
Compile both docs (`pdflatex`, exit 0 required) before pushing.

## Constraints

- Fix allowlist for Part A: `scripts/mars_mantle_agnostic_l2.py`, the
  regenerated artifact, and the TASK-044 Result section. No solver-module
  changes.
- Commit and push as you go (standing authorization), including the
  proposal repo.
- Ledger entries per the standing format; TASK-044 to VERIFIED (or to
  CORRECTED+VERIFIED with the amended verdict) and TASK-045 to DONE.
- Prose standard: never "genuine" or "honest". Suites green in both
  repos' senses (pytest; pdflatex exit 0).
