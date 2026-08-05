# TASK-017 — Codex handoff: test-suite runtime lanes

**Status:** IN-PROGRESS (Codex 5.6)
**Owner:** CODEX
**Parent:** suite grew to ~16 min wall (Mars lateral coupled solves). Split
into a fast default lane and a documented full lane, changing NO test
logic.

This spec is self-contained. You (Codex) have no other context.

## GUARDRAILS — hard limits

1. **Files you may write** (nothing else):
   - `pylov3d/pyproject.toml` (pytest config only)
   - Files under `pylov3d/tests/` — but ONLY these edit kinds: adding
     `@pytest.mark.slow` decorators, adding `import pytest` if missing,
     and adding a short comment on why a test is slow. The diff must show
     ONLY additions; no test logic, assertion, fixture, or parameter may
     change.
   - This file (append to Completion notes only)
2. No state-changing git. No environment changes; no network. Runner:
   `venvLOV3Dconv/bin/python` from repo root; if broken, STOP and report.
3. `src/`, `docs/HANDOFF.md`, `data/` read-only. No deletions; 500-line cap.
4. On ambiguity: stop, record in Completion notes, finish unambiguous parts.

## Goal

1. Measure: `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q --durations=40`
   (record the top of the list in your Completion notes).
2. Mark every test (or parametrized case via module-level pytestmark where a
   whole module qualifies) with wall > 20 s as `@pytest.mark.slow`.
   Constraint: every test MODULE must keep at least one unmarked test, so
   the fast lane still touches every subsystem. If marking a >20 s test
   would empty a module, leave the single fastest such test unmarked and
   note it.
3. `pylov3d/pyproject.toml`: add `addopts = "-m 'not slow'"` to
   `[tool.pytest.ini_options]` (the `slow` marker is already registered).
4. Document IN pyproject comments: full lane =
   `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q -m ""` (empty
   marker expression overrides addopts) — verify that exact command runs
   the full 465 and write the verified command, not an assumed one.

## Done criteria

- Fast lane: `... -m 'not slow'` (i.e., the bare default command)
  **< 5 min wall** and 0 failures.
- Full lane: exactly **465 passed** (the current total), 0 failures.
- `git diff` on test files shows only added lines.

Append Completion notes: durations top-10, which tests were marked, fast-lane
count + wall, full-lane count + wall, the verified full-lane command.

## Completion notes

_(Codex appends here.)_

### 2026-08-05 — completed on the live Machine A tree

Untouched-suite measurement command:
`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q --durations=40`

Result: **491 passed, 2 warnings in 990.12 s**. Top 10 durations:

1. 296.57 s call — `tests/test_mars_lateral.py::TestTruncationSensitivity::test_lmax5_k2_shift_within_20_percent`
2. 239.26 s call — `tests/test_mars_lateral.py::TestLinearity::test_order1_mode_scales_linearly`
3. 152.92 s call — `tests/test_mars_lateral.py::TestJaxEquivalence::test_jax_matches_numpy_coupled`
4. 91.74 s call — `tests/test_mars_lateral.py::TestForcingModePerturbation::test_k2_shift_much_smaller_than_observational_uncertainty`
5. 35.99 s call — `tests/test_moon.py::TestPocoMCSmoke::test_micro_run_completes_and_recovers_k2`
6. 20.85 s call — `tests/test_forward.py::TestPocoMCSmoke::test_micro_run_completes_and_recovers_k2`
7. 15.27 s call — `tests/test_jax_propagator.py::TestJaxScan::test_scan_matches_python_loop_jax`
8. 14.60 s call — `tests/test_jax_propagator.py::TestJaxPropagator::test_k2_analytic`
9. 14.53 s call — `tests/test_jax_propagator.py::TestJaxPropagator::test_k2_matches_numpy`
10. 10.25 s setup — `tests/test_mapping.py::TestPlotMapSmoke::test_returns_figure_no_exception`

Newly marked `slow` after measurement:

- `tests/test_mars_lateral.py::TestLinearity::test_order1_mode_scales_linearly`
- `tests/test_mars_lateral.py::TestJaxEquivalence::test_jax_matches_numpy_coupled`
- `tests/test_mars_lateral.py::TestForcingModePerturbation::test_k2_shift_much_smaller_than_observational_uncertainty`

The other measured tests over 20 s were already marked `slow`: the Mars
truncation-sensitivity test and the Moon and forward PocoMC smoke tests.
The pre-existing two-test `TestFitReproducibility` slow class was retained.
Every affected test module still has unmarked tests; no exception to the
module-touch constraint was needed. The test-file diff contains only the
three decorator additions above and no test logic changes.

Fast default lane:
`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`

- **483 passed, 8 deselected, 2 warnings in 213.92 s** (215.33 s real wall),
  0 failures.

Verified full lane:
`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q -m ""`

- **491 passed, 2 warnings in 1204.33 s** (1206.11 s real wall), 0 failures.

Ambiguity/deviation: the spec says the current full total is exactly 465,
but the live tree already contained concurrent untracked Moon work at task
start, including `pylov3d/tests/test_moon.py`. The untouched measurement and
verified full lane both collected **491**, 26 more than the stated baseline.
No tests were deleted, hidden from the full lane, or altered to manufacture
the stale 465 count; the exact required full-lane command is green on all
tests present in the live tree.
