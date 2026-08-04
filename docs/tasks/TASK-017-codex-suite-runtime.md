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
