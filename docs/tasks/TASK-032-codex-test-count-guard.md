# TASK-032 (Codex): refresh the test-count claims and stop them drifting again

## Goal

Two parts. First, correct the stale test/file counts in `README.md`. Second —
and this is the part that matters — add a test that makes the claim
self-checking, so it cannot silently go stale a fourth time.

## Why

The counts have now drifted three separate times as the suite grew
(483 → 488 → 496 → 502/516 → 546 → current). Every correction so far has been
manual and reactive, noticed only when someone happened to look. `README.md`
currently claims **546 tests across 39 test files, 521 in the fast lane**. The
actual numbers at the time of writing are **653 collected, 624 in the fast
lane, 43 test files** — roughly 150 tests adrift.

A hand-maintained count in prose is a fact that rots. A test that asserts it
does not.

## Standing guardrails (MUST follow — every Codex task)

1. **Writable allowlist** (everything else read-only):
   - `README.md`
   - `pylov3d/tests/test_repo_consistency.py` (new file; name it as you see
     fit if that clashes)
2. **No state-changing git** (no commit, push, branch, stash). Leave changes
   uncommitted.
3. No environment changes, no network. Neither is needed.
4. Never touch `src/`, `docs/HANDOFF.md`, `data/`, `LICENSE`, `NOTICE`,
   `pyproject.toml`, or any module under `pylov3d/` other than the new test
   file.
5. Stop on ambiguity and report rather than guessing.
6. The fast suite must stay green.

## Part A — correct the README

Determine the real numbers yourself rather than trusting the ones above; they
will have moved if other work has landed. Use:

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q --collect-only   # fast lane
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q --collect-only -m ""   # full
ls pylov3d/tests/test_*.py | wc -l
```

Update the sentence at `README.md:130` to the measured values, preserving its
existing structure and the list of what the suite covers. Change nothing else
in the README.

## Part B — the guard (the actual deliverable)

Add a test that fails when the README's claim stops matching reality.

Requirements, in order of importance:

- **It must not be circular.** A test that recomputes the count and compares
  it to itself proves nothing. Parse the numbers out of `README.md` and
  compare them against counts obtained from the live test session.
- **It must not be brittle in a way that annoys.** A test that fails on every
  single added test would be worse than no test — it would be disabled within
  a week. Prefer one of:
  - assert the README figures are within a small tolerance (say a few
    percent, or ±10 tests) of actual, so ordinary growth is fine but a
    150-test drift fails; **or**
  - assert exact equality but make the failure message state the exact
    replacement sentence, so fixing it is a copy-paste.

  Pick one, implement it, and **say in the test's docstring why you chose
  that trade-off**. Either is defensible; an unexamined choice is not.
- **Getting the counts from inside a running pytest session is the subtle
  part.** `request.session.items` gives collected items for the current run,
  but that reflects the *current* invocation's selection — running
  `pytest path/to/test_repo_consistency.py` alone would see one item, not the
  whole suite, and a naive assertion would then fail spuriously. Handle this.
  Options include skipping unless the whole suite is being collected,
  or invoking collection in-process. **If you cannot make it robust against
  partial invocation, say so and implement the file-count check only** — a
  reliable partial guard beats a flaky full one.
- Count test *files* by globbing `pylov3d/tests/test_*.py`.
- Mark it `slow` only if it genuinely is; it should be fast.

## Verify

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/test_repo_consistency.py -q
```

Both must pass — the second is specifically the partial-invocation case that
must not produce a spurious failure.

Then confirm the guard actually guards: temporarily edit the README number to
something wrong, watch the test fail, and restore it. Report that you did
this and what the failure message looked like.

## Done criteria

README counts correct; a non-circular guard that passes both as part of the
full suite and when invoked alone; the deliberate brittleness trade-off
explained in the docstring; and a report of the demonstrated failure.
