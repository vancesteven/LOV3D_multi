# TASK-033 (Codex): make the README guard robust to in-flight work

## Context — the guard is good, and it has already found something

TASK-032's guard (`pylov3d/tests/test_repo_consistency.py`) is well built:
non-circular, subprocess-based so it survives partial invocation, and it
documents its tolerance trade-off. A's verification confirmed it catches a
+11-test drift, passes a +9 one, and hands back the exact replacement
sentence on failure. None of that needs changing.

But it is currently **failing the whole suite**, and for a reason worth
fixing rather than papering over: the *file* count is exact, while the test
counts are tolerant. Another machine's in-progress work (a new
`test_moon_lateral.py`, uncommitted) pushes the file count from 44 to 45, so
every developer on the repo gets a red suite until someone updates a prose
sentence in the README.

That is the "brittle in a way that annoys" outcome the TASK-032 spec warned
against — a guard that gets disabled within a week is worse than no guard.
The reasoning behind the exact file count was defensible (new files are
infrequent; the README makes an exact inventory claim), but it interacts
badly with multi-machine development, where uncommitted work on one machine
should not red-light another's suite.

## What to change

Make the guard resilient to in-flight work while keeping it able to catch
the drift class it exists for (roughly 150 tests / several files).

You choose the mechanism. Reasonable options, not exhaustive:

- Give the file count the same tolerance treatment as the test counts.
- Count only files **tracked by git** (`git ls-files pylov3d/tests/test_*.py`),
  so uncommitted work in progress is invisible to the guard and the claim
  becomes "what the repository contains" rather than "what this working tree
  happens to contain".
- Keep the check exact but make it a warning rather than a failure when the
  discrepancy is small and the extra files are untracked.

**State in the docstring which you chose and what it trades away.** The
existing docstring does this well for the ±10 tolerance; match that standard.
If you pick the git-tracked option, note that it makes the guard silent about
a genuinely stale README in a dirty tree — that is an acceptable trade, but
it should be written down, not discovered later.

## Also

Re-run the count refresh: the suite has grown again since TASK-032 landed
(B's TASK-030 and the in-flight Moon lateral work). Set the README to the
values your chosen mechanism should compare against, measured, not assumed.

## Standing guardrails (MUST follow)

1. **Writable allowlist**: `README.md`, `pylov3d/tests/test_repo_consistency.py`.
   Everything else read-only.
2. **No state-changing git** — no commit, push, branch, stash. Note that if
   you use `git ls-files` for counting, that is a *read-only* query and is
   fine; the prohibition is on changing state.
3. No environment changes, no network.
4. Never touch `src/`, `docs/HANDOFF.md`, `data/`, `LICENSE`, `NOTICE`,
   `pyproject.toml`, or any other module under `pylov3d/`.
5. **Do not modify, commit, or delete the untracked Moon lateral files**
   (`pylov3d/moon_lateral.py`, `pylov3d/tests/test_moon_lateral.py`,
   `scripts/moon_lateral_spectrum.py`, and the two
   `docs/figures/proposal/moon_lateral_spectrum.*` artifacts). They are
   another task's work in progress and are not yours to touch — they are
   precisely the condition you are making the guard tolerate.
6. Stop on ambiguity and report rather than guessing.

## Verify

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/test_repo_consistency.py -q
```

Both green, **with the untracked Moon lateral files still present** — that is
the actual acceptance condition, not a clean tree.

Then demonstrate the guard still guards: introduce a large drift in the
README (say +200 tests, or a wrong file count well outside whatever
tolerance you chose), watch it fail, restore, and report the failure message.

## Done criteria

Suite green with the in-flight files present; guard still catches large
drift; the chosen mechanism and its trade-off written into the docstring;
README counts measured and current.
