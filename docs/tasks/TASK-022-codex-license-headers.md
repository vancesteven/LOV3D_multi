# TASK-022 (Codex): Per-file license headers + README count refresh

## Goal

Add a short Apache-2.0 license header to every Python source file in the
`pylov3d` package and the figure/analysis scripts, identifying the project as
a derivative work of LOV3D. Refresh the stale test counts in README.md.
Purely mechanical; zero logic changes.

## Standing guardrails (MUST follow — same as every Codex task)

1. **Writable allowlist** (everything else read-only):
   - `pylov3d/**/*.py`
   - `scripts/proposal_figures/*.py`
   - `scripts/*.py` (Python only — do NOT touch `*.m` MATLAB files)
   - `README.md`
2. **No state-changing git** (no commit, no push, no branch, no stash). Leave
   changes uncommitted.
3. No environment changes (no pip install, no venv edits).
4. Never touch `src/` (upstream MATLAB), `docs/HANDOFF.md`, `data/`,
   `LICENSE`, `NOTICE`, `pyproject.toml`.
5. Each file stays under 500 lines of *added* content (headers are ~8 lines;
   trivially satisfied).
6. Stop on ambiguity: if any file's existing top-of-file content makes header
   placement unclear (unusual encoding lines, embedded license text), skip it
   and list it in your report instead of guessing.
7. The fast test suite must stay green (see Verify).

## Header text

Insert at the very top of each `.py` file — AFTER a shebang line (`#!...`)
and AFTER any `# -*- coding -*-` line if present, BEFORE the module
docstring:

```python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.
```

Rules:
- Idempotent: if a file already contains `SPDX-License-Identifier`, skip it.
- Exactly one blank line between the header block and whatever follows.
- Do not reflow, reformat, or otherwise modify any other line of any file.
- Include `__init__.py` files and test files.

## README refresh

In `README.md`, update the test-suite counts to: fast lane 502 tests
(~2 min), full lane 516 tests, 38 test files. Verify the 38 by counting
`pylov3d/tests/test_*.py`. If the counted number differs, use the counted
number. Do not change any other README content.

## Verify (run before finishing)

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
```

Must report 502 passed (or more if other work landed; zero failures). Also
run a syntax sanity pass:

```
venvLOV3Dconv/bin/python -m compileall -q pylov3d scripts
```

## Done criteria

- Every in-scope `.py` file has exactly one SPDX header, correctly placed.
- Suite green; compileall clean.
- Report: number of files modified, any skipped files with reasons.
