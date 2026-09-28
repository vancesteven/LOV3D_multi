# LOV3d-genai

worktree for AI-led work on the Python port line, branched from `agent/task-046-multibasis-energy`.

- Agent name for this tree: `claude-lov3d-genai`
- Branch `lov3d-genai`, pushes to `myfork`
- Python env: not recorded here — confirm with Steve before running anything
- Pushes go to `myfork`, **not** `origin` (`origin` is upstream `mroviranavarro/LOV3D_multi` and rejects with 403).


## Agent lanes

Claude Code is the **manager lane**: planning, scientific review and
adjudication, integration, and pushes. Delegate implementation and
reconnaissance passes to subagents; keep adjudication for the manager.

Codex is the **delegate lane**, restricted to tasks explicitly queued in
`plans/CODEX-QUEUE.md`. Its instructions are `AGENTS.md`. It commits locally and
never pushes. **Scientific adjudication is never delegated to Codex.**

Codex is mechanically review-only in the sandbox (`codex exec --sandbox
read-only`). Request a review with `codex-review <repo-dir> [git-range]`; the
wrapper writes `coordination/reviews/<timestamp>-<sha>.md` and appends its own
audit entry.

## State files

Forward-looking — what to do next, and who owns it:

- `plans/STATUS.md` — current focus, work in flight, blockers
- `plans/CODEX-QUEUE.md` — Codex's queue and the claim/report protocol

Backward-looking — what happened, and cross-repo messages:

- `coordination/audit.md` — append-only, one entry per completed action
- `coordination/inbox/<agent>.md` — messages addressed to an agent in this repo
- `coordination/open-questions.md` — anything needing Steve's sign-off
- `~/src/coordination/inbox/<agent>.md` — **cross-repo** messages (e.g. thrak ↔ lov3d)

**Queue freshness:** any session that pushes commits, integrates artifacts, or
changes a queue must refresh the `Updated:` lines and affected sections of
`plans/STATUS.md` and the relevant queue file **in the same session**.

Sibling repos are all visible under `~/src`. Never edit another repo's files
directly — hand work across via its inbox.

## Verification discipline

A change is NOT "done" until its specified behavior has been observed in the
running system. Compiling, importing, or running a smoke pass that does not
exercise the specific change is not verification.

For this repo: a targeted `pytest` test under `tests/` must exercise the changed path. For port work, numerical parity against the reference implementation is the bar — not merely that the Python runs.

If you cannot verify (no env, no display, no time), say so explicitly and label
the change `implemented, unverified` — never `done` or `fixed`.

## Status vocabulary

Status entries must use exactly this vocabulary:

- `verified` — observed working under documented reproduction steps; **cite the
  artifact** (test name, PDF path, screenshot).
- `implemented, unverified` — code written and syntax-checked, but the targeted
  behavior has not been observed.
- `not implemented` — planned, no code written.

Do NOT use `done`, `fixed`, `complete`, `syntax clean`, or `review passed`.
Those describe intermediate states, not whether the problem is gone.

**Layering a new fix on top of an `implemented, unverified` change is forbidden
without verifying the prior change first.** A chain of unverified fixes is a
chain of unknowns.

## Escalation

Stop and write to `coordination/open-questions.md` — do not proceed — for
anything ambiguous, destructive (deletions, force-pushes, dependency upgrades),
or scientifically consequential. **Never change a scientific assumption to make
a test or gate pass.**

Commit coordination and plan files with the work they describe, so the record
and the code state cannot drift apart.
