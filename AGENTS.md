# Codex project instructions — LOV3d-genai

Read `CLAUDE.md` completely before planning or changing this repository. Its
scientific constraints, verification discipline, and status vocabulary are
binding on you too.

## Your lane

Claude Code is the manager lane (planning, scientific review and adjudication,
integration, pushes). You are the delegate lane: implementation and
reconnaissance tasks **explicitly queued** in `plans/CODEX-QUEUE.md`. Scientific
adjudication is never delegated to you.

## Workflow

- Read `plans/STATUS.md` first for current state, then `plans/CODEX-QUEUE.md`
  for your tasks and the claim/report protocol. Work only queued tasks unless
  Steve directs otherwise.
- Commit locally on `lov3d-genai` with clear messages. **Do not push** — the
  manager reviews and pushes. Record commit hashes in your queue report.
- Report status strictly in the `CLAUDE.md` vocabulary: `verified` (cite the
  artifact), `implemented, unverified`, or `not implemented`. Never `done`,
  `fixed`, or `complete`.
- Escalate rather than proceed when a task touches scientific assumptions, test
  fixtures, thresholds, or anything beyond its written scope. Never change a
  scientific assumption to make a test pass.
- Append one entry per completed action to `coordination/audit.md`.
- **Preserve unrelated working-tree changes.** This tree may carry uncommitted
  work that is not yours.
