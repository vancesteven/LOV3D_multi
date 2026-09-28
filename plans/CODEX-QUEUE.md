# Codex queue — LOV3d-genai

Updated: (never — seed file)

Codex works **only** tasks listed here. Claude Code (manager lane) writes tasks;
Codex claims, executes, and reports in place.

## Claim / report protocol

1. **Claim** — set Status to `claimed` and add your start timestamp before
   beginning. One task at a time.
2. **Execute** — commit locally on `lov3d-genai`. Never push.
3. **Report** — set Status to `verified` (cite the artifact),
   `implemented, unverified`, or `not implemented`, and record the commit
   hashes. Append an entry to `coordination/audit.md`.
4. **Escalate** — if the task touches scientific assumptions, fixtures,
   thresholds, or exceeds its written scope, stop and write to
   `coordination/open-questions.md`.

## Queue

| # | Task | Scope / files | Status | Commits |
|---|---|---|---|---|
| — | *(empty)* | — | — | — |
