# pylov3d — Machine Coordination Channel

> Single shared coordination file. Every session on any machine: **pull, read
> this file, work, update this file, commit `coord:`, push.** Keep this file
> under ~150 lines — task detail lives in `docs/tasks/`, not here.

---

## Sync protocol (git is the channel, not Dropbox)

Machines have **separate clones**; the fork on GitHub is the single source of
truth. Refer to the push target by URL — remote *names* differ per machine:

- **Push target (always):** `https://github.com/vancesteven/LOV3D_multi`
  (Machine A: remotes `origin` *and* `myfork`; Machine B: remote `myfork`.)
- **Never push to** `https://github.com/mroviranavarro/LOV3D_multi` (upstream).

**Locking = git non-fast-forward rejection.** To claim a task:
1. `git pull` → confirm task is `free` in the ledger below.
2. Edit ledger (set Owner), commit `coord: claim TASK-NNN`, **push immediately**.
3. Push rejected → someone else moved first: pull, re-read ledger, retry or stand down.

Only then start work. Same dance to release (`coord: release TASK-NNN`, with a
log line). Coordination commits (`coord:` prefix, touching only `docs/HANDOFF.md`
and `docs/tasks/`) are pre-authorized — no per-commit user approval needed.
**Code commits still require user approval** (standing rule unchanged).

---

## Task ledger

One row per task. Owner ∈ {A, B, CODEX, free}. Never work a task you don't own.

| ID | Task | Owner | Status | Spec |
|---|---|---|---|---|
| TASK-001 | 3D coupled JAX port — increments 3–4 (scan + API) | A | READY (001a verified) | `docs/tasks/TASK-001-jax-coupled-3d.md` |
| TASK-001a | 3D coupled JAX port — increments 1–2 (static precompute + traced build) | A (review done) | VERIFIED — uncommitted, 2 test nits open | `docs/tasks/TASK-001a-codex-increments-1-2.md` |

Statuses: `QUEUED → IN-PROGRESS → DONE → VERIFIED` (verification by an Opus-tier
reviewer, done by a *different* driver than the implementer when practical).

---

## Model routing (economical defaults)

| Engine | Where | Use for | Cost posture |
|---|---|---|---|
| **Fable 5** | A, main loop | Orchestration, protocol, hard design calls, final math review | Most expensive — keep turns short, delegate early |
| **Opus 5** | A, subagent | Review/verification, delicate impl (e.g. A1/A2 coupled unroll) | Escalation tier |
| **Sonnet 5** | A, subagent | Default implementation + tests | Workhorse |
| **Haiku 4.5** | A, subagent | Recon, search, file inventory | Cheapest |
| **Codex 5.6** | A, separate tool | Self-contained, well-specified tasks run in parallel (test suites, mechanical ports) | Parallel track |
| **Opus 4.8** | B, main loop | Tasks B claims in the ledger; same delegation rules as before | — |

**Codex handoffs:** Codex shares no conversation context. Every handoff is a
spec file in `docs/tasks/` that stands alone: goal, exact files, constraints,
the verify command (`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`),
and done-criteria. Ledger Owner = CODEX while it runs. Codex output is always
reviewed by an Opus-tier agent before Status moves DONE → VERIFIED.

---

## Environment notes (non-obvious)

- **Python venv:** `venvLOV3Dconv/bin/python` on BOTH machines (venvs are
  per-clone, not synced). Machine B: JAX 0.10.0. Machine A: rebuilt 2026-08-02 —
  JAX 0.10.2, py3nj compiled from source (needed `brew install gcc` +
  `CC=gcc-16 CXX=g++-16 FC=gfortran-16` for OpenMP), plus pyalma3 (module name
  is `alma`) and matplotlib. Full suite on A: 322 = 318 baseline + 4 new.
  Tests: `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`
- JAX x64 enabled at import; complex128 OK on CPU. First JIT call slow (~10–30s).
- `matlab/` is source material only — do not scan/grep/modify.

---

## Standing rules (both machines)

- Do NOT implement code unless explicitly instructed — summarize & propose first.
- Code commits only when the user asks. `coord:` commits pre-authorized (above).
- Keep files < 500 lines; tests in `pylov3d/tests/`; no working files in repo root.
- graphify re-runs only with user confirmation: `graphify run pylov3d/ --exclude pylov3d/tests/`

---

## Project status

- ✅ M1 1D solver · ✅ M2 3D lateral/mode coupling · ✅ M3 MATLAB cross-validation
- 🔄 **M4:** PyALMA3 benchmark ✅ · JAX 1D loop ✅ · JAX 1D lax.scan ✅ · 3D coupled port ⬜ (TASK-001)
- Suite: 318 passed as of `3c41217`.

---

## Coordination log

_Newest on top. Format: `[machine][YYYY-MM-DD] note`. Cap at ~15 entries —
prune from the bottom when adding (git history keeps the rest)._

- `[A][2026-08-02]` TASK-001a assessed and VERIFIED. Codex output correct: Opus 5 adversarial review (line-by-line + randomized differential harness to 2.2e-15 + 27-mutant mutation test, 25 killed / 2 equivalent-or-gap) → APPROVE, zero defects. Suite in rebuilt machine-A venv: 322 passed, 0 failed. Open nits before commit: test_jit_smoke compares JAX to itself (swap tuple unpack), Nreo≥2 coupling axis untested (fixture has Nreo=1). Increment 3 note: close over `static`, do not pass as jit arg (non-hashable; traced-pytree path degrades deg-0 rows to dynamic scatter).
- `[A][2026-08-01]` TASK-001 go-ahead received. Increments 1–2 delegated to Codex 5.6 (TASK-001a spec written; user launches Codex). Design upgrade over original plan: all traced quantities enter linearly, so the traced build is static-tensor einsum contractions — no 27×N² trace-time unroll, flat compile cost in N. A holds increments 3–4, blocked until 001a verified.
- `[A][2026-08-01]` Formalized coordination protocol (this rewrite): git-push locking, task ledger, model routing incl. Codex 5.6 handoff convention. Moved 3D-port plan to `docs/tasks/TASK-001-jax-coupled-3d.md`. TASK-001 still awaiting user go-ahead.
- `[B][2026-08-01]` Pushed `3c41217` (JAX scan increment + multilayer viscoelastic test; suite 318 green). Wrote initial handoff. Yielded to Machine A.
