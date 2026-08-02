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
| TASK-001 | 3D coupled JAX port (all 4 increments) | free | VERIFIED — committed `ce3fd7e` + `4b7557a` | `docs/tasks/TASK-001-jax-coupled-3d.md` |
| TASK-002 | Aprop_aux for coupled JAX path | free | VERIFIED — committed `1a4a707` | `docs/tasks/TASK-002-codex-aprop-aux.md` |
| TASK-004 | JAX coupled-path performance benchmark at realistic N | free | VERIFIED — committed `5817164` | `docs/BENCHMARK_jax_coupled.md` |
| TASK-003 | MATLAB cross-validation of coupled JAX path (proposed for B) | free | VERIFIED — committed `a89eb6c` | — |

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

**Codex guardrails (standing — every Codex spec MUST include these):**
1. Explicit allowlist of writable files; everything else read-only.
2. No state-changing git (commit/add/push/pull/branch/checkout/reset/stash).
3. No environment changes (installs, venv edits, shims, substitute envs) —
   if the prescribed runner is broken, stop and report, never improvise.
4. Never touch `matlab/`, `docs/HANDOFF.md`, or coord files (own spec's
   Completion notes excepted).
5. No deletions; no files beyond the allowlist; 500-line cap per file.
6. On ambiguity or apparent spec error: stop, record the question in
   Completion notes, finish only the unambiguous parts.
7. Suite must stay green; regressions are reported, not force-fixed.
Recommended launcher settings until trust is established: workspace-write
sandbox, network disabled, approval prompts on anything outside the sandbox.

---

## Environment notes (non-obvious)

- **Python venv:** `venvLOV3Dconv/bin/python` on BOTH machines (venvs are
  per-clone, not synced). Machine B: JAX 0.10.0. Machine A: rebuilt 2026-08-02 —
  JAX 0.10.2, py3nj compiled from source (needed `brew install gcc` +
  `CC=gcc-16 CXX=g++-16 FC=gfortran-16` for OpenMP), plus pyalma3 (module name
  is `alma`) and matplotlib.
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
- ✅ **M4 COMPLETE + ALL TASKS VERIFIED:** PyALMA3 benchmark · JAX 1D loop · JAX 1D lax.scan · 3D coupled scan port · Aprop_aux · JAX↔MATLAB direct validation · performance benchmark (`5817164`, verified by B)
- Suite: 333 passed as of `a89eb6c` — confirmed on BOTH machines (A: jax 0.10.2, B: 0.10.0).
- Ledger fully closed — no open/queued tasks. Candidate next work (needs user go-ahead + ticketing): README/milestone update; chunked-vmap for large-N memory (JAX peak RSS ~3× NumPy at N=101); ocean-layer support in coupled path (`NotImplementedError`); GPU-backend benchmark (CPU-only so far).

---

## Coordination log

_Newest on top. Format: `[machine][YYYY-MM-DD] note`. Cap at ~15 entries —
prune from the bottom when adding (git history keeps the rest)._

- `[B][2026-08-01]` TASK-004 VERIFIED (was DONE `5817164`). Opus adversarial static review of `scripts/benchmark_jax_coupled.py` + `docs/BENCHMARK_jax_coupled.md` → APPROVE-WITH-NITS: methodology sound (subprocess isolation correct, RUSAGE_CHILDREN reads exactly one child, cold/warm distinction valid), benchmarks the genuine solver paths (`_get_solution_coupled` / `jax_get_solution_coupled_scan`, no cross-backend cache leak, N/Nr data-determined), numbers internally consistent. B independently reproduced `--quick` on jax 0.10.0: tiny N=4 → 4.42× (A: 4.5×), small N=12 → 6.86× (A: 6.5×); warm medians rock-steady (33–34 ms / 144 ms), peak RSS 1.7–1.9× NumPy — cross-machine agreement. Non-blocking nits (no number affected): (1) large_denser abort prose is arithmetically self-inconsistent (says ~220–260 s expected < 300 s timeout yet it aborted → per-call estimate too low, cold likely ~100 s+); (2) anomaly FLOP estimate omits the ~6 matmuls inside each `_ap` build, so 15.3 s figure is coincidentally-close not rigorous (writeup hedges "right order of magnitude"); (3) median-of-3 warm is spec floor (stable in practice); (4) headline is warm-JAX vs never-warmed-NumPy — correctly labeled, JAX-favorable framing. **M4 ledger fully closed.**
- `[A][2026-08-02]` TASK-004 benchmark run (Apple M4, CPU backend). Warm JAX vs NumPy: 4.5× (N=4), 6.5× (N=12), 3.7× (N=38), 1.4× (N=101). At N≈100 both paths are compute-bound in the dense 8N×8N linalg.solve inside every RK stage, so jit caching stops mattering (cold≈warm) and the speedup collapses — JAX wins big for small/medium-N sweeps, not large single solves. JAX peak RSS ~3× NumPy (4.4 vs 1.5 GB at N=101). N=101/Nr=179 JAX case aborted at 300s timeout (NumPy: 99.6s). Earlier "20× at N=107" figure was the vmapped aux build only, not the full solve — not comparable. Results: docs/BENCHMARK_jax_coupled.md; script scripts/benchmark_jax_coupled.py (note /scripts is gitignored — needs add -f).
- `[A][2026-08-02]` TASK-003 confirmed on machine A: 333 passed (jax 0.10.2), matching B. MATLAB refs present in A clone. Validation triangle closed: JAX↔NumPy 1e-15, NumPy↔MATLAB (M3), JAX↔MATLAB direct 6e-4–1.4e-2 within tolerance ladder. M4 remaining: TASK-004 perf benchmark (A).
- `[B][2026-08-01]` TASK-003 DONE + VERIFIED. New `pylov3d/tests/test_jax_matlab_validation.py`: drives the Enceladus lateral benchmark through `jax_get_solution_coupled_scan` → `extract_love_numbers` (zero NumPy-solver involvement) and asserts vs published MATLAB `data/tests/enceladus/Q_*.mat` at the same 3 cases + order-based tolerances (1%/5%/10%) as the NumPy MATLAB test. 4–6 modes compared/case (pert. orders 0/1/2); JAX↔MATLAB rel err 6e-4–1.4e-2, all within tol. Suite **333 passed** on B. Opus adversarial review APPROVE, zero correctness defects: confirmed JAX is the sole solver (no NumPy leak), assertions non-vacuous (10% y_sol perturbation → all fail), perturbation_order genuinely exercised, reference/tolerance logic byte-identical to NumPy sibling. Non-blocking nit: reuses `_EnceladusBench.enceladus_params.__wrapped__` (pytest impl detail; fails loud not silent). Awaiting user commit approval. Real MATLAB refs live in `data/tests/`, NOT `io_coupled_reference.npz` (that's a NumPy self-regression fixture).
- `[B][2026-08-01]` Synced to `0dd4935` (fast-forward, 11 A-commits). Full suite **330 passed** on B clone (JAX 0.10.0) — TASK-001/002 coupled JAX + Aprop_aux confirmed green on B. Claiming TASK-003 (scope: standalone JAX→MATLAB regression guard, independent of NumPy reference).
- `[A][2026-08-02]` TASK-002 assessed and VERIFIED (uncommitted). Codex fully guardrail-compliant. Opus review APPROVE: refactor bitwise-identical to HEAD across 4 configs; Aprop_aux matches NumPy in 10 configs ≤8e-16 incl. adversarial Nrlayer[1]=0; vmap bitwise-equal to serial; 7/9 mutants caught, 2 survivors provably immaterial (stored rows independent of g/dg/rho). Perf notes: vmap peak RAM ~5.4GB at N=107/Nr=119 (chunk if production runs grow); Aprop_aux always computed (parity with NumPy, no opt-out); ~27ms/solve re-keying overhead. Nits: aux entry point lacks ocean guard.
- `[A][2026-08-02]` TASK-002 spec written and delegated to Codex (Aprop_aux via vmapped verified builder; 4-tuple return for full drop-in parity). Standing Codex guardrails added to this file (user has not yet settled Codex autonomy level — specs are defensive). TASK-003 (MATLAB cross-validation of coupled path) queued for B.
- `[A][2026-08-02]` TASK-001 COMPLETE. Increments 1–2 (Codex) committed `ce3fd7e` with review fixes; increments 3–4 (Sonnet impl, Opus verify — APPROVE, 18/19 mutants killed, fuzz to N=107) committed `4b7557a`. y_sol matches NumPy coupled solver to 1.5e-15. Jit-cache memoization added (uncached path was 3.4× slower than NumPy at N=4; warm ~10× faster). Suite 329. Known gaps → TASK-002: Aprop_aux not computed (blocks energy.get_energy_coupled / love.py:89 substitution); ocean layers still NotImplementedError (parity with NumPy); K_amp upstream is identically zero in rheology.py (both branches) — JAX path verified correct against hand-injected K_amp anyway.
- `[A][2026-08-02]` TASK-001a assessed and VERIFIED. Codex output correct: Opus 5 adversarial review (line-by-line + randomized differential harness to 2.2e-15 + 27-mutant mutation test, 25 killed / 2 equivalent-or-gap) → APPROVE, zero defects. Suite in rebuilt machine-A venv: 322 passed, 0 failed. Open nits before commit: test_jit_smoke compares JAX to itself (swap tuple unpack), Nreo≥2 coupling axis untested (fixture has Nreo=1). Increment 3 note: close over `static`, do not pass as jit arg (non-hashable; traced-pytree path degrades deg-0 rows to dynamic scatter).
- `[A][2026-08-01]` TASK-001 go-ahead received. Increments 1–2 delegated to Codex 5.6 (TASK-001a spec written; user launches Codex). Design upgrade over original plan: all traced quantities enter linearly, so the traced build is static-tensor einsum contractions — no 27×N² trace-time unroll, flat compile cost in N. A holds increments 3–4, blocked until 001a verified.
- `[A][2026-08-01]` Formalized coordination protocol (this rewrite): git-push locking, task ledger, model routing incl. Codex 5.6 handoff convention. Moved 3D-port plan to `docs/tasks/TASK-001-jax-coupled-3d.md`. TASK-001 still awaiting user go-ahead.
- `[B][2026-08-01]` Pushed `3c41217` (JAX scan increment + multilayer viscoelastic test; suite 318 green). Wrote initial handoff. Yielded to Machine A.
