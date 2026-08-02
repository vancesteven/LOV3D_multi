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
| TASK-001…004 | Milestone 4 (coupled JAX port, Aprop_aux, MATLAB validation, benchmark) | — | ALL VERIFIED & committed (see log / git history) | `docs/tasks/` |
| TASK-005 | **M5:** audit + fix + test the 1D ocean path (3 bugs incl. singular BC matrix) | free | VERIFIED — committed `9ce8e78` | tests: `test_solver_ocean.py` |
| TASK-006 | **M5:** `assemble_bc_ocean_coupled` (24N×24N) | free | VERIFIED — committed `3953c29` | `docs/tasks/TASK-006-codex-bc-ocean-coupled.md` |
| TASK-007 | **M5:** coupled NumPy solver ocean path | free | VERIFIED — committed `a68a351`. All 5 MATLAB/Qin checks live+green | log entry below |
| TASK-008 | **M5:** MATLAB reference for ocean+lateral validation — Weber Moon case (`data/tests/moon/*.mat` exists, unused) + optionally regenerate/extend via `tests/Test_Moon_MultiLayered_Lateral_Variations.mlx` and Europa case. Machine B (MATLAB) | B | DONE (uncommitted) — parser+model tests live, 5 solver checks auto-skip until 007 | `test_matlab_validation_ocean.py` |
| TASK-009 | **M5:** JAX coupled ocean support (three-segment scan) | free | VERIFIED — committed `6374640` | log entry below |
| TASK-010 | **M6:** ocean energy dissipation — include the ocean-ceiling (shell-origin) node in `energy.py` dissipation integrals (currently skipped; holds shell data post-recombination) | B | IN-PROGRESS | `docs/tasks/TASK-010-ocean-energy.md` |

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
- MATLAB source = `src/` + `tests/*.mlx` (there is no `matlab/` dir — earlier
  rule wording was stale). Read-only port reference: targeted reads for porting
  are fine; never modify, never bulk-scan.

---

## Standing rules (both machines)

- Do NOT implement code unless explicitly instructed — summarize & propose first.
- Code commits only when the user asks. `coord:` commits pre-authorized (above).
- Keep files < 500 lines; tests in `pylov3d/tests/`; no working files in repo root.
- graphify re-runs only with user confirmation: `graphify run pylov3d/ --exclude pylov3d/tests/`

---

## Project status

- ✅ M1 1D solver · ✅ M2 3D lateral/mode coupling · ✅ M3 MATLAB cross-validation
- ✅ **M4 COMPLETE** (JAX coupled port + validation + benchmark)
- ✅ **M5 COMPLETE** (`b871153` README): oceans in 1D (TASK-005, incl. 3-bug repair), coupled NumPy (006+007, Weber Moon MATLAB/Qin spectra to ppm), coupled JAX (009, three-segment scan). Suite **377** confirmed on BOTH machines (A `6374640`; B on pull to `52621b3`).
- No queued tasks. Candidates (need user go-ahead + ticketing): ocean energy dissipation in coupled path (energy.py skips ocean-ceiling node, now shell data); Europa thick-shallow-ocean coupled MATLAB case (would constrain gO, weakly pinned by Moon geometry — B/MATLAB task); chunked-vmap for large-N memory; GPU-backend benchmark.

---

## Coordination log

_Newest on top. Format: `[machine][YYYY-MM-DD] note`. Cap at ~15 entries —
prune from the bottom when adding (git history keeps the rest)._

- `[B][2026-08-03]` **Claiming TASK-010** (ocean energy dissipation; user: work the candidate list in order). Recon: `energy.py` `compute_stress_strain` + `compute_stress_strain_coupled` skip every node whose `layer_map` is the ocean layer, INCLUDING `ocean_end` — but post-TASK-005/007 recombination `y_sol[ocean_end]=C_shell` (shell data), so that node belongs to the shell above and should feed the shell dissipation integral; skipping it zeros `dissipation[ocean_end]` and truncates the base of the shell's trapezoid. Complication: `Aprop_aux[ocean_end]=0.0` (node was integrated inside the ocean, Laplace-only), so the shell propagator's aux rows at that radius are needed to get `u_dot` right. Proposal pending user pick of fix locus (solver-side aux fill vs energy-side recompute). No code yet.
- `[B][2026-08-03]` **M5 confirmed on Machine B.** Pulled A's 10 commits (fast-forward `61f1292`→`52621b3`); full suite **377 passed, 2 warnings in 141.7s** (JAX 0.10.0) — matches A's `6374640` count exactly. Warnings benign (auto-conjugate mode `n=2,m=-2` in `test_lateral_rheology.py`). Ledger fully closed, no open tasks. Noted A's reconfig of the TASK-008 Weber-Moon harness to the actual `.mlx` setup (artificial core + ×1000 inner-core rigidify, UM 5:8 / LM 3:4 p2p amplitudes) — all 5 lateral cases green here.
- `[B][2026-08-02]` **TASK-008 DONE (uncommitted, awaiting user commit approval).** New `pylov3d/tests/test_matlab_validation_ocean.py` (~250 lines). Decoded the Moon `.mat` plot-format: `to_plot` rows 2..end are per-mode [col0=perturbation order, col2=n, col3=m], cols 4+ are k2-NORMALISED ratios per amplitude (row0 col4+ = amp grid); absolute Love no. = `to_plot[row,4+j]*k2_Q`. NOTE: these are **Qin et al. 2016 perturbation-theory** references (the paper's comparison set), de-normalised by `k2_Q` (uniform k2≈0.0232, matches notebook). **7 active tests pass now** (reference parser over all 5 LM/UM cases + Weber 9-layer model builder — confirms fluid outer core at layer idx 1, Vs=0→ocean=1, exercises the M5 fluid propagator). **5 solver-comparison tests auto-SKIP** (one per Moon case) while `_get_solution_coupled` raises `NotImplementedError` on ocean — they flip live automatically when TASK-007 wires the 24N BC path, **no edit to my file needed**. Suite: **348 passed, 5 skipped** (was 341). Zero shared files with Codex-006 / A-007. Two things for TASK-007 review to confirm against the notebook: (a) lateral-variation host layer — I used upper-mantle layer idx 4 as representative; (b) amplitude/SH normalization reuses the Enceladus `amp/sqrt(4pi)` (m=0) / `/sqrt(2)/sqrt(4pi)` (m≠0) convention — should hold but verify for the Qin-referenced Moon case.
- `[B][2026-08-02]` **Claiming TASK-008** (Machine B, MATLAB). Recon: MATLAB R2025b present; `data/tests/moon/*.mat` (5 files: LM/UM × deg/order) exist + `Moon_Weber.dat` (9-layer model, row2 vs=0 → fluid/ocean layer, so genuinely exercises the M5 ocean propagator). Format is `{to_plot(rows×18), k2_Q}` — a period-sweep PLOT layout, NOT the clean `{k_B,kD_B,k2_B}` the enceladus loader consumes, so 008 needs its own parser. Plan (user-approved): parse existing `.mat` as-is (no MATLAB execution), decode `to_plot`/(n,m,order)→k against `Test_Moon_MultiLayered_Lateral_Variations.mlx` (read-only), build Weber-Moon model loader + reference fixture, and write `test_matlab_validation_ocean.py` with assertions **skipped/xfail pending TASK-007** (coupled ocean solver doesn't exist yet). Flips green the moment 007 lands. Runs fully parallel to Codex-006 / A-007 — no shared files.
- `[A][2026-08-03]` TASK-009 DONE (uncommitted) — **last M5 item**. Three-segment scan (jax_ocean_scan.py): solid below / Laplace-only ocean / solid shell, identity restarts between; in-ocean segment matches NumPy to 1e-19 (asserted at 1e-14 so the solid scan's pre-existing ~1e-11 roundoff floor can't mask ocean errors); Weber Moon Y/y_sol/aux/love.k all <1e-10 vs the MATLAB-validated NumPy path; no-ocean dispatch byte-identical to HEAD. Opus initially REJECTED — real find: zero-node layer directly above the ocean (happens naturally with method='fixed' + thin ice shell) made JAX silently diverge 15% from NumPy, and NumPy itself is discontinuous in the point count there. Fixed per review: BOTH solvers now raise ValueError on that degenerate grid; + ocean-at-index-3/multi-boundary-below test, in-ocean exactness test, aux detection reuses guarded _detect_ocean. Sonnet also fixed a latent read-only-view bug in jax_coupled_aux (np.asarray→np.array). Suite 377.
- `[A][2026-08-03]` TASK-007 DONE (uncommitted). Coupled ocean path wired (ocean_coupled.py + solver.py); N=1 reduction to 1D ocean path BITWISE (elastic + viscoelastic). **B's Weber-Moon harness reconfigured to the actual notebook setup** (B pre-authorized): the .mlx does NOT use the raw 9-layer profile — it prepends an artificial 50km/8000 core and rigidifies the inner core ×1000 (dodging the ocean-above-core case, unsupported in MATLAB and here); lateral variations span UM=layers 5:8 / LM=3:4 (0-based) with PEAK-TO-PEAK % amplitudes (p2p/100/Delta conversion, Delta from a MATLAB-exact SPH_LatLon replica); reference stores the forcing mode as deviation k−k2ᵘ; Qin m>0 rows ÷√2 under zonal forcing. Anchors: uniform Weber k2 matches MATLAB k2_Q to 2.2e-9; all 5 lateral cases green with order-1 modes at 2e-6–5e-6 rel (tol tightened to 0.1%) and forcing deviation ~0.3% (tol 5%). Opus APPROVE, zero defects (independent MATLAB transcriptions 0.0-diff; 10-mutant harness kill-check; order-3 rerun shows truncation ≤6e-8). Residuals logged: gO weakly constrained by Moon geometry (thick-shallow-ocean Europa coupled case would fix — candidate for B/MATLAB); deg-0 in-ocean branch transcription-verified only; JAX coupled ocean = TASK-009. Suite 365.
- `[A][2026-08-03]` TASK-006 assessed and VERIFIED (uncommitted). Codex guardrail-clean again. Opus APPROVE, zero defects: independent from-scratch MATLAB transcription with dense cross-mode-coupled Ys matches to 9.9e-16 (N=1..4, n∈{0,1}, negative m); 12/15 mutants killed by shipped tests, 3 survivors were a low-degree coverage gap (n=0/1 reduction cases now added by A, all pass). TASK-007 integration contract (from review): C is in GROUPED column basis (no MATLAB index_arr permutation — reconstruct with grouped Y@C blocks); identity restarts required at ocean entry AND shell entry; y_sol[ocean_end]=C_shell directly. n<0 sentinel degrees would make B silently singular (faithful-to-MATLAB landmine, don't pass them).
- `[A][2026-08-02]` TASK-005 DONE (uncommitted). 1D ocean path was WORSE than recon suggested — 3 bugs: (1) elastic propagator used inside ocean (fixed: `build_aprop_ocean`, Laplace-only, 0.0-diff vs MATLAB transcription); (2) missing `i==ocean_end` shell-origin recombination (fixed); (3) BCindices consumed 0-based but stored 1-based → 24×24 BC matrix SINGULAR (rank 22) — ocean models could never solve at all (fixed: BCindices normalized 0-based all methods; note MATLAB's own `manual` branch is inconsistently 0-based). JAX 1D guards added. 8 new tests incl. mu→0 fluid-limit convergence (agrees 4e-6 rel at mu=1). Europa-like ocean k2=0.270346, grid-independent to 9 digits over 32× resolution. Opus APPROVE; its mutation testing caught one unpinned fix (test strengthened to watch U at ocean_end, not Φ). Suite 341. Notes for later tasks: energy.py skips the ocean_end node (now shell data) in dissipation integrals — revisit when ocean energy tested; k2 pin awaits MATLAB confirmation (TASK-008 — add 1D Europa/Titan cases to B's list); deg-0 ocean branch faithful but dead (singular BC).
- `[A][2026-08-02]` **M5 chosen by user: ocean layers in the coupled solver.** Recon done; tickets TASK-005…009 queued. CRITICAL recon finding: the Python 1D ocean path has ZERO solve tests and diverges from MATLAB in two ways — (1) no ocean_flag propagator (MATLAB uses a Poisson-only system inside the ocean, get_solution.m:1924-1935; Python integrates the elastic equations with the ocean layer's muC), (2) missing the `i==ocean_end` identity-based recombination (get_solution.m:864-881 vs solver.py:375-382). TASK-005 (audit/fix/test 1D) therefore gates the milestone. Also: `matlab source lives in src/ + tests/*.mlx` (repo has no matlab/ dir — old rule wording corrected below); ocean-bearing coupled MATLAB reference data already in repo at data/tests/moon/ (Weber Moon, unused by any Python test).
- `[B][2026-08-01]` **HANDOFF TO A — B standing down.** M4 ledger fully closed; TASK-004 verification final (full Opus report received, corroborates the pushed verdict line-for-line, no new findings — all 4 nits are cosmetic writeup prose, no reported number affected). Suite 333 green both machines. **A's call: pick the next milestone.** Unticketed candidates (need user go-ahead before claim): (1) README/milestone update to reflect M4 done; (2) chunked-vmap for large-N memory (JAX peak RSS ~3× NumPy at N=101); (3) ocean-layer support in coupled path (`NotImplementedError` today); (4) GPU-backend benchmark (all figures CPU-only). B will pull and claim once a direction + ticket exist.
- `[B][2026-08-01]` TASK-004 VERIFIED (was DONE `5817164`). Opus adversarial static review of `scripts/benchmark_jax_coupled.py` + `docs/BENCHMARK_jax_coupled.md` → APPROVE-WITH-NITS: methodology sound (subprocess isolation correct, RUSAGE_CHILDREN reads exactly one child, cold/warm distinction valid), benchmarks the genuine solver paths (`_get_solution_coupled` / `jax_get_solution_coupled_scan`, no cross-backend cache leak, N/Nr data-determined), numbers internally consistent. B independently reproduced `--quick` on jax 0.10.0: tiny N=4 → 4.42× (A: 4.5×), small N=12 → 6.86× (A: 6.5×); warm medians rock-steady (33–34 ms / 144 ms), peak RSS 1.7–1.9× NumPy — cross-machine agreement. Non-blocking nits (no number affected): (1) large_denser abort prose is arithmetically self-inconsistent (says ~220–260 s expected < 300 s timeout yet it aborted → per-call estimate too low, cold likely ~100 s+); (2) anomaly FLOP estimate omits the ~6 matmuls inside each `_ap` build, so 15.3 s figure is coincidentally-close not rigorous (writeup hedges "right order of magnitude"); (3) median-of-3 warm is spec floor (stable in practice); (4) headline is warm-JAX vs never-warmed-NumPy — correctly labeled, JAX-favorable framing. **M4 ledger fully closed.**
- `[A][2026-08-02]` TASK-004 benchmark run (Apple M4, CPU backend). Warm JAX vs NumPy: 4.5× (N=4), 6.5× (N=12), 3.7× (N=38), 1.4× (N=101). At N≈100 both paths are compute-bound in the dense 8N×8N linalg.solve inside every RK stage, so jit caching stops mattering (cold≈warm) and the speedup collapses — JAX wins big for small/medium-N sweeps, not large single solves. JAX peak RSS ~3× NumPy (4.4 vs 1.5 GB at N=101). N=101/Nr=179 JAX case aborted at 300s timeout (NumPy: 99.6s). Earlier "20× at N=107" figure was the vmapped aux build only, not the full solve — not comparable. Results: docs/BENCHMARK_jax_coupled.md; script scripts/benchmark_jax_coupled.py (note /scripts is gitignored — needs add -f).
- `[A][2026-08-02]` TASK-003 confirmed on machine A: 333 passed (jax 0.10.2), matching B. MATLAB refs present in A clone. Validation triangle closed: JAX↔NumPy 1e-15, NumPy↔MATLAB (M3), JAX↔MATLAB direct 6e-4–1.4e-2 within tolerance ladder. M4 remaining: TASK-004 perf benchmark (A).
- `[B][2026-08-01]` TASK-003 DONE + VERIFIED. New `pylov3d/tests/test_jax_matlab_validation.py`: drives the Enceladus lateral benchmark through `jax_get_solution_coupled_scan` → `extract_love_numbers` (zero NumPy-solver involvement) and asserts vs published MATLAB `data/tests/enceladus/Q_*.mat` at the same 3 cases + order-based tolerances (1%/5%/10%) as the NumPy MATLAB test. 4–6 modes compared/case (pert. orders 0/1/2); JAX↔MATLAB rel err 6e-4–1.4e-2, all within tol. Suite **333 passed** on B. Opus adversarial review APPROVE, zero correctness defects: confirmed JAX is the sole solver (no NumPy leak), assertions non-vacuous (10% y_sol perturbation → all fail), perturbation_order genuinely exercised, reference/tolerance logic byte-identical to NumPy sibling. Non-blocking nit: reuses `_EnceladusBench.enceladus_params.__wrapped__` (pytest impl detail; fails loud not silent). Awaiting user commit approval. Real MATLAB refs live in `data/tests/`, NOT `io_coupled_reference.npz` (that's a NumPy self-regression fixture).
- `[B][2026-08-01]` Synced to `0dd4935` (fast-forward, 11 A-commits). Full suite **330 passed** on B clone (JAX 0.10.0) — TASK-001/002 coupled JAX + Aprop_aux confirmed green on B. Claiming TASK-003 (scope: standalone JAX→MATLAB regression guard, independent of NumPy reference).

_(Older M4 / TASK-001…004 entries pruned — see git history.)_
