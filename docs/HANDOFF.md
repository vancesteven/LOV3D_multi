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
| TASK-008 | **M5:** MATLAB/Qin reference harness (Weber Moon) | free | VERIFIED — committed `61f1292`; all 5 checks live+green since 007 | `test_matlab_validation_ocean.py` |
| TASK-009 | **M5:** JAX coupled ocean support (three-segment scan) | free | VERIFIED — committed `6374640` | log entry below |
| TASK-010 | ocean energy dissipation (ocean-ceiling node in `energy.py`) | free | **ON HOLD — B STAND DOWN** (user direction 2026-08-03, supersedes "work the candidate list"). Recon preserved in spec; resume later | `docs/tasks/TASK-010-ocean-energy.md` |
| TASK-011 | **M6:** Mars 1D reference model | free | VERIFIED — committed `4836150` (2 science-review rounds) | `docs/MARS_MODEL.md`, `pylov3d/mars.py` |
| TASK-012 | **M6:** body-agnostic forward-model + MC framework + mapping + Mars fit map | free | VERIFIED — committed `ecef078` (with 013) | log below |
| TASK-015 | **M6:** full pocoMC Mars posterior production run (n_active≥256, Nrbase=100, tens of minutes; publish corner plot + medians vs point fit). **Machine B** — compute-heavy, token-light | free | QUEUED — claimable now (`scripts/mars_pocomc.py`, everything committed) | — |
| TASK-016 | **M6:** lateral-field stage (Airy, n_lv≤4, fixed forward — user-approved design) | A | IN-PROGRESS | `docs/tasks/TASK-016-design.md` |
| TASK-013 | **M6:** Mars SH data loaders (GMM-3 SHADR + MarsTopo719) | free | VERIFIED — committed `ecef078` (with 012) | `docs/tasks/TASK-013-codex-mars-sh-loaders.md` |
| TASK-014 | **M6:** MATLAB Mars cross-checks | B (pt 2 open) | **part 1 VERIFIED — committed `a07bbf8`**: MATLAB↔Python k2/h2/l2 to ~1e-12; h2/k2=1.8676 CONFIRMED real (coarse-4-layer feature, not a port bug); eta0 empty-vs-NaN convention gotcha documented. Part 2 (coupled Mars lateral sweeps + MATLAB coupled cross-val) claimable | `scripts/mars_1d_cross_check.m` |

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
- **ON HOLD (user decision 2026-08-03) — ALL ocean-world follow-ons**, resume later: (1) TASK-010 ocean energy dissipation (B's recon preserved in `docs/tasks/TASK-010-ocean-energy.md`); (2) Europa thick-shallow-ocean coupled MATLAB case (constrains gO — B/MATLAB); (3) chunked-vmap for large-N memory; (4) GPU-backend benchmark; (5) aux NaN-factory cleanup (compute_aprop_aux_coupled evaluates the elastic propagator at in-ocean nodes and discards it).
- **NEW DIRECTION: M6 = Mars model**, staged: TASK-011 first (1D radial model fit to published bulk constraints — mass/MoI/k2/InSight core — with full source citations and a science-accuracy review), then TASK-012 (lateral variations from gravity/topography fields; data-source decision deferred to its ticketing). B: stand down from TASK-010, await Mars tickets.

---

## Coordination log

_Newest on top. Format: `[machine][YYYY-MM-DD] note`. Cap at ~15 entries —
prune from the bottom when adding (git history keeps the rest)._

- `[B][2026-08-04]` **TASK-014 part 1 DONE (uncommitted).** New `scripts/mars_1d_cross_check.m` runs native LOV3D on the exact 4-layer Mars model (radii/densities/mu/Ks copied verbatim from `build_mars_model()`, elastic). **MATLAB ↔ Python agree to ~1e-12 on all three Love numbers**: k2=0.169000000000 (rel 2.0e-12), h2=0.315632205682 (1.2e-12), l2=0.051595952202 (8.6e-13). **Adjudicates the open item: h2/k2 = 1.867646 is CONFIRMED REAL** — the native MATLAB solver reproduces it bit-for-bit, so the ratio sitting above the published ~1.6–1.7 is a genuine coarse-4-layer/elastic-parameterization feature, NOT a Python port bug. Mars now has the same MATLAB anchor the Moon has. One porting gotcha found & documented in the script: MATLAB `get_rheology` treats a layer as elastic only when `eta0` is EMPTY/absent — `eta0=NaN` wrongly enters the viscoelastic branch and poisons the solve with NaN (Python uses NaN=elastic, opposite convention). Python suite unaffected (test_mars.py 18 passed; only a MATLAB script + docs added). Awaiting user commit approval (code/script commit). Part 2 (heavy coupled Mars lateral sweeps + MATLAB coupled cross-val) now unblocked (012 done) — claimable next.
- `[B][2026-08-04]` **Claiming TASK-014 part 1** (MATLAB LOV3D cross-check of the Mars 1D model). Plan: build a MATLAB driver that runs the native LOV3D 1D solver on the exact 4-layer Mars model from `docs/MARS_MODEL.md` (liquid core 1830 km / LM 2340 km / UM / 50 km crust; same densities + mu_lm/um=96.5/67.5 GPa), and confirm Python's elastic k2=0.169, h2=0.3156, l2=0.0516. Gives Mars the same MATLAB anchor the Moon has and adjudicates the open h2/k2=1.87 item (is it a real coarse-4-layer artifact or a Python bug?). Token-light, MATLAB-side. Deprioritizing TASK-015 per A's note (converged n_active=256/Nrbase=15 posterior already exists).
- `[A][2026-08-05]` TASK-016 build round 1 done (461 tests) → Opus dual review REJECT, fix round dispatched. Core confirmed correct 3 independent ways (CSH conversion incl. first-ever imaginary amplitudes through the solver — verified by a rotation-covariance test; rigidity sign checked against Mars geography). Adopted science upgrade from review: Airy elevation re-referenced to the GMM-3 low-degree areoid (removes the single-point rigidity-positivity violation, peak δt 40.6→34.2 km). Review also found: k2 lateral shift is FIRST order via (4,0)↔(2,0) even-parity self-coupling (k2 = linear probe of degree-4 zonal crust — proposal-relevant); (5,±4) order mislabel is a parity/toroidal selection effect, root cause = get_couplings discarding ST when collapsing mode labels (couplings.py:143-147, note for a future fix); latent mu_scale threading bug fixed before it could bite the MC stage; npz for B's MATLAB run was gitignored — !/data/mars exception being added.
- `[A][2026-08-04]` NASA proposal figure set committed `3554001` (docs/figures/proposal/: interior model, converged pocomc corner plot ESS=4119 + archived chain, MOLA pipeline map, tidal response, Qin validation pedigree; regenerable via scripts/proposal_figures/). Supports the user's Solar System science program proposal. TASK-015 (B production MC) partly superseded: a converged n_active=256/Nrbase=15 posterior now exists; B's 015 remains useful only at higher n_active or full Nrbase=100 — B may deprioritize in favor of 014 part 1 (MATLAB Mars cross-check).
- `[A][2026-08-04]` TASK-012+013 fix round complete, final Opus verdict APPROVE both. Fixes verified hard: Legendre.m Rapp-recursion port (no-CS, stable to lmax=719; orthonormality quadrature exact; MarsTopo719 degree-1 error 2743 m → 4.5e-13 m); R_core ridge broken (log-posterior now decays as the exact Stähler Gaussian, spread 0→2.0 nats), with honest framing: tidal/bulk data contribute ZERO R_core information — the marginal IS the Stähler prior, now properly in the likelihood; Hellas integration test adjudicated sound (C20 removal destroys 0.11% of Hellas signal vs 20 km of flattening) with Olympus Mons matched to 0.13 km as free corroboration. Suite 451 (24 forward + 21 mapping + 11 sh_data + refactored harness). Logged for later: mapping memory ~1.5 GB at full-res lmax=719 synthesis (accumulate per-order to fix); quick-run remains banner-flagged non-converged; shape-C20 proxy is not the true areoid (Hellas floor -6.8 vs published -8.2 km rel. areoid).
- `[A][2026-08-04]` TASK-012 build round 1 done (31 tests) + Codex TASK-013 done CONCURRENTLY in the same tree (8 tests; zero file conflicts — disjoint-allowlist protocol worked; both agents noticed and ignored each other's files). Combined Opus review: 013 APPROVE (loaders bitwise vs independent parses); 012 REJECT — headline finds: (1) Condon-Shortley sign retained in sh_to_latlon → odd-m maps 180°-rotated in lon exactly when composed with 013's no-CS data; (2) factorial underflow → NaN maps at the data's own lmax (120/719); (3) Mars posterior EXACTLY FLAT along R_core (4 params/3 constraints — quoted R_core "measurement" was resampled prior; fix = add the unused Stähler 1830±40 as a 4th Gaussian constraint); (4) quick-run credible intervals ~2× under-dispersed (ESS 45). Fix round dispatched to the build agent (incl. porting the trusted MATLAB Legendre.m stable recursion and a Hellas-basin integration test tying 012+013 together).
- `[A][2026-08-04]` TASK-011 committed `4836150`. M6 stage-2 split ticketed: 012 (A, science design), 013 (Codex, mechanical SH loaders — after A lands the data files, since Codex runs without network), 014 (B — part 1 claimable NOW: MATLAB cross-check of the Mars 1D model, cheap tokens, uses MATLAB, adjudicates the h2/k2=1.87 question; part 2 later: heavy coupled sweeps). B: pick up 014 part 1 whenever.
- `[A][2026-08-04]` TASK-011 DONE (uncommitted). Mars 1D: 4 layers (liquid core 1830 km via native fluid-CMB BC / LM to 2340 km olivine-wadsleyite / UM / 50 km crust); exact 2×2 density solve to mass GM/G + MEAN MoI 0.36310 (J2-corrected from polar 0.3644 — Opus science review caught the mean/polar confusion, a 2.6σ effect); mantle-mu bisection to elastic k2=0.169 (residual <1e-12). Fitted: rho_core=6128 (Stähler band ✓), rho_lm=4136 (mass-balancing shell — documented as exceeding the literal Khan-2021 range), mu_lm/um=96.5/67.5 GPa (Vs 4.83/4.46 ✓). h2=0.3156, l2=0.0516 pinned. Two Opus review rounds: 1st caught 10 citation errors in MY spec (GMM-3→MRO120D, k2=0.174→Konopliv 2020 not "InSight RISE", MoI→Konopliv 2011, missing Khan/Samuel 2023 small-core caveat) + 7 code defects; 2nd verified all 17 fixed but caught a fabricated Khan-2023 title introduced by the fix (now corrected + Huang 2022 PNAS added). Suite 395 (18 Mars tests). Known open: h2/k2=1.87 above typical published ~1.6-1.7 (coarse 4-layer artifact, pinned for future cross-validation); bodies.py id 40 Ae/Aobli are synchronous-satellite formulas, not meaningful for Mars.
- `[A][2026-08-03]` **DIRECTION CHANGE (user): all ocean-world follow-ons ON HOLD, including TASK-010 — B stand down** (this supersedes the earlier "work the candidate list" instruction; B had done recon only, no code — preserved in the spec file for resumption). New milestone M6: **Mars model matching gravity + topographic constraints.** Scoping on A now; Mars tickets will appear here. Note: this was also the coordination protocol's first real push collision (A editing HANDOFF while B pushed e66cb3c) — non-FF rejection caught it as designed; resolved by rebasing A's edits onto B's version.
- `[B][2026-08-03]` **Claiming TASK-010** (ocean energy dissipation; user: work the candidate list in order). Recon: `energy.py` `compute_stress_strain` + `compute_stress_strain_coupled` skip every node whose `layer_map` is the ocean layer, INCLUDING `ocean_end` — but post-TASK-005/007 recombination `y_sol[ocean_end]=C_shell` (shell data), so that node belongs to the shell above and should feed the shell dissipation integral; skipping it zeros `dissipation[ocean_end]` and truncates the base of the shell's trapezoid. Complication: `Aprop_aux[ocean_end]=0.0` (node was integrated inside the ocean, Laplace-only), so the shell propagator's aux rows at that radius are needed to get `u_dot` right. Proposal pending user pick of fix locus (solver-side aux fill vs energy-side recompute). No code yet.
- `[B][2026-08-03]` **M5 confirmed on Machine B.** Pulled A's 10 commits (fast-forward `61f1292`→`52621b3`); full suite **377 passed, 2 warnings in 141.7s** (JAX 0.10.0) — matches A's `6374640` count exactly. Warnings benign (auto-conjugate mode `n=2,m=-2` in `test_lateral_rheology.py`). Ledger fully closed, no open tasks. Noted A's reconfig of the TASK-008 Weber-Moon harness to the actual `.mlx` setup (artificial core + ×1000 inner-core rigidify, UM 5:8 / LM 3:4 p2p amplitudes) — all 5 lateral cases green here.
- `[B][2026-08-02]` **TASK-008 DONE (uncommitted, awaiting user commit approval).** New `pylov3d/tests/test_matlab_validation_ocean.py` (~250 lines). Decoded the Moon `.mat` plot-format: `to_plot` rows 2..end are per-mode [col0=perturbation order, col2=n, col3=m], cols 4+ are k2-NORMALISED ratios per amplitude (row0 col4+ = amp grid); absolute Love no. = `to_plot[row,4+j]*k2_Q`. NOTE: these are **Qin et al. 2016 perturbation-theory** references (the paper's comparison set), de-normalised by `k2_Q` (uniform k2≈0.0232, matches notebook). **7 active tests pass now** (reference parser over all 5 LM/UM cases + Weber 9-layer model builder — confirms fluid outer core at layer idx 1, Vs=0→ocean=1, exercises the M5 fluid propagator). **5 solver-comparison tests auto-SKIP** (one per Moon case) while `_get_solution_coupled` raises `NotImplementedError` on ocean — they flip live automatically when TASK-007 wires the 24N BC path, **no edit to my file needed**. Suite: **348 passed, 5 skipped** (was 341). Zero shared files with Codex-006 / A-007. Two things for TASK-007 review to confirm against the notebook: (a) lateral-variation host layer — I used upper-mantle layer idx 4 as representative; (b) amplitude/SH normalization reuses the Enceladus `amp/sqrt(4pi)` (m=0) / `/sqrt(2)/sqrt(4pi)` (m≠0) convention — should hold but verify for the Qin-referenced Moon case.
- `[B][2026-08-02]` **Claiming TASK-008** (Machine B, MATLAB). Recon: MATLAB R2025b present; `data/tests/moon/*.mat` (5 files: LM/UM × deg/order) exist + `Moon_Weber.dat` (9-layer model, row2 vs=0 → fluid/ocean layer, so genuinely exercises the M5 ocean propagator). Format is `{to_plot(rows×18), k2_Q}` — a period-sweep PLOT layout, NOT the clean `{k_B,kD_B,k2_B}` the enceladus loader consumes, so 008 needs its own parser. Plan (user-approved): parse existing `.mat` as-is (no MATLAB execution), decode `to_plot`/(n,m,order)→k against `Test_Moon_MultiLayered_Lateral_Variations.mlx` (read-only), build Weber-Moon model loader + reference fixture, and write `test_matlab_validation_ocean.py` with assertions **skipped/xfail pending TASK-007** (coupled ocean solver doesn't exist yet). Flips green the moment 007 lands. Runs fully parallel to Codex-006 / A-007 — no shared files.
- `[A][2026-08-03]` TASK-009 DONE (uncommitted) — **last M5 item**. Three-segment scan (jax_ocean_scan.py): solid below / Laplace-only ocean / solid shell, identity restarts between; in-ocean segment matches NumPy to 1e-19 (asserted at 1e-14 so the solid scan's pre-existing ~1e-11 roundoff floor can't mask ocean errors); Weber Moon Y/y_sol/aux/love.k all <1e-10 vs the MATLAB-validated NumPy path; no-ocean dispatch byte-identical to HEAD. Opus initially REJECTED — real find: zero-node layer directly above the ocean (happens naturally with method='fixed' + thin ice shell) made JAX silently diverge 15% from NumPy, and NumPy itself is discontinuous in the point count there. Fixed per review: BOTH solvers now raise ValueError on that degenerate grid; + ocean-at-index-3/multi-boundary-below test, in-ocean exactness test, aux detection reuses guarded _detect_ocean. Sonnet also fixed a latent read-only-view bug in jax_coupled_aux (np.asarray→np.array). Suite 377.
- `[A][2026-08-03]` TASK-007 DONE (uncommitted). Coupled ocean path wired (ocean_coupled.py + solver.py); N=1 reduction to 1D ocean path BITWISE (elastic + viscoelastic). **B's Weber-Moon harness reconfigured to the actual notebook setup** (B pre-authorized): the .mlx does NOT use the raw 9-layer profile — it prepends an artificial 50km/8000 core and rigidifies the inner core ×1000 (dodging the ocean-above-core case, unsupported in MATLAB and here); lateral variations span UM=layers 5:8 / LM=3:4 (0-based) with PEAK-TO-PEAK % amplitudes (p2p/100/Delta conversion, Delta from a MATLAB-exact SPH_LatLon replica); reference stores the forcing mode as deviation k−k2ᵘ; Qin m>0 rows ÷√2 under zonal forcing. Anchors: uniform Weber k2 matches MATLAB k2_Q to 2.2e-9; all 5 lateral cases green with order-1 modes at 2e-6–5e-6 rel (tol tightened to 0.1%) and forcing deviation ~0.3% (tol 5%). Opus APPROVE, zero defects (independent MATLAB transcriptions 0.0-diff; 10-mutant harness kill-check; order-3 rerun shows truncation ≤6e-8). Residuals logged: gO weakly constrained by Moon geometry (thick-shallow-ocean Europa coupled case would fix — candidate for B/MATLAB); deg-0 in-ocean branch transcription-verified only; JAX coupled ocean = TASK-009. Suite 365.
- `[A][2026-08-03]` TASK-006 assessed and VERIFIED (uncommitted). Codex guardrail-clean again. Opus APPROVE, zero defects: independent from-scratch MATLAB transcription with dense cross-mode-coupled Ys matches to 9.9e-16 (N=1..4, n∈{0,1}, negative m); 12/15 mutants killed by shipped tests, 3 survivors were a low-degree coverage gap (n=0/1 reduction cases now added by A, all pass). TASK-007 integration contract (from review): C is in GROUPED column basis (no MATLAB index_arr permutation — reconstruct with grouped Y@C blocks); identity restarts required at ocean entry AND shell entry; y_sol[ocean_end]=C_shell directly. n<0 sentinel degrees would make B silently singular (faithful-to-MATLAB landmine, don't pass them).
- `[A][2026-08-02]` TASK-005 DONE (uncommitted). 1D ocean path was WORSE than recon suggested — 3 bugs: (1) elastic propagator used inside ocean (fixed: `build_aprop_ocean`, Laplace-only, 0.0-diff vs MATLAB transcription); (2) missing `i==ocean_end` shell-origin recombination (fixed); (3) BCindices consumed 0-based but stored 1-based → 24×24 BC matrix SINGULAR (rank 22) — ocean models could never solve at all (fixed: BCindices normalized 0-based all methods; note MATLAB's own `manual` branch is inconsistently 0-based). JAX 1D guards added. 8 new tests incl. mu→0 fluid-limit convergence (agrees 4e-6 rel at mu=1). Europa-like ocean k2=0.270346, grid-independent to 9 digits over 32× resolution. Opus APPROVE; its mutation testing caught one unpinned fix (test strengthened to watch U at ocean_end, not Φ). Suite 341. Notes for later tasks: energy.py skips the ocean_end node (now shell data) in dissipation integrals — revisit when ocean energy tested; k2 pin awaits MATLAB confirmation (TASK-008 — add 1D Europa/Titan cases to B's list); deg-0 ocean branch faithful but dead (singular BC).
- `[A][2026-08-02]` **M5 chosen by user: ocean layers in the coupled solver.** Recon done; tickets TASK-005…009 queued. CRITICAL recon finding: the Python 1D ocean path has ZERO solve tests and diverges from MATLAB in two ways — (1) no ocean_flag propagator (MATLAB uses a Poisson-only system inside the ocean, get_solution.m:1924-1935; Python integrates the elastic equations with the ocean layer's muC), (2) missing the `i==ocean_end` identity-based recombination (get_solution.m:864-881 vs solver.py:375-382). TASK-005 (audit/fix/test 1D) therefore gates the milestone. Also: `matlab source lives in src/ + tests/*.mlx` (repo has no matlab/ dir — old rule wording corrected below); ocean-bearing coupled MATLAB reference data already in repo at data/tests/moon/ (Weber Moon, unused by any Python test).

_(Older M4 / M5-early / TASK-001…005 entries pruned — see git history.)_
