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
| TASK-016 | **M6:** lateral-field stage (Airy + areoid correction, n_lv≤4) | free | VERIFIED — committed `462d78a` | `docs/tasks/TASK-016-design.md` |
| TASK-014 pt 2 | **M6:** B — MATLAB coupled cross-check of the Mars lateral model | B | **DONE — committed `8a134e3`**: native LOV3D coupled solver on the 4-layer Mars model + committed crust `mu_variable` field reproduces the Python lateral spectrum — N=115 coupled modes (exact), k2_uniform identical to 12 digits, forcing-mode k2 rel err **2.95e-13**, k2 lateral shift 5.5174e-5 confirmed, non-forcing spectrum matches mode-for-mode. Awaiting VERIFIED (different-driver review). | `scripts/mars_lateral_cross_check.m` |
| TASK-017 | test-suite runtime lanes | free | VERIFIED — committed `c14a268` (fast 483/~2min, full 496 via `-m ""`) | `docs/tasks/TASK-017-codex-suite-runtime.md` |
| TASK-018 | **M7:** Moon instantiation (Weber reference, 4-param MC, citation-verified constraints, core_layer_index framework fix) | free | VERIFIED — committed `c14a268` (3 Opus rounds) | `docs/MOON_MODEL.md` |
| TASK-020 | **CRITICAL (user): commit verification artifacts for both Mars MATLAB cross-checks** — the 1D (k2/h2/l2 ~1e-12) and coupled-lateral (N=115, 2.95e-13) numbers currently exist only as prose; a MATLAB-less reader cannot verify them. B: re-run both drivers, save console output + a small .mat of the computed Love numbers/spectrum to `data/tests/mars/` (mirroring `data/tests/moon/` practice), commit. Also state MATLAB version used | B | **DONE — committed `08a38b6`** (MATLAB R2025b, 25.2.0.3150157 Update 4). Both drivers re-run; `.log`+`.mat` for each in `data/tests/mars/`; 1D k2/h2/l2 ~1e-12, coupled N=115 & k2 rel 2.95e-13 reproduced; artifacts scipy-readable (MATLAB-less verify OK); MARS_MODEL.md cites them. **VERIFIED (A, 2026-08-07)** — both .mat scipy-verified on A: 1D k2/h2/l2 match Python pins to ~1e-13; lateral N=115 + full spectrum present. | `data/tests/mars/` |
| TASK-019 | **M7:** properly-resolved Moon posterior under the shipped bounds (n_active>=64, n_effective>=128; the quoted 363.7±33.2 km reference run predates the density floor). **Machine B** — compute-heavy, token-light | B | **DONE — committed `e165ee0`.** pocoMC n_active=64/n_effective=128/Nrbase=50, dynamic term; n_samples=4105, ESS=4087.7, wall 123 min. R_fluid_core=**321 km (−13.8/+19.1)** (core_rho_scale 0.900 at floor, mu_scale 0.965, mantle_rho_scale 1.01); observables match all 4 constraints. 321 km < stale pre-floor 363.7±33.2 km, pulled toward as-built Weber ~327 km — mass/MoI mechanism under the 0.88 floor, as moon_mc.py predicts. Python suite still green (488 passed). **VERIFIED (A, 2026-08-07)** — chain re-analyzed on A: ESS 4087.70 and all medians reproduce. MOON_MODEL.md reference-posterior section updated (`2470969`). | `docs/figures/proposal/moon_posterior_chain.npz` |
| TASK-013 | **M6:** Mars SH data loaders (GMM-3 SHADR + MarsTopo719) | free | VERIFIED — committed `ecef078` (with 012) | `docs/tasks/TASK-013-codex-mars-sh-loaders.md` |
| TASK-014 | **M6:** MATLAB Mars cross-checks (both parts) | free | **VERIFIED (A, 2026-08-05)** — pt 1 `a07bbf8` (1D, ~1e-12); pt 2 `8a134e3` (coupled lateral: N=115 exact, k2 rel 2.95e-13, spectrum mode-for-mode; A checked the Python-side anchors, the npz contract in the driver, and that pylov3d/ is untouched — MATLAB execution B-attested per part-1 precedent) | `scripts/mars_1d_cross_check.m`, `scripts/mars_lateral_cross_check.m` |
| TASK-021 | **Hydration-front tidal signature (proposal RQ1 made quantitative):** serpentinization front in the crust shell (f_h, serpentinite mu/K reductions, lateral distribution tied to ingested crustal-thickness fields) -> mu_variable + K_amp through the validated lateral machinery -> predicted k2 shift + Love spectrum vs f_h, detectability statement vs sigma_k2=0.006 | A | **DONE — committed `370234d`** (2 Opus review rounds + verified fix round). Headline: Δk2 max ~5% of σ_k2 at f_h=0.5, robust across 12× reference-crust span; mean:lateral ~65:1 at lmax=4; k2 = bulk-hydration meter, front location lives in off-(2,0) spectrum. 20 tests; fast lane 502. Awaiting VERIFIED (B cross-check welcome) | `pylov3d/mars_hydration.py`, `docs/MARS_MODEL.md` §5, fig7 |
| TASK-021b | **B: hydration-amplitude sweeps** — run `get_love_hydrated` over the f_h × serpentinite-bracket grid at higher radial resolution (Nrbase≥50) + lmax=4, confirm A's Δk2 curve + ~65:1 mean:lateral ratio on independent hardware, optionally extend to lmax=6 to check (4,0)↔(2,0) channel convergence. Compute-heavy, token-light. Entry: `pylov3d/mars_hydration.py` docstring + `docs/MARS_MODEL.md` §5 | B | **DONE — committed `1a0d26a`** (`scripts/mars_hydration_sweep.py`; artifacts under `docs/figures/proposal/mars_hydration_sweep.{npz,png}`). Independent sweep at **Nrbase=50** (finer than A's 30), lmax=4, wall 3621 s: reproduces A's central-ratio mean/lateral/ratio to **<0.15% on every f_h** (e.g. f_h=0.5: mean 3.2032e-4, lateral 5.622e-6, ratio 57.0 vs A's 3.203e-4/5.62e-6/57). **mean:lateral 57–64:1 confirmed** on independent hardware+grid (A's ~60:1, not 65:1, corroborated). Detectability reproduced (Δk2 max 5.43% of σ_k2 at f_h=0.5). **lmax=6 convergence check** (Nrbase=30, central): lateral term shifts only 0.33%/1.09%/2.05% at f_h=0.1/0.3/0.5 (N 115→219) — (4,0)↔(2,0) channel converged at lmax=4, A's config vindicated. No pylov3d module modified. Awaiting VERIFIED (A). | — |

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

- `[B][2026-08-08]` **TASK-021b DONE — committed `1a0d26a`** (`scripts/mars_hydration_sweep.py`; artifacts under `docs/figures/proposal/mars_hydration_sweep.{npz,png}` — `scripts/output/` is gitignored, so placed with the Mars/Moon MC-posterior artifacts). Independent cross-check of A's TASK-021 at **higher radial resolution** (Nrbase=50 vs A's 30), lmax=4, on independent hardware; full f_h×{low,central,high} sweep, wall 3621 s (15 coupled solves). **Reproduces A's central-ratio numbers to <0.15% on every f_h:** mean/lateral/ratio at f_h=0.1 = 6.3357e-5/9.889e-7/64.1 (A: 6.336e-5/9.89e-7/64), f_h=0.3 = 1.9106e-4/3.136e-6/60.9 (A: 61), f_h=0.5 = 3.2032e-4/5.622e-6/57.0 (A: 57). **mean:lateral 57–64:1 confirmed** — corroborates A's corrected ~60:1 (not 65:1). Detectability reproduced: Δk2 never approaches σ_k2=0.006 (max 5.43% at f_h=0.5); precision to resolve f_h=0.1 = 6.43e-5. **lmax=6 convergence check** (Nrbase=30, central, N 115→219): lateral term shifts only 0.33%/1.09%/2.05% at f_h=0.1/0.3/0.5 — the (4,0)↔(2,0) first-order channel is converged at lmax=4, vindicating A's validated config; residual grows with f_h as expected and is <0.04% of total Δk2. Note: lmax=6 at Nrbase=50 needs >15 GB RAM (near-OOM here), so the convergence check ran at Nrbase=30 (angular question, radial-independent). No `pylov3d` module modified — driver only calls `mars_hydration`. Awaiting VERIFIED (A).
- `[A][2026-08-07]` **Corrections to the entry below + the `370234d` commit message** (caught by an Opus pass while porting TASK-021 results into the proposal docs): (1) D3's fix does NOT raise — it threads `K_variable` through `process_lateral_variations` so the coupling is applied; the regression test pins non-collapse, not an exception (the module's only raise is for an unknown scenario name). (2) The mean:lateral dominance is ~57–64:1 over the swept f_h range (write "~60:1"), not "~65:1". (3) "Lateral ≤1.5% of total" conflates channels: the whole lateral term is <2% (1.72% at f_h=0.5), while the K-convention spread alone is ~0.2% of total. docs/MARS_MODEL.md was already correct on all three; commit message stands as-is (immutable), this entry is the correction of record.
- `[A][2026-08-07]` **TASK-021 DONE — committed `370234d`** (user gate passed). Fix round verified against the reviewer's own measured numbers: D1 dominance restated ~65:1 at validated lmax=4 (500:1 was an lmax=2 truncation artifact); D2 K convention adopted as parity-derived K_nm=3·δK with the ~4× convention band documented (standing caveat: unvalidated in either codebase — raise with LOV3D authors); D3 K-only silent no-op now raises; D4 sign test at lmax=4; D5 normalization pin (kills a −31% mutant); S1 reference-crust sensitivity table (17 GPa ×0.20 … peridotite ×2.40 — dominant systematic, null conclusion robust); S2 reframing ("k2 measures how much hydration, not where; front-diagnostic observable is the off-(2,0) spectrum"). 20 hydration tests green (incl. slow), fast lane 502. Fig7 in proposal set. **TASK-021b cut for B** (independent sweep + lmax-convergence check, claimable now).
- `[A][2026-08-07]` TASK-021 round 1 + Opus dual review (REJECT-narrow, fix round dispatched). Core verified exact; headline conclusion (hydration k2 signal undetectable at current sigma_k2) ROBUST — survives a 12x reference-crust span per the review's own solves. Key corrections in flight: mean/lateral dominance is ~65:1 at validated lmax=4 (the ~500:1 was an lmax=2 truncation artifact — lmax=2 deletes the (4,0) first-order channel); K_amp scaling convention is UNVALIDATED anywhere in either codebase (MATLAB zeroes K_amp on all paths; 3.8x spread across defensible readings; conclusion-insensitive) — flagged as a standing caveat for any future lateral-K work; reference-crust choice is the dominant systematic (InSight in-situ ~17 GPa x0.2 vs peridotite x2.4); framing corrected to "k2 measures how much hydration, not where — the front-diagnostic observable is the off-(2,0) spectrum". Suite 513 full-lane green pre-fix.
- `[B][2026-08-07]` **Claiming TASK-021b** (independent hydration-amplitude sweep, A's TASK-021 headline cross-check). Plan: reproduce A's f_h × serpentinite-bracket sweep via `get_love_hydrated` at higher radial resolution (**Nrbase=50, lmax=4**) on independent hardware, confirm the Δk2(f_h) curve + ~65:1 mean:lateral ratio; then an **lmax=6** convergence check on the (4,0)↔(2,0) channel. Compute-heavy, token-light — reuses `pylov3d/mars_hydration.py` (no pylov3d module modified). Sweep script/artifacts pending user commit approval.
- `[B][2026-08-06]` **TASK-019 DONE — committed `e165ee0`.** Resolved Moon posterior via pocoMC (`scripts/moon_pocomc.py`, Moon analogue of `mars_pocomc.py`): n_active=64, n_effective=128, Nrbase=50, dynamic termination; n_samples=4105, Kish **ESS=4087.7**, wall 123 min. Medians ±1σ: core_rho_scale 0.900 (−0.0146/+0.0278, sits at the 0.88 floor), mu_scale 0.965 (−0.0096/+0.0104), **R_fluid_core 321 km (−13.8/+19.1)**, mantle_rho_scale 1.01 (−0.0012/+0.0011). Median-model observables match all 4 constraints (mass 7.344e22 kg, moi_mean 0.3929, core 321.2 km, k2 0.02419). **R_fluid_core=321 km < stale pre-floor 363.7±33.2 km**, pulled toward the as-built Weber ~327 km: a residual mass/MoI pull under the shipped 0.88 core-density floor, exactly as `moon_mc.py`'s docstring predicts (mass/MoI mechanism, not anelastic). Chain+pairplot committed under `docs/figures/proposal/` (`moon_posterior_chain.npz`, `moon_posterior_pairplot.png`), mirroring the Mars MC precedent. Python suite still green (488 passed) after the pocomc install (mpmath 1.4.1→1.3.0). Awaiting VERIFIED (A).
- `[B][2026-08-05]` **Claiming TASK-019** (resolved Moon posterior under the shipped bounds; A flagged the quoted 363.7±33.2 km reference run as predating the density floor). Plan: read `pylov3d/moon_mc.py` to fix the exact sampler entry point + parameterization, run pocoMC at `n_active>=64` / `n_effective>=128`, report medians+credible intervals vs the stale reference, and persist the chain/summary as an artifact (mirroring the Mars MC practice). Compute-heavy, token-light — no framework changes intended (TASK-018 `core_layer_index` fix already in).
- `[B][2026-08-05]` **TASK-020 DONE — committed `08a38b6`.** Re-ran both native-MATLAB Mars cross-check drivers (**MATLAB R2025b, 25.2.0.3150157 Update 4**) and committed verification artifacts to a new `data/tests/mars/` (mirroring `data/tests/moon/`): `mars_1d_cross_check.{log,mat}` and `mars_lateral_cross_check.{log,mat}`. 1D reproduced k2=0.169000000000 / h2=0.315632205682 / l2=0.051595952202 (rel ~1e-12, h2/k2=1.8676); coupled reproduced **N=115 modes (exact)**, k2_uniform=0.169000000002, k2_forcing=0.169055174106 (**rel 2.95e-13**), k2_shift=5.517410e-5, full n/m/complex-k spectrum saved. Both `.mat` confirmed scipy-readable (a MATLAB-less reader can verify every headline number). Added save-artifact blocks to the two `.m` drivers, the `data/tests/mars` .gitignore whitelist exception, and citations in both MARS_MODEL.md cross-val notes. Python suite untouched. Awaiting VERIFIED (A). Next: TASK-019 (resolved Moon posterior).
- `[B][2026-08-05]` **Claiming TASK-020** (CRITICAL — priority over 019). Re-run both committed Mars MATLAB drivers (`scripts/mars_1d_cross_check.m`, `scripts/mars_lateral_cross_check.m`), capture full console output and a small `.mat` of the computed Love numbers/spectrum into a new `data/tests/mars/` (mirroring the `data/tests/moon/` `.mat` practice), add the `data/tests/mars` .gitignore whitelist exception, and commit so a MATLAB-less reader can verify the ~1e-12 (1D) and 2.95e-13 (coupled-lateral, N=115) numbers that currently live only as prose. Will also record the MATLAB version (R2025b). Then 019.
- `[B][2026-08-04]` **TASK-014 part 2 DONE — committed `8a134e3`.** New `scripts/mars_lateral_cross_check.m` runs the native LOV3D *coupled* solver on the exact 4-layer Mars model + the committed crust-layer complex-SH `mu_variable` field (`data/mars/mars_mu_variable_lateral.npz`, read directly via a minimal in-script `.npy` parser — no Python bridge; complex-SH path used directly, p2p percent conversion bypassed; `eta0` omitted on all 4 layers). **MATLAB ↔ Python agree essentially bit-for-bit** (matched apples-to-apples: `method='variable'`, `Nrbase=30`, `perturbation_order=2`): **N = 115 coupled modes (exact)**; k2_uniform = 0.169000000002 (identical to 12 digits); forcing-mode (2,0) k2 = 0.169055174106 (**rel err 2.95e-13**); k2 lateral shift = 5.517410e-5 vs Python 5.517410435e-5 (7.9e-8 residual on the *shift* is pure float64 cancellation from differencing two ~0.169 values). Non-forcing spectrum matches mode-for-mode: (3,0) largest at -7.29e-5, (2,±2) pair at +3.22e-5±2.03e-5i, (3,±1)/(3,±3) pairs, etc. The Mars lateral model now has the same native-MATLAB anchor the 1D model got in part 1. Doc note added to `docs/MARS_MODEL.md` §4. Python suite untouched (only a MATLAB script + docs added). Both parts of TASK-014 now complete; **awaiting a different-driver VERIFIED review** (A, when convenient). Coupling files cached under `data/couplings/` (gitignored) — regenerated automatically on first run.
- `[B][2026-08-04]` **Claiming TASK-014 part 2** (MATLAB coupled cross-check of the Mars lateral model). Plan: extend the part-1 MATLAB driver to run the native LOV3D *coupled* solver on the same 4-layer Mars model plus the lateral mu-variation field committed in `data/mars/mars_mu_variable_lateral.npz` (23 amplitudes + provenance; MATLAB layer idx = python+1; eta0 OMITTED for elastic), and compare the resulting coupled Love spectrum against the `docs/MARS_MODEL.md` §"Lateral variations" table (k2 lateral shift target +5.52e-5). Gives the Mars lateral model the same native-MATLAB anchor the 1D model now has. Token-light, MATLAB-side. First: read the npz + the Lateral-variations table to fix the exact comparison contract before writing the driver.
- `[B][2026-08-04]` **TASK-014 part 1 DONE (uncommitted).** New `scripts/mars_1d_cross_check.m` runs native LOV3D on the exact 4-layer Mars model (radii/densities/mu/Ks copied verbatim from `build_mars_model()`, elastic). **MATLAB ↔ Python agree to ~1e-12 on all three Love numbers**: k2=0.169000000000 (rel 2.0e-12), h2=0.315632205682 (1.2e-12), l2=0.051595952202 (8.6e-13). **Adjudicates the open item: h2/k2 = 1.867646 is CONFIRMED REAL** — the native MATLAB solver reproduces it bit-for-bit, so the ratio sitting above the published ~1.6–1.7 is a genuine coarse-4-layer/elastic-parameterization feature, NOT a Python port bug. Mars now has the same MATLAB anchor the Moon has. One porting gotcha found & documented in the script: MATLAB `get_rheology` treats a layer as elastic only when `eta0` is EMPTY/absent — `eta0=NaN` wrongly enters the viscoelastic branch and poisons the solve with NaN (Python uses NaN=elastic, opposite convention). Python suite unaffected (test_mars.py 18 passed; only a MATLAB script + docs added). Awaiting user commit approval (code/script commit). Part 2 (heavy coupled Mars lateral sweeps + MATLAB coupled cross-val) now unblocked (012 done) — claimable next.
- `[B][2026-08-04]` **Claiming TASK-014 part 1** (MATLAB LOV3D cross-check of the Mars 1D model). Plan: build a MATLAB driver that runs the native LOV3D 1D solver on the exact 4-layer Mars model from `docs/MARS_MODEL.md` (liquid core 1830 km / LM 2340 km / UM / 50 km crust; same densities + mu_lm/um=96.5/67.5 GPa), and confirm Python's elastic k2=0.169, h2=0.3156, l2=0.0516. Gives Mars the same MATLAB anchor the Moon has and adjudicates the open h2/k2=1.87 item (is it a real coarse-4-layer artifact or a Python bug?). Token-light, MATLAB-side. Deprioritizing TASK-015 per A's note (converged n_active=256/Nrbase=15 posterior already exists).
- `[A][2026-08-05]` TASK-018 build round 1 done (Moon instantiation: Weber 10-layer reference reusing the M5 MATLAB anchor, GRAIL/LLR constraint table with a corrected GM citation, 3-param MC with measured tension — the elastic Weber profile undershoots GRAIL k2 by 4.8σ, consistent with the anelastic monthly-tide contribution, under review). Build also caught a framework defect: forward.compute_observables "core_radius_km" hardwires layer 0 — correct for Mars, wrong for any body whose fluid core is interior (Moon: layer 2); local workaround shipped, proper forward.py fix under review. Suite 491. Opus dual review round 1 in flight.
- `[A][2026-08-05]` TASK-016 build round 1 done (461 tests) → Opus dual review REJECT, fix round dispatched. Core confirmed correct 3 independent ways (CSH conversion incl. first-ever imaginary amplitudes through the solver — verified by a rotation-covariance test; rigidity sign checked against Mars geography). Adopted science upgrade from review: Airy elevation re-referenced to the GMM-3 low-degree areoid (removes the single-point rigidity-positivity violation, peak δt 40.6→34.2 km). Review also found: k2 lateral shift is FIRST order via (4,0)↔(2,0) even-parity self-coupling (k2 = linear probe of degree-4 zonal crust — proposal-relevant); (5,±4) order mislabel is a parity/toroidal selection effect, root cause = get_couplings discarding ST when collapsing mode labels (couplings.py:143-147, note for a future fix); latent mu_scale threading bug fixed before it could bite the MC stage; npz for B's MATLAB run was gitignored — !/data/mars exception being added.
_(Older M4 / M5 / TASK-001…013 entries, the 2026-08-04 proposal-figure-set + TASK-012/013 fix-round notes, and the 2026-08-03 M6 direction-change note pruned — see git history.)_
