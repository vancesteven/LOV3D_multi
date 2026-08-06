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
- `[A][2026-08-04]` NASA proposal figure set committed `3554001` (docs/figures/proposal/: interior model, converged pocomc corner plot ESS=4119 + archived chain, MOLA pipeline map, tidal response, Qin validation pedigree; regenerable via scripts/proposal_figures/). Supports the user's Solar System science program proposal. TASK-015 (B production MC) partly superseded: a converged n_active=256/Nrbase=15 posterior now exists; B's 015 remains useful only at higher n_active or full Nrbase=100 — B may deprioritize in favor of 014 part 1 (MATLAB Mars cross-check).
- `[A][2026-08-04]` TASK-012+013 fix round complete, final Opus verdict APPROVE both. Fixes verified hard: Legendre.m Rapp-recursion port (no-CS, stable to lmax=719; orthonormality quadrature exact; MarsTopo719 degree-1 error 2743 m → 4.5e-13 m); R_core ridge broken (log-posterior now decays as the exact Stähler Gaussian, spread 0→2.0 nats), with honest framing: tidal/bulk data contribute ZERO R_core information — the marginal IS the Stähler prior, now properly in the likelihood; Hellas integration test adjudicated sound (C20 removal destroys 0.11% of Hellas signal vs 20 km of flattening) with Olympus Mons matched to 0.13 km as free corroboration. Suite 451 (24 forward + 21 mapping + 11 sh_data + refactored harness). Logged for later: mapping memory ~1.5 GB at full-res lmax=719 synthesis (accumulate per-order to fix); quick-run remains banner-flagged non-converged; shape-C20 proxy is not the true areoid (Hellas floor -6.8 vs published -8.2 km rel. areoid).
- `[A][2026-08-04]` TASK-012 build round 1 done (31 tests) + Codex TASK-013 done CONCURRENTLY in the same tree (8 tests; zero file conflicts — disjoint-allowlist protocol worked; both agents noticed and ignored each other's files). Combined Opus review: 013 APPROVE (loaders bitwise vs independent parses); 012 REJECT — headline finds: (1) Condon-Shortley sign retained in sh_to_latlon → odd-m maps 180°-rotated in lon exactly when composed with 013's no-CS data; (2) factorial underflow → NaN maps at the data's own lmax (120/719); (3) Mars posterior EXACTLY FLAT along R_core (4 params/3 constraints — quoted R_core "measurement" was resampled prior; fix = add the unused Stähler 1830±40 as a 4th Gaussian constraint); (4) quick-run credible intervals ~2× under-dispersed (ESS 45). Fix round dispatched to the build agent (incl. porting the trusted MATLAB Legendre.m stable recursion and a Hellas-basin integration test tying 012+013 together).
- `[A][2026-08-04]` TASK-011 committed `4836150`. M6 stage-2 split ticketed: 012 (A, science design), 013 (Codex, mechanical SH loaders — after A lands the data files, since Codex runs without network), 014 (B — part 1 claimable NOW: MATLAB cross-check of the Mars 1D model, cheap tokens, uses MATLAB, adjudicates the h2/k2=1.87 question; part 2 later: heavy coupled sweeps). B: pick up 014 part 1 whenever.
- `[A][2026-08-04]` TASK-011 DONE (uncommitted). Mars 1D: 4 layers (liquid core 1830 km via native fluid-CMB BC / LM to 2340 km olivine-wadsleyite / UM / 50 km crust); exact 2×2 density solve to mass GM/G + MEAN MoI 0.36310 (J2-corrected from polar 0.3644 — Opus science review caught the mean/polar confusion, a 2.6σ effect); mantle-mu bisection to elastic k2=0.169 (residual <1e-12). Fitted: rho_core=6128 (Stähler band ✓), rho_lm=4136 (mass-balancing shell — documented as exceeding the literal Khan-2021 range), mu_lm/um=96.5/67.5 GPa (Vs 4.83/4.46 ✓). h2=0.3156, l2=0.0516 pinned. Two Opus review rounds: 1st caught 10 citation errors in MY spec (GMM-3→MRO120D, k2=0.174→Konopliv 2020 not "InSight RISE", MoI→Konopliv 2011, missing Khan/Samuel 2023 small-core caveat) + 7 code defects; 2nd verified all 17 fixed but caught a fabricated Khan-2023 title introduced by the fix (now corrected + Huang 2022 PNAS added). Suite 395 (18 Mars tests). Known open: h2/k2=1.87 above typical published ~1.6-1.7 (coarse 4-layer artifact, pinned for future cross-validation); bodies.py id 40 Ae/Aobli are synchronous-satellite formulas, not meaningful for Mars.
_(Older M4 / M5 / TASK-001…010 entries + the 2026-08-03 M6 direction-change note pruned — see git history.)_
