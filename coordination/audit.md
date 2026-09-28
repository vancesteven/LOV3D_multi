# Audit log — LOV3d-genai

Append-only. One entry per completed action, newest at the bottom. Never edit
or delete existing entries. Format is defined in `AGENTS.md`.

## 2026-08-28T00:00Z — claude-lov3d-genai
- Did: Rebuilt a working Linux venv (.venv-linux, ignored) after venvLOV3Dconv-linux lost exec bits; found `py3nj>=0.6` pin unsatisfiable (PyPI tops out at 0.2.x; verified reference venv uses 0.2.1) and relaxed it to `py3nj>=0.2`.
- Files: pylov3d/pyproject.toml
- Verification: import smoke test (wigner3j(2,2,2;0,0,0) = -0.239046, correct); full suites below run in this venv
- Handoff: none

## 2026-08-28T00:10Z — claude-lov3d-genai
- Did: Ran the validation order from MARS_3D_ALTERATION_GRAVITY_STATUS_2026-08-21.md at 5b8d7e4: gravity-convention suite 32/32 passed; mars_alteration_gravity_demo.py clean (off-target leakage 5.9e-20); run_science_benchmarks.py 164/164 passed. Status doc's "not yet rerun" caveat is now discharged.
- Files: none (validation only)
- Verification: pytest + demo + science benchmarks, all green
- Handoff: none

## 2026-08-28T00:20Z — claude-lov3d-genai
- Did: Fast-forwarded lov3d-genai 5b8d7e4 -> 0fccfd1, picking up the six mass-preserving radial profile reduction commits from myfork/agent/task-046-multibasis-energy, and reran run_science_benchmarks.py on the new tip: 167/167 passed (includes test_profile_reduction.py).
- Files: none (merge + validation)
- Verification: run_science_benchmarks.py, 167 passed in 318 s
- Handoff: Publication gate in docs/RADIAL_PROFILE_REDUCTION_2026-08-21.md is still open — Love-number convergence vs. target layer count has not been demonstrated for any reduced profile.

## 2026-08-28T01:00Z — claude-lov3d-genai
- Did: Implemented the open publication gate from docs/RADIAL_PROFILE_REDUCTION_2026-08-21.md: pylov3d/profile_convergence.py reduces the same high-resolution artifact to a sequence of layer counts, solves degree-2 elastic Love numbers per reduction, and reports successive |dk2|/|k2|; added scripts/radial_reduction_convergence.py (accepts an artifact or --synthetic Mars-like fixture) and pylov3d/tests/test_profile_convergence.py (registered in run_science_benchmarks.py). Also fixed reduced_shells_to_interior_model: a multi-shell fluid run touching the center now converts as liquid core (mu=0, ocean unset, matching build_mars_model) instead of ocean flags the solver rejects at layer index < 2.
- Files: pylov3d/profile_convergence.py, pylov3d/profile_reduction.py, pylov3d/tests/test_profile_convergence.py, scripts/radial_reduction_convergence.py, scripts/run_science_benchmarks.py, docs/RADIAL_PROFILE_REDUCTION_2026-08-21.md
- Verification: new tests 10/10; full run_science_benchmarks.py 174/174 passed; CLI on the 64-shell fixture shows |dk2|/|k2| falling to 5.2e-4 at 16 layers
- Handoff: Gate machinery is ready; still needs a real PlanetProfile Mars radial artifact to run for science (none exists in-repo). Convergence here is elastic-only, per the reducer's stated scope.

## 2026-08-28T02:00Z — claude-lov3d-genai
- Did: Preserved the three untracked MATLAB TASK-046 anchor files found in the sibling main checkout (~/src/LOV3d_multi, which had them only as untracked working files) by committing them here; also committed the /external/ ignore rule that existed there only as an uncommitted .gitignore edit.
- Files: data/tests/io/io_identical_coefficients_anchor.mat, data/tests/io/io_uniform_radial_anchor.mat, data/tests/io/io_raw_grid_energy_anchor.mat, .gitignore
- Verification: io_compare_identical_coefficients_anchor.py -> strict solver parity PASS (worst k relerr 1.3e-11); io_compare_uniform_radial_anchor.py -> alpha=0.9999999999559, diffs ~1e-9 except MATLAB-zeroed surface row; io_raw_grid_energy_anchor.mat loads via scipy with expected keys (no Python consumer script; it is the authoritative MATLAB Gate C artifact)
- Handoff: LOV3d_multi still carries the originals as untracked files plus regenerable CSV/log outputs; per protocol I did not modify that tree.

## 2026-09-11T00:00Z — claude-lov3d-genai
- Did: Delegated a subagent to reconcile external/SSS_2025_Mars with its upstream and update its documentation for pylov3d developments since 2026-08-21, then assess the proposal text. Result: proposal main fast-forwarded ef119b6 -> 0723ca2 (37 upstream commits incl. Task 3/MCR removal and today's Overleaf edits); stale local Task 1.3 edit + fig3 regeneration preserved on branch local-wip-2026-08-18 (ef22626); three new local commits 51e5de5 (PYLOV3D_CAPABILITY_UPDATE_2026-09-11.md + PROPOSAL_COMPLETION_STATUS.md refresh, 69 -> 174 tests), a48d0b6 (Methods_Models.tex: lateral-K validated, anchors committed, PlanetProfile on-ramp), 9fe2ff1 (PROPOSAL_ASSESSMENT_2026-09-11.md). Task 2 untouched: zProposal.tex identical to 0723ca2. Not pushed to the proposal's origin; origin/overleaf-2026-08-21-2222 left unmerged as superseded.
- Files: external/SSS_2025_Mars/* (separate repo, ignored here); plans/STATUS.md (refreshed per CLAUDE.md queue-freshness rule)
- Verification: md/doc updates verified against pylov3d docs and git log (cite: commits above); Methods_Models.tex edits implemented, unverified (no TeX toolchain here; static \ref/\cite check found no unresolved keys)
- Handoff: Steve to review PROPOSAL_ASSESSMENT_2026-09-11.md §7, apply Task 2 items himself (stray apostrophe at zProposal.tex:173), and push/sync the proposal repo with Overleaf. Protocol rewrite (AGENTS.md, CLAUDE.md, plans/) found uncommitted in this tree and left as-is.

## 2026-09-12T00:00Z — claude-lov3d-genai
- Did: Pushed the three proposal documentation commits (51e5de5, a48d0b6, 9fe2ff1) from external/SSS_2025_Mars to its origin main (fast-forward 0723ca2 -> 9fe2ff1) after Steve asked about push capability. Branch local-wip-2026-08-18 left local-only (preservation branch).
- Files: none in this repo
- Verification: git push output shows fast-forward; main == origin/main
- Handoff: Overleaf will pick up the new .md files and the Methods_Models.tex hunks on next sync; zProposal.tex unchanged.

## 2026-09-28T00:00Z — claude-lov3d-genai
- Did: Assessed proposal (external/SSS_2025_Mars @ 4ab83ce, fast-forwarded from 9fe2ff1) against Berne et al. 2026 (papers/berne2026tidal.pdf, read-only). Three Claude general-purpose subagents (Opus 5.5): science validity, pyLOV3D capability gap, science directions; Codex CLI 0.149.0 (gpt-5.6-sol) read-only fact-check; C1-C3 confirmed, C4 confirmed by manager spot-check, C5 raised open question (logged in open-questions.md). Manager adjudicated; corrected a subagent COM-COF number (300 K -> 1.46 km, not 2.9 km). Wrote PROPOSAL_ASSESSMENT_2026-09-28_BERNE2026.md in the proposal repo; added Berne parity/extension roadmap (B1-B10) to plans/STATUS.md.
- Files: external/SSS_2025_Mars/PROPOSAL_ASSESSMENT_2026-09-28_BERNE2026.md (separate repo, uncommitted); plans/STATUS.md
- Verification: review only; citations spot-checked against zProposal.tex/Methods_Models.tex and the paper text; COM-COF and water-budget numbers recomputed. No code changed.
- Handoff: Steve to apply F1-F3 in zProposal.tex (not edited by agents). Roadmap B1-B3 are the first LOV3D tasks; open physics question on m'!=0 annual forcing recorded in STATUS.md.

## 2026-09-28T18:30Z — claude-lov3d-genai
- Did: Roadmap B1 (extended Love tensor K^{l'm'}_{lm}, complex + real 4pi cos/sin basis) and B4 (degree-3/4 forcing anchor), implemented by a Claude coder subagent (Opus 5.5) and reviewed by the manager. Zenodo example recon (Claude general-purpose subagent, Opus 5.5) findings logged in open-questions.md: non-zero annual m'!=0 forcing, odd-m sine sign convention differs from our src, and a 2-layer toy model rather than ED Table 2.
- Files: pylov3d/extended_love.py, pylov3d/tests/test_extended_love.py, pylov3d/tests/test_love_degree_n.py, plans/STATUS.md, coordination/open-questions.md
- Verification: manager re-ran test_love_degree_n.py 4/4, test_extended_love.py 14/14 (fast), and slow test_mars_columns_match_matlab 1/1 (m'=0,1,2 vs native MATLAB spectra, max abs 1.2e-12)
- Handoff: B2 parity against Zenodo k2_responses.txt with both sine conventions; the sqrt2 flag in mars_detectability.required_stokes_amplitude (m_forcing_solve!=0 branch)

## 2026-09-28T21:30Z — claude-lov3d-genai
- Did: Manager verification and integration of the uncommitted Berne-roadmap code left by the earlier 2026-09-28 session: B1 extended-Love tensor (pylov3d/extended_love.py + test_extended_love.py) and B4 degree-n forcing anchor (test_love_degree_n.py). Full run 19/19 passed in 963 s including the slow MATLAB-anchor test (columns m'=0,1,2 vs mars_lateral_cross_check.mat to 1e-10; forced-mode 1e-10 relative). B1 and B4 marked verified in plans/STATUS.md (B4: analytic half only; PyALMA3/MATLAB cross-check open). Also committed the operative protocol rewrite (548ebbc), the second Zenodo open-questions entry, and git-ignored /papers/ (third-party PDF stays untracked).
- Files: pylov3d/extended_love.py, pylov3d/tests/test_extended_love.py, pylov3d/tests/test_love_degree_n.py, plans/STATUS.md, coordination/open-questions.md, .gitignore, AGENTS.md, CLAUDE.md, plans/CODEX-QUEUE.md
- Verification: pytest -m "" on the two new files, 19 passed (16:03); fast lane 18 passed earlier; no existing modules modified, so the 174-test suite is unaffected by construction
- Handoff: next roadmap items are B2 (Berne 1D reference + mantle-l=1 MATLAB anchor; needs ED Table 2 sourced outside the Zenodo archive) and B3 (ephemeris forcing + annual projection; will also decide the two open questions in coordination/open-questions.md). Sine-convention decisive test from OQ item 2 is still to be run.

## 2026-09-29T00:30Z — claude-lov3d-genai
- Did: Executed next steps 1 and 2. (1) Transcribed Berne et al. 2026 Extended Data Table 2 from the paper's page-20 table image (liquid core 0-1830 km rho 6200 K 142.8 GPa; mantle to 3340 km rho 3600 K 130.2 mu 79.5; crust to 3390 km rho 3300 K 104.3 mu 61.0) into pylov3d/berne2026.py with analytic transcription pins; elastic k2 = 0.176316, within 1.2 sigma of 0.169+/-0.006; model's 1.6-1.9% mass/MoI misfits documented as Berne's, not transcription error. (2) Ran the decisive sine-convention test: pylov3d on the Zenodo 2-layer example matches the shipped native-MATLAB k2_responses.txt to 1.0e-12 worst relative over 365 modes x 5 forcings under run_forward_shear.m's (-1)^m sine map; the canonical get_rheology.m map misses by 7.1e-2. OQ 2026-09-28 item 2 resolved (input-interpretation difference, solvers agree; resolution appended to open-questions.md). Zenodo anchors (CC-BY-4.0) committed under data/tests/berne_zenodo/; helpers in berne2026.py (a refactor sign bug on the canonical -M term was caught by a structural check against the validated builder and fixed before commit); durable test + diagnostic script added and registered in the science benchmark suite (fast classes only).
- Files: pylov3d/berne2026.py, pylov3d/tests/test_berne2026.py, pylov3d/tests/test_berne_zenodo_parity.py, scripts/berne_zenodo_convention_check.py, data/tests/berne_zenodo/*, scripts/run_science_benchmarks.py, plans/STATUS.md, coordination/open-questions.md
- Verification: test_berne2026.py 5/5; test_berne_zenodo_parity.py 3/3 including the slow (2,-1)-column anchor test (251 s, worst rel <= 1e-10); full two-convention sweep logged at 1.0e-12 vs 7.1e-2; benchmark registrations collect-checked
- Handoff: B2 verified (anchor far exceeds the <=1e-9 bar). Next: B3 ephemeris forcing + annual projection, which also tests OQ item 1 (fixed-local-time forcing) and the (3,2)/(3,3) mechanism question. For Steve: OQ item 1 and whether the paper's production pipeline shares the script sine convention are author-facing questions; the resolution text in open-questions.md is written to support that email if you choose to send it.

## 2026-09-29T02:30Z — claude-lov3d-genai
- Did: Built and verified the pointwise-connectivity higher-degree study (Fig-3 lateral extension), per Steve's direction treating results as PRELIMINARY pending Berne's updated calculations. Machinery: pylov3d/mars_hydration_connectivity.py (pointwise Voigt/Hill/Reuss on the crustal-thickness hydration field; MATLAB-faithful transform pair with (-1)^m conversion and half-cell phase removal, corrected round trip 4e-16; pointwise-Voigt reductions match the linear pipeline to 1e-13/6e-16; a wrong-direction Jensen assertion in an early test draft was corrected: convex Reuss/Hill pointwise means sit ABOVE mean-only). 12-combo sweep completed (~13 min): delta_k2_mean 6.3e-5..8.0e-4 (consistent with Fig-3 mean-only ranges); k_2m splitting 1e-8..1.1e-6 with connectivity-sensitive m-ordering (Voigt m=0 largest, Reuss m=2 largest); degree-3 response to (2,0) forcing 3.8e-6..3.2e-5, ~30-60x above the splitting at this lmax_out=2 truncation.
- Files: pylov3d/mars_hydration_connectivity.py, pylov3d/tests/test_mars_hydration_connectivity.py, scripts/mars_connectivity_higher_degree.py, scripts/run_science_benchmarks.py, data/tests/mars/connectivity_higher_degree_preliminary.csv, plans/STATUS.md
- Verification: test_mars_hydration_connectivity.py 6/6 including the slow tensor-sanity solve (74 s); sweep ran clean, clip_fraction 0 everywhere; CSV header carries the PRELIMINARY caveat and full numerics config
- Handoff: when Berne's updated calculations arrive, rerun the sweep against them and revisit lmax_out (2 -> 4) convergence; the l=3-dominance finding should be checked at lmax_out=4 before any proposal use.

## 2026-09-29T03:00Z — claude-lov3d-genai
- Did: Regenerated proposal Figure 3 with scripts/proposal_figures/fig_composite_inference_structure.py from the connectivity CSV (copied read-only from ~/src/LOV3d_multi, now committed here for reproducibility) and pushed both fig3_composite.pdf and .png to the proposal's origin main (f9a223d). Discharges PROPOSAL_COMPLETION_STATUS item 2 (figure regeneration/copy). Generator reported 4288 posterior samples, ESS=4093.
- Files: external/SSS_2025_Mars/figures/fig3_composite.{pdf,png} (separate repo, pushed); data/tests/mars/serpentinite_connectivity_sensitivity.csv
- Verification: generator ran clean and wrote both artifacts; visual check not performed (Steve closing machine); check the compiled figure on next Overleaf pass
- Handoff: confirm Figure 3 renders correctly at printed size on Overleaf.
