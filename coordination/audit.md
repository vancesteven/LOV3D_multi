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
