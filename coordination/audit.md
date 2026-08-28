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
