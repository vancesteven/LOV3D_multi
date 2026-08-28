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
