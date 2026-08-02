# TASK-006 — Codex handoff: coupled ocean boundary conditions (24N×24N)

**Status:** IN-PROGRESS (Codex 5.6)
**Owner:** CODEX
**Parent:** Milestone 5 (ocean layers in the coupled solver). TASK-005 (committed
`9ce8e78`) fixed and tested the 1D ocean path; this task builds the coupled
BC assembler that TASK-007 (Claude) will wire into `_get_solution_coupled`.
Do not deviate from the interface below without recording it in this file.

This spec is self-contained. You (Codex) have no other context.

## GUARDRAILS — hard limits

1. **Files you may write** (nothing else):
   - `pylov3d/bc_ocean_coupled.py` (create — the existing
     `boundary_conditions.py` is near the 500-line cap, do NOT touch it)
   - `pylov3d/tests/test_bc_ocean_coupled.py` (create)
   - This file (append to Completion notes only)
2. No state-changing git (commit/add/push/pull/branch/checkout/reset/stash).
3. No environment changes (installs, venv edits, shims, substitute envs).
   The only runner is `venvLOV3Dconv/bin/python` from the repo root; if it is
   broken, STOP and report in Completion notes.
4. `src/` and `tests/*.mlx` are read-only MATLAB reference — targeted reads
   fine, never modify. Never touch `docs/HANDOFF.md`.
5. No deletions; no files beyond the allowlist; 500-line cap per file.
6. On ambiguity or apparent spec error: stop, record the question in
   Completion notes, finish only the unambiguous parts.
7. Full suite stays green: `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`
   (currently **341 passed**). Regressions are reported, not force-fixed.

## Goal

Implement `assemble_bc_ocean_coupled` — the multi-mode (N modes) version of
the 1D ocean boundary-condition assembler — as a faithful transcription of
MATLAB `src/get_solution.m:696-838` (the 24N branch; system sizing at
:609-617, forcing RHS at :833-836).

References you must read first:
- `pylov3d/boundary_conditions.py:268-438` — `assemble_bc_ocean` (the 1D
  N=1 version; recently verified faithful to MATLAB row by row). Your
  function must reduce to it exactly at N=1.
- `pylov3d/boundary_conditions.py:135-261` — `assemble_bc_no_ocean_coupled`
  (argument conventions, per-mode n/m handling, forcing normalization).
- `src/get_solution.m:696-838` — the coupled ocean BC assembly itself.

## Interface (fixed — TASK-007 consumes this)

```python
def assemble_bc_ocean_coupled(
    Y_cmb,            # (8N, 8N) fundamental matrix at the CMB (identity)
    Y_surf,           # (8N, 8N) shell-segment matrix at the surface
    Y_ocean_start,    # (8N, 8N) sub-ocean-segment matrix at the ocean floor
    Y_ocean_end,      # (8N, 8N) ocean-segment matrix at the ocean ceiling
    n_s, m_s,         # (N,) mode degrees and orders
    gc, Rc, rho2, rhoK_surface, Gg, rho1,   # same meaning as the siblings
    gO, gI, rhoO, rho_below_ocean, rho_above_ocean,  # same as assemble_bc_ocean
    forcing,          # Forcing or list[Forcing]
) -> tuple[np.ndarray, np.ndarray]:   # B (24N, 24N), B2 (24N, 1) or (24N,)
```

Note the argument order follows `assemble_bc_ocean` (Y matrices first, in
the order cmb/surf/ocean_start/ocean_end — check the actual 1D signature and
match its style; record the final order in Completion notes).

**Column layout (fixed):** columns `[0:8N)` = sub-ocean segment coefficients,
`[8N:16N)` = ocean segment, `[16N:24N)` = shell segment — matching MATLAB's
`j` / `8*Nmodes+j` / `16*Nmodes+j` and the solver's future
`C_below/C_ocean/C_shell` split.

**Row layout (fixed):** per-mode 24-row blocks exactly as MATLAB: mode k
(0-based) owns rows `24k .. 24k+23`, holding BC1-BC24 in the same order as
the 1D `assemble_bc_ocean` (CMB×4, surface×4, ocean floor×3, floor potential
×2, in-ocean zeros×6, ocean ceiling×3, ceiling potential×2). Forcing RHS
goes to row `24k + 7` for the mode(s) matching the forcing (n, m), with the
same normalization as the siblings.

Per-mode values: everywhere the 1D assembler uses the scalar degree n, the
coupled one uses `n_s[k]` for mode k (and `m_s[k]` for the forcing match).
Y-matrix slicing: the 1D rows index components 0..7 of an 8×8 Y; here the
mode-k components live at the same offsets within mode-k's blocks of the 8N
state — mirror how `assemble_bc_no_ocean_coupled` indexes (U_k = 3k,
R_k = 3N+3k, Phi_k = 6N+2k, dPhi_k = 6N+2k+1 — verify against that function
and MATLAB before assuming).

## Tests — `pylov3d/tests/test_bc_ocean_coupled.py`

1. **N=1 exact reduction (the load-bearing test):** with `n_s=[2]`,
   `m_s=[0]`, random complex well-conditioned 8×8 Y matrices (fixed seed)
   and physical-ish scalars, `assemble_bc_ocean_coupled` must equal
   `assemble_bc_ocean` entry-for-entry (max abs diff < 1e-14) for both B and
   B2. Repeat for `n_s=[3], m_s=[1]` and for a list-of-forcings input.
2. **Multi-mode structure:** for `n_s=[2,4], m_s=[0,0]` with random Ys:
   - shape (48, 48); B2 nonzero only at row 7 (the n=2,m=0 mode) when the
     forcing is (2,0);
   - rows `24k..24k+23` reference only mode-k component rows of the Y
     matrices (build Ys that are zero except mode-k's rows/cols to verify
     block separation);
   - the 6 in-ocean-zero rows of each mode contain exactly one 1.0 in the
     ocean column block.
3. **Degenerate sanity:** B is finite everywhere (no NaN/inf) for n_s
   containing a degree-1 mode.

Done-criteria: new tests pass, full suite stays at 341 + yours.
Append to Completion notes: what you built, final signature, test counts,
max N=1 reduction error, file line counts, any deviations.

## Completion notes

- `[CODEX][2026-08-02]` Implemented `assemble_bc_ocean_coupled` as a faithful
  24N×24N transcription with contiguous BC1–BC24 mode-row blocks, grouped
  coupled-state indexing, and `[below, ocean, shell]` coefficient columns.
  Final signature order is exactly `Y_cmb, Y_surf, Y_ocean_start,
  Y_ocean_end, n_s, m_s, gc, Rc, rho2, rhoK_surface, Gg, rho1, gO, gI,
  rhoO, rho_below_ocean, rho_above_ocean, forcing`. Focused suite: 5 passed.
  Full current-tree suite: 353 passed, 5 skipped (the five pre-existing
  TASK-008 checks await TASK-007; the spec's 341 baseline predates TASK-008).
  Maximum N=1 reduction error across B and B2: `0.0`. Final line counts:
  `bc_ocean_coupled.py` 125, `test_bc_ocean_coupled.py` 114. No deviations.
