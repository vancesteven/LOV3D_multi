# TASK-002 — Codex handoff: Aprop_aux for the coupled JAX path

**Status:** IN-PROGRESS (Codex 5.6)
**Owner:** CODEX
**Parent:** completes the 3D coupled JAX port (TASK-001, committed `ce3fd7e` + `4b7557a`) by adding the auxiliary stress/strain recovery matrix, making `jax_get_solution_coupled_scan` a full drop-in for the NumPy `_get_solution_coupled`.

This spec is self-contained. You (Codex) have no other context.

## GUARDRAILS — read first, these are hard limits

1. **Files you may write** (nothing else, no exceptions):
   - `pylov3d/jax_coupled.py` (modify, only as described below)
   - `pylov3d/jax_coupled_aux.py` (create)
   - `pylov3d/tests/test_jax_coupled_scan.py` (modify, only as described below)
   - This file (append to Completion notes only)
2. **No git commands that change state**: no commit, add, push, pull, branch, checkout, reset, stash. Read-only git (status, diff, log, show) is fine.
3. **No environment changes**: no pip/conda/brew installs, no venv creation or modification, no shims or substitute environments. The only test runner is `venvLOV3Dconv/bin/python` from the repo root. It exists and has jax 0.10.2, py3nj, pytest, pyalma3, matplotlib. If it is missing or broken, STOP and record that in Completion notes — do not improvise an environment.
4. **Never read, grep, or modify `matlab/`.** Never modify `docs/HANDOFF.md` or other files under `docs/`.
5. **No file deletion. No new files beyond the one listed.** Keep every file you touch under 500 lines.
6. **If the spec is ambiguous or an instruction seems wrong**, stop at that point, write the question and what you observed into Completion notes, and finish whatever unambiguous parts remain. Do not invent interface changes beyond the recorded-deviation mechanism.
7. Full suite must stay green: `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q` — currently **329 passed**. Any regression = stop and report, do not force-fix unrelated tests.

## Goal

NumPy reference `solver._get_solution_coupled` (pylov3d/solver.py:391-542) returns `(y_sol, r_grid, Y, Aprop_aux)` where `Aprop_aux` has shape `(Nr+1, 3N, 8N)`: the first 3N rows of the coupled propagator matrix evaluated at every grid point (solver.py:450-505). The JAX port `jax_get_solution_coupled_scan` (pylov3d/jax_coupled.py) currently returns only the first three. Add `Aprop_aux` so the JAX function returns the same 4-tuple.

Reference semantics (read solver.py:450-505 carefully):
- **Index 0 (CMB, r=Rc):** built with **layer 1** properties: `muC[1]`, `lam[1]`, `rho[1]`, amplitudes `muC_amp[1,:]`/`K_amp[1,:]`, and gravity from `compute_gravity(Rc, rho[1], M_at_boundary[1], R[0], Gg)`.
- **Index k≥1:** built with the current point's layer properties (`i_layer = layer_map[k]`) at radius `r_curr = r_grid[k]`, gravity from `compute_gravity(r_curr, rho_k, M_inner_k, R_inner_k, Gg)` (solver.py:497-499).
- Stored rows: `Ap[:3N, :]` of the full 8N×8N propagator.

## Implementation plan

### Step 1 — factor per-point arrays out of `propagate_coupled_jax_scan`

In `pylov3d/jax_coupled.py`, extract the existing grid/property construction (radial grid, `layer_map`, `M_at_boundary`, and the per-step xs arrays, currently inside `propagate_coupled_jax_scan`) into a module-level helper:

```
_grid_and_props(model, numerics, couplings, lateral) ->
    (r_grid, layer_map, M_at_boundary, xs_arrays_dict)
```

`propagate_coupled_jax_scan` must call this helper and keep **numerically identical** behavior (the delta-rho `prev_layer=1 if k_idx==1` convention included). This is a pure extraction — do not change any formula. The existing tests (Y match to 1e-12) protect you.

### Step 2 — new module `pylov3d/jax_coupled_aux.py`

Public function:

```
compute_aprop_aux_coupled(model, forcing, numerics, couplings, lateral)
    -> Aprop_aux (Nr+1, 3N, 8N) complex128 numpy array
```

- Use `_grid_and_props` + `_get_cached_scan`'s static dict (import `_get_cached_scan` from `.jax_coupled` and take its first tuple element — this reuses the memoized static tensors) OR call `_precompute_coupled_static` behind your own small memo; prefer reusing `_get_cached_scan`.
- Build per-point (length Nr+1) arrays r, g, dg, muC, lam, rho, muC_amp, K_amp per the reference semantics above (note the CMB layer-1 special case at index 0; `compute_gravity` imported from `pylov3d.propagator`).
- Evaluate with a jitted `jax.vmap` of `_build_aprop_coupled_jax` over those arrays, slice `[:, :3N, :]`, return as numpy. Memoize the jitted vmapped builder the same way `_SCAN_CACHE` does (keyed on `(n_s.tobytes(), Coup.tobytes(), Gg)`), so repeated calls do not recompile.

### Step 3 — 4-tuple return

`jax_get_solution_coupled_scan` returns `(y_sol, r_grid, Y, Aprop_aux)` by calling `compute_aprop_aux_coupled` (import inside `jax_coupled.py`; if that creates a circular import, do the aux call in `jax_coupled_aux.py` via a thin wrapper and note it). Update the docstring (remove the "does not compute Aprop_aux" caveat). Update the existing unpack sites in `pylov3d/tests/test_jax_coupled_scan.py` (fixture and the K_amp test) to the 4-tuple.

## Tests (extend `pylov3d/tests/test_jax_coupled_scan.py`)

1. Fixture: capture the reference `Aprop_aux` (currently discarded as `_`) and the JAX one from the 4-tuple.
2. `test_aprop_aux_matches_reference`: shape equal AND max relative error (global-max normalized, same `_relative_error` helper) < 1e-12 over the full `(Nr+1, 3N, 8N)` array.
3. In the K_amp injection test, additionally assert the Aprop_aux from the K-injected case matches its NumPy reference < 1e-12 (this exercises the CMB layer-1 amplitude row and per-layer amp selection in the aux path).

## Done criteria

`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/test_jax_coupled_scan.py pylov3d/tests/test_jax_coupled_build.py -q` green, then full suite green (expect 329 + new tests). Append to Completion notes: what you built, test counts, measured Aprop_aux max relative error, file line counts, any deviations.

## Completion notes

_(Codex appends here.)_
