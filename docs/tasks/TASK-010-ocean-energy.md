# TASK-010 — Ocean energy dissipation (ocean-ceiling node)

**Owner:** B · **Milestone:** M6 (post-M5 cleanup) · **Status:** IN-PROGRESS

## Goal
Make `energy.py` account correctly for the **ocean-ceiling node** (`ocean_end`)
in tidal-dissipation integrals for ocean-bearing models, in both the 1D
(`compute_stress_strain` / `get_energy`) and coupled
(`compute_stress_strain_coupled` / `get_energy_coupled`) paths.

## Root cause
Both stress/strain builders skip every grid node whose `layer_map` entry is the
ocean layer (ocean flag == 1), including `ocean_end`. After the TASK-005/007
recombination, `y_sol[ocean_end]` holds **shell** data (`C_shell`, the shell's
identity restart at its base), so that node physically belongs to the solid
shell above and should contribute to the shell's dissipation integral. Skipping
it forces `dissipation[ocean_end] = 0`, truncating the base of the shell's
trapezoidal radial integral → underestimated shell heating.

## Complication
`Aprop_aux[ocean_end] = 0.0` in the solver, because that node was integrated
with the ocean (Laplace-only) propagator. Stress/strain at `ocean_end` need the
**shell** propagator's first-3 (`_aux`) rows evaluated at `r = r_grid[ocean_end]`
with the shell layer's material — not the ocean's zeros.

## Candidate approaches (pick one — user decision pending)
- **A — solver-side aux fill (preferred):** in `solver.py`, after the ocean
  loop, overwrite `Aprop_aux[ocean_end]` with the shell propagator's aux rows at
  `r_grid[ocean_end]` (shell material). Then in `energy.py`, treat `ocean_end`
  as belonging to `ocean_layer + 1`. Single source of truth for the propagator;
  small solver change. Must confirm no other consumer of `Aprop_aux[ocean_end]`
  relies on the current zero.
- **B — energy-side recompute:** leave the solver untouched; in `energy.py`,
  detect `ocean_end`, assign shell material, and recompute the shell aux rows
  on the fly. No solver change, but duplicates propagator logic in energy.

## Done criteria
- 1D + coupled ocean models include `ocean_end` in shell dissipation.
- Grid-refinement convergence: the total integral converges (skipped node was a
  fixed truncation, not a discretization one — verify the correction shrinks the
  gap vs a fine-grid reference).
- No change to non-ocean models (regression: existing `test_energy.py` green).
- New tests in `pylov3d/tests/` covering an ocean model 1D and coupled.
- Cross-check against MATLAB `get_energy.m` behavior at the shell origin.
- Suite stays green (currently 377).

## Constraints
- Files < 500 lines; tests in `pylov3d/tests/`; propose before implementing.
- MATLAB `src/` is read-only reference.
