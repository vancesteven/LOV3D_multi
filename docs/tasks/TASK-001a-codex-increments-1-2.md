# TASK-001a — Codex handoff: JAX coupled propagator, increments 1–2

**Status:** IN-PROGRESS (Codex 5.6)
**Owner:** CODEX
**Parent:** TASK-001 (`docs/tasks/TASK-001-jax-coupled-3d.md`). Increments 3–4
(lax.scan loop + public API) stay with Claude and depend on this task's
interfaces — do not deviate from the contracts below without recording the
deviation in this file.

This spec is self-contained. You (Codex) have no other context.

## Repository facts

- Repo: `/Users/svance/LOV3D_multi`, branch `python-conversion`.
- Python: **always** `venvLOV3Dconv/bin/python` (has JAX 0.10.0 + pytest; the
  base conda python has neither).
- Test command: `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`
  (currently 318 passed; keep it green).
- `matlab/` is reference material — never read, grep, or modify it.
- Do NOT commit. Leave changes in the working tree; Claude reviews, then the
  user approves the commit.
- New files only: `pylov3d/jax_coupled.py` and
  `pylov3d/tests/test_jax_coupled_build.py`. Do not modify existing files.
  Keep each file under 500 lines.

## Goal

Port the NumPy coupled-propagator *assembly* to JAX so it can later run inside
`jax.lax.scan`. Reference implementation to match numerically:
`build_aprop_coupled` in `pylov3d/propagator.py:497-623`, plus its helpers
`build_A1_A2_coupled` (:448), `_coupling_A1_A2` (:414), `_a1a2_geometric`
(:340), `build_A1_A2` (:25), `build_others` (:182). Study the existing 1D JAX
port `pylov3d/jax_scan.py` for conventions (x64 config at import, complex128,
static-dict pattern).

## Key mathematical fact (drives the whole design)

Every traced quantity enters **linearly** as a scalar coefficient on static
(geometry-only) matrices:

- Diagonal A1/A2 blocks (`build_A1_A2`): row 0 of each 6×3 block scales with
  `(3*lam + 2*muC)`; rows 1–5 scale with `2*muC`. The unit-material factors
  are exactly `_a1a2_geometric(n)` (see its docstring).
- Off-diagonal coupling (`_coupling_A1_A2`): slot 0 contributes
  `K_amp[ireo] * Cp[0] * A1g(na)[0,:]` to row 0; slots 1–25 contribute
  `2 * muC_amp[ireo] * Cp[slot] * A1g(na)[source,:]` to row `target`
  (`_COUP_ROWS` mapping). All `Cp` values and geometric rows are static.
- `build_others`: A71, A72, A81, A82, A11, A12 are linear in `rho` (A11/A12
  also carry the constant `4*pi*Gg`); A13, A9, A100, A101, A102 depend only
  on `n`.

Therefore: precompute static tensors once in NumPy (increment 1), and the
traced build (increment 2) is scalar-times-matrix sums plus two `jnp.einsum`
contractions. Do NOT port the Python loops of `build_A1_A2_coupled` into
traced code, and do NOT reproduce the `if km == 0 and mm == 0: continue`
short-circuit or the `if Cp[slot] != 0` guards in traced code — in the
precompute they just mean some static entries are zero, which is harmless.

## Increment 1 — `_precompute_coupled_static(n_s, Coup, Gg) -> dict`

Inputs: `n_s` (N,) int array of mode degrees; `Coup` (N, N, 27, Nreo) complex
coupling tensor; `Gg` float. All NumPy work; convert to `jnp.array(...,
dtype=jnp.complex128)` at the end. Required dict keys (exact names — Claude's
increments 3–4 will consume them):

| Key | Shape | Content |
|---|---|---|
| `G1_bulk`, `G2_bulk` | (6N, 3N) | block-diagonal, block i = row 0 of `_a1a2_geometric(n_s[i])` (other rows zero) |
| `G1_shear`, `G2_shear` | (6N, 3N) | block-diagonal, block i = rows 1–5 of `_a1a2_geometric(n_s[i])` (row 0 zero) |
| `G1_K`, `G2_K` | (Nreo, 6N, 3N) | slot-0 coupling: for each (i, j, ireo) with `Coup[i,j,26,ireo] != 0`, add `Cp[0] * A_g(n_s[j])[0,:]` into rows `6i`, cols `3j:3j+3` |
| `G1_mu`, `G2_mu` | (Nreo, 6N, 3N) | slots 1–25: add `2 * Cp[slot] * A_g(n_s[j])[source,:]` into row `6i + target`, cols `3j:3j+3`, using `_COUP_ROWS` for target/source |
| `A3_inv` | (3N, 3N) | block-diagonal inverse of per-mode `build_A3(n)` |
| `A4`, `A5` | (3N, 6N) | block-diagonal per-mode `build_A4` / `build_A5` |
| `A13` | (3N, 3N) | identity |
| `A9` | (2N, 2N) | identity |
| `A100`, `A101`, `A102` | (2N, 2N) | block-diagonal, per-mode entries from `build_others` (n-dependent only) |
| `P71`, `P72` | (3N, 3N) | unit-rho patterns of A71/A72 (i.e. `build_others(n, rho=1, Gg)` blocks) |
| `P81`, `P82` | (3N, 2N) | unit-rho patterns of A81/A82 |
| `P11`, `P12` | (2N, 3N) | unit-rho patterns of A11/A12 (these DO include the `4*pi*Gg` factors) |
| `deg0_modes` | Python list[int] | mode indices k with `n_s[k] == 0` (plain list, not a jnp array) |

Reuse the existing NumPy helpers (`_a1a2_geometric`, `build_A3`, `build_A4`,
`build_A5`, `build_others`, `_COUP_ROWS`) — import them from
`pylov3d.propagator`; do not re-derive the formulas.

## Increment 2 — `_build_aprop_coupled_jax(r, g, dg, muC, lam, rho, muC_amp, K_amp, static) -> (8N, 8N)`

Traced scalars: `r, g, dg, muC, lam, rho`; traced vectors: `muC_amp, K_amp`
(each (Nreo,) complex). `static` is the dict from increment 1. Must be
traceable under `jax.jit` (no Python branching on traced values).

Assembly:

```
A1 = (3*lam + 2*muC) * G1_bulk + 2*muC * G1_shear \
     + jnp.einsum('r,rij->ij', K_amp, G1_K) \
     + jnp.einsum('r,rij->ij', muC_amp, G1_mu)     # same pattern for A2
A71 = rho * P71   (likewise A72, A81, A82, A11, A12 from their P patterns)
```

Then mirror the block layout of `build_aprop_coupled` (propagator.py:576-597)
**exactly**, including the A1/A2 role swap (A2 → Adotx, A1 → Ax) and the
`A6 = 0` omission. Then apply the degree-0 special case (propagator.py:599-619)
for each k in `static["deg0_modes"]` using `.at[...].set` row overrides —
these mode indices are static, so a Python loop over them at trace time is
correct. Finally `Aprop = jnp.linalg.solve(Adotx, Ax)`.

## Tests — `pylov3d/tests/test_jax_coupled_build.py`

Find the existing coupled-solver test fixture (grep `pylov3d/tests/` for
`coupled` — there is an Io 3-layer lateral-variation pipeline) and reuse its
model/couplings/lateral setup to get real `n_s`, `Coup`, `muC_amp`, `K_amp`.
Required tests:

1. **Direct equivalence:** at ≥3 radii spanning CMB to surface (with the
   matching `g`, `dg` from `pylov3d.propagator.compute_gravity` and the
   correct per-layer material values), `_build_aprop_coupled_jax` matches
   NumPy `build_aprop_coupled` with max relative error < 1e-10.
2. **Zero-amplitude reduction:** with `muC_amp = K_amp = 0`, the coupled JAX
   build equals NumPy `build_aprop_coupled` called with zero amplitudes
   (coupling terms vanish; block-diagonal structure).
3. **JIT smoke test:** wrap the build in `jax.jit` and confirm same result
   (first call may take ~10–30 s to compile — that is normal).
4. If the fixture includes a degree-0 mode, assert the overridden rows match;
   if not, construct a minimal `n_s` containing 0 to cover the degree-0 path
   (a synthetic `Coup` of zeros is acceptable for this sub-test).

Done-criteria: all new tests pass AND the full suite stays green:
`venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`.

When finished, append a short completion note (what was built, test counts,
any interface deviations) to the bottom of this file.

## Completion notes

- `[CODEX][2026-08-01]` Implemented static geometry/coupling precomputation
  and traceable 8N×8N JAX assembly in `pylov3d/jax_coupled.py`, with direct,
  zero-amplitude, JIT, degree-0, and nonzero K/mu coupling coverage in
  `pylov3d/tests/test_jax_coupled_build.py`. Focused tests: 4 passed. Full
  suite: 314 passed, 1 skipped (green; historical handoff count was 318).
  Interface deviation: added `static["Gg"]` because the required builder
  signature omits `Gg` but the degree-0 Poisson override requires it.
  Verification used an isolated JAX 0.10.0 environment because the prescribed
  `venvLOV3Dconv/bin/python` is absent in this clone; a temporary SymPy-backed
  `py3nj` shim replaced the unavailable compiled dependency for test execution.
  MATLAB is available only on Machine B, so any MATLAB-side validation should
  be handed off there after Machine A's Claude review.
