# TASK-001 — 3D coupled (8N×8N) JAX scan port

**Status:** IN-PROGRESS — increments 1–2 delegated to Codex (TASK-001a); increments 3–4 with Machine A, blocked until 001a verified
**Owner:** A (increments 3–4) / CODEX (increments 1–2, see `TASK-001a-codex-increments-1-2.md`)
**Origin:** Machine B plan, 2026-08-01 (moved out of HANDOFF.md by Machine A)

> **Design revision (Machine A, 2026-08-01):** the A1/A2 "unroll mode×slot
> loops at trace time" strategy below is superseded. All traced quantities
> (muC, lam, rho, muC_amp, K_amp) enter *linearly* as scalar coefficients on
> static geometry, so the traced build is precomputed-tensor contractions
> (`jnp.einsum`) — no trace-time unroll, compile cost flat in N. Interface
> contract lives in TASK-001a.

Extends `jax_scan.py` (8×8) to the coupled **8N×8N** system (N = # GSH modes, ~5–20).
NumPy reference to match: `solver._get_solution_coupled` (pylov3d/solver.py:~390–542)
and `propagator.build_aprop_coupled` (pylov3d/propagator.py:497–623).

## Mapping (static vs traced)

| Piece | Static/traced | Strategy |
|---|---|---|
| N, n_s, m_s | static | compile-time; sets all shapes |
| A3⁻¹ block-diag | static (n_s only) | precompute frozen (3N×3N) |
| `Coup` tensor (N,N,27,Nreo) | static (pre-loop) | frozen array; `if Cp[slot]!=0` resolves at trace time |
| A1/A2 coupled (6N×3N) | geometry static + traced μC/λ | unroll mode×slot loops at trace time into `.at[].set/.add` |
| A71/A72/A81/A82/A11/A12 | traced rho | inline block-diagonal from traced rho (as 1D scan) |
| μC_amp/K_amp | traced per-layer | pass as `xs`; DROP the `if km==0 and mm==0` short-circuit (mul-by-zero safe) |
| degree-0 special case | static per mode | Python conditional at trace time |
| Δρ discontinuity | per-mode at boundaries | `Y.at[dPhi_row,:].add(4π·Gg·Δρ·Y[U_row,:])` looped over N in step |
| BC assembly | — | keep `assemble_bc_no_ocean_coupled` in NumPy |

## Increment plan (new file `pylov3d/jax_coupled.py` + tests)

1. `_precompute_coupled_static(n_s, Coup)` → frozen A3⁻¹, A4, A5, A13, A9, A100/101/102 + coupling geometry.
2. `_build_aprop_coupled_jax(...)` — 8N×8N assembly, traced (r,g,dg,μC,λ,rho,μC_amp,K_amp); mirror A1/A2 role-swap + degree-0.
3. `_make_jit_scan_coupled(static, n_s, Gg)` — lax.scan over radius, Δρ correction in step.
4. `propagate_coupled_jax_scan` / `jax_get_love_coupled_scan` — drop-in for `_get_solution_coupled`; k-vector via NumPy BC solve.

## Verification (reviewers = Opus tier)

- scan vs NumPy `_get_solution_coupled` rel 1e-5 on the Io 3-layer lateral pipeline (`test_love_coupled` fixture).
- N=1 single-mode must match the 1D scan.
- multi-mode N≥3 with non-trivial Im(k).

**Risk:** A1/A2 coupled unrolling (27 slots × N² pairs) is the delicate part; XLA compile grows with N (one-time).
