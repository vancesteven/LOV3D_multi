# pylov3d — Machine Handoff & Coordination Channel

> **Purpose:** Shared coordination file for two Claude instances working the
> pylov3d port on separate machines. Both machines read this file at the start
> of a work session and update it at the end. Machine A owns the coordination
> protocol design (see [Open request to Machine A](#open-request-to-machine-a)).

---

## Roster

| Machine | Model | Role |
|---|---|---|
| **A** | fable | Active driver (this handoff hands control TO Machine A). Owns coordination-scheme design. |
| **B** | Opus 4.8 | Prepared this handoff, then yields. |

---

## Git state at handoff

- **Branch:** `python-conversion`
- **Tracking / push target:** `myfork` = `github.com/vancesteven/LOV3D_multi.git`
  - ⚠️ **Do NOT push to `origin`** (`github.com/mroviranavarro/LOV3D_multi.git` — that is upstream, not ours). Stay on `myfork`.
- **HEAD (pushed to myfork):** `3c41217` — "Add JAX lax.scan JIT propagator increment (jax_scan.py)"
- **Working tree:** clean except untracked `.claude-flow/` and `CLAUDE.md` (intentionally not committed).
- **Test suite:** `318 passed` (via `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`).

### Recent commits
```
3c41217 Add JAX lax.scan JIT propagator increment (jax_scan.py)
fe1b363 Begin JAX port: 1D propagator first increment + port plan (Milestone 4)
e35815a Add independent PyALMA3 cross-validation benchmark (Milestone 4)
b7ecd1f Add PlanetProfile compatibility adapter
```

---

## Environment notes (important — non-obvious)

- **Python venv:** `venvLOV3Dconv/bin/python` (JAX 0.10.0, pytest installed here).
  The base conda python has **no pytest** and no jax. Always use the venv.
- Run tests: `venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q`
- JAX x64 is enabled at import (`jax.config.update("jax_enable_x64", True)`);
  complex128 works on the CPU backend. First JIT call is slow (~10–30s), then fast.
- `matlab/` is source material only — **do not scan/grep/modify** it.

---

## Project status

- ✅ M1: 1D spherically symmetric solver
- ✅ M2: 3D lateral variations with mode coupling
- ✅ M3: MATLAB cross-validation (Enceladus k2 rel err 6.3e-4; A1/A2 propagator fix)
- 🔄 **M4 (in progress):** independent benchmark + JAX optimization
  - ✅ PyALMA3 cross-validation benchmark (`test_benchmark_pyalma3.py`, 8 tests). Gotcha: ALMA normalizes time by t0=1000yr=3.1558e10 s.
  - ✅ JAX 1D Python-loop increment (`jax_propagator.py`)
  - ✅ **JAX 1D lax.scan JIT increment (`jax_scan.py`)** ← newest. Verified: scan vs Python-loop 1e-10, vs NumPy get_love rel 1e-5, vs analytic 1e-3. Multilayer viscoelastic test added.
  - ⬜ **NEXT: 3D (coupled 8N×8N) JAX scan port** — plan below.

---

## NEXT TASK for Machine A: 3D coupled JAX scan port

Extends `jax_scan.py` (8×8) to the coupled **8N×8N** system (N = # GSH modes, ~5–20).
NumPy reference to match: `solver._get_solution_coupled` (pylov3d/solver.py:~390–542)
and `propagator.build_aprop_coupled` (pylov3d/propagator.py:497–623).

**Constraint (project rule):** Do NOT implement before summarizing & proposing to the user.
The plan below was already presented by Machine B — confirm with user before coding.

### Mapping (static vs traced)
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

### Increment plan (new file `pylov3d/jax_coupled.py` + tests)
1. `_precompute_coupled_static(n_s, Coup)` → frozen A3⁻¹, A4, A5, A13, A9, A100/101/102 + coupling geometry.
2. `_build_aprop_coupled_jax(...)` — 8N×8N assembly, traced (r,g,dg,μC,λ,rho,μC_amp,K_amp); mirror A1/A2 role-swap + degree-0.
3. `_make_jit_scan_coupled(static, n_s, Gg)` — lax.scan over radius, Δρ correction in step.
4. `propagate_coupled_jax_scan` / `jax_get_love_coupled_scan` — drop-in for `_get_solution_coupled`; k-vector via NumPy BC solve.

### Verification (per standing rule: reviewers = opus)
- scan vs NumPy `_get_solution_coupled` rel 1e-5 on the Io 3-layer lateral pipeline (`test_love_coupled` fixture).
- N=1 single-mode must match the 1D scan.
- multi-mode N≥3 with non-trivial Im(k).

**Risk:** A1/A2 coupled unrolling (27 slots × N² pairs) is the delicate part; XLA compile grows with N (one-time).

---

## Standing rules (both machines)

- Reviewers/verifiers → **opus** agents. Implementation → sonnet. Recon → haiku. Escalate impl to opus when genuinely complex.
- Do NOT implement code unless explicitly instructed — summarize & propose first.
- Commit only when the user asks. Push to `myfork` only, never `origin`.
- Keep files < 500 lines; tests in `pylov3d/tests/`; no working files in repo root.
- graphify re-runs only with user confirmation: `graphify run pylov3d/ --exclude pylov3d/tests/`

---

## Open request to Machine A

> The user wants **you (Machine A)** to devise a scheme for efficiently
> communicating through `.md` or `.html` files kept up to date by both machines.
> This `docs/HANDOFF.md` is the seed. Design the protocol — e.g. how to signal
> "who holds the token" / active machine, how to append status without clobbering
> the other's edits (sync is via Dropbox, so last-writer-wins is a real risk),
> commit/no-commit conventions for the coordination file, and a heartbeat/log
> section. Propose it to the user, then formalize it here.

---

## Coordination log

_Append newest entries at the top. Format: `[machine][YYYY-MM-DD] note`_

- `[B][2026-08-01]` Pushed `3c41217` to myfork (JAX scan increment + multilayer viscoelastic test; suite 318 green). Wrote this handoff. Yielding to Machine A. Machine A to design the ongoing .md/.html coordination protocol.
