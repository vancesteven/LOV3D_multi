# JAX Port Plan: 1D Propagator Hot Path

## Goal

Port the 1D (spherically-symmetric, uncoupled) forward propagation
pipeline to JAX so that it can be JIT-compiled and (later) batched across
many planetary models or tidal frequencies in a single device call.

---

## 1. Hot-path identification

The dominant cost in a single Love-number evaluation is the radial loop in
`solver.get_solution`, which calls:

1. `propagator.build_aprop` — constructs the 8×8 propagator matrix at a
   given radius by assembling A1–A5 sub-matrices and solving a linear
   system (`Adotx \ Ax`).  Called **6 times per RK step** (6 stages of
   Cash-Karp).
2. `solver.cash_karp_increment` — one RK5 step: 6 matrix-product chains
   (each an 8×8 matmul), one linear combination.
3. The outer loop over `Nr` radial points (typically 500–2000 for
   single-layer, 2–4× more for multi-layer).

The matrix is 8×8 and complex.  For a typical Nrbase=500 run there are
~3000 `build_aprop` evaluations.  Each involves several 3×3/6×3 numpy
matmuls, a 6×3 inverse, and an 8×8 `np.linalg.solve`.  These are fine
individually but have high per-call Python overhead when looped.

**JIT over the entire radial loop is therefore the main leverage point.**

The sub-matrix builders (`build_A1_A2`, `build_A3`, etc.) contain Python
`if n <= 0` branches.  For a single mode with fixed `n=2` (the standard
tidal degree), these branches are static at trace time and can be removed
or handled via `jax.lax.cond`.  For the first increment, `n=2` is hard-
wired to avoid the branching complexity.

---

## 2. JIT / vmap / scan strategy

### `build_aprop_jax`

Implement the 8×8 assembly as a pure function of `(r, g, dg, muC, lam,
rho, Gg)` with `n=2` specialization (the forced tidal degree).
All sub-matrix constants that depend only on `n` are precomputed once and
passed as static arrays; only `r`, `g`, `dg`, and material parameters are
traced.  The result is a single `jnp.linalg.solve(Adotx, Ax)`.

### `cash_karp_step_jax`

Implements one RK5 stage using `lax.fori_loop` stage unrolling (or just
a fixed sequence of 6 explicit calls).  Returns `(Y_new, Aprop_at_r)`.
Full unrolling is preferred at size-8 because loop overhead in JAX is
negligible for fixed-length loops.

### `propagate_1d_jax`

The key radial loop is expressed as `jax.lax.scan`:

```python
def step(carry, inputs):
    Y_prev, ... = carry
    r_curr, muC_k, lam_k, rho_k, M_inner_k, R_inner_k = inputs
    dr = r_curr - r_prev
    inc, Ap = cash_karp_step_jax(r_prev, dr, n, muC_k, lam_k, rho_k, Gg,
                                  M_inner_k, R_inner_k)
    Y_new = (I8 + inc) @ Y_prev
    return (Y_new, r_curr), Y_new
```

`lax.scan` XLA-compiles the full loop body into a single kernel.  The
carry thread `r_prev` avoids storing the full `Y` trajectory if only
`Y_surf` is needed for Love-number extraction (memory-efficient).  The
stacked outputs are saved when `Y_all` is required for energy integrals.

### Density discontinuity at layer boundaries

In the NumPy code, `Y_old[7, :] += 4π·Gg·ΔΡ · Y_old[0, :]` at layer
boundaries.  Inside `lax.scan`, this is handled by providing a per-point
`delta_rho` array (non-zero only at boundary indices) and applying:

```python
Y_corrected = Y_prev.at[7, :].add(4*jnp.pi*Gg * delta_rho[k] * Y_prev[0, :])
```

This is fully vectorizable with no Python branching.

### vmap strategy

Once `propagate_1d_jax` is compiled, it can be `vmap`-ped over:
- A batch of `(muC, lam)` values for frequency sweeps (e.g., variable Td).
- A batch of interior models for ensemble calculations.

### Boundary conditions

`assemble_bc_no_ocean` and `np.linalg.solve(B, B2)` remain in NumPy for
the first increment.  They run once per evaluation (not in the hot loop)
and JAX-ifying them is a lower priority.

---

## 3. Complex-dtype and float64

JAX defaults to float32.  We use:

```python
jax.config.update("jax_enable_x64", True)
```

This must be set **before** any JAX import in the module that uses complex128.
`constants.py` already calls this at import time, so any module importing
from `pylov3d` inherits it.  The JAX propagator explicitly uses
`jnp.complex128` in array constructors to be safe.

---

## 4. Static vs traced shapes

| Variable | How used | Notes |
|---|---|---|
| `n` (degree) | Static — used in `n <= 0` checks in A-matrix builders | Hard-wire `n=2` in first increment; generalize with `lax.cond` later |
| `Nrbase`, `Nr` | Static at `jit` boundary | Must not change between calls; use `jax.jit(static_argnums=...)` if variable |
| `n_layers` | Static | Never traced |
| `8×8` matrix shapes | Fully static | XLA handles fixed-shape matmuls optimally |
| `r_grid`, `muC_k`, etc. | Traced | Vary continuously; pass as arrays |

For the scan, `r_grid` and the per-point material property arrays are
pre-built as JAX arrays of shape `(Nr,)` and passed as `xs` to `lax.scan`.

---

## 5. Extension to mode coupling (8N×8N)

The coupled propagator (`build_aprop_coupled`) replaces the single-mode
8×8 system with an 8N×8N system.  N is typically 5–20 for near-resonant
modes.  Extending the JAX port:

- `N` must be a static compile-time constant (or handled with a fixed
  maximum size + masking).
- `build_A1_A2_coupled` involves loops over mode pairs; these translate
  directly to JAX vectorized indexing with scatter-add.
- The `lax.scan` strategy is unchanged; only the carry shape grows.
- For N ≤ 16, XLA can still unroll the 8N×8N matmuls efficiently.
- A practical approach: compile a separate kernel per N value (N=1, N=3,
  N=7 etc.) using `functools.partial` and `jit`.

The Wigner coupling coefficients (`couplings.Coup`, shape `(N, N, 27,
Nreo)`) are static inputs (computed once before the loop) and passed as
frozen arrays into the compiled function.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| `n <= 0` Python branch in A-matrix builders | Hard-wire `n=2` for first increment; wrap with `lax.cond` later |
| `np.linalg.inv(A3)` inside loop | Precompute `A3_inv` once (it depends only on `n`, which is static) |
| `np.linalg.solve(Adotx, Ax)` — JAX fallback to XLA `jnp.linalg.solve` | Verified to work; slightly slower than LAPACK for 8×8 but still beneficial when amortized over scan |
| Dropbox sync latency on first JIT compile | First call is slow (~5–30s); subsequent calls are fast. Tests use `jax.block_until_ready`. |
| complex128 on macOS Metal backend | Tested: JAX 0.10.0 on macOS uses CPU backend by default; complex128 is fully supported |
| Numerical difference from NumPy due to FP order-of-operations | Tolerance set to 1e-5 relative (not 1e-12) to accommodate XLA FP reordering |

---

## 7. First increment scope

The file `pylov3d/jax_propagator.py` implements:

1. A3 precomputation for n=2 (static).
2. `build_aprop_jax(r, g, dg, muC, lam, rho, Gg)` — 8×8 propagator, n=2 only.
3. `cash_karp_step_jax(...)` — one RK5 step returning `(inc, Aprop_at_r)`.
4. `propagate_1d_jax(model, forcing, numerics)` — full radial scan returning
   `Y_surf` (and intermediate `Y_all` if requested).
5. `jax_get_love_k2(model, forcing, numerics)` — convenience wrapper that
   calls the boundary-condition assembly in NumPy and returns the complex k2.

Test `pylov3d/tests/test_jax_propagator.py` verifies:
- JAX k2 ≈ analytic k2 = 0.038704 to 1e-6.
- JAX k2 ≈ NumPy `get_love` k2 to rel 1e-5.
