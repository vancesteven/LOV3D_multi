# TASK-035 (Machine B): native-MATLAB anchor for the Moon lateral spectrum

## Why

Every load-bearing Mars result in this project has an independent
native-MATLAB cross-check: the 1-D model (~1e-12), the lateral spectrum
(N=115, 2.95e-13, `8a134e3`), the diagonal k2m solves (7e-12, `78e7c92`),
and the two first-order zonal channels (`cbae776`). **The Moon lateral stage
(`67dd097`) has none.** It is now the only unanchored link, exactly as the
first-order channel result was before TASK-029 closed it.

That matters because the Moon lateral numbers are about to be used: TASK-034
compares the predicted k2m splitting against GRAIL's actual measurement, and
that comparison is only as trustworthy as the spectrum underneath it.

## Method — follow your own TASK-029 improvement

TASK-029's spec suggested extending the existing lateral driver. You did
something better: exported the exact field to `.npz` and had MATLAB read
*that*, so both codes consume the identical field rather than independently
re-deriving it. That isolates the physics from the ingest, which is the right
trade when the ingest already has its own anchor. **Do the same here.**

1. **Export.** A read-only Python step writing the committed Moon crust-layer
   `mu_variable` field (from `pylov3d.moon_lateral`, `lmax=4`, the shipped
   configuration) to `.npz` — the analogue of
   `scripts/export_mars_dwak_mu_variable.py`.
2. **Solve.** Native LOV3D on the as-built ten-layer Weber model —
   **including its fluid outer core**, which is the part with no Mars
   precedent, since the Mars model's core sits at the centre. Use the
   validated settings: `method='variable'`, `Nrbase=30`,
   `perturbation_order=2`, unit `(2,0)` monthly forcing.
3. **Compare** against the committed Python results:
   - uniform Weber `k2` = 0.02315914223
   - lateral forcing-mode `k20` = 0.02316054935, `Delta k20` = +1.40712e-6
   - `N` = 115 coupled modes
   - largest off-forcing pairs: `(2,±2)` 3.13471e-6, `(2,±1)` 2.76868e-6,
     `(3,±3)` 2.01884e-6

## What to watch

- **The fluid outer core is the novel part.** If MATLAB and Python disagree
  anywhere, that is the first place to look, and a disagreement there would
  be a genuine finding rather than a nuisance — the ocean/fluid-layer path
  has its own history in this project (TASK-005 through 009).
- Report agreement per mode, not just on `Delta k20`. A forcing-mode match
  with an off-mode mismatch would be important and easy to miss.
- If agreement is worse than the Mars lateral precedent (2.95e-13), say so
  plainly and investigate rather than reporting a looser tolerance as a pass.

## Constraints

- MATLAB driver plus a read-only Python export; **do not modify
  `pylov3d/moon_lateral.py` or any solver module.**
- Save console output and a small `.mat` to `data/tests/moon/`, matching the
  TASK-020/029 artifact practice, so a MATLAB-less reader can verify.
- Record the MATLAB version.
- Prose standard: never "genuine" or "honest".
- Suite green.

## Done criteria

Per-mode agreement reported against the Python values above; artifacts
committed under `data/tests/moon/`; a note in `docs/MOON_MODEL.md`; and an
explicit statement of whether the Moon lateral stage now has an anchor of the
same quality as the Mars one. Ledger to DONE awaiting VERIFIED.
