# pylov3d scientific validation workflow

This is the canonical record of **test order, scientific rationale, and current
acceptance state** for the Python conversion. `docs/SCIENCE_VALIDATION.md` is the
compact publication-facing matrix; `docs/tasks/TASK-046-diagnostic-log.md`
preserves the chronological failed-run history.

The order matters. Later stages combine more physics and are harder to diagnose.
A failure at a later stage must not be interpreted as invalidating earlier stages
that have independent references.

## Validation ladder

### Stage 1 — local algebra and bookkeeping: PASS

Run focused invariants before any expensive planetary comparison:

```bash
pytest -q \
  pylov3d/tests/test_energy_multibasis.py \
  pylov3d/tests/test_energy_couplings_matlab_order.py
```

These tests establish native per-forcing basis support, quadratic forcing
scaling, and MATLAB column-major grouping in the GSH energy-coupling tensor.

Latest recorded focused results include:

```text
2 passed in 0.99 s
1 passed after near-machine-precision tolerance correction
```

The original exact-equality regression failed only at `2.6e-16` relative from
floating-point summation order; the corrected regression still rejects the
substantively wrong C-order grouping.

### Stage 2 — publication-facing core matrix: previously PASS, rerun pending

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

Last complete recorded summary:

```text
58 passed in 88.57s (0:01:28)
```

A later expanded run emitted a complete stream of passing tests but its final
pytest summary was not captured. The next final promotion of TASK-046 should be
followed by a full rerun with commit and dependency provenance.

### Stage 3 — authoritative Io physical closure: PASS

The original upstream `Consistency_test_Energy.m` supplies full `mu_latlon` and
`eta_latlon` fields. Reproducing that native raw-grid path gives:

```text
retained asthenosphere rheology modes: 6
retained degree range: 2..4
active solution counts m=[0,-2,+2]: [43 41 41]
```

The retained modes are

```text
(2,-2), (2,0), (2,2), (4,-2), (4,0), (4,2)
```

The production Python rheology processor now reproduces this six-mode spectrum
and `[43,41,41]` closure. The earlier `[125,125,125]` coefficient-input result
is a basis-mismatch artifact and must not appear in physical acceptance criteria.

Regression:

```bash
pytest -q pylov3d/tests/test_io_rheology_spectrum_parity.py
```

### Stage 4 — uniform radial fields and energy contraction: PASS

At `Nrbase=50`, the point-by-point MATLAB/Python comparison established:

```text
k relerr                     = 3.08e-11
state U..dPhi relL2          = 3.58e-11
u_GSH relL2                  = 3.96e-11
stress relL2                 = 3.85e-11
strain relL2                 = 5.42e-11
E00 profile relL2            = 7.56e-11   (excluding outermost row)
Python E00                   = 2.166877846599
MATLAB E00                   = 2.166877841569
best profile scale           = 0.999999999956
```

The large earlier energy discrepancy was traced to NumPy using C-order for the
translation of MATLAB `reshape(Caux,9,[])`. The corrected code uses
`reshape(..., order='F')`. The only residual all-row discrepancy is MATLAB's
explicitly zeroed outermost auxiliary stress/strain row.

Diagnostic:

```bash
python scripts/io_compare_uniform_radial_anchor.py
```

### Stage 5 — raw-grid MATLAB Gate C: PASS within measured transform floor

Authoritative MATLAB raw-grid anchor at `Nrbase=50`:

```text
active counts: [43 41 41]

m=+0 k_lat = +7.331226411605e-01 -1.477975363870e-02 i
m=-2 k_lat = +7.343584891776e-01 -1.557193356211e-02 i
m=+2 k_lat = +7.343584891750e-01 -1.557193356303e-02 i

direct energy uniform/lateral = 2.166877841569 / 2.228354218886
Love energy   uniform/lateral = 2.214402434848 / 2.277217563440
mismatch uniform/lateral      = 2.14615883% / 2.14574775%
```

Final Python raw-grid Gate C run:

```text
native lateral mode counts: [43, 41, 41]

relerr k_lat:
  m=0   1.687e-4
  m=-2  1.902e-4
  m=+2  1.902e-4

relerr E_direct lateral = 7.542e-3
relerr E_Love   lateral = 7.541e-3

direct/Love mismatch:
  uniform 2.1462%
  lateral 2.1457%

MATLAB raw-grid Gate C assertions: PASS
  (within documented raw-grid transform floor)
```

Run with:

```bash
python scripts/io_energy_gate_bc_multibasis.py --nrbase 50 --assert-matlab
```

#### Why this gate is 1%, not 1e-8

After the SciPy-to-LOV3D coefficient normalization was corrected by the expected
`1/sqrt(4*pi)` factor, the MATLAB raw-grid transform still leaves a small
finite-grid asymmetry between its `+m` and `-m` complex coefficients. Python's
coefficient representation preserves the symmetric field. The raw-grid
response differences are therefore bounded by the transform itself rather than
by the coupled radial solver. The measured ~0.8% coefficient asymmetry and
~0.75% energy difference define the empirical transform floor.

Use the cheap diagnostic to inspect this layer directly:

```bash
python scripts/io_rheology_amplitude_parity.py
```

### Stage 6 — strict identical-coefficient solver parity: READY TO RUN

This is the final TASK-046 numerical-hygiene gate. It removes the raw-grid SH
transform from the cross-language comparison.

First generate a canonical six-mode MATLAB artifact:

```bash
/Applications/MATLAB_R2025b.app/bin/matlab -batch \
  "run('scripts/io_matlab_identical_coefficients_anchor.m')"
```

The MATLAB script:

1. constructs the already-vetted native raw-grid Io rheology;
2. averages the small `+m/-m` transform asymmetry for degrees 2 and 4;
3. applies those six exact lateral coefficients to the uniform complex
   asthenosphere background, which already has ~1e-11 MATLAB/Python parity; and
4. archives forcing-mode Love numbers plus direct and Love-derived energy.

Then give Python the **same six coefficient values** from the MAT artifact:

```bash
python scripts/io_compare_identical_coefficients_anchor.py
```

Provisional strict acceptance:

```text
mode counts identical
worst forcing-mode k relerr < 1e-8
direct-energy relerr         < 1e-8
Love-energy relerr           < 1e-8
```

This test does not assess the raw-grid transform. It isolates coupling
construction, radial propagation, Love extraction, coupled stress/strain
recovery, and multibasis energy contraction.

### Stage 7 — full publication-facing suite and promotion

After Stage 6 passes:

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

Record:

- git commit;
- Python executable/environment;
- NumPy/SciPy/JAX/PyALMA3 versions;
- final test count;
- wall time;
- MATLAB version and reference artifacts.

Only after that rerun should TASK-046 be promoted from a task-specific Gate C
into the standard publication-facing benchmark runner.

## Interpretation hierarchy

Validation claims should state their level explicitly:

1. **unit / invariant:** local algebra, signs, limits, scaling, data-shape behavior;
2. **analytic:** closed-form physics recovered;
3. **parent-code parity:** Python reproduces an equivalent native MATLAB LOV3D calculation;
4. **independent-code validation:** Python agrees with a separate implementation such as PyALMA3;
5. **planetary/science validation:** a defensible planetary model simultaneously satisfies relevant observables and convergence tests.

TASK-046 demonstrates why physical input equivalence matters. The retired
125-mode anchor compared incompatible coefficient conventions; the raw-grid
benchmark recovered the actual six-mode Io problem, and the identical-
coefficient lane now separates transform accuracy from strict solver parity.

## Current TASK-046 order

1. run `io_matlab_identical_coefficients_anchor.m`;
2. run `io_compare_identical_coefficients_anchor.py`;
3. repair only if strict coefficient-identical parity fails;
4. rerun the complete science benchmark suite;
5. update the PR from draft validation work to merge-ready only after the full
   suite result and provenance are recorded.

For chronological rejected hypotheses and historical outputs, see
`docs/tasks/TASK-046-diagnostic-log.md`.