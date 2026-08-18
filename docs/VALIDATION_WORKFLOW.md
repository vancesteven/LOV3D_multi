# pylov3d scientific validation workflow

This is the canonical record of **test order, scientific rationale, and observed results** for validation of the Python conversion. It complements `docs/SCIENCE_VALIDATION.md`, which is the compact publication-facing matrix, and the task notes under `docs/tasks/`, which preserve implementation details and failed diagnostic history.

The order is intentional. Later stages combine more physics and are harder to diagnose. A failure at a later stage must not be interpreted as invalidating earlier stages that have independent passing references.

## Validation ladder

### Stage 1 — local algebra and bookkeeping

**Question:** does a new implementation preserve required array, basis, sign, scaling, and limiting-case invariants before any expensive planetary comparison?

For TASK-046:

```bash
pytest -q pylov3d/tests/test_energy_multibasis.py
```

**Why first:** these tests isolate bookkeeping from planetary physics. If this stage fails, a MATLAB comparison is not diagnostically useful.

**Latest verified TASK-046 result:**

```text
2 passed in 0.73 s
```

The focused tests establish finite multibasis contraction, support for genuinely distinct forcing closures, and the required quadratic forcing scaling `E(F/2)=E(F)/4`. They are structural/physical-invariant tests only; they do not establish MATLAB parity.

### Stage 2 — publication-facing core science matrix

**Question:** does pylov3d reproduce known behavior across qualitatively different physical regimes before adding a new compound benchmark?

Run:

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

The selected matrix includes analytic elasticity, archived MATLAB lateral-heterogeneity cases, a fluid-core Moon case, multilayer Mars structure, dissipation sanity checks, and independent PyALMA3 elastic/Maxwell comparisons.

**Why second:** this protects against fixing one complicated benchmark by accidentally breaking already validated physics elsewhere.

**Verified baseline:**

```text
58 passed in 88.57s (0:01:28)
```

This is the last complete final pytest summary captured in the validation record. Do not infer a later count from a partial stream of dots; record the next complete final line when rerun.

### Stage 3 — establish the physically authoritative MATLAB target

**Question:** are Python and MATLAB actually solving the same physical lateral-rheology field before any Love-number or energy comparison is interpreted?

TASK-046 initially archived a coefficient-input MATLAB cross-check at `Nrbase=50`:

```text
N coupled modes = [125,125,125]
```

That script fed Python `scipy.special.sph_harm_y` coefficients through MATLAB's different coefficient-input convention. Its own header warned that the basis equivalence had not been established.

The authoritative upstream Io test, `tests/Consistency_test_Energy.m`, instead supplies full `mu_latlon` and `eta_latlon` maps. Reproducing that original raw-grid path gives:

```text
retained asthenosphere rheology modes: 6
retained degree range: 2..4
active solution counts for m=[0,-2,+2]: [43 41 41]
```

The six retained modes are

```text
(2,-2), (2,0), (2,2), (4,-2), (4,0), (4,2)
```

and the independent Python MATLAB-work-grid diagnostic gives the same `[43,41,41]` closure.

**Conclusion:** `[43,41,41]` is the physical Io closure. The old `[125,125,125]` coefficient-path result is retained only as a basis-mismatch diagnostic artifact and must not appear in publication-facing acceptance criteria.

### Stage 4 — uniform Love and direct-energy control

**Question:** before lateral complexity is reintroduced, does the already-correct uniform tidal solution produce MATLAB-equivalent post-solve fields and direct energy?

The uniform Love response is already strongly validated. Even at low radial resolution:

```text
k_uni = +0.7337217052 - 0.0151236753 i
E_Love = 2.2144024577
```

which agrees essentially exactly with the MATLAB uniform anchor.

TASK-046 then exposed and fixed several post-solve energy defects:

1. the multi-forcing uniform calculation had been routed through a simplified radial contraction instead of MATLAB's full GSH angular contraction;
2. coupled stress recovery retained interleaved-per-mode indexing assumptions even though the solver stores fields in grouped blocks;
3. off-diagonal lateral-rheology constitutive terms were missing from recovered stress;
4. post-solve stress used the wrong operational A1/A2 pairing;
5. the CMB auxiliary-field node was labelled core and skipped, whereas MATLAB treats it as the first solid layer.

The A1/A2 point is now source-verified: MATLAB `get_A1A2.m` returns `[A1,A2]`, but `get_solution.m` intentionally calls it as `[A2,A1] = get_A1A2(...)`. Python's corrected recovery now follows that same operational convention.

After the catastrophic scale bug was removed, the uniform direct-energy ladder became:

```text
Nrbase=10  E_direct=-2.399564717933
Nrbase=20  E_direct=-2.614859604596
Nrbase=50  E_direct=-2.754371543041
MATLAB Nrbase=50 coefficient-path uniform E00 = +2.166877841600
```

The discrepancy grows rather than shrinks with radial refinement. Therefore it is **not** a coarse-grid error and should not be treated by empirical renormalization or a sign flip.

#### Required next diagnostic

Generate a complete MATLAB uniform radial anchor:

```bash
/Applications/MATLAB_R2025b.app/bin/matlab -batch "run('scripts/io_matlab_uniform_radial_anchor.m')"
```

Then compare the complete `Nrbase=50` radial fields:

```bash
python scripts/io_compare_uniform_radial_anchor.py
```

The comparison reports pointwise/blockwise errors for:

- `U,V,W,R,S,T,Phi,dPhi`;
- GSH displacement `u`;
- six GSH stress components;
- six GSH strain components.

**Why this comes before any further energy changes:** the uniform Love number is already correct, so a pointwise field comparison will identify exactly where the remaining direct-energy disagreement first appears.

### Stage 5 — Python lateral rheology spectrum parity

**Question:** does Python's general viscoelastic lateral-rheology processor reproduce the six-mode physical spectrum found by the raw-grid MATLAB benchmark?

Current general Python processing retains four rheology modes and gives:

```text
[29,29,29]
```

The MATLAB-faithful work-grid calculation gives six modes and:

```text
[43,41,41]
```

The target is therefore **six retained modes / `[43,41,41]`**, not 125.

This stage should be closed with a dedicated regression on the retained `(n,m)` spectrum and active-mode counts before running expensive lateral solves.

### Stage 6 — replacement raw-grid MATLAB Love/energy anchor

Only after Stages 4 and 5 are closed should a new full `Nrbase=50` MATLAB lateral anchor be generated from the **raw-grid** Io field used by `Consistency_test_Energy.m`.

Archive at minimum:

- forcing-mode complex `k` for `m=0,-2,+2`;
- active-mode counts;
- selected coupled complex `k` coefficients;
- direct `E00`;
- Love-derived energy;
- direct-vs-Love mismatch;
- exact rheology spectrum and numerics.

This replacement anchor supersedes the coefficient-path lateral values for physical validation.

### Stage 7 — final Python Gate B/C

Only after the replacement raw-grid anchor exists should the Python compound benchmark become a hard pass/fail gate.

The current `scripts/io_energy_gate_bc_multibasis.py --assert-matlab` still contains obsolete coefficient-path lateral targets and must **not** be used as a publication acceptance test until those assertions are replaced.

Final acceptance should require:

1. uniform radial-field/direct-energy parity;
2. six-mode lateral rheology spectrum and `[43,41,41]` closure;
3. forcing-mode complex-Love parity against the raw-grid MATLAB anchor;
4. direct-energy parity against the raw-grid MATLAB anchor;
5. direct-vs-Love mismatch within a declared numerical tolerance based on the parent-code convergence test.

### Stage 8 — promote into the standard science matrix

Only after Stage 7 passes should the Io viscoelastic+lateral+energy benchmark be added to `scripts/run_science_benchmarks.py` as a publication-facing case.

Then rerun the full suite and record:

- git commit;
- Python executable/environment;
- relevant NumPy/SciPy/JAX/PyALMA3 versions;
- test count;
- wall time;
- MATLAB version and reference artifacts used.

## Interpretation hierarchy

Validation claims should state their level explicitly:

1. **unit / invariant:** local algebra, signs, limits, scaling, data-shape behavior;
2. **analytic:** closed-form physics recovered;
3. **parent-code parity:** Python reproduces an equivalent native MATLAB LOV3D calculation;
4. **independent-code validation:** Python agrees with a separate implementation such as PyALMA3;
5. **planetary/science validation:** a physically defensible planetary model simultaneously satisfies relevant observables and convergence tests.

Parent-code parity requires more than matching numbers: the input physical field and basis conventions must also be equivalent. TASK-046's retired 125-mode anchor is the explicit cautionary example.

## Current TASK-046 blocker order

Resolve in this order:

1. rerun `pytest -q pylov3d/tests/test_energy_multibasis.py` after the CMB parity correction;
2. generate `io_uniform_radial_anchor.mat` with MATLAB;
3. run `python scripts/io_compare_uniform_radial_anchor.py` and repair any remaining uniform radial-field discrepancy;
4. modify the general Python viscoelastic rheology processor to reproduce the six raw-grid modes and `[43,41,41]` closure;
5. generate the replacement full raw-grid MATLAB `Nrbase=50` lateral Love/energy anchor;
6. replace obsolete coefficient-path assertions in the Python Gate B/C driver;
7. run the final compound Gate B/C;
8. rerun and record the complete publication-facing science suite.

For chronological failed runs, rejected hypotheses, and exact numerical outputs, see `docs/tasks/TASK-046-diagnostic-log.md`.