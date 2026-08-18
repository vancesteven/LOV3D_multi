# pylov3d scientific validation workflow

This is the canonical record of **test order, scientific rationale, and observed results** for validation of the Python conversion. It complements `docs/SCIENCE_VALIDATION.md`, which is the compact publication-facing matrix, and the individual task files under `docs/tasks/`, which contain implementation details.

The order is intentional. Later stages combine more physics and are harder to diagnose. A failure at a later stage must not be interpreted as invalidating earlier stages that have independent passing references.

## Validation ladder

### Stage 1 — local algebra and bookkeeping

**Question:** does a new implementation preserve required array, basis, sign, and limiting-case invariants before any expensive planetary comparison?

Run the focused unit tests for the feature being changed. For TASK-046 multibasis energy:

```bash
pytest -q pylov3d/tests/test_energy_multibasis.py
```

**Why first:** these tests isolate bookkeeping from planetary physics. If this stage fails, a MATLAB comparison is not diagnostically useful.

**TASK-046 result:** PASS, reported by the user on 2026-08-18 for commit `5325350` on `agent/task-046-multibasis-energy`.

The tests establish that the multibasis field-contraction path reproduces the existing coupled-energy result when bases are identical and remains finite/nonzero when different forcings use genuinely distinct native bases. This is structural validation only; it does not establish MATLAB parity.

### Stage 2 — publication-facing core science matrix

**Question:** does pylov3d reproduce known behavior across qualitatively different physical regimes before adding a new complex benchmark?

Run:

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

The selected matrix includes analytic elasticity, archived MATLAB lateral-heterogeneity cases, a fluid-core Moon case, multilayer Mars structure, dissipation sanity checks, and independent PyALMA3 elastic/Maxwell comparisons.

**Why second:** this protects against fixing one complicated benchmark by accidentally breaking already validated physics elsewhere.

**Verified baseline:** on 2026-08-17 in `/Users/svance/mamba/envs/PPcl/bin/python`, the pre-energy-expansion matrix returned

```text
58 passed in 88.57s (0:01:28)
```

The expanded runner was subsequently started and progressed through its selected tests without an early failure, but no final count was captured in this record. Do not replace the 58/58 result with an inferred result; record a new final line when available.

### Stage 3 — parent-code numerical anchor

**Question:** for the exact target model, what does native MATLAB LOV3D compute?

For TASK-046, B ran the native MATLAB anchor derived from `tests/Consistency_test_Energy.m` and archived:

- `data/tests/io/io_energy_cross_check.log`
- `data/tests/io/io_energy_cross_check.mat`

At `Nrbase=50`, MATLAB gives:

```text
k_uni(2,m)       = +0.7337217069 - 0.0151236751 i   for m=0,-2,+2
k_lat(2,0)       = +0.7325399703 - 0.0153355564 i
k_lat(2,+/-2)    = +0.7381214321 - 0.0198692819 i
N coupled modes  = [125, 125, 125]
E_direct uni/lat = 2.1668778416 / 2.8404609804
E_Love   uni/lat = 2.2144024348 / 2.9026033327
mismatch uni/lat = approximately 2.19% / 2.19%
```

**Why before Python tuning:** this gives an external numerical target and prevents a Python-only internal-consistency test from validating a shared implementation error or an incorrectly normalized quantity.

The MATLAB run also established that the previously observed Python ~1e7 lateral energy mismatch is not an intrinsic limitation of the LOV3D formulation. Native MATLAB closes the two independent energy estimates to about 2.2%.

### Stage 4 — cheap Python diagnostic rung

**Question:** before spending substantial runtime at the reference resolution, does the Python target case reproduce the correct qualitative structure and identify which subsystem still disagrees?

Run:

```bash
python scripts/io_energy_gate_bc_multibasis.py --nrbase 10
```

**Why before `Nrbase=50`:** this rung is diagnostic, not publication-facing. It should expose basis, spectrum, and normalization errors quickly.

**Observed 2026-08-18 result:** FAIL as a quantitative Gate B/C check, but highly diagnostic:

```text
TASK-046 Gate B/C, Nrbase=10
native lateral mode counts: [29, 29, 29]

forcing (2, 0):  k_uni=+0.7337217052-0.0151236753i  k_lat=+0.7316354724-0.0139412820i
forcing (2,-2):  k_uni=+0.7337217052-0.0151236753i  k_lat=+0.7360972172-0.0170264646i
forcing (2,+2):  k_uni=+0.7337217052-0.0151236753i  k_lat=+0.7360972172-0.0170264646i

direct energy uniform/lateral: -1.7735878556e-03 / 2.5770609705e+05
Love energy   uniform/lateral:  2.2144024577e+00 / 2.4784838914e+00
direct/Love mismatch: 100.0801% / 10397631.3687%
wall time: 17.9 s
```

#### What this result establishes

1. **The 1-D/mean viscoelastic solver and Love-number normalization are healthy.** Even at `Nrbase=10`, uniform `k` agrees with the MATLAB `Nrbase=50` anchor to roughly a few parts in 1e-9, and the uniform Love-derived energy (`2.2144024577`) agrees with MATLAB (`2.2144024348`) to about 1e-8 relative.
2. **The uniform direct-energy path has an independent normalization/assembly defect.** It returns `-1.77e-3` instead of the MATLAB `2.1669`, despite the Love response being correct. This cannot be blamed on lateral coupling or the multibasis change.
3. **The Python lateral rheology/closure does not yet reproduce the MATLAB target spectrum.** Python produces 29 native modes per forcing whereas the MATLAB anchor has 125. Mode count is controlled by the rheology spectrum and perturbation closure, not radial resolution, so running `Nrbase=50` cannot by itself fix this discrepancy.
4. **The lateral Love response is therefore not yet a valid Gate C comparison.** Its difference from MATLAB must first be traced to the lateral `mu`/`eta` spectrum, filtering/cutoff rules, spherical-harmonic convention, or active-mode construction.
5. **The multibasis bookkeeping fix remains useful but is not sufficient.** Stage 1 passed; Stage 4 shows there are upstream differences in the fields being contracted and a separate direct-energy normalization issue.

This failed rung is part of the scientific record and must remain documented. Do not report TASK-046 as closed until the two blockers above are resolved.

### Stage 5 — full Python MATLAB-anchor check

Only after the Stage-4 blockers are resolved, run:

```bash
python scripts/io_energy_gate_bc_multibasis.py --nrbase 50 --assert-matlab
```

**Acceptance target:** forcing-mode complex Love numbers reproduce the archived MATLAB values; each forcing uses the same physically intended rheology spectrum/closure as MATLAB; the direct and Love-derived energies agree with the archived values within the declared numerical tolerances; and the direct-vs-Love mismatch is below 3% for both uniform and lateral cases.

A `Nrbase=50` run should **not** be used as a brute-force attempt while Stage 4 still reports 29 modes and the incorrect uniform direct energy. Those discrepancies are resolution-independent diagnostics.

### Stage 6 — promote into the standard science matrix

Only after Stage 5 passes should the Io viscoelastic+lateral+energy benchmark be added to `scripts/run_science_benchmarks.py` as a publication-facing validation case.

Then rerun the full suite and record:

- git commit;
- Python executable/environment;
- dependency versions relevant to NumPy/SciPy/JAX/PyALMA3;
- test count;
- wall time;
- reference artifacts used.

## Interpretation hierarchy

Validation claims should state their level explicitly:

1. **unit / invariant:** local algebra, signs, limits, data-shape behavior;
2. **analytic:** closed-form physics recovered;
3. **parent-code parity:** Python reproduces archived native MATLAB LOV3D output;
4. **independent-code validation:** Python agrees with a separate implementation such as PyALMA3;
5. **planetary/science validation:** a physically defensible planetary model simultaneously satisfies relevant observables and convergence tests.

Passing a higher-cost internal test is not a substitute for an external anchor. Conversely, a failure in a newly added compound benchmark does not erase already passed analytic, MATLAB, or PyALMA3 validations in other regimes.

## Current TASK-046 blocker order

Resolve these in this order:

1. **uniform direct-energy parity:** reproduce MATLAB `E_direct_uni=2.1668778416` for the already-correct uniform solution. This isolates the direct stress-strain normalization/assembly without lateral complications;
2. **lateral rheology spectrum parity:** reproduce MATLAB's retained rheology modes and resulting 125-mode forcing closures from the same Io `mu`/`eta` fields;
3. **lateral forcing-mode Love parity:** verify all three complex forcing-mode `k` values against the MATLAB archive;
4. **multibasis lateral direct energy:** only then interpret the union-field energy contraction quantitatively;
5. **`Nrbase=50 --assert-matlab`:** final Gate B/C closure.

This order minimizes confounding: first fix a 1-D energy quantity whose Love solution is already proven correct, then establish that Python and MATLAB are solving the same 3-D rheology problem, and only afterward judge the multi-forcing energy contraction.
