# TASK-046 — Io viscoelastic + lateral-rheology + energy validation

## Scientific purpose

Close the most important remaining validation gap for the Mars hydration application: demonstrate that pylov3d can propagate **lateral variations in a viscoelastic rheology** and produce a dissipative response that is internally consistent with the work done by the tidal forcing.

This benchmark is deliberately based on the upstream MATLAB `tests/Consistency_test_Energy.m` case. That case is stronger than a generic regression because it combines:

- a four-layer Io model including a fluid core;
- viscoelastic mantle/asthenosphere layers;
- lateral variations in both shear modulus and viscosity;
- three degree-2 eccentricity-tide forcing components (`m=0,-2,+2`);
- second-order mode coupling;
- direct stress-strain dissipation; and
- an independent global dissipation estimate from the imaginary Love-number response.

The MATLAB test checks convergence of the fractional mismatch between those two energy estimates as radial resolution increases.

## Why this matters for Mars hydration

For the SSS Mars application, hydration is not expected to map only onto a real-valued elastic shear-modulus perturbation. Temperature, hydration, melt fraction, and mineralogy can change both elastic moduli and anelastic/viscous response. A validated laterally heterogeneous complex rheology is therefore a prerequisite for interpreting mechanically weak or dissipative hydrated domains.

Io is used here as a high-signal numerical validation target, not as a Mars analogue. Once this benchmark is closed, the same machinery can be driven by physically consistent Mars properties supplied by PlanetProfile.

## Upstream reference configuration

Mirror `tests/Consistency_test_Energy.m`:

- radii [km]: `965, 1591.6, 1791.6, 1821.6`;
- densities [kg m^-3]: `5150, 3244, 3244, 3244`;
- solid-layer bulk modulus: `200e12 Pa`;
- shear moduli [Pa]: mantle `6e10`, asthenosphere `7.8e5`, crust `6.5e10`;
- viscosities [Pa s]: mantle `1e20`, asthenosphere `1e11`, crust `1e23`;
- orbital frequency `4.1086e-5 s^-1`;
- forcing modes `(2,0)`, `(2,-2)`, `(2,+2)` with the MATLAB amplitudes;
- lateral asthenosphere fields derived from the prescribed degree-2 heating pattern and mapped into both `mu` and `eta`.

The upstream script uses `Nbase=[5,10,20,50,100,200,500,1000]`. The Python publication benchmark should use a shorter convergence ladder by default and retain the full ladder as an opt-in slow benchmark.

## Acceptance criteria

### Gate A — lateral complex-rheology response

1. The uniform and lateral models both solve with finite complex Love spectra.
2. The lateral `mu`+`eta` model excites non-forcing modes absent from the 1-D solution.
3. The forcing-mode imaginary Love response remains nonzero and has a physically consistent sign.
4. Reducing the lateral amplitudes toward zero converges to the uniform viscoelastic result.

### Gate B — energy consistency

For both the uniform and laterally varying cases, compare:

1. integrated direct stress-strain dissipation from the full MATLAB-equivalent generalized-spherical-harmonic angular contraction; and
2. the forcing-work estimate constructed from the imaginary Love spectra using the same forcing normalization as the MATLAB test.

The fractional mismatch must decrease with radial refinement. A plain radial `Im(conj(stress)*strain)` contraction is **not** equivalent to MATLAB `get_energy.m` for the three-component eccentricity forcing, even when the interior is laterally uniform.

### Gate C — MATLAB numerical anchor

Generate and commit a compact MATLAB reference artifact for a tractable radial resolution, containing at minimum:

- forcing-mode complex `k` values for all three forcing components;
- selected first- and second-order coupled complex `k` coefficients;
- direct integrated energy;
- Love-number-derived energy;
- radial resolution and all model/forcing parameters.

Then compare Python against those archived values. This gate is what upgrades the benchmark from internal consistency to independent implementation parity.

## Implementation sequence

1. Add a Python helper that reproduces the analytical degree-2 heating pattern used to construct the MATLAB lateral `mu` and `eta` fields.
2. Confirm `process_lateral_variations` produces complex `muC_amp` from simultaneous `mu_variable` and `eta_variable` inputs.
3. Solve the three forcing components for uniform and lateral models.
4. Add the short radial convergence ladder and compare direct versus Love-number energy.
5. Add a MATLAB export script derived from `Consistency_test_Energy.m` that writes a compact `.mat` reference without plotting.
6. Archive the reference under `data/tests/io/` with a provenance log.
7. Add the resulting pytest node(s) to `scripts/run_science_benchmarks.py` only after Gates A-C pass.

## Important caution

The current simple Io energy fixture in `pylov3d/tests/test_energy.py` uses `Ks0=200e16`, whereas the upstream MATLAB consistency test specifies `200e12 Pa`. The publication benchmark must use the MATLAB value and the existing fixture should be audited to determine whether `200e16` is intentional, a unit-convention artifact, or a typo. Do not silently reuse it.

The audit performed during TASK-046 found that `200e16` occurs in the older MATLAB `scripts/multiple_layers_example.m`, while the energy-consistency benchmark uses `200e12`. The TASK-046 Io model therefore uses `200e12` and treats the older value as a likely inherited example-script typo unless contrary provenance is found.

## Results to date

### Gate A / multibasis bookkeeping

`pylov3d/tests/test_energy_multibasis.py` passed on 2026-08-18 on branch `agent/task-046-multibasis-energy` after commit `5325350`. This validates the bookkeeping strategy of preserving native forcing bases and forming a union only at the field-contraction stage. It does not establish MATLAB parity.

### Gate C MATLAB anchor

B completed the native MATLAB anchor and committed `data/tests/io/io_energy_cross_check.{log,mat}`. At `Nrbase=50`:

```text
k_uni(2,m)       = +0.7337217069 - 0.0151236751 i
k_lat(2,0)       = +0.7325399703 - 0.0153355564 i
k_lat(2,+/-2)    = +0.7381214321 - 0.0198692819 i
N coupled modes  = [125, 125, 125]
E_direct uni/lat = 2.1668778416 / 2.8404609804
E_Love   uni/lat = 2.2144024348 / 2.9026033327
mismatch uni/lat = approximately 2.19% / 2.19%
```

This establishes that the original MATLAB formulation does not exhibit the ~1e7 mismatch seen in the early Python driver.

### Cheap Python diagnostic rung (`Nrbase=10`)

Run on 2026-08-18:

```text
native lateral mode counts: [29, 29, 29]
k_uni = +0.7337217052 - 0.0151236753 i for all three forcings
k_lat(2,0)    = +0.7316354724 - 0.0139412820 i
k_lat(2,+/-2) = +0.7360972172 - 0.0170264646 i
E_direct uni/lat = -1.7735878556e-03 / 2.5770609705e+05
E_Love   uni/lat =  2.2144024577e+00 / 2.4784838914e+00
mismatch uni/lat = 100.0801% / 10397631.3687%
```

Interpretation:

1. The uniform complex Love response is already essentially identical to the MATLAB anchor, even at low radial resolution.
2. The uniform Love-derived energy is also essentially identical to MATLAB. Therefore the uniform direct-energy discrepancy is a separate assembly defect, not a solver or lateral-coupling error.
3. Python retains only 29 lateral solution modes per forcing versus 125 in MATLAB. Because active-mode count is set by the rheology spectrum and perturbation closure rather than radial resolution, this must be resolved before a `Nrbase=50` run can be interpreted as Gate C.
4. The lateral forcing-mode Love numbers therefore do not yet represent the same 3-D problem as the MATLAB anchor.
5. The multibasis fix is necessary but not sufficient: the fields being contracted still differ from the parent-code target.

### Root cause 1 — uniform direct energy used the wrong contraction

Inspection on 2026-08-18 showed that `pylov3d.energy.get_energy()` is a simplified single-mode radial contraction,

```text
Im(sum(conj(stress) * strain))
```

whereas MATLAB `get_energy.m` retains the full generalized-spherical-harmonic angular contraction even for a laterally uniform body under the three eccentricity-tide forcings. MATLAB first forms the union of the forcing `(n,m)` modes, constructs the positive/negative-`m` stress and strain fields, applies the energy-coupling tensor, `2*pi` factors and parity phases, then performs the radial integral.

Therefore the `Nrbase=10` uniform direct-energy failure was primarily a **driver-path error**, not evidence that the validated 1-D solver or stress/strain fields were wrong. Commit `f59ea9f` changes `scripts/io_energy_gate_bc_multibasis.py` so the uniform control is evaluated through `get_energy_coupled_multibasis` using three native one-mode bases: `[(2,0)]`, `[(2,-2)]`, and `[(2,+2)]`.

The simplified `get_energy()` remains useful as a local invariant/sanity utility, but it must not be described as MATLAB-equivalent for a multi-component physical tide.

### Root cause 2 — viscoelastic rheology is spectrally truncated too early

Inspection of MATLAB `get_rheology.m` identified a second independent mismatch. For coefficient-based viscoelastic lateral rheology, MATLAB uses

```text
l_max_base_value = 30
```

as the working spatial resolution, evaluates the nonlinear Maxwell complex shear-modulus field on that grid, re-expands it through degrees `0..(2*l_max-1)` (degree 59 for the default), and only then applies `Numerics.rheology_cutoff=2` plus the minimum-amplitude guard.

The Python `process_lateral_variations()` currently instead sets its nonlinear working/analysis ceiling from the maximum degree present in the input `mu_variable`/`eta_variable` coefficients and re-analyses only to roughly twice that input degree. For the Io degree-2 starting pattern this truncates the nonlinear harmonic tail **before** the two-decade rheology cutoff can decide which generated harmonics are significant.

This is the leading explanation for the observed `[29,29,29]` Python solution closures versus MATLAB `[125,125,125]`. The next code change should reproduce MATLAB's degree-30 working resolution and degree-59 re-expansion for coefficient-based viscoelastic layers, followed by the same amplitude filtering. A regression should pin the retained rheology spectrum and the resulting 125-mode closure before any expensive radial convergence run.

## Current blocker order

Resolve in this order to minimize confounding:

1. **Re-run uniform direct-energy control after `f59ea9f`.** The uniform calculation now uses the MATLAB-style angular contraction. It should move from the spurious ~100% mismatch toward the MATLAB ~2.19% value.
2. **Lateral rheology-spectrum parity.** Match MATLAB's degree-30 working grid / degree-59 nonlinear re-expansion and filtering. Pin the retained `muC` spectrum and require 125 solution modes for each Io forcing.
3. **Lateral forcing-mode Love parity.** Match all three archived complex `k` values before judging the energy contraction.
4. **Multibasis direct lateral energy.** With the same physical fields and closures as MATLAB, compare the union-field contraction quantitatively.
5. **Final Gate B/C.** Run `python scripts/io_energy_gate_bc_multibasis.py --nrbase 50 --assert-matlab` and require the declared Love-number and energy tolerances plus <3% direct-vs-Love mismatch.
6. **Promotion.** Only after final Gate B/C passes, add the Io benchmark to `scripts/run_science_benchmarks.py` as a publication-facing case.

See `docs/VALIDATION_WORKFLOW.md` for the canonical test-order rationale and full result log.

## Relation to lateral K

This task does not depend on lateral bulk-modulus variation. `K_variable` remains a separate blocked validation item because the upstream MATLAB rheology preprocessing appears not to propagate it despite constitutive support in `get_solution.m`. That issue should not prevent validation of lateral `mu`+`eta`, which is directly relevant to Mars hydration rheology.
