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

1. integrated direct stress-strain dissipation from `pylov3d.energy.get_energy`; and
2. the forcing-work estimate constructed from the imaginary Love spectra using the same forcing normalization as the MATLAB test.

The fractional mismatch must decrease with radial refinement. Do **not** hard-code a tight publication tolerance until the MATLAB normalization and Python normalization have been checked term by term.

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

## Relation to lateral K

This task does not depend on lateral bulk-modulus variation. `K_variable` remains a separate blocked validation item because the upstream MATLAB rheology preprocessing appears not to propagate it despite constitutive support in `get_solution.m`. That issue should not prevent validation of lateral `mu`+`eta`, which is directly relevant to Mars hydration rheology.
