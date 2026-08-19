# pylov3d science validation matrix

This document is the compact, publication-facing validation record for the
Python conversion. It is intentionally distinct from the full regression test
suite: the goal here is to show that qualitatively different physical regimes
reproduce analytic solutions, archived MATLAB LOV3D results, or an independent
Love-number implementation.

For the required testing order, rationale, exact commands, failed diagnostic
runs, and result history, see `docs/VALIDATION_WORKFLOW.md` and
`docs/tasks/TASK-046-diagnostic-log.md`.

Run the core matrix with

```bash
python scripts/run_science_benchmarks.py
```

and add the independent PyALMA3 viscoelastic comparison with

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

## Verified publication-facing baseline

A complete independent-code run was recorded on 2026-08-17:

```text
58 passed in 88.57s (0:01:28)
```

A later expanded run including the energy tests produced a complete stream of
passing tests, but its final pytest summary was not preserved in the validation
record. Do not infer a new publication-facing count from dots alone; the next
full rerun should record commit, environment, dependency versions, count, and
wall time.

## Validation matrix

| Physical regime | Body / model | Observable | Reference | Current acceptance criterion | Test |
|---|---|---|---|---|---|
| Homogeneous elastic limit | Uniform self-gravitating sphere | degree-2 Love numbers | closed-form elastic-sphere solution | encoded analytic tolerance | `test_analytical.py` |
| Elastic lateral heterogeneity | Enceladus | coupled Love spectrum and forcing-mode `k2` | archived MATLAB LOV3D | uniform `k2` <0.1%; first-order modes <1%; second-order modes <5% | `test_matlab_validation.py` |
| Fluid layer + lateral heterogeneity | Weber Moon | 1-D `k2` and coupled Love spectrum | archived MATLAB/Qin reference | uniform `k2` <1e-6; coupled spectrum within encoded ceilings | `test_matlab_validation_ocean.py` |
| Multilayer planetary structure | Four-layer Mars | mass, MoI, Love numbers, density ordering | Mars bulk constraints plus MATLAB artifacts | mass <0.1%; MoI and `k2` within stated uncertainties; regression pins | selected `test_mars.py` |
| Dissipation invariants | elastic sphere + viscoelastic Io | energy integral and heating sign/scaling | constitutive invariants | elastic ~0; viscoelastic non-zero; forcing scaling correct | selected `test_energy.py` |
| Independent elastic implementation | uniform sphere | complex `k2` | PyALMA3 + analytic result | agreement within 1% | `test_benchmark_pyalma3.py` |
| Independent Maxwell implementation | fluid core + Maxwell mantle | complex `k2` | PyALMA3 | Re/Im agreement within 0.1% | `test_benchmark_pyalma3.py` |
| Viscoelastic lateral rheology + multibasis energy | Io eccentricity tide | mode closure, complex forcing-mode `k`, direct and Love-derived dissipation | native MATLAB raw-grid `Consistency_test_Energy.m` path | six retained rheology modes; `[43,41,41]` closure; raw-grid end-to-end response within documented 1% transform floor; direct/Love mismatch <3% | `test_io_rheology_spectrum_parity.py`, `io_energy_gate_bc_multibasis.py` |

## TASK-046 status: raw-grid Gate C passes

TASK-046 now has a physically faithful native-MATLAB raw-grid anchor at
`Nrbase=50`. The original Io `mu_latlon`/`eta_latlon` path retains six complex
rheology modes,

```text
(2,-2), (2,0), (2,2), (4,-2), (4,0), (4,2)
```

and produces active solution counts

```text
[43, 41, 41]
```

for forcing `m=[0,-2,+2]`. The earlier `[125,125,125]` coefficient-input result
is retained only as a basis-mismatch diagnostic artifact and is not a physical
acceptance target.

The final Python raw-grid Gate C run reported:

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
```

The raw-grid MATLAB transform leaves a small finite-grid asymmetry between the
`+m` and `-m` complex rheology coefficients of the otherwise symmetric Io
pattern. After correcting the SciPy-to-LOV3D SH normalization, the remaining
coefficient difference is approximately 0.8%, consistent with the remaining
~0.75% lateral-energy difference. Python intentionally preserves the symmetric
coefficient field rather than reproducing this discretization asymmetry.
Therefore the raw-grid Gate C is an end-to-end physical validation with a
measured 1% transform floor, not a strict solver-only parity test.

Uniform validation is much tighter. At `Nrbase=50`, MATLAB and Python agree in
primary state, GSH displacement, stress, strain, and the interior `E00(r)`
profile at approximately `1e-10` relative once the MATLAB zeroed outermost
auxiliary row is excluded. The energy-coupling bug was a NumPy/MATLAB reshape
ordering error and is now protected by
`test_energy_couplings_matlab_order.py`.

## Final strict solver-parity lane

A second TASK-046 lane now removes the raw-grid transform from the comparison.
`scripts/io_matlab_identical_coefficients_anchor.m` constructs the native
raw-grid rheology, symmetrizes the small `+/-m` transform difference, and applies
those exact six lateral coefficients to the already-validated uniform complex
rheology background. `scripts/io_compare_identical_coefficients_anchor.py` then
loads that artifact and gives Python the identical coefficients and background.

This is the remaining numerical-hygiene gate for TASK-046. It tests only:

1. active-mode closure and coupling construction;
2. coupled radial solution and Love spectra;
3. coupled stress/strain reconstruction; and
4. multibasis direct-energy contraction.

The provisional strict tolerance is `1e-8` relative for forcing-mode `k`, direct
energy, and Love-derived energy. If this passes, TASK-046 can be described as
strict parent-code solver parity in addition to the already passing raw-grid
end-to-end physical validation.

## Lateral bulk-modulus status

Lateral bulk-modulus (`K`) heterogeneity remains a separate validation gap. The
Python coupled propagator contains a `K_amp` constitutive path, but both the
current Python rheology preprocessor and the upstream MATLAB preprocessing path
appear not to propagate a non-zero lateral bulk-modulus spectrum consistently.
Agreement with zero response would therefore not establish physical correctness.
See `docs/LATERAL_K_VALIDATION.md`.

## Remaining high-value work

1. run and archive the strict identical-coefficient TASK-046 anchor;
2. rerun the complete publication-facing science benchmark suite and record its
   final summary and dependency provenance;
3. resolve the lateral `K` path with a genuinely non-zero reference case;
4. generate a machine-readable validation table for the methods paper.

## Interpretation for publication

Validation claims should distinguish five levels:

* **unit / physics invariant:** local algebra, signs, scaling, limits;
* **analytic validation:** a closed-form limit is recovered;
* **parent-code parity:** Python reproduces an equivalent native MATLAB LOV3D calculation;
* **independent-code validation:** Python agrees with PyALMA3 where implementations overlap;
* **planetary/science validation:** a defensible planetary model simultaneously satisfies relevant observables and convergence tests.

TASK-046 is the cautionary example for why parent-code parity requires physical
input equivalence, not only numerical similarity: the retired 125-mode anchor
matched an incompatible coefficient convention, whereas the raw-grid benchmark
recovered the actual six-mode Io physics.