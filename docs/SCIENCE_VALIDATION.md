# pylov3d science validation matrix

This document is the compact, publication-facing validation record for the
Python conversion. It is intentionally distinct from the full regression test
suite: the goal here is to show that qualitatively different physical regimes
reproduce analytic solutions, archived MATLAB LOV3D results, or an independent
Love-number implementation.

Run the core matrix with

```bash
python scripts/run_science_benchmarks.py
```

and add the independent PyALMA3 viscoelastic comparison with

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

The benchmark runner disables the repository's default `not slow` filter so
that the selected science cases cannot silently disappear because of pytest
marker configuration.

## Verified benchmark run

A full independent-code run was executed on 2026-08-17 using
`/Users/svance/mamba/envs/PPcl/bin/python`:

```text
python scripts/run_science_benchmarks.py --with-pyalma3
58 passed in 88.57s (0:01:28)
```

This result should be treated as the current integrated science-validation
baseline for the `python-conversion` branch. Future publication tables should
record the exact git commit and dependency versions in addition to the test
count and wall time.

## Validation matrix

| Physical regime | Body / model | Observable | Reference | Current acceptance criterion | Test |
|---|---|---|---|---|---|
| Homogeneous elastic limit | Uniform self-gravitating sphere | degree-2 Love numbers | closed-form elastic-sphere solution | numerical result within the analytic tolerance encoded in the test | `test_analytical.py` |
| Elastic lateral heterogeneity | Enceladus, rigid interior/ocean plus elastic shell with degree-1/2 shear-modulus variations | coupled Love-number spectrum and forcing-mode `k2` | archived MATLAB LOV3D outputs from the published Enceladus lateral-variation benchmark | uniform `k2` <0.1% relative; first-order spectral modes <1%; second-order modes <5% | `test_matlab_validation.py` |
| Fluid layer + lateral heterogeneity | Weber Moon with fluid outer core and upper/lower-mantle lateral shear-modulus variations | 1-D `k2` and coupled Love-number spectrum | archived MATLAB/Qin reference arrays used by the LOV3D Moon benchmark | uniform `k2` <1e-6 relative; measured order-1 coupled errors are ~2e-6 to 5e-6 relative, with test ceiling 0.1%; forcing-mode deviation uses a 5% ceiling | `test_matlab_validation_ocean.py` |
| Multilayer planetary structure with density discontinuities | Four-layer Mars reference model | mass, mean moment of inertia, `k2`, `h2`, `l2`, radial density ordering | published Mars bulk constraints plus pylov3d/MATLAB cross-check artifacts in `data/tests/mars/` | mass within 0.1%; MoI within stated uncertainty; `k2` within observational uncertainty; `h2/l2` regression pins | selected classes in `test_mars.py` |
| Independent elastic implementation | Uniform elastic sphere | complex `k2` (imaginary part ~0) | PyALMA3 plus analytic elastic sphere | pylov3d and PyALMA3 agree within 1%; both agree with analytic solution | `test_benchmark_pyalma3.py` |
| Independent Maxwell viscoelastic implementation | Fluid core + Maxwell mantle, forcing period 1 day, viscosity 1e15 Pa s | real and imaginary parts of complex `k2` | PyALMA3 | Re(`k2`) and Im(`k2`) agree within 0.1% | `test_benchmark_pyalma3.py` |

## What this matrix establishes

The core conversion is not validated by a single favorable planet or a single
solver path. The selected cases exercise: (1) the analytic elastic limit,
(2) density jumps and multilayer propagation, (3) a zero-shear fluid layer,
(4) spectral coupling from lateral rheology, and (5) complex viscoelastic
response. The Enceladus and Moon cases compare against archived outputs from
the original MATLAB LOV3D workflow, while the PyALMA3 case is an independent
implementation and therefore protects against a bug shared by the Python port
and its MATLAB parent.

The Mars entry is deliberately included as a planetary-structure validation,
not merely as another numerical parity test. Its mass, moment of inertia,
density ordering, and Love numbers must remain simultaneously physical. The
separate committed MATLAB artifacts under `data/tests/mars/` provide the
stronger code-to-code parity record for the same reference model.

## Lateral bulk-modulus status: upstream functionality is incomplete

The next planned validation target was lateral bulk-modulus (`K`) heterogeneity.
Inspection of both implementations shows that this cannot yet be treated as a
straight parent-code parity benchmark.

The Python `process_lateral_variations()` API accepts `K_variable`, and the
coupled propagator has a `K_amp` path in its constitutive coupling matrices.
However, the current rheology processing fills `K_amp` with zero. The original
MATLAB source shows the same deeper problem: `get_rheology.m` parses
`K_variable`, but the elastic branch explicitly sets
`rheology_variable(:,3)=0`, while the viscoelastic branch populates the complex
shear-modulus column but does not populate the bulk-modulus column. In
`get_solution.m`, column 3 is nevertheless read as `K_nm` and used in the
isotropic constitutive coupling term.

Therefore a MATLAB-vs-Python `K` comparison made without first resolving this
path could give a misleading result: agreement with zero response would verify
shared omission rather than physical correctness. The next step is to derive
and test the intended normalization of the lateral bulk-modulus coefficient,
repair the parent and/or Python path, and only then archive a non-zero MATLAB
reference case. See `docs/LATERAL_K_VALIDATION.md`.

## Known gaps before a methods-paper validation claim

This matrix does **not** yet constitute exhaustive validation of every LOV3D
feature. The remaining high-value additions are:

1. resolve the dormant/broken lateral **bulk modulus** (`K`) path, including its
   normalization in the isotropic stress term, then create a non-zero archived
   reference case;
2. add an archived MATLAB reference for a genuinely **viscoelastic lateral**
   case, which would exercise complex rheology and spectral coupling
   simultaneously;
3. add a dedicated end-to-end **tidal dissipation / energy** benchmark;
4. generate a publication table from machine-readable benchmark output rather
   than transcription from pytest prose.

These are feature-coverage gaps, not evidence of failure in the regimes already
listed above. They should be closed before claiming complete feature parity
with MATLAB LOV3D.

## Interpretation for publication

A methods paper should distinguish three validation levels:

* **analytic validation:** a known closed-form limit is recovered;
* **parent-code parity:** pylov3d reproduces archived MATLAB LOV3D outputs;
* **independent-code validation:** pylov3d agrees with PyALMA3 where the two
  implementations overlap.

That distinction matters because parent-code parity establishes a faithful port,
but independent-code and analytic agreement are what test the underlying
physics most strongly.