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

The final expanded independent-code suite was run on 2026-08-18 using
`/Users/svance/mamba/envs/PPcl/bin/python` after promotion of the fast TASK-046
regressions:

```text
69 passed in 404.51s (0:06:44)
```

This supersedes the earlier recorded baseline of 58 tests. The added coverage
includes native multibasis energy bookkeeping, MATLAB column-major GSH energy
coupling, and the authoritative six-mode Io viscoelastic rheology closure.
MATLAB reference artifacts used by TASK-046 were generated with MATLAB R2025b.
A package-version snapshot should be archived separately rather than inferred
from unrelated environments.

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
| Viscoelastic lateral rheology + multibasis energy | Io eccentricity tide | mode closure, complex forcing-mode `k`, direct and Love-derived dissipation | native MATLAB raw-grid `Consistency_test_Energy.m` path plus identical-coefficient solver anchor | six retained rheology modes; `[43,41,41]` closure; raw-grid end-to-end response within documented 1% transform floor; strict identical-input solver parity <1e-8; direct/Love mismatch <3% | `test_energy_multibasis.py`, `test_energy_couplings_matlab_order.py`, `test_io_rheology_spectrum_parity.py`, archived Gate C scripts |

## TASK-046 status: numerical validation closed

TASK-046 has a physically faithful native-MATLAB raw-grid anchor at `Nrbase=50`.
The original Io `mu_latlon`/`eta_latlon` path retains six complex rheology modes,

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
relerr k_lat = 1.687e-4 .. 1.902e-4
relerr E_direct lateral = 7.542e-3
relerr E_Love   lateral = 7.541e-3
direct/Love mismatch uniform/lateral = 2.1462% / 2.1457%
MATLAB raw-grid Gate C assertions: PASS
```

The raw-grid MATLAB transform leaves a small finite-grid asymmetry between the
`+m` and `-m` complex rheology coefficients of the otherwise symmetric Io
pattern. After correcting the SciPy-to-LOV3D SH normalization, the remaining
coefficient difference is approximately 0.8%, consistent with the ~0.75%
lateral-energy difference. Python intentionally preserves the symmetric field
rather than reproducing this discretization asymmetry. Thus this raw-grid gate
is an end-to-end physical validation with a measured 1% transform floor.

Uniform validation is much tighter. At `Nrbase=50`, MATLAB and Python agree in
primary state, GSH displacement, stress, strain, and the interior `E00(r)`
profile at approximately `1e-10` relative once the MATLAB zeroed outermost
auxiliary row is excluded. The energy-coupling bug was a NumPy/MATLAB reshape
ordering error and is now protected by `test_energy_couplings_matlab_order.py`.

### Strict identical-coefficient solver parity

The raw-grid transform was then removed completely. MATLAB symmetrized its six
retained coefficients and both codes solved the same lateral rheology field on
the same uniform complex-rheology background.

Observed strict comparison:

```text
mode counts Python/MATLAB: [43, 41, 41] / [43, 41, 41]
worst forcing-mode k relerr = 9.936e-12
direct-energy relerr        = 2.215e-09
Love-energy relerr          = 3.429e-11
Python direct/Love mismatch = 2.14574731%
strict identical-coefficient solver parity: PASS
```

This closes parent-code solver parity for coupling construction, radial
propagation, Love extraction, coupled stress/strain recovery, generalized-
spherical-harmonic energy coupling, and multibasis direct-energy contraction.

## Lateral bulk-modulus status

Lateral bulk-modulus (`K`) heterogeneity remains a separate validation gap. The
Python coupled propagator contains a `K_amp` constitutive path, but both the
current Python rheology preprocessor and the upstream MATLAB preprocessing path
appear not to propagate a non-zero lateral bulk-modulus spectrum consistently.
Agreement with zero response would therefore not establish physical correctness.
See `docs/LATERAL_K_VALIDATION.md`.

## Remaining high-value work

1. resolve the lateral `K` path with a genuinely non-zero reference case;
2. capture a machine-readable environment/dependency provenance snapshot;
3. generate a machine-readable validation table for the methods paper;
4. extend planetary/science validation of the Mars hydration/serpentinization cases as the proposal workflow matures.

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
recovered the actual six-mode Io physics and the identical-coefficient lane
established strict solver parity.
