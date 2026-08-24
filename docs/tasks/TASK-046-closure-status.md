# TASK-046 closure status

## Numerical status

TASK-046 has passed both the physically faithful raw-grid end-to-end gate and a strict identical-coefficient MATLAB/Python solver-parity gate.

### Raw-grid end-to-end Gate C

The original MATLAB `Consistency_test_Energy.m` raw-grid Io rheology path retains six lateral rheology modes,

```text
(2,-2), (2,0), (2,2), (4,-2), (4,0), (4,2)
```

with native perturbation closures

```text
[43, 41, 41]
```

for forcing `m=[0,-2,+2]`. Python reproduces the same closure. At `Nrbase=50`, the final raw-grid Python comparison passes within the documented finite-grid transform floor:

```text
relerr k_lat = 1.69e-4 .. 1.90e-4
relerr E_direct lateral = 7.542e-3
relerr E_Love lateral   = 7.541e-3
direct/Love mismatch uniform/lateral = 2.1462% / 2.1457%
```

The remaining ~0.75% raw-grid response difference is consistent with the finite-grid MATLAB SH transform asymmetry in the `+m/-m` rheology coefficients and is not a coupled-solver discrepancy.

### Strict identical-coefficient solver parity

To remove the transform from the comparison, MATLAB symmetrizes its six retained rheology coefficients and both codes solve the identical coefficient field on the same uniform complex-rheology background.

Observed result:

```text
mode counts Python/MATLAB: [43, 41, 41] / [43, 41, 41]

m=+0 k relerr = 8.132e-12
m=-2 k relerr = 9.936e-12
m=+2 k relerr = 6.836e-12

direct energy:
  Python = 2.222075086653
  MATLAB = 2.222075081733
  relerr = 2.215e-09

Love energy:
  Python = 2.270800732202
  MATLAB = 2.270800732280
  relerr = 3.429e-11

Python direct/Love mismatch = 2.14574731%
strict identical-coefficient solver parity: PASS
```

This validates, independently of the raw-grid transform:

1. lateral perturbation closure and coupling construction;
2. coupled radial propagation;
3. forcing-mode Love-number extraction;
4. coupled stress/strain reconstruction;
5. generalized-spherical-harmonic energy coupling; and
6. multibasis direct-energy contraction.

## Root causes closed during TASK-046

The task identified and repaired several independent translation defects:

- native forcing bases were incorrectly merged before the solve;
- coupled state recovery assumed interleaved rather than grouped field storage;
- off-diagonal lateral constitutive terms were omitted in post-solve stress recovery;
- the operational MATLAB A1/A2 convention was translated incorrectly;
- the CMB auxiliary node was assigned to the wrong material layer;
- MATLAB `reshape(Caux,9,[])` column-major grouping was translated using NumPy C-order;
- the viscoelastic nonlinear work grid and cutoff differed from MATLAB;
- SciPy orthonormal SH coefficients were passed to the LOV3D solver without the required `1/sqrt(4*pi)` convention conversion.

The earlier `[125,125,125]` MATLAB coefficient-input anchor is retired as a basis-mismatch artifact. It must not be used as a physical validation target.

## Publication-facing suite: PASS

After promoting the fast TASK-046 regressions into `scripts/run_science_benchmarks.py`, the complete science suite was rerun with independent PyALMA3 validation on 2026-08-18.

Command:

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

Recorded runner command and result:

```text
/Users/svance/mamba/envs/PPcl/bin/python -m pytest -q -m "" \
  pylov3d/tests/test_analytical.py \
  pylov3d/tests/test_matlab_validation.py \
  pylov3d/tests/test_matlab_validation_ocean.py \
  pylov3d/tests/test_mars.py::TestMass \
  pylov3d/tests/test_mars.py::TestMoI \
  pylov3d/tests/test_mars.py::TestK2 \
  pylov3d/tests/test_mars.py::TestLoveNumberSanity \
  pylov3d/tests/test_mars.py::TestDensityProfile \
  pylov3d/tests/test_energy.py::TestGetEnergy::test_elastic_zero_dissipation \
  pylov3d/tests/test_energy.py::TestGetEnergy::test_io_nonzero_dissipation \
  pylov3d/tests/test_energy.py::TestGlobalDissipation \
  pylov3d/tests/test_energy_multibasis.py \
  pylov3d/tests/test_energy_couplings_matlab_order.py \
  pylov3d/tests/test_io_rheology_spectrum_parity.py \
  pylov3d/tests/test_benchmark_pyalma3.py

69 passed in 404.51s (0:06:44)
```

Environment/provenance captured here:

- Python executable: `/Users/svance/mamba/envs/PPcl/bin/python`
- environment name: `PPcl`
- MATLAB reference generation: MATLAB R2025b (`/Applications/MATLAB_R2025b.app`)
- strict MATLAB artifacts: `data/tests/io/io_uniform_radial_anchor.mat`, `io_raw_grid_energy_anchor.mat`, `io_identical_coefficients_anchor.mat`
- package-version snapshot: not yet archived; capture separately rather than infer versions from unrelated environments.

The fast TASK-046 regressions are part of the routine publication-facing suite. The expensive `Nrbase=50` MATLAB/Python Gate C remains an archived cross-language validation rather than a routine test dependency.

## Status

**TASK-046 numerical validation and publication-facing regression promotion are complete.** The branch is merge-ready subject to normal PR review/CI. Remaining scientific validation work such as the lateral bulk-modulus (`K`) path is explicitly separate from TASK-046.
