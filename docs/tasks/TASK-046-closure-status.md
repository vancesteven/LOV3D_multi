# TASK-046 closure status

## Numerical status

TASK-046 has now passed both the physically faithful raw-grid end-to-end gate and a strict identical-coefficient MATLAB/Python solver-parity gate.

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

## Remaining promotion step

The numerical core of TASK-046 is closed. Before the PR is marked merge-ready, rerun the complete publication-facing science benchmark suite on the final branch head:

```bash
python scripts/run_science_benchmarks.py --with-pyalma3
```

Record the final pytest count, wall time, Python executable/environment, key dependency versions, and MATLAB version/reference artifacts. The fast TASK-046 regressions are now included in `scripts/run_science_benchmarks.py`; the expensive `Nrbase=50` MATLAB/Python Gate C remains an archived cross-language validation rather than a routine unit-test dependency.
