# TASK-046 diagnostic log

This file records chronological diagnostic runs and the scientific interpretation of each result. It complements `docs/VALIDATION_WORKFLOW.md` (canonical test order) and `TASK-046-io-viscoelastic-lateral-energy-validation.md` (task specification).

## Native MATLAB Gate-C anchor

At `Nrbase=50`, the coefficient-input MATLAB cross-check reported:

```text
k_uni(2,m)       = +0.7337217069 - 0.0151236751 i
k_lat(2,0)       = +0.7325399703 - 0.0153355564 i
k_lat(2,+/-2)    = +0.7381214321 - 0.0198692819 i
N coupled modes  = [125, 125, 125]
E_direct uni/lat = 2.1668778416 / 2.8404609804
E_Love   uni/lat = 2.2144024348 / 2.9026033327
mismatch uni/lat = ~2.19% / ~2.19%
```

**Status:** the uniform quantities remain a useful numerical anchor, but the lateral 125-mode result is provisional because that script fed Python `sph_harm_y` coefficients through MATLAB's coefficient-input convention. The original upstream `Consistency_test_Energy.m` instead supplies full lat/lon fields. See `TASK-046-anchor-reassessment.md` and the raw-grid diagnostic.

## Python diagnostic 1: simplified uniform energy path

`Nrbase=10` before commit `f59ea9f`:

```text
native lateral mode counts: [29, 29, 29]
k_uni = +0.7337217052 - 0.0151236753 i
k_lat(2,0)    = +0.7316354724 - 0.0139412820 i
k_lat(2,+/-2) = +0.7360972172 - 0.0170264646 i
E_direct uni/lat = -1.7735878556e-03 / 2.5770609705e+05
E_Love   uni/lat =  2.2144024577e+00 / 2.4784838914e+00
mismatch uni/lat = 100.0801% / 10397631.3687%
```

Interpretation:

- uniform Love numbers and Love-derived energy are already correct;
- direct energy is broken downstream of the validated Love solution;
- the lateral closure is not yet MATLAB-equivalent.

## Python diagnostic 2: full angular contraction on existing recovered fields

After commit `f59ea9f`, reported 2026-08-18:

```text
TASK-046 Gate B/C, Nrbase=10
native lateral mode counts: [29, 29, 29]

forcing (n=2,m=+0)  k_uni=+0.7337217052-0.0151236753i  k_lat=+0.7316354724-0.0139412820i
forcing (n=2,m=-2)  k_uni=+0.7337217052-0.0151236753i  k_lat=+0.7360972172-0.0170264646i
forcing (n=2,m=+2)  k_uni=+0.7337217052-0.0151236753i  k_lat=+0.7360972172-0.0170264646i

direct energy uniform/lateral: 5.9942543799e+04  2.5770609705e+05
Love energy   uniform/lateral: 2.2144024577e+00  2.4784838914e+00
direct/Love mismatch: uniform=2706839.9057%  lateral=10397631.3687%
wall time: 17.9 s
```

Interpretation:

1. Routing the uniform case through the full GSH angular contraction was necessary but not sufficient.
2. Because uniform `k` and `E_Love` remain correct, the remaining uniform failure is specifically in post-solve stress/strain reconstruction.
3. Inspection of the coupled Python energy path found two independent lateral-specific defects:
   - the solver stores coupled state by field blocks, but the old energy recovery retained interleaved-per-mode indexing assumptions;
   - the old energy recovery used only diagonal mean `A1/A2` blocks and omitted the off-diagonal lateral-rheology constitutive terms that MATLAB includes in `get_A1A2`.

## Python diagnostic 3: solver-consistent lateral field recovery

After commits `130c5ae`, `c10cd51`, and `63cbfac`, reported 2026-08-18:

```text
TASK-046 Gate B/C, Nrbase=10
native lateral mode counts: [29, 29, 29]

k values unchanged from diagnostic 2
E_direct uniform/lateral = 5.9942543799e+04 / 6.6828074342e+04
E_Love   uniform/lateral = 2.2144024577e+00 / 2.4784838914e+00
```

Interpretation:

- lateral direct energy fell by a factor of about 3.86 after correcting grouped state ordering and off-diagonal constitutive recovery;
- Love-number solutions remained unchanged, localizing these defects to post-solve energy reconstruction.

## Harness failures discovered during diagnostic 3

Two test-harness issues were exposed and preserved rather than hidden:

1. `scripts/io_uniform_energy_diagnostic.py` initially lacked the repository-root `sys.path` insertion and failed with `ModuleNotFoundError: pylov3d`. Commit `0740360` fixes the launch path.
2. The original multibasis unit test asserted exact equality to the legacy `get_energy_coupled` result. After switching the new path to solver-consistent coupled stress recovery, that equality was no longer a valid invariant. Commit `cb94378` replaces it with the physically required quadratic forcing-amplitude scaling, `E(F/2)=E(F)/4`.

## Python diagnostic 4: corrected A1/A2 recovery pairing

Inspection of the Python propagator showed that the translated `build_A1_A2` names do not correspond directly to the symbols used in MATLAB's post-processing comments: the forward Python ODE uses the returned `A2` matrix in the radial-derivative block and returned `A1` in the angular `/r` block. The energy recovery had been using the opposite pairing.

Commit `68fb191` made post-solve stress recovery consistent with the actual Python forward ODE. User run at `Nrbase=10`, 2026-08-18:

```text
TASK-046 uniform energy diagnostic, Nrbase=10
r range: 0.52975406 .. 1.00000000

direct E, solver-consistent endpoint handling: -2.399564717933e+00
direct E, MATLAB zero-surface convention:      -2.394038555198e+00
MATLAB coefficient-path target at Nrbase=50:    +2.1668778416e+00

surface profile value: +6.722038017069e+00
```

The largest shell contributions are now O(10^-1 to 10^0), rather than O(10^4), and are distributed through the outer mantle/asthenosphere rather than concentrated solely at the surface.

Interpretation:

1. This resolves the catastrophic direct-energy scale error: the magnitude is now O(1), consistent with the independently validated Love-derived dissipation.
2. The zero-surface convention changes the result by only ~0.23%, so the surface endpoint is not the source of the discrepancy.
3. At coarse `Nrbase=10`, `|E_direct|=2.3996`, about 10.7% above the coefficient-path MATLAB `Nrbase=50` magnitude 2.1669. This difference must now be separated into radial-resolution convergence and convention differences.
4. The remaining sign reversal is likely a stress/strain or energy-sign convention issue. Do not flip it ad hoc until compared at matched radial resolution and against the raw-grid MATLAB path.
5. `pytest -q pylov3d/tests/test_energy_multibasis.py` passes: `2 passed in 0.73 s`.

## Rheology-spectrum diagnostic

The MATLAB-style degree-30 work grid / degree-59 nonlinear re-expansion retained 6 rheology modes and produced active solution counts `[43, 41, 41]`, not `[125,125,125]`:

```text
current Python retained rheology modes: 4
current Python active solution counts: [29, 29, 29]
MATLAB-work retained rheology modes: 6
MATLAB-work active solution counts: [43, 41, 41]
MATLAB-work retained degree range: 2..4
spectrum/closure hypothesis: NOT YET CONFIRMED
```

This falsifies the earlier hypothesis that the low Python work-degree alone explains the 29-vs-125 discrepancy. The 125-mode coefficient-path anchor is now treated as provisional because of the documented SH-basis mismatch. The authoritative next comparison is MATLAB's original raw lat/lon input path.

## Code changes through diagnostic 4

- `853a565`: added `scripts/io_uniform_energy_diagnostic.py`.
- `0740360`: fixed standalone repo import path.
- `130c5ae`: added `pylov3d/energy_fields.py` with grouped state ordering and coupled constitutive recovery.
- `c10cd51`: changed multibasis energy to use solver-consistent field recovery.
- `63cbfac`: wired the Io Gate B/C driver to pass each forcing's native couplings/lateral rheology.
- `cb94378`: replaced obsolete legacy-energy equality testing with quadratic forcing-amplitude scaling.
- `68fb191`: corrected the A1/A2 pairing in post-solve stress recovery to match the Python forward propagator.
- `63ad709`: updated the uniform diagnostic to use the solver-consistent recovery.
- `8c40655`: added the rheology-spectrum diagnostic.
- `dc69540`: added the MATLAB raw-grid closure diagnostic.
- `31610fa`: documented why the coefficient-path 125-mode lateral anchor is provisional.

## Current testing order

1. **Uniform direct-energy radial convergence** at multiple `Nrbase` values. Do not change sign conventions during this test.
2. **MATLAB raw-grid closure diagnostic** using the original `Consistency_test_Energy.m` input path.
3. Once the raw-grid active-mode spectrum is known, reconcile Python lateral rheology/closure against that physical anchor.
4. Only then rebuild the quantitative lateral Gate-C reference and run the full direct-vs-Love energy comparison.

Do not use `--nrbase 50 --assert-matlab` with the old coefficient-path lateral targets as a pass/fail gate until the raw-grid anchor replaces them.
