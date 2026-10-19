# TASK-046 diagnostic log

This file records chronological diagnostic runs and the scientific interpretation of each result. It complements `docs/VALIDATION_WORKFLOW.md` (canonical test order) and `TASK-046-io-viscoelastic-lateral-energy-validation.md` (task specification).

## Native MATLAB Gate-C coefficient-path anchor

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

**Status:** the uniform quantities remain a useful numerical anchor. The lateral `125/125/125` closure is no longer a physical acceptance target because that script fed Python `sph_harm_y` coefficients through MATLAB's different coefficient-input convention. The original upstream `Consistency_test_Energy.m` instead supplies full lat/lon fields. The raw-grid diagnostic below resolves this point decisively.

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

This is also the actual MATLAB operational convention. MATLAB `get_A1A2.m` returns `[A1,A2]`, but `get_solution.m` intentionally calls it as `[A2,A1] = get_A1A2(...)` before both propagation and auxiliary stress recovery. Therefore the corrected Python pairing is parent-code faithful, not merely an internal consistency choice.

Commit `68fb191` made post-solve stress recovery consistent with that convention. User run at `Nrbase=10`, 2026-08-18:

```text
TASK-046 uniform energy diagnostic, Nrbase=10
r range: 0.52975406 .. 1.00000000

direct E, solver-consistent endpoint handling: -2.399564717933e+00
direct E, MATLAB zero-surface convention:      -2.394038555198e+00
MATLAB coefficient-path target at Nrbase=50:    +2.1668778416e+00

surface profile value: +6.722038017069e+00
```

Interpretation:

1. This resolves the catastrophic direct-energy scale error: the magnitude is now O(1), consistent with the independently validated Love-derived dissipation.
2. The zero-surface convention changes the result by only ~0.23%, so the surface endpoint is not the source of the discrepancy.
3. The remaining sign reversal must not be flipped ad hoc; first compare pointwise fields with MATLAB.
4. `pytest -q pylov3d/tests/test_energy_multibasis.py` passes.

## Uniform radial-convergence diagnostic

The corrected uniform direct energy was then run at matched physics for `Nrbase=10,20,50`:

```text
Nrbase=10  E_direct=-2.399564717933   |E|=2.399564717933
Nrbase=20  E_direct=-2.614859604596   |E|=2.614859604596
Nrbase=50  E_direct=-2.754371543041   |E|=2.754371543041

MATLAB coefficient-path Nrbase=50 E_direct = +2.166877841600
```

The magnitude relative errors against the MATLAB `Nrbase=50` uniform anchor are approximately 10.74%, 20.67%, and 27.11%, respectively.

Interpretation:

- the discrepancy does **not** converge away with radial refinement;
- the outer-surface endpoint remains negligible at each resolution;
- because the uniform complex Love number remains essentially exact, the remaining mismatch is isolated to the post-solve radial auxiliary fields or their energy contraction;
- a scalar integrated comparison is no longer sufficient for diagnosis.

## Rheology-spectrum diagnostic

The MATLAB-style degree-30 work grid / degree-59 nonlinear re-expansion retained 6 rheology modes and produced active solution counts `[43, 41, 41]`, not `[125,125,125]`:

```text
current Python retained rheology modes: 4
current Python active solution counts: [29, 29, 29]
MATLAB-work retained rheology modes: 6
MATLAB-work active solution counts: [43, 41, 41]
MATLAB-work retained degree range: 2..4
```

This falsified the earlier hypothesis that the low Python work-degree alone explained 29 versus 125.

## MATLAB raw-grid closure diagnostic: authoritative closure resolved

The original physical Io input path from `tests/Consistency_test_Energy.m` was then run directly in MATLAB using `mu_latlon` and `eta_latlon`, avoiding all cross-code coefficient-basis assumptions.

Observed 2026-08-18:

```text
TASK-046 MATLAB raw-grid closure diagnostic
retained asthenosphere rheology modes: 6
retained degree range: 2..4
retained (n,m,Re(muC),Im(muC)):
  ( 2, -2)  -9.60735661e-08  +5.65536188e-08
  ( 2, +0)  +2.85014363e-07  -1.57998590e-07
  ( 2, +2)  -9.69543634e-08  +5.50299311e-08
  ( 4, -2)  +9.10944254e-10  -1.88740912e-09
  ( 4, +0)  -1.69998059e-09  +4.52976873e-09
  ( 4, +2)  +9.40625838e-10  -1.87279354e-09
active solution counts for m=[0,-2,+2]: [43 41 41]
previous coefficient-path anchor counts: [125 125 125]
Python MATLAB-work-grid diagnostic counts: [43 41 41]
```

Interpretation:

1. **`[43,41,41]` is the authoritative closure for the intended Io physical field.** MATLAB raw-grid and the Python MATLAB-work-grid diagnostic agree exactly in active-mode count.
2. The previous `[125,125,125]` coefficient-path result is a basis-mismatched diagnostic artifact and must not be used as a publication-facing acceptance criterion.
3. The Python general rheology processor still retains only four modes / `[29,29,29]`; it should now be changed specifically to reproduce the six-mode raw-grid physical spectrum, not to chase 125.
4. The six retained degree/order pairs are `(2,-2),(2,0),(2,2),(4,-2),(4,0),(4,2)`.

## CMB auxiliary-field convention

Inspection of MATLAB `get_solution.m` shows that the CMB node participates in the first solid layer's auxiliary-field loop. Python's forward solver already constructs `Aprop_aux[0]` with first-solid-layer properties, but the TASK-046 recovery layer map had labelled that point as core and skipped its stress/strain reconstruction.

Commit `2a4fbee` changes the post-solve recovery map so the CMB node belongs to Python layer index 1, matching MATLAB's first solid layer for auxiliary fields.

## Point-by-point uniform radial anchor: field parity established

The authoritative MATLAB uniform anchor at `Nrbase=50` reports:

```text
Nr=198
Nrlayer=[0 86 61 51]
k=+0.7337217069-0.0151236751i
direct E00=2.166877841569e+00
```

The Python point-by-point comparison reported on 2026-08-18:

```text
shape: (199,24)
max |dr|: 0
k relerr: 3.08e-11

excluding the outermost row:
state U..dPhi relL2 = 3.58e-11
u_GSH          relL2 = 3.96e-11
stress         relL2 = 3.85e-11
strain         relL2 = 5.42e-11
```

Every non-identically-zero stress/strain component agrees at roughly `3e-11` to `7e-11` relative L2. The only large all-row discrepancy comes from the known MATLAB convention of leaving the outermost recovered stress/strain row zero, while Python evaluates it.

Interpretation:

1. **Uniform forward propagation is parent-code equivalent.** The primary state and Love number agree to ~1e-11.
2. **Uniform GSH displacement, stress, and strain recovery are parent-code equivalent throughout the interior.** The earlier A1/A2 and CMB fixes are validated quantitatively.
3. Therefore the remaining ~27% direct-energy discrepancy is **not** in the solver, rheology, constitutive stress recovery, strain recovery, radial grid, or layer assignment.
4. The remaining discrepancy is now isolated to the GSH energy contraction or its treatment of the assembled `+m/-m` forcing fields, phase factors, coupling coefficients, profile sign, or final radial integral.
5. The `ComplexWarning` previously emitted when printing MATLAB `E00` was only an output-casting nuisance; the comparison now uses `real_if_close`.

Commit `1733639` extends `scripts/io_compare_uniform_radial_anchor.py` to compare the Python and MATLAB `E00(r)` profiles directly, including a global-sign test and best-fit scalar diagnostic. This is the next required rung.

## Code changes through this diagnostic stage

- `853a565`: added `scripts/io_uniform_energy_diagnostic.py`.
- `0740360`: fixed standalone repo import path.
- `130c5ae`: added `pylov3d/energy_fields.py` with grouped state ordering and coupled constitutive recovery.
- `c10cd51`: changed multibasis energy to use solver-consistent field recovery.
- `63cbfac`: wired the Io Gate B/C driver to pass each forcing's native couplings/lateral rheology.
- `cb94378`: replaced obsolete legacy-energy equality testing with quadratic forcing-amplitude scaling.
- `68fb191`: corrected the A1/A2 pairing in post-solve stress recovery to match MATLAB/Python propagation.
- `63ad709`: updated the uniform diagnostic to use solver-consistent recovery.
- `8c40655`: added the rheology-spectrum diagnostic.
- `dc69540`: added the MATLAB raw-grid closure diagnostic.
- `31610fa`: documented why the coefficient-path 125-mode lateral anchor was provisional.
- `2a4fbee`: restored MATLAB CMB ownership in auxiliary stress/strain recovery.
- `7dff75c`: added MATLAB `Nrbase=50` uniform radial-field export.
- `d6a9404`: added Python point-by-point radial-field comparison.
- `053e4b9`: made the MATLAB anchor robust to forcing-dependent struct outputs by using cells.
- `f4e54ca`: removed unnecessary non-scalar rheology exports from the MATLAB anchor.
- `1733639`: added direct Python/MATLAB `E00(r)` profile comparison.

## Current testing order

1. **Run the extended energy-profile comparison using the already generated MATLAB anchor:**
   `python scripts/io_compare_uniform_radial_anchor.py`.
2. If the profile differs only by a global sign (`relL2(-Python,MATLAB) << 1` and best-fit `alpha ~ -1`), document and repair the sign convention only.
3. If the best-fit scalar is not approximately `+/-1`, inspect the GSH energy coupling tensor and `+m/-m` assembly term-by-term before touching normalization.
4. In parallel, update Python's viscoelastic rheology processing to reproduce the authoritative six-mode raw-grid spectrum and `[43,41,41]` closure.
5. Only after uniform energy-profile parity and lateral spectrum parity are established should a new raw-grid `Nrbase=50` lateral energy anchor replace the obsolete coefficient-path Gate-C assertions.

Do not use `--nrbase 50 --assert-matlab` with the old 125-mode lateral targets.