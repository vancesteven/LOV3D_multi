# TASK-046 diagnostic log

This file records chronological diagnostic runs and the scientific interpretation of each result. It complements `docs/VALIDATION_WORKFLOW.md` (canonical test order) and `TASK-046-io-viscoelastic-lateral-energy-validation.md` (task specification).

## Native MATLAB Gate-C anchor

At `Nrbase=50`, native MATLAB LOV3D gives:

```text
k_uni(2,m)       = +0.7337217069 - 0.0151236751 i
k_lat(2,0)       = +0.7325399703 - 0.0153355564 i
k_lat(2,+/-2)    = +0.7381214321 - 0.0198692819 i
N coupled modes  = [125, 125, 125]
E_direct uni/lat = 2.1668778416 / 2.8404609804
E_Love   uni/lat = 2.2144024348 / 2.9026033327
mismatch uni/lat = ~2.19% / ~2.19%
```

This is the quantitative parent-code target.

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
- the lateral closure is not MATLAB-equivalent (29 vs 125 modes).

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
2. Because uniform `k` and `E_Love` remain correct, the remaining uniform failure is specifically in post-solve stress/strain reconstruction or its radial endpoint convention.
3. Inspection of MATLAB `get_solution.m` shows the outermost `u_dot`, stress and strain row is intentionally not populated in its layer loop; Python currently evaluates that row. A dedicated uniform diagnostic measures whether this endpoint dominates the direct energy.
4. Inspection of the coupled Python energy path found two independent lateral-specific defects:
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

- uniform energy is unchanged, as expected, because this patch addressed coupled/lateral field recovery only;
- lateral direct energy fell from `2.5770609705e+05` to `6.6828074342e+04`, a factor of about 3.86, demonstrating that grouped state ordering and off-diagonal constitutive terms were materially important;
- the remaining large direct-energy scale error is therefore not solely a lateral constitutive-recovery issue, because the uniform control has essentially the same order-of-magnitude failure;
- Love-number solutions remain unchanged, further localizing this family of defects to post-solve energy reconstruction rather than the tidal forward solve.

## Harness failures discovered during diagnostic 3

Two test-harness issues were also exposed and are preserved rather than hidden:

1. `scripts/io_uniform_energy_diagnostic.py` initially lacked the repository-root `sys.path` insertion used by the other standalone scripts and failed with `ModuleNotFoundError: pylov3d`. Commit `0740360` fixes the script launch path.
2. The original multibasis unit test asserted exact equality to the legacy `get_energy_coupled` result. After switching the new path to solver-consistent coupled stress recovery, that equality is no longer a valid invariant: the old routine intentionally represents the superseded reconstruction. The observed arrays differed substantially (`[0.052857,-0.014867,0.047332]` versus `[0.034882,0.498189,0.152882]`). Commit `cb94378` replaces this obsolete parity assertion with the physically required quadratic forcing-amplitude scaling, `E(F/2)=E(F)/4`.

## Code changes after diagnostic 2/3

- `853a565`: added `scripts/io_uniform_energy_diagnostic.py`.
- `0740360`: fixed standalone repo import path in that diagnostic.
- `130c5ae`: added `pylov3d/energy_fields.py`, recovering fields with the solver's grouped state ordering and, when lateral couplings are supplied, the same `build_A1_A2_coupled` constitutive matrices used by the forward propagator.
- `c10cd51`: changed multibasis energy to use the solver-consistent field recovery.
- `63cbfac`: wired the Io Gate B/C driver to pass each forcing's native couplings and the lateral rheology into field recovery.
- `cb94378`: replaced obsolete legacy-energy equality testing with quadratic forcing-amplitude scaling.

## Next required runs

Run in this order:

```bash
python scripts/io_uniform_energy_diagnostic.py
pytest -q pylov3d/tests/test_energy_multibasis.py
```

Interpret the first command before changing any global energy normalization. The second protects the revised multibasis bookkeeping and physical amplitude scaling.

The ordinary `Nrbase=10` Gate B/C driver should be repeated after the independent rheology-spectrum patch described below. Do not run or interpret `--nrbase 50 --assert-matlab` until both the uniform direct-energy reconstruction and the 29-vs-125 lateral spectrum mismatch are resolved.

## Remaining independent spectrum issue

MATLAB coefficient-based viscoelastic rheology uses a degree-30 working representation and re-expands the nonlinear complex shear field through degree 59 before applying its two-decade rheology cutoff. The current Python path derives its working degree from the low-degree input spectrum and therefore truncates nonlinear harmonics before the cutoff. This remains the leading explanation for 29 versus 125 active solution modes and is the next spectrum-level repair after the uniform diagnostic is understood.