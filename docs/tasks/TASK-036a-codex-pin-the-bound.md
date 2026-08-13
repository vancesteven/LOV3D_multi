# TASK-036a (Codex): pin the amplitude bound, and compute the Mars equivalent

## Context

`docs/tasks/TASK-036-design-lateral-amplitude-wall.md` diagnoses why both the
Mars and Moon lateral stages hit `max|dmu/mu_bar| -> 1`. **Read it first.**

The short version: `delta_mu/mu_bar = dt * (mu_c - mu_m) / (T * mu_c)` is the
*exact* Voigt volume-fraction average inside a fixed shell of thickness `T`.
It is linear in `dt` because volume fraction is linear in `dt` — not because
anything was expanded to first order. So the bound is shell geometry, not a
solver approximation, and no solver setting can move it.

That diagnosis currently rests on one person reading the code and running
numbers by hand. Your job is parts 1 and 4 of TASK-036's scope: make it
checkable, and extend it to Mars. **No solver runs are needed** — this is
closed-form arithmetic on committed constants.

## Part A — pin the bound with tests

Add tests asserting the algebraic relationship, so the diagnosis is recorded
in something executable rather than in prose that can drift:

1. That `moon_lateral._dmu_ddt_coeff()` equals `(mu_c - mu_m) / (T * mu_c)`
   computed independently from the module's own `LAYER_MU`,
   `CRUST_LAYER_INDEX`, `MANTLE_LAYER_INDEX` and `CRUST_THICKNESS_M`. Compute
   the expected value in the test from those constants — do not hardcode
   `-3.0348e-05`, which would only restate the implementation.
2. That the unity crossing is where the diagnosis says: `|dt|` at
   `|dmu/mu_bar| = 1` is **32.95 km** for the Moon. Assert to a sensible
   tolerance and note in the docstring that this is *below* the 40 km shell,
   so the binding factor is the rigidity contrast (1.2139) rather than the
   shell being full — that is the non-obvious part and it should be written
   down where someone will see it.
3. That the reported margins reproduce:
   `crustal_thickness_diagnostics(lmax=L)['max_abs_dmu_over_mubar']` is
   0.9902 / 1.1531 / 1.2897 for L = 4 / 5 / 6. These are cheap (no solver).
   Mark `slow` only if measured to be slow.
4. That the guard actually guards: `mu_variable_from_topography` raises for
   `lmax=5` and `lmax=6`. If a test for this already exists in
   `test_moon_lateral.py`, do not duplicate it — say so in your report
   instead.

Put these in a new file or extend `pylov3d/tests/test_moon_lateral.py`, your
choice; say which and why.

## Part B — the same arithmetic for Mars

`pylov3d/mars_lateral.py` has the analogous constants (`AIRY_FACTOR`, the
crust/mantle moduli, the 50 km crust). TASK-028 reported the Mars fields
reach `max|dmu/mu_bar| = 0.9689` for the DWAK InSight model.

Compute and report, in a short section appended to `docs/MARS_MODEL.md`:

- the Mars coefficient and the `|dt|` at which the Mars bound reaches unity;
- the shell thickness `T` that would give Mars's worst committed field a 0.8
  margin, the Mars counterpart of the Moon's 64.5 km figure;
- whether Mars's binding factor is the contrast or the shell fullness, as it
  was for the Moon.

**If Mars turns out to be limited for a different reason than the Moon, that
is the interesting result** — say so prominently rather than forcing a
parallel.

## Standing guardrails (MUST follow)

1. **Writable allowlist**: `pylov3d/tests/test_moon_lateral.py`, a new test
   file under `pylov3d/tests/` if you prefer, and `docs/MARS_MODEL.md`
   (append only — do not edit existing sections). Everything else read-only.
2. **No state-changing git**; leave changes uncommitted.
3. No environment changes, no network.
4. **Do not modify `pylov3d/moon_lateral.py`, `pylov3d/mars_lateral.py`, or
   any solver module.** This task is diagnosis, not migration. Changing a
   shell thickness anywhere would move published numbers.
5. Citation rule: you have no network, so **any citation you write is from
   memory and must be flagged for A to verify.** Prefer not citing.
6. Prose standard: never "genuine" or "honest".
7. Stop on ambiguity and report.

## Verify

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
```

Green. Then confirm the new tests are not vacuous: perturb one constant in a
scratch copy (not the module) or reason explicitly about what each assertion
would catch, and report that reasoning. A test that restates the
implementation catches nothing — that lesson has cost this project twice.

## Done criteria

Bound pinned by non-vacuous tests; Mars arithmetic reported with its own
numbers; an explicit statement of whether Mars and the Moon are limited by
the same factor.
