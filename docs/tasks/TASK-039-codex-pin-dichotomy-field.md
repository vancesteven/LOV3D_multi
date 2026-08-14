# TASK-039 (Codex): pin the dichotomy field and its committed artifacts

## Context

The PI resolved the TASK-031 plan/implementation divergence on 2026-08-14:
`pylov3d.moon_lateral` now defaults to `include_degree1=True`, retaining the
nearside–farside crustal dichotomy (commit `69a71ab`). The regenerated
artifacts carry the new field: `Δk20 = +2.14124e-6`, and a **new dominant
off-forcing pair (3,±1) at 6.37279e-6** — the dichotomy's direct
first-order imprint.

The field flip and its solves are done and documented
(`docs/MOON_MODEL.md`, TASK-031/034 sections). What is *not* yet done is
pinning the new state in executable form, so it cannot drift the way prose
can. That is your job — same shape as your TASK-036a work, which held up
well. **No solver runs are needed**; everything below is either
constant-level arithmetic or reading committed npz artifacts.

## Part A — pin the committed artifacts

Add tests (new file under `pylov3d/tests/`, or extend
`pylov3d/tests/test_moon_lateral.py` — your choice, say which and why)
asserting:

1. `docs/figures/proposal/moon_lateral_spectrum.npz` internal consistency:
   `degree_one_removed == False`, `mode_count == 115`,
   `delta_k2 == approx(2.14124e-6, rel=1e-5)`, and the largest off-forcing
   `|k|` belongs to `(3,±1)` at `approx(6.37279e-6, rel=1e-5)`. Locate the
   maximum by scanning the `n`/`m`/`k` arrays — do not hardcode the index.
2. `docs/figures/proposal/moon_k2m_vs_grail.npz`: the three `Δk2m` are
   `approx(+2.1412e-6 / +1.0606e-6 / +1.9250e-6)` and every
   `|Δk2m|/σ_GRAIL < 0.01` — the three-tier-null guard. Also
   `delta_k2m[0]` equals the spectrum npz's `delta_k2` to 1e-9 relative
   (same configuration, must agree).
3. `data/moon/moon_mu_variable_lateral.npz`: 23 entries, the three
   degree-1 rows present, and the Condon–Shortley symmetry
   `f(n,−m) = (−1)^m conj(f(n,m))` holds exactly.

## Part B — pin the coupling channels behind the (3,±1) dominance

The doc claims (3,±1) dominance follows from the `(2,0)×(1,±1)→(3,±1)`
first-order channel. Make the channel claims executable via
`pylov3d.couplings.coupling_coefficients(n, m, na, ma, nb, mb)`
(signature: equation mode, source mode, rheology/field mode; returns 27
slots, slot 26 is the any-nonzero flag):

1. `coupling_coefficients(3, 1, 2, 0, 1, 1)`: `max|C|` over slots 0–25 is
   `approx(0.7171371656, rel=1e-9)` — the channel exists and is strong.
2. `coupling_coefficients(2, 0, 2, 0, 1, 0)`: **all 26 slots exactly
   zero** — the zonal dichotomy term `dt(1,0)` cannot touch the forcing
   mode at first order (parity 2+1+2 odd). This is the non-obvious pin:
   it means the +52% rise in `Δk20` under the dichotomy is carried by
   second-order paths and by the sectoral degree-1 terms, not by `(1,0)`
   directly. State that in the test docstring.
3. `coupling_coefficients(2, 1, 2, 0, 1, 1)`: `max|C|` is
   `approx(0.7071067812, rel=1e-9)`. Assert the value only — do **not**
   assert an interpretation (the 27-slot array mixes parity families, and
   an earlier interpretive claim about a "missing channel" in this project
   was wrong; values are what the tests should hold).

## Part C — flag consistency

A cheap test that `crustal_thickness_variation(lmax=4)` and
`crustal_thickness_variation(lmax=4, include_degree1=False)` differ
*exactly* on the three degree-1 keys and nowhere else. This pins the flag
to its advertised meaning — it must not touch any other coefficient.

## Standing guardrails (MUST follow)

1. **Writable allowlist**: `pylov3d/tests/test_moon_lateral.py`, a new
   test file under `pylov3d/tests/`, and `README.md` **only if** the test
   count guard forces the counts sentence to change. Everything else
   read-only.
2. **No state-changing git**; leave changes uncommitted. (The commit gate
   for the *driving machines* is gone, but your output is still committed
   by A after review — unchanged from your previous tasks.)
3. No environment changes, no network, no solver runs.
4. **Do not modify `pylov3d/moon_lateral.py`, `pylov3d/couplings.py`, or
   any solver module.**
5. Citation rule: you have no network, so any citation you write is from
   memory and must be flagged for A to verify. Prefer not citing.
6. Prose standard: never "genuine" or "honest".
7. Stop on ambiguity and report.

## Verify

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
```

Green. Then confirm the new tests are non-vacuous: for each assertion,
state what committed-artifact or code change it would catch. A test that
restates the implementation catches nothing.

## Done criteria

Artifacts, channels, and flag pinned by non-vacuous tests; suite green;
report which file you used and the non-vacuity reasoning.
