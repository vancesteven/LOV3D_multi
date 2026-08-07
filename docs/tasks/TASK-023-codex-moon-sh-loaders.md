# TASK-023 (Codex): Moon SH field-data loader hardening + tests

## Goal

The Moon gravity/shape spherical-harmonic data files are now committed under
`data/moon/` (TASK-024). This task (a) makes `load_shape` delimiter-agnostic
so it accepts the whitespace-delimited form these products ship in natively,
and (b) adds loader + physical-validation tests for the Moon files, mirroring
what `test_sh_data.py` and `test_mapping.py` already do for Mars.

No new loader functions are needed — `pylov3d.sh_data.load_shadr` and
`load_shape` already parse both Moon files as stored. This is hardening plus
test coverage, not new capability.

## Standing guardrails (MUST follow — same as every Codex task)

1. **Writable allowlist** (everything else read-only):
   - `pylov3d/sh_data.py` (the one narrow change described below only)
   - `pylov3d/tests/test_sh_data.py`
   - `pylov3d/tests/test_mapping.py`
   - `pylov3d/tests/test_moon_sh_data.py` (new file, if you prefer to keep
     Moon tests separate — either layout is acceptable, pick one and say which)
2. **Never modify `data/moon/**` or `data/mars/**`.** These are checksummed
   archival products; `data/moon/SOURCES.md` pins their sha256. If a test
   needs a variant file, write it to `tmp_path`.
3. **No state-changing git** (no commit, push, branch, stash). Leave changes
   uncommitted.
4. No environment changes (no pip install). No network access is needed or
   available — both data files are already on disk.
5. Never touch `src/` (upstream MATLAB), `docs/HANDOFF.md`, `LICENSE`,
   `NOTICE`, `pyproject.toml`, or any `pylov3d` module other than
   `sh_data.py`.
6. Under 500 lines added per file.
7. Stop on ambiguity: if a physical expectation below does not hold when you
   actually compute it, **do not tune the test until it passes** — report the
   discrepancy with the number you measured. A failing physical check is a
   finding, not a bug in your test.
8. Fast suite must stay green (see Verify).

## Part A — delimiter-agnostic `load_shape`

`pylov3d/sh_data.py:120` currently hardcodes:

```python
table = np.loadtxt(stream, delimiter=",", ndmin=2, dtype=np.float64)
```

Wieczorek `.shape` products ship in two forms in the wild: comma-delimited
(Mars's `MarsTopo719.shape`) and whitespace/fixed-width (the Moon's native
`MoonTopo2600p.shape`, e.g. `   0  0  1737151.19826508  0.000000000000000`).
The committed `MoonTopo719.shape.gz` was reformatted to commas during
truncation specifically to work around this limitation
(`data/moon/SOURCES.md`, "Format deviation"). Removing the limitation means a
future fetch need not manufacture a reformatted file at all.

Change `load_shape` to accept either form. Suggested approach: attempt the
comma parse, and on `ValueError` retry with `delimiter=None` (numpy's
any-whitespace default), rewinding the stream between attempts — or sniff the
first non-blank line for a comma. Either is fine; keep it small and
readable, and keep the existing validation, error messages, and return
contract byte-identical in behavior.

Requirements:
- Mars files must load **bit-identically** to before (test this: assert the
  returned `clm`/`slm` arrays are exactly equal to a pre-change reference you
  capture at the start, or simply that existing Mars tests still pass
  unchanged).
- A whitespace-delimited fixture written in `tmp_path` must load to the same
  arrays as the equivalent comma-delimited fixture.
- A file that is malformed in **both** delimiter conventions must still raise
  a clear error, not silently misparse. Add a test for this — the current
  failure mode with a wrong delimiter is a silent misparse into one column,
  which is exactly what makes this worth guarding.
- Do **not** change `load_shadr`.

## Part B — Moon loader tests

Mirror the existing Mars tests. Read `pylov3d/tests/test_sh_data.py` first
and follow its fixture/parametrization style; where a Mars test has an
obvious Moon analogue, parametrize over both bodies rather than copy-pasting.

Cover, for `data/moon/grgm900c_120_sha.tab` (via `load_shadr`):
- Header round-trip: `r0` = 1738.0 km, `GM` = 4902.799967088640 km³/s²,
  `lmax` = 120, 4π normalization flag.
- J2 consistency: J2 = −C̄20·√5 ≈ 2.0327e-4. Assert against the published
  GRAIL-era lunar J2 ≈ 2.033e-4 with a tolerance loose enough to be a real
  check and tight enough to catch a normalization error (~1%).
- Mass consistency: GM/G against `pylov3d.bodies` catalog id 31 ("Moon")
  `Mass`, to ~5 significant figures.
- Full-triangle completeness at the declared lmax (the truncation must not
  have left a partial degree block): 7,380 data rows for lmax=120 with no
  explicit degree-0 row.

Cover, for `data/moon/MoonTopo719.shape.gz` (via `load_shape`):
- Degree-0 term = 1737151.19826508 m; mean radius matches
  `pylov3d.moon.MOON["R"]` = 1737.151 km.
- Full triangle: 259,560 rows for lmax=719.
- Gzip/plain equivalence (mirror `test_shape_gzip_and_plain_agree`).

## Part C — Moon physical/geographic validation test

Mirror `test_mapping.py`'s `TestMarsTopoHellasIntegration` (read it first —
it synthesizes real topography onto a lat/lon grid via `sh_to_latlon` and
asserts the global minimum lands inside the Hellas basin box, plus a
peak-to-peak sanity range). Build the Moon analogue.

Suggested target: **South Pole–Aitken basin** as the global topographic
minimum. Do **not** trust these coordinates blindly — establish the expected
box from the data itself plus a documented published value, and state in the
test docstring where the expectation came from, as the Mars test does. Note
two Moon-specific complications the Mars test does not have, and handle or
document them:
1. The shape model is in a **principal-axis** frame; check whether the
   longitude convention matches what `sh_to_latlon` assumes (the Mars test's
   convention is the reference).
2. The Moon's shape is dominated by a large center-of-figure/center-of-mass
   offset (degree-1 terms), so the global min/max locations depend on whether
   degree 1 is retained. State explicitly which you use and why.

A defensible weaker check is acceptable if the strong one proves fragile —
e.g. assert the farside mean radius exceeds the nearside mean radius (the
well-established nearside/farside crustal dichotomy), which is robust to
frame conventions. **Report which check you shipped and why.**

Mark any test slower than ~5 s with the existing `slow` marker (see
`pylov3d/tests/conftest.py` and how `test_mapping.py` marks its heavy
synthesis tests).

## Verify (run before finishing)

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/test_sh_data.py pylov3d/tests/test_mapping.py -q -m ""
```

Fast lane must be green with zero failures. Existing Mars tests must pass
**unchanged** — if you had to modify a Mars test to accommodate Part A, that
is a signal Part A changed behavior it should not have; report it instead.

## Done criteria

- `load_shape` accepts comma and whitespace forms; Mars loading unchanged;
  malformed input still errors loudly.
- Moon gravity + shape loader tests pass with the values above.
- One Moon physical/geographic validation test shipped.
- Report: which test-file layout you chose, which geographic check shipped
  and why, any physical expectation that did **not** verify (with the number
  you measured), and the final test counts.
