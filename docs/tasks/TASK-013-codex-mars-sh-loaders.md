# TASK-013 — Codex handoff: Mars spherical-harmonic data loaders

**Status:** IN-PROGRESS (Codex 5.6)
**Owner:** CODEX
**Parent:** M6 (Mars). The data files are already fetched, verified, and
documented in `data/mars/SOURCES.md` (read it first). A separate agent is
concurrently building `pylov3d/forward.py` / `pylov3d/mapping.py` /
`pylov3d/mars_mc.py` in this same working tree — you will see uncommitted
files that are not yours. IGNORE all files outside your allowlist.

This spec is self-contained. You (Codex) have no other context.

## GUARDRAILS — hard limits

1. **Files you may write** (nothing else):
   - `pylov3d/sh_data.py` (create)
   - `pylov3d/tests/test_sh_data.py` (create)
   - This file (append to Completion notes only)
2. No state-changing git (commit/add/push/pull/branch/checkout/reset/stash).
3. No environment changes; no network. Runner:
   `venvLOV3Dconv/bin/python` from the repo root. If broken, STOP and report.
4. `src/`, `tests/*.mlx`, `docs/HANDOFF.md`, `data/mars/*` are read-only.
5. No deletions; 500-line cap per file.
6. On ambiguity: stop, record the question in Completion notes, finish the
   unambiguous parts.
7. Full suite must stay green. Baseline is a moving target this session
   (another agent is adding tests concurrently): record the pass count you
   observe BEFORE your change and verify the count after = before + yours,
   with zero failures either way.

## Goal

`pylov3d/sh_data.py`: loaders turning the two raw files in `data/mars/`
into plain NumPy coefficient arrays, with validation. No pyshtools
dependency — parse the text formats directly.

### 1. `load_shadr(path) -> dict`

Parses a PDS SHADR spherical-harmonic file (`data/mars/gmm3_120_sha.tab`).
Format: first row is a header
`r0_km, GM_km3s2, uncertainty, lmax, lmax, normalized_flag, ref_lon, ref_lat`
(comma-separated, fixed-width-ish floats); subsequent rows are
`l, m, Clm, Slm, sigmaC, sigmaS`. Return dict with keys:
`r0_m` (float, meters), `gm` (float, m^3/s^2 — convert from km^3/s^2),
`lmax` (int), `clm` and `slm` ((lmax+1, lmax+1) float64, [l, m] indexed,
zeros where undefined), `sigma_clm`/`sigma_slm` same shape. Degree 0 C00 is
absent from the data rows (starts at l=1): set C00 = 1.0 (standard for
normalized gravity fields).

### 2. `load_shape(path) -> dict`

Parses a Wieczorek `.shape` file (`data/mars/MarsTopo719.shape.gz` — accept
both gzipped and plain paths, sniff by extension). Rows: `l, m, Clm, Slm`
(comma-separated), coefficients in METERS (this is a shape/radius
expansion, not a potential). Return `lmax`, `clm`, `slm` as above (C00 here
IS present in the file and equals the mean radius ~3.3895e6 m — keep it).

### 3. `truncate(coeffs_dict, lmax_new) -> dict`

Return a copy truncated to a lower lmax (both loaders' outputs).

## Tests — `pylov3d/tests/test_sh_data.py`

Validation values come from `data/mars/SOURCES.md` — treat them as the
ground truth to assert against:

1. GMM-3: header parse (r0 = 3.396e6 m, GM = 4.2828372854187757e13 m^3/s^2
  to 1e-9 relative, lmax = 120); `C20 == -8.750211323545289e-4` to 1e-15
  abs; C00 == 1.0; clm/slm shapes; slm[:, 0] all zero (m=0 sine terms);
  sigma arrays nonnegative and nonzero somewhere.
2. MarsTopo719: C00 == 3.38950012207057e6 m to 1e-6 relative; lmax == 719;
  spot-check the l=1 row values against the file's own first lines
  (read them independently in the test with a 3-line manual parse);
  gz and (if you create one in tmp_path by gunzipping) plain-file loads
  agree exactly.
3. truncate: lmax honored, coefficients preserved, originals unmodified.
4. Determinism: loading twice gives identical arrays.
5. A degree-2 physical check tying gravity to the 1D model work: J2 =
  -C20 * sqrt(5) ≈ 1.9566e-3 within 1e-6 (this is the J2 used for the
  mean-MoI correction in `docs/MARS_MODEL.md` — assert consistency with
  the value 1.9555e-3 cited there only loosely, abs 2e-6, and note the
  two products differ slightly).

Done criteria: your tests pass; suite failure count 0; pass count =
(baseline you measured) + (your test count). Append Completion notes:
what you built, measured baseline/after counts, line counts, deviations.

## Completion notes

_(Codex appends here.)_
