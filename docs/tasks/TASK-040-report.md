# TASK-040 audit report: dichotomy transition

Audited `python-conversion` at `4dd2eb8` on 2026-08-15. This was a
read-only audit apart from this report. No coupled solve was run.

## Findings

### 1. The committed field export carries stale degree-1 provenance

- `scripts/export_moon_mu_variable.py:10` still frames the exporter solely as
  TASK-035, and `scripts/export_moon_mu_variable.py:18-21` says C00 **and
  degree 1 are always removed**. The shipped API defaults to
  `include_degree1=True`, and the committed export contains 23 entries,
  including `(1,-1)`, `(1,0)`, and `(1,1)`.
- `scripts/export_moon_mu_variable.py:59-65` embeds the TASK-035 description
  into the artifact's `readme` key. Consequently,
  `data/moon/moon_mu_variable_lateral.npz[readme]` identifies the current
  23-entry dichotomy export as the TASK-035 export, even though TASK-035
  anchored the superseded 20-entry field and TASK-038 produced the current
  artifact.
- `scripts/export_moon_mu_variable.py:82-84` calls its listed keys provenance,
  but `scripts/export_moon_mu_variable.py:121-133` writes `include_c20` and
  does not write `include_degree1`. The load-bearing transition flag is
  therefore absent from the artifact's explicit provenance; it can only be
  inferred from the three degree-1 rows.
- `scripts/export_moon_mu_variable.py:154-159` describes this export's spatial
  maximum and margin as 32.63 km and 0.9902. The current spectrum artifact
  records 32.61618237523627 km and 0.9898438019553393; 0.9902 is the
  degree-1-removed field's margin.

### 2. The regenerated MATLAB log identifies the TASK-038 run as TASK-035

- `scripts/moon_lateral_cross_check.m:249-255` still emits a TASK-035 log
  banner, and `data/tests/moon/moon_lateral_cross_check.log:3` therefore says
  `TASK-035: native-MATLAB anchor` while the remainder of that log contains
  the TASK-038 dichotomy-field values (`k20=0.023161283468`,
  `Delta k20=+2.141241e-6`, dominant `(3,+/-1)`). The script's file header
  correctly says it was re-anchored in TASK-038, but that qualification is
  absent from the standalone artifact.

### 3. The k2m driver presents the superseded margin as the current cutoff

- `scripts/moon_k2m_vs_grail.py:12-15` describes the default TASK-031 field
  used by the driver and says its lmax=4 margin is 0.9902. The driver now
  follows the dichotomy-retaining default, whose artifact-of-record margin is
  0.9898438019553393 (reported as 0.9898); 0.9902 belongs to the explicitly
  degree-1-removed field.

## Clean audit areas

### Cross-artifact numbers

Clean. The spectrum artifact records `N=115`,
`k2_uniform=0.02315914222851756`,
`k20=0.023161283468225102`, and
`Delta k20=2.1412397075426526e-6`. The k2m artifact's m=0 `k2m`,
`delta_k2m`, uniform baseline, and mode count equal those values exactly;
its full deltas are 2.1412397075426526e-6, 1.060559087218138e-6, and
1.92499784601452e-6 with mode counts 115/114/111.

The MATLAB `.mat` has the same 115 `(n,m)` modes as the spectrum npz. For the
111 modes above `|k| > 1e-12`, the independently recomputed median relative
error is 1.4603e-11 and the worst is 1.3875e-10 at `(5,5)`, matching the
handoff and model document. Its forcing-mode relative error is 7.1902e-13.
The log's rounded values, `docs/MOON_MODEL.md`, current `docs/HANDOFF.md`
claims, and TASK-039 test pins agree with the artifacts of record. Old tables
in `docs/MOON_MODEL.md` are explicitly headed or introduced as
degree-1-removed/superseded.

### Perturbation-order breakdown

Clean for current claims. Running
`get_active_modes(2, variations, 2, 0)` directly on the committed export's 23
`(n,m)` pairs gives exactly 1 zeroth-order, 41 first-order, and 73
second-order modes (115 total). Removing its three degree-1 pairs and repeating
the derivation gives the same 1/41/73 split for the old 20-pair field.

The remaining `1/42/73` strings are historical or corrective: the TASK-035
handoff records preserve the original report, while the TASK-038 row and the
2026-08-15 verification entries explicitly identify and correct the
116-mode arithmetic error. No occurrence presents 1/42/73 as the current
dichotomy-field result.

### Superseded-number labeling

Clean except for Findings 1 and 3. Occurrences of the old `1.40712e-6`,
`3.13471e-6`, and `7.2081e-8` signatures in the model document are in columns
or sections explicitly labeled degree-1-removed/superseded. Other `0.9902`
occurrences describe the historical TASK-031b/036 field or tests that
explicitly pass `include_degree1=False`. The `2.95e-13` occurrences describe
the Mars anchor precedent or are explicitly comparative/historical, not
claims for the current Moon run.

### MATLAB `py_*` constants

Clean. Every constant at `scripts/moon_lateral_cross_check.m:227-242` equals
the corresponding spectrum-npz value at the digits written, and the `.mat`
embeds the same values. The four pair constants use the negative-m member of
each archived pair exactly:

| Constant | Archived mode | Exact value |
|---|---:|---:|
| `py_k31_abs` | `(3,-1)` | 6.372785331949207e-6 |
| `py_k22_abs` | `(2,-2)` | 3.0301228895909613e-6 |
| `py_k21_abs` | `(2,-1)` | 2.6352536444942246e-6 |
| `py_k33_abs` | `(3,-3)` | 2.0205378604363597e-6 |

The positive-m partners differ only at floating-point noise, so using the
same pair reference does not affect the reported agreement.

### Test-artifact coupling

Clean. `pylov3d/tests/test_moon_lateral.py` is unchanged from TASK-039 commit
`68ac0be`. Its pins still scan the spectrum for the dominant off-forcing mode,
assert the three k2m deltas and cross-artifact m=0 equality, require all three
degree-1 export rows with exact Condon-Shortley symmetry, pin the coupling
channels, and require the flag to change exactly three coefficients. The
full prescribed gate completed with **651 passed, 30 deselected, 2 warnings**
in 128.92 s.
