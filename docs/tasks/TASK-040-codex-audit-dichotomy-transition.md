# TASK-040 (Codex): audit the dichotomy transition end-to-end

## Why you, and why now

The 2026-08-14 field change (degree-1 dichotomy retained) touched code,
tests, four npz artifacts, two figures, a MATLAB driver, MATLAB-run
artifacts, `docs/MOON_MODEL.md`, `docs/HANDOFF.md`, and the proposal.
Machine A implemented the flip and the doc updates; machine B ran the
MATLAB re-anchor (TASK-038); A verified B. **You are the only engine that
has implemented none of it**, which makes you the right auditor of the
whole. This is an inspection task: find inconsistencies, do not fix them.

Relevant commits, newest first: `81ec788` (A verifies 038), `bb8e843`
(B's TASK-038), `68ac0be` (your TASK-039 + the .m prep), `69a71ab` (the
field flip). Read the `[A]`/`[B]` 2026-08-14/15 ledger entries in
`docs/HANDOFF.md` first for the intended state.

## What to audit

1. **Cross-artifact number consistency.** The same quantities are quoted
   in `data/tests/moon/moon_lateral_cross_check.log`, the `.mat` (load
   with scipy), `scripts/moon_lateral_cross_check.m` (reference
   constants), `docs/figures/proposal/moon_lateral_spectrum.npz`,
   `docs/figures/proposal/moon_k2m_vs_grail.npz`, `docs/MOON_MODEL.md`,
   `docs/HANDOFF.md`, and the test pins in
   `pylov3d/tests/test_moon_lateral.py`. Every place a number appears, it
   must either match the artifact of record (the npz files) or be
   explicitly labeled as describing the superseded degree-1-removed
   field. Report every mismatch with file:line.
2. **Order breakdown.** Re-derive 1/41/73 yourself via
   `pylov3d.couplings.get_active_modes(2, variations, 2, 0)` on the
   shipped field's `(n, m)` pairs (cheap, no solve) and grep the repo for
   any remaining `1/42/73` presented as a current claim rather than as
   the corrected history.
3. **Superseded-field leakage.** Grep for the old field's signature
   numbers — `1.40712e-6` (old Δk20), `0.9902` (old lmax=4 margin),
   `3.13471e-6` (old dominant pair), `7.2081e-8` (old Δk21),
   `2.95e-13`-adjacent anchor claims — and check each occurrence is
   framed as superseded/historical, not as the shipped value. The docs
   deliberately keep old tables; the audit is about *labeling*, not
   presence.
4. **The .m constants.** Verify every `py_*` constant in
   `scripts/moon_lateral_cross_check.m` equals the corresponding npz
   value at the precision written (17 significant digits).
5. **Test-artifact coupling.** Your TASK-039 pins assert against the
   committed npz. B's TASK-038 overwrote `data/moon/
   moon_mu_variable_lateral.npz` (re-export; array content should be
   identical). Confirm the suite still passes and that no pin silently
   weakened.

## Guardrails (MUST follow)

1. **Read-only audit.** Writable allowlist: a single new report file
   `docs/tasks/TASK-040-report.md`. Nothing else — no fixes, no doc
   edits, no test edits. Findings belong in the report.
2. No state-changing git; leave the report uncommitted.
3. No environment changes, no network, no coupled solves
   (`get_active_modes` and pytest are fine).
4. Citation rule: no network, so do not cite.
5. Prose standard: never "genuine" or "honest".
6. Stop on ambiguity and report.

## Verify

```
venvLOV3Dconv/bin/python -m pytest pylov3d/tests/ -q
```

## Done criteria

`docs/tasks/TASK-040-report.md` with: every mismatch found (file:line,
what it says, what the artifact of record says), the 1/41/73 re-derivation
result, the superseded-number labeling sweep result, and an explicit
"clean" statement for each audit area where nothing was found. A finding
count of zero is a valid outcome if it is the true one.
