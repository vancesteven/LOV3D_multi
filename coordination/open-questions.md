# Open questions — LOV3d-genai

Anything ambiguous, destructive, or scientifically consequential goes here and
work **stops** until Steve resolves it. Include what you were doing, what the
options are, and your recommendation.

## 2026-09-28 — claude-lov3d-genai: annual m'!=0 forcing in Berne et al. 2026
Codex (gpt-5.6-sol) and manager reasoning agree the annual body-fixed degree-2 solar forcing is effectively (2,0); first-order (2,0) x l=1 coupling excites only (3,0),(3,1). Berne report (3,2),(3,3) deviations (~2-2.6 sigma) and invoke an l=2,m=1 obliquity-tide mechanism [p3]. Unresolved; roadmap B3 (reproduce Zenodo script) will test it. Steve: decide whether to raise with Berne/Rovira-Navarro before the proposal cites mechanism details. Details: external/SSS_2025_Mars/PROPOSAL_ASSESSMENT_2026-09-28_BERNE2026.md §6.

## 2026-09-28 — claude-lov3d-genai: Berne Zenodo example (10.5281/zenodo.20823473) findings
Recon by Claude general-purpose subagent (Opus 5.5); files in /tmp/claude-1001/berne-zenodo (not in repo). Manager adjudication below.
1. **Annual m'!=0 forcing is non-zero in the example** (`forcing_keplerian.txt`: C21A = 3.80e-9 vs C20A = -1.31e-9). A toy Keplerian eq. 5/7 projection gives ~1e-14 for m'!=0 with a correctly rotating body-fixed longitude, but O(1e-9) when the Sun's body-fixed longitude is sampled once per sol (fixed local time). Manager view: this is NOT necessarily an error — MGS/ODY/MRO are Sun-synchronous, so diurnal/semidiurnal tides sampled at fixed local time can alias into annual signals, and the fixed-local-time forcing may be the physically right model of what the orbits see. But the paper's Methods also say empirical accelerations absorb diurnal tides [p8]. Unresolved; it controls most of the degree-2 and fed degree-3 signal in the example. Suggest asking Berne/Rovira-Navarro how the forcing file was generated.
2. **Sine-term sign convention.** Zenodo `run_forward_shear.m:103-112` maps odd-m sine coefficients with an extra (-1)^M relative to our `src/get_rheology.m`, `pylov3d/mars_lateral._real_sh_to_complex_mu_variable`, and Zenodo's own `example.m:136-137`. Affects (1,-1),(3,-1),(3,-3) structure, S21 forcing, S21/S31/S33 outputs; not equivalent to a rotation. Decisive test planned: run pylov3d on the example's 2-layer model with both conventions and see which reproduces the shipped MATLAB `k2_responses.txt`.
3. The example uses a non-dimensional 2-layer "Mars-like" model (mu_eff=1.72, r_core/R=0.54), not Extended Data Table 2 (not in archive; image-only in paper). B2 must source ED Table 2 elsewhere; the example is still a usable parity anchor (solver is stock LOV3D_open, not LOV3D_multi).
