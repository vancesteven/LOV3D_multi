# TASK-041 (A): Mars 3D spatial tidal-response maps

## Why

Everything the project has produced for Mars 3D is spectral — lists of
k_nm coefficients. Nobody has synthesized the actual spatial field: the
time-varying tidal gravity/displacement anomaly over Mars's surface that
the off-forcing spectrum implies. That map is the missing "3D Mars"
deliverable (the PI noticed its absence directly), and it converts "the
spatial information lives in the off-forcing spectrum" from an abstract
claim into a spatial detectability statement for the MaQuIs-class
mission argument: where the signal peaks (Tharsis? dichotomy boundary?
Hellas?), and at what amplitude in measurable units.

A gap found while scoping: Mars has **no committed full-spectrum
artifact**. The Moon has `moon_lateral_spectrum.npz` (n, m, k, h, l +
provenance); Mars's artifact of record is the MATLAB anchor `.mat`,
which carries k only. Displacement maps need h, so part 0 fixes the
asymmetry.

## Scope

**Part 0 — commit the Mars spectrum artifact.** One coupled solve of the
shipped `pylov3d.mars_lateral` configuration (lmax=4, N=115,
`method='variable'`, Nrbase=30, `perturbation_order=2`, unit (2,0)
forcing) saving `docs/figures/proposal/mars_lateral_spectrum.npz` with
n, m, k, h, l, uniform references, and provenance keys (mirror the Moon
artifact's layout, plus `include_degree1`-style provenance if
applicable). Gate: the k array must match the committed `.mat` anchor
per-mode at the established precision before anything downstream uses it.

**Part 1 — synthesis.** Driver `scripts/mars_response_maps.py`:

- Physical scale: reuse `solar_tide_amplitude_parameter` from
  `pylov3d.mars_detectability` — do not invent a second convention. The
  scope caveat from TASK-026 carries over verbatim: the spectrum is the
  response to a unit (2,0) forcing; the real solar tide has m=0/1/2
  components at distinct frequencies. State it, don't re-litigate it.
- Map the **lateral part only**: off-forcing modes plus the forcing-mode
  *shift* Δk20 (subtract the uniform response — the map is the anomaly a
  laterally uniform Mars would not show).
- Observables, amplitude of the periodic signal (|complex amplitude| per
  grid point, via `sh_to_latlon` on the real-form coefficients):
  1. surface gravity perturbation from the induced potential,
     `(n+1)/R · k_nm · U2 · Y_nm`, in µGal;
  2. radial displacement, `h_nm · U2/g · Y_nm`, in mm.
  State both conventions explicitly in the module docstring (the (n+1)/R
  factor's derivation one-liner included).
- Report: peak amplitude and location for each observable; amplitude at
  Tharsis rise, dichotomy boundary, Hellas, InSight site; fraction of
  total map variance by degree.

**Part 2 — figure.** `scripts/proposal_figures/` two-panel map (gravity
µGal, displacement mm), Tharsis/dichotomy/Hellas annotated, colorbar in
physical units. PNG+PDF to `docs/figures/proposal/`.

**Part 3 — documentation restructure.** `docs/MARS_MODEL.md` is titled
"Mars 1D Radial Reference Model" while half the file is 3D work — the
title actively hides the capability. Retitle to cover both, add a short
top-level overview section mapping capability → task sections (1D model,
MC, lateral spectrum, crust substitution, hydration, detectability,
response maps), and add § TASK-041 with the results.

## Constraints

- New driver + figure script only; no solver-module changes.
- Model routing: Sonnet implements, Opus reviews (standing split).
- Anything that touches the proposal text is a separate follow-up once
  the numbers are verified.
- Prose standard: never "genuine" or "honest". Suite green.

## Done criteria

Spectrum artifact committed and anchored against the `.mat`; maps +
figure committed with peak locations and amplitudes reported; MARS_MODEL
restructured; ledger DONE → different-driver verification.
