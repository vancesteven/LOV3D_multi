# Berne et al. 2026 Zenodo example — parity anchor files

Source: "Example Code for 'Tidal Tomography Reveals a Thermal Anomaly
Beneath Mars's Crustal Dichotomy'", A. Berne, I. Matsuyama,
M. Rovira-Navarro, Zenodo, doi:10.5281/zenodo.20823473, license CC-BY-4.0.

- `shear_modulus_coefficients.txt` — real-SH shear-modulus lateral
  variations (percent peak-to-peak), input to the archive's
  `Scripts/run_forward_shear.m`.
- `k2_responses.txt` — the shipped native-MATLAB LOV3D_open Love-number
  responses for degree-2 forcings m' = -2..2 on the archive's
  non-dimensional 2-layer Mars-like model (mu_eff = 1.72,
  r_core/R = 0.54, rho_core/rho_mantle = 1.57, Ks/mu = 3,
  Numerics: Nr = 200, perturbation_order = 2, rheology_cutoff = 2).

These two files are the cross-solver anchor for
`pylov3d/tests/test_berne_zenodo_parity.py`, which also settles the
sine-term real->complex convention question recorded in
`coordination/open-questions.md` (2026-09-28, item 2).
