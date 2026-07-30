# MATLAB cross-validation for Enceladus benchmark

**Labels:** validation, matlab

## Description
Cross-validate pylov3d Milestone 2 (lateral variations) against MATLAB LOV3D reference data for the Enceladus 2-layer benchmark case from Rovira-Navarro et al. (2024).

## Current Status: RESOLVED (2026-07-29)
- Test framework: `pylov3d/tests/test_matlab_validation.py`
- MATLAB reference data: `data/tests/enceladus/*.mat`
- **Uniform k2 now matches:** pylov3d 0.0151953 vs MATLAB 0.0151858 (rel err 6.3e-4)
- **Amplitude sweep matches:** 0.03% / 0.13% / 0.01% at 5% / 10% / 25%
- Full suite: 298 passed, 3 skipped, 0 failed.

## Root Cause (the original hypothesis below was WRONG)
The failure was NOT parameter translation — normalized Gg, R, rho, mu, Ks all
match MATLAB to 6 significant digits. Two real bugs:

1. **A1/A2 role swap in `pylov3d/propagator.py`** (`build_aprop` and
   `build_aprop_coupled`): the constitutive matrix A1 (angular /r term) and A2
   (pure-moduli d/dr term) were assigned to each other's blocks. Fixed so
   A2 -> Adotx (d/dr), A1 -> Ax (/r). Masked in CI because `test_analytical.py`
   only checked h2 quantitatively and k2>0 (never k2's value). Fix makes the
   analytic homogeneous-sphere k2 EXACT (0.038704; pre-fix 0.0897, 2.3x wrong).
   Cross-checked against Beuthe (2014) thin-shell formula (0.01505, <1%).
2. **Core Delta_rho0**: MATLAB sets Interior_Model(1).Delta_rho0=0 for the
   ice-ocean interface; the test now passes Delta_rho0=[0.0, rho_core-rho_ice].

Plus a test-harness bug in `test_amplitude_sweep`: it selected the constant
order=0 (2,0) reference row instead of the amplitude-varying order=1 row —
fixed by filtering on `order==1`.

## Remaining (optional follow-up)
- [ ] `test_lateral_love_spectra` still SKIPS: requests exactly 5%/10% amplitude
      but MATLAB reference `amp` grid has 1.06, 2.11, 5.29%... Needs the
      amp3(peak-to-peak) vs amp2(complex-SH) scaling reconciled to run.

### Superseded original root-cause hypothesis (kept for history)
Model parameters from MATLAB's .mlx file not correctly translated to pylov3d's
dimensional input format. [This was incorrect — see Root Cause above.]

## Files
- Test: `pylov3d/tests/test_matlab_validation.py`
- MATLAB data: `data/tests/enceladus/{Q,B}_{10,11,20}.mat`
- MATLAB source: `tests/Test_Enceladus_Two_Layer_Lateral_Variations.mlx`

## Note
This is methodical debugging work - recommend using Opus for efficient parameter matching.
