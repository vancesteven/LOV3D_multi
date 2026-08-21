# Mars 3D alteration-to-gravity forward model

Date: 2026-08-21
Branch: `agent/task-046-multibasis-energy`

## Scientific role

This development replaces the earlier single-thickness/single-density gravity sensitivity rung with a forward path that can consume a radially resolved 3D density-anomaly field from the Task-1 composition/alteration model:

`composition + candidate alteration -> delta rho(lat,lon,r) -> q_lm -> GMM-3 C_lm/S_lm -> orbital gravity`.

It remains a forward model, not an attribution claim. Porosity, crustal thickness, compensation, thermal structure, composition, and alteration can all contribute to the same density coefficients and must be included in the eventual covariance/inference problem.

## Exact finite-shell equation

For a unit-norm real spherical-harmonic density coefficient `rho_lm(r)`, the external potential coefficient at reference radius R is

`q_lm = 4*pi / ((2*l+1)*M*R**l) * integral rho_lm(r) r**(l+2) dr`.

`mars_gravity_coefficients.finite_shell_potential_coefficient` evaluates the piecewise-constant shell integral analytically. The production expression uses radius ratios rather than raw powers of metre-valued radii, keeping the calculation finite through GMM-3 degree 120.

The historical thin-sheet expression remains available and is tested as the small-thickness limit of the exact shell solution.

## 3D map convention

`mars_alteration_gravity.layered_density_gravity_coefficients` accepts density maps with shape

`(nz, 2*lmax, 4*lmax)`

on the native cell-centred LOV3D equiangular grid. The existing MATLAB-faithful `LatLon_SPH` transform contains a deterministic m-dependent half-cell longitude phase. That phase is required for MATLAB raw-grid regression, but it is explicitly removed for new physical Mars density maps so the returned cosine/sine coefficients refer to the supplied geographic coordinates.

The map transform produces unit-norm physical density coefficients, the finite-shell integration produces orthonormal `q_lm`, and `orthonormal_gravity_arrays_to_gmm3` then converts to normalized GMM-3 `C_lm/S_lm`, including the 3389.5 km to 3396 km reference-radius change.

## Candidate-alteration density map

The first explicit material bridge is

`delta_rho = f_h * f_reactive * (rho_hydrated - rho_dry)`.

This deliberately isolates alteration density from other density terms. `f_h` is a candidate altered fraction of the locally reactive component, not a statement that fracture accessibility implies complete hydration.

## Tests added

`pylov3d/tests/test_mars_gravity_coefficients.py` now checks:

- exact finite-shell analytic equivalence;
- thin-sheet limiting behavior;
- depth-dependent shell weighting;
- cancellation of radial density moments;
- numerical stability at spherical-harmonic degree 120;
- existing compensation and altitude attenuation behavior.

`pylov3d/tests/test_mars_alteration_gravity.py` checks:

- alteration-density scaling;
- uniform maps produce no nonzero-degree signal;
- removal of the MATLAB longitude half-cell phase for a non-axisymmetric C/S pair;
- end-to-end degree-2 map -> exact finite-shell gravity agreement;
- array-level conversion to GMM-3 normalization;
- input validation.

The new tests are included in `scripts/run_science_benchmarks.py`.

## Diagnostic

Run

```bash
python scripts/mars_alteration_gravity_demo.py
```

or specify a mode and shell, for example

```bash
python scripts/mars_alteration_gravity_demo.py \
  --degree 21 --order 4 --density-coeff -250 \
  --top-depth-km 5 --bottom-depth-km 25 --altitude-km 300
```

The diagnostic synthesizes a known physical density harmonic, analyzes the map, integrates the finite shell, converts to GMM-3 normalization, and reports the orbital radial-gravity amplitude plus off-target harmonic leakage.

## Validation order

Before interpreting any new 3D gravity result:

```bash
pytest -q pylov3d/tests/test_mars_gravity_coefficients.py \
          pylov3d/tests/test_mars_alteration_gravity.py \
          pylov3d/tests/test_mars_gravity_normalization.py \
          pylov3d/tests/test_mars_gmm3.py \
          pylov3d/tests/test_mars_gravity_background.py

python scripts/mars_alteration_gravity_demo.py
python scripts/run_science_benchmarks.py
```

The first command isolates gravity conventions and physics. The diagnostic then checks a readable end-to-end synthetic case. The publication-facing suite is last because it is broader and includes the slow TASK-046 raw-grid Gate C.

## Current result status

The implementation, tests, benchmark registration, and diagnostic are committed. They have not yet been rerun in the user's development environment after the latest finite-shell and 3D-map changes, so no new pass count is claimed here.

## Next science step

The next useful input is not another arbitrary single harmonic. It is a real Task-1 density field built from a compositionally consistent dry reference plus a candidate alteration field. That will let the same posterior sample predict gravity, seismic properties, rigidity/tidal response, magnetic source geometry, and conductivity, turning the current separate discriminability calculations into a common-state forward model.
