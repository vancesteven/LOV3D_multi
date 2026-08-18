# TASK-046 anchor reassessment

## Why the 125-mode MATLAB anchor is now provisional

The first native-MATLAB Gate-C artifact (`data/tests/io/io_energy_cross_check.{log,mat}`) used `Interior_Model(3).mu_variable` and `eta_variable` coefficients exported from Python. The MATLAB driver itself documented that those coefficients were produced in pylov3d's `scipy.special.sph_harm_y` basis, while MATLAB `get_rheology.m` interprets coefficient inputs through LOV3D/SPH_Tools conventions. The script explicitly warned that the basis equivalence had not been established and recommended rebuilding the original raw lat/lon fields if the comparison became suspect.

That suspicion is now justified by the 2026-08-18 Python spectrum diagnostic:

```text
current Python retained rheology modes: 4
current Python active solution counts: [29, 29, 29]

MATLAB-work retained rheology modes: 6
MATLAB-work active solution counts: [43, 41, 41]
MATLAB-work retained degree range: 2..4

native MATLAB coefficient-path target active solution counts: [125, 125, 125]
spectrum/closure hypothesis: NOT YET CONFIRMED
```

The MATLAB-work calculation deliberately reproduced the parent code's degree-30 working representation, nonlinear Maxwell transformation, degree-59 re-expansion, and separate real/imaginary two-decade rheology filtering. It still produced only six retained rheology modes and 43/41/41 active solution modes. Therefore the earlier hypothesis that Python's low working degree alone explained 29 versus 125 is rejected.

The leading interpretation is now that the 125-mode anchor is at least partly a consequence of feeding Python-basis coefficients into MATLAB's coefficient-input convention. It remains useful as a regression for that exact code path, but **must not be treated as the authoritative physical Io benchmark** until the basis issue is resolved.

## Authoritative next anchor

The original upstream `tests/Consistency_test_Energy.m` does not use coefficient inputs for the Io asthenosphere. It constructs the prescribed heating pattern on a lat/lon grid, derives full `mu_latlon` and `eta_latlon` fields, and passes those grid structs to `get_rheology.m`.

`scripts/io_matlab_raw_grid_closure_diagnostic.m` now reproduces that original path and reports:

1. the retained asthenosphere complex-rheology `(n,m)` spectrum after MATLAB filtering; and
2. the active-mode counts for the `(2,0)`, `(2,-2)`, and `(2,+2)` forcing components.

This cheap closure diagnostic must be run before generating a replacement full `Nrbase=50` MATLAB energy anchor. If the raw-grid MATLAB result is close to the Python MATLAB-work result (six retained rheology modes and approximately 43/41/41 active solution modes), the earlier 125-mode target should be formally demoted to a basis-mismatched diagnostic artifact.

## Uniform direct-energy diagnostic result

The 2026-08-18 radial diagnostic also ruled out the outer-surface endpoint convention as the dominant source of the Python uniform direct-energy error. The largest shell contributions occur well below the surface, around normalized radii approximately 0.88--0.96, and zeroing the surface row leaves those dominant contributions unchanged.

A stronger implementation inconsistency was then identified: the Python propagator uses the matrices returned by `build_A1_A2` with `A2` as the radial-derivative contribution and `A1/r` as the angular contribution, while the post-solve stress recovery still used `A1*udot + A2*u/r`. Commit `68fb191` changes solver-consistent field recovery to use the same convention as the Python forward propagator. `scripts/io_uniform_energy_diagnostic.py` was updated in `63ad709` to test that corrected recovery directly.

## Current testing order

1. Run the updated uniform diagnostic and determine whether solver-consistent stress recovery collapses the direct-energy error.
2. Run `pytest -q pylov3d/tests/test_energy_multibasis.py` to retain the local bookkeeping and quadratic-forcing-scaling checks.
3. Run the raw-grid MATLAB closure diagnostic. Do not compare against 125 as a physical acceptance criterion until this result is known.
4. Only after the authoritative raw-grid rheology/closure is established should the Python lateral processor be adjusted for parity.
5. Generate a new raw-grid `Nrbase=50` MATLAB energy anchor, then perform the final direct/Love and complex-Love comparison.

The validation record should continue to preserve the original 125-mode artifact, but publication-facing claims must use the raw-grid benchmark because it matches the actual upstream Io science test.
