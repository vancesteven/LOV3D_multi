# Controlled PlanetProfile radial reduction for pylov3d

Date: 2026-08-21
Branch: `agent/task-046-multibasis-energy`

PlanetProfile radial models can contain many more shells than pylov3d's current JAX static limit (`MAX_LAYERS=16`). Direct profile import therefore refuses to decimate a high-resolution structure silently.

## Implemented reduction

`pylov3d.profile_reduction` provides an explicit first-pass reducer for elastic radial artifacts. Adjacent shells are merged greedily subject to three hard rules:

1. total mass is preserved exactly by choosing the merged density from the summed shell masses,
2. a merge never crosses a fluid/solid boundary defined by shear modulus,
3. among allowed adjacent pairs, the merge with the smallest axial-moment perturbation is chosen.

Bulk and shear moduli are volume-averaged within a merged shell. This is a numerical handoff approximation, not a claim of unique effective-medium physics.

## Required diagnostics

For every reduction report:

- original and reduced layer count,
- relative mass change,
- change in C/MR^2,
- mass and C/MR^2 mismatch relative to artifact metadata when supplied.

Mass closure is guaranteed by construction to floating-point precision. C/MR^2 is a diagnostic and must not be assumed preserved.

Run:

```bash
python scripts/reduce_radial_artifact.py path/to/profile.npz --layers 16
```

## Publication gate

Bulk closure alone is not sufficient for a tidal model. Before choosing a reduced profile for science, calculate Love numbers over a sequence of target layer counts and demonstrate convergence. If the desired convergence cannot be reached below `MAX_LAYERS`, increase the static layer limit or develop a more specialized reduction rather than relaxing the convergence requirement.

This gate is executable (2026-08-28): `pylov3d.profile_convergence.love_number_convergence` reduces the same high-resolution artifact independently to each target layer count, solves degree-2 elastic Love numbers per reduction, and `successive_k2_differences` reports the relative k2 movement between successive counts. Run

```bash
python scripts/radial_reduction_convergence.py path/to/profile.npz --layers 6 8 10 12 14 16
```

or `--synthetic 64` for the built-in Mars-like fixture. Choose a layer count only where `|dk2|/|k2|` is far below the science error budget.

Conversion note: a multi-shell fluid region touching the planet center is converted as a liquid core (`mu=0`, ocean flag unset), matching `build_mars_model` and the solver's convention; only fluid shells above a solid interior are flagged as ocean layers.

No viscoelastic averaging rule is asserted here. The current reducer is an elastic-structure handoff and must be extended separately before reducing strongly varying viscosity/anelastic structure.
