# Mars mid-crust seismic hydration diagnostic

## Purpose

This diagnostic connects the proposal-facing Mars hydration endmembers to the InSight mid-crust seismic constraint without yet imposing a poroelastic or metamorphic interpretation. It evaluates each homogeneous isotropic solid-modulus case against the Wright et al. (2024) observational vector through `pylov3d.mars_seismic`.

Observed target (Wright et al. 2024):

- Vp = 4.10 +/- 0.20 km/s
- Vs = 2.50 +/- 0.30 km/s
- rho = 2589 +/- 157 kg/m^3
- approximate depth interval = 11.5--20 km

The equivalent homogeneous isotropic moduli at the observed mean are K = 21.95 GPa and mu = 16.18 GPa. These are a diagnostic conversion, not an assertion that the real mid-crust is homogeneous or nonporous.

## Result

The current global reference crust is far too stiff for the local mid-crust constraint when treated as a homogeneous solid:

| case | K (GPa) | mu (GPa) | Vp (km/s) | Vs (km/s) | chi2 |
|---|---:|---:|---:|---:|---:|
| global reference crust | 70.00 | 30.00 | 6.16 | 3.22 | 115.59 |
| low:Voigt, fh=0.5 | 44.45 | 18.90 | 5.19 | 2.70 | 29.98 |
| low:Hill, fh=0.5 | 37.11 | 15.64 | 4.73 | 2.46 | 9.99 |
| low:Reuss, fh=0.5 | 29.76 | 12.38 | 4.23 | 2.19 | 1.50 |
| central:Voigt, fh=0.5 | 59.15 | 22.20 | 5.85 | 2.93 | 79.03 |
| central:Hill, fh=0.5 | 58.15 | 20.83 | 5.76 | 2.84 | 70.23 |
| central:Reuss, fh=0.5 | 57.16 | 19.46 | 5.67 | 2.74 | 61.93 |
| high:Voigt, fh=0.5 | 70.70 | 27.15 | 6.43 | 3.24 | 141.28 |
| high:Hill, fh=0.5 | 70.70 | 27.00 | 6.42 | 3.23 | 140.43 |
| high:Reuss, fh=0.5 | 70.69 | 26.85 | 6.41 | 3.22 | 139.58 |

## Interpretation

1. Seismology is already discriminating material assumptions that tidal k2 alone cannot. The low-property, connected-weak Reuss limit is the only simple hydrated-solid case in this first sweep that approaches the observed Vp/Vs/rho vector (chi2 = 1.50).
2. This is **not evidence that the InSight mid-crust is serpentinite**. The calculation uses Wright's density as a diagnostic for the hydrated cases and does not yet propagate self-consistent density changes, fracture porosity, liquid saturation, pore aspect ratio, anisotropy, or metamorphic assemblages.
3. The result makes the next model-development step concrete: compare hydrated-solid, fractured-dry, fluid-saturated, and mixed alteration models against one common seismic likelihood.
4. The proposal-facing inference should treat porosity, saturation, pore geometry, protolith/mineralogy, hydration fraction, and mechanical connectivity as separate parameters. A seismic fit alone is expected to be non-unique; the scientific payoff is in combining it with gravity/density, tides/compliance, remanence/history, and future EM conductivity.

## Next validation/development rung

Implement at least two independent rock-physics branches against the same `mars_seismic` likelihood:

- porous/fractured dry-to-saturated branch using a published effective-medium + Gassmann/Biot formulation;
- hydrated/metamorphosed solid assemblage branch using Perple_X-derived K, mu, rho tables and explicit serpentinite/alteration fractions.

The purpose is not to reproduce one published interpretation by construction, but to quantify the degeneracy among liquid-filled porosity, alteration, and mixtures before adding gravity/tidal/magnetic information.
