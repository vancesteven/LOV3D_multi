# Proposal figure captions (draft)

Figures for the NASA Solar System Workings proposal, demonstrating the
working `pylov3d` Mars pipeline. Each figure is provided as a vector PDF
(for print layout) and a 300-dpi PNG (for slides/review). Source scripts:
`scripts/proposal_figures/fig{1..5}_*.py` (each runnable standalone from
the repo root with `venvLOV3Dconv/bin/python`); shared style constants in
`scripts/proposal_figures/common.py`.

All categorical colors use the fixed, colorblind-validated order
`#0072B2, #E69F00, #009E73, #CC79A7, #D55E00` (Okabe-Ito subset; validated
with `dataviz/scripts/validate_palette.js`, light mode: ALL CHECKS PASS —
CVD separation and surface contrast flagged WARN but within the tool's
"legal with secondary encoding" band, which every figure here satisfies via
direct labels).

---

## Figure 1 — `fig1_mars_interior_model`

**Size:** full-width (7.2 in), three-panel (two profiles + one residual
strip stacked below).

**Draft caption:** *The TASK-011 fitted 1-D reference interior model for
Mars (`pylov3d.mars.build_mars_model`), spanning core to surface: liquid
core, lower mantle, upper mantle, and crust, each shown as a step profile
of density (left) and shear modulus (right) against radius (0 km at the
planet's center, 3389.5 km at the surface). Layer densities are the exact
solution of the mass + mean-moment-of-inertia linear system; the shared
mantle shear-modulus scale factor is bisected against the observed tidal
k2. The bottom strip shows the resulting fit residuals for mass, mean
moment-of-inertia factor, and k2, each expressed in units of the published
observational uncertainty and computed live (not hardcoded) via
`pylov3d.mars.mars_moi_factor` and `pylov3d.love.get_love` — all three sit
at zero to within numerical precision, confirming the deterministic fit is
exact to its own targets.*

**Suggested placement:** Early in the Mars case-study section, as the
first "here is the model" figure before the posterior (Fig. 2) and its
tidal/topographic consequences (Figs. 3-4).

---

## Figure 2 — `fig2_mars_posterior`

**Size:** full-width square (7.2 x 7.2 in), 4x4 corner plot.

**Draft caption:** *Posterior distribution of the four free parameters of
the TASK-012 Mars forward model (core density, lower-mantle density,
shared mantle shear-modulus scale, and core radius), from a real `pocomc`
(Preconditioned Monte Carlo) run — n_active=256, Nrbase=15 (a
discretization verified to leave k2 unchanged to 1e-12 relative to the
production Nrbase=100 grid), fixed seed (0), 4301 posterior samples,
effective sample size (ESS) = 4119 (wall time ~882 s = 14.7 min; well
above the ESS>400 acceptance threshold, so no n_active=512 rerun was
needed). Posterior medians +/- 1 sigma: rho_core = 6115 (+95/-98) kg/m^3,
rho_lm = 4136 (+101/-110) kg/m^3, mu_scale = 0.979 (+0.070/-0.064),
R_core = 1836 (+36/-36) km — all consistent with the TASK-011 point fit
(rho_core=6128, rho_lm=4137, mu_scale=0.965, R_core=1830). Diagonal panels are 1-D marginal
histograms; off-diagonal panels are 2-D density (hexbin, single-hue blue).
The orange cross marks the TASK-011 deterministic point fit in every
panel. On the core-radius diagonal, the dashed curve is the independent
Stahler et al. (2021) InSight-seismology prior (1830 +/- 40 km) — the
posterior closely tracks it. This is by design and should be stated
plainly in the text: core density, lower-mantle density, and the shear-modulus scale
are constrained by the mass + mean moment-of-inertia + k2 data; the core
radius itself carries essentially no information from those three
observables (a documented degeneracy — see `pylov3d.mars_mc`,
"Identifiability") and is prior/seismology-driven, not data-driven, in
this parameterization. Chain saved to
`docs/figures/proposal/mars_posterior_chain.npz` for reproducibility.*

**Suggested placement:** Immediately after Fig. 1, as the uncertainty
quantification companion to the point fit — this is the figure that shows
the full Bayesian pipeline (not just a deterministic solve) actually runs
end to end on Mars.

---

## Figure 3 — `fig3_mars_topography_pipeline`

**Size:** full-width (7.2 in), single map panel.

**Draft caption:** *Real MOLA topography (`data/mars/MarsTopo719.shape.gz`)
rendered end to end through the committed pipeline:
`pylov3d.sh_data.load_shape` -> `truncate` to spherical-harmonic degree
90 -> subtract the C00 (mean radius) and C20 (dominant rotational
flattening) terms as an areoid proxy -> `pylov3d.mapping.sh_to_latlon`
synthesis on a 0.5-degree grid. The global minimum lands in the Hellas
impact basin (measured: lat -32.5, lon 62.5E, -6.9 km relative to the
areoid) and the global maximum at Olympus Mons (measured: lat 17.5, lon
133.5W, +22.5 km), matching both bodies' known locations and the
literature summit height of Olympus Mons (~21.9 km) closely. Because a
sign flip, phase error, or normalization bug in the loader or synthesis
step would each independently break this agreement, the figure is a
strong end-to-end validation of the spherical-harmonic pipeline against
the real planet, not just a unit test.*

**Suggested placement:** Can stand alone in a "data pipeline validation"
subsection, or pair with Fig. 4 as "here is the same pipeline machinery
applied first to real topography, then to the tidal response."

---

## Figure 4 — `fig4_mars_tidal_response`

**Size:** full-width (7.2 in), two map panels with one shared colorbar.

**Draft caption:** *Degree-2 tidal response of the TASK-011 fitted Mars
model, computed live via `pylov3d.love.get_love`: radial displacement
h2 . P-bar_2^0(sin lat) (left) and gravitational potential perturbation
k2 . P-bar_2^0(sin lat) (right), sharing one diverging color scale since
both are the same normalized degree-2 zonal shape function scaled by a
different Love number. Fitted values: k2 = 0.169 (matching the observed
0.169 +/- 0.006 from Konopliv, Park & Folkner 2016), h2 = 0.3156, l2 =
0.0516. The m=0 (zonal, longitude-independent) pattern is shown for
single-panel clarity; because Love numbers for a spherically symmetric
1-D reference model depend on degree n only, not order m, these are
exactly the amplitudes of Mars's real, longitude-varying m=2
solar-semidiurnal tide — only the longitude pattern shown here would
differ from the actual tide.*

**Suggested placement:** Immediately after Fig. 3 (shared "spherical
harmonic pipeline" visual language), and/or as the closing "this is what
the fitted model predicts" figure of the Mars case study.

---

## Figure 5 — `fig5_validation_pedigree`

**Size:** single-column (3.5 in), square 1:1 scatter.

**Draft caption:** *Independent validation of the pylov3d coupled 3-D
solver's lateral-variation (perturbation) machinery: |k_pylov3d| vs.
|k_reference| for the five published Weber-Moon MATLAB/Qin test cases
(`pylov3d/tests/test_matlab_validation_ocean.py`), an ocean-bearing
(fluid-outer-core) body chosen specifically to exercise the same
fluid-layer boundary-condition code path used for Mars's liquid core.
Points are colored by perturbation order (1 = direct coupling, 2 =
second-order/forcing-mode coupling); the forcing-mode deviation follows
the tests' exact convention (subtract the uniform k2, apply the
sqrt(2) forcing-geometry factor for the relevant m). Across 16 compared
modes spanning three decades in |k|, agreement is tight: worst-case
relative error 0.27% (moon_3D_LM_10, degree n=2, order m=0, perturbation
order 2 — the forcing-mode deviation term, whose reference value is itself
a small difference of two O(1) numbers), typical order-1 agreement
4.0e-06 relative (median). This validates the coupled 3-D solver to a few
parts in 10^-6 - 10^-3 against an independent perturbation-theory
reference on an ocean-bearing body — the same machinery, applied to
Mars's liquid core, underlies Figs. 1-4.*

**Suggested placement:** Either as the closing "pedigree" figure of the
whole proposal figure set (recommended — it is the credibility anchor for
everything shown for Mars), or in an appendix/supplementary-validation
section if space is tight in the main text.

---

## Figure 6 — `fig6_mars_lateral_spectrum`

**Size:** full-width (7.6 in), two panels (map + spectrum).

**Draft caption:** *TASK-016: the Mars Love-number spectrum excited by
laterally varying crustal rigidity. Left: the crust shell's fractional
rigidity perturbation delta-mu/mu-bar (%), from Airy-compensated
MarsTopo719 topography (degree <= 4, C00 and C20 removed, referenced to
the low-degree GMM-3 areoid rather than the bare sphere; Airy factor
rho_c/(rho_m - rho_c) = 5.8; max|delta_t| = 34.2 km, at Tharsis (lat -8,
lon -106) — precisely where real Mars is known to be substantially
flexurally/dynamically supported rather than Airy-compensated, so this is
also where the model's own approximation is least trustworthy). Right:
the resulting coupled Love-number spectrum |k_nm|, computed live via
`pylov3d.mars_lateral.mars_lateral_love_spectrum` (NumPy coupled path,
`pylov3d.love.get_love` with `mu_variable`), modes labeled (n, m) and
colored by perturbation order (0 = the forcing mode itself, 1 = direct
forcing-rheology coupling, 2 = second-order); +/-m pairs are always shown
or dropped together (they carry identical |k| by rotational covariance).
Real MarsTopo719 has both cosine and sine content at essentially every
(n, m) up to degree 4, which activates N=115 coupled modes at
perturbation_order=2 — well above the design doc's N~15-30 estimate; the
panel shows the forcing mode plus the largest-amplitude modes per order
(of 115 total), annotated in the title. The forcing mode (2,0) dominates
by ~3 orders of magnitude over the largest order-1 mode and ~6 over
order-2, and its own value shifts by only ~5.5e-5 from the uniform
k2 = 0.169 (the lateral rigidity variation is a perturbation on the tidal
response, not a re-fit of it) — driven in part (~a third) by the (4,0)
crustal harmonic alone, the one degree in this cutoff that self-couples
the degree-2 tide at *first* order (see docs/MARS_MODEL.md section 5).
Wall time ~130-180 s at the production grid (Nrbase=30, machine-dependent;
converged to 1.4e-11 relative in k2 against Nrbase=15).*

**Suggested placement:** After Fig. 4/5, as the "proposed work" figure —
this is the Mars application of the mode-coupling machinery Fig. 5
validates against MATLAB, applied to the real MOLA/Airy crustal load
rather than a synthetic test case.

---

## Notes on reproducing

- `venvLOV3Dconv/bin/python scripts/proposal_figures/fig1_mars_interior_model.py`
- `venvLOV3Dconv/bin/python scripts/proposal_figures/fig2_mars_posterior.py`
  (re-runs pocomc from scratch only if
  `docs/figures/proposal/mars_posterior_chain.npz` is absent or `--resample`
  is passed; otherwise re-plots from the saved chain — long pole, ~8-15 min
  wall on a real run)
- `venvLOV3Dconv/bin/python scripts/proposal_figures/fig3_mars_topography_pipeline.py`
- `venvLOV3Dconv/bin/python scripts/proposal_figures/fig4_mars_tidal_response.py`
- `venvLOV3Dconv/bin/python scripts/proposal_figures/fig5_validation_pedigree.py`
- `venvLOV3Dconv/bin/python scripts/proposal_figures/fig6_mars_lateral_spectrum.py`
  (coupled solve at N=115 modes, ~130-180 s wall, machine-dependent)

No file under `pylov3d/` was modified to produce any of these figures.
