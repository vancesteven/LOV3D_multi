# TASK-042 (design only): laterally varying Mars mantle — the second 3D layer

## Question

The Mars 3D stage varies one layer: crustal rigidity from Airy-compensated
topography. The solver already supports `mu_variable` in any layer. The
science question this ticket designs (and does NOT implement): what does a
defensible laterally varying *mantle* look like for Mars, and what tidal
signature does it add?

Physical target: the Tharsis thermal anomaly / elastic-lithosphere
thickness variation — long-wavelength mantle temperature structure that
perturbs rigidity at depth, where the tidal strain energy for k2 actually
concentrates. A mantle-depth anomaly couples to the tide differently from
a crustal one (different radial sensitivity kernel), so the two stages are
distinguishable in principle — that distinguishability is the science.

## Design deliverables (no solves)

1. **Parameterization options**, each with its data source and free
   parameters, kept pluggable per the standing Mars modeling roadmap
   (open parameterization now, Perple_X/PlanetProfile petrology later;
   per-eval cost low for pocoMC):
   a. thermal: δT(θ,φ) at fixed depth from a Tharsis-centered
      low-degree model, δµ via a stated dµ/dT;
   b. lithospheric: elastic-thickness map (published Te models) mapped to
      an effective µ perturbation of the lithospheric layer;
   c. agnostic: low-degree δµ/µ̄ coefficients as free parameters bounded
      by positivity — the pocoMC-ready option.
2. **Positivity bound for the mantle layer** — the fixed-shell Voigt
   arithmetic redone for the mantle layer's thickness and contrast, before
   any solve (the TASK-036 lesson: the bound is checkable in closed form).
3. **Radial sensitivity argument**: where dµ perturbations move k2 most,
   from the existing 1-D solution's stress/strain profile — predicts
   whether a mantle stage is a bigger or smaller lever than the crust
   stage per unit δµ/µ̄.
4. **Cost estimate**: N at lmax=2–4 with two laterally varying layers
   (mode closure may grow), wall/RSS extrapolated from committed timings.
5. **Decision memo**: recommend one parameterization to implement first,
   with the test that would falsify its usefulness.

## Constraints

Design document to `docs/tasks/TASK-042-design.md` when executed; no code,
no solver runs beyond reading committed artifacts. Any engine may claim.
Prose standard: never "genuine" or "honest".
