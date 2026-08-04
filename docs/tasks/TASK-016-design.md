# TASK-016 — Design: Mars lateral-variation stage (draft for user approval)

**Status:** APPROVED (user, 2026-08-04): Airy (fewer assumptions; the
interpreted seismic crustal model can be imposed later), n_lv <= 4, fixed
forward runs first. IMPLEMENTATION IN PROGRESS (A/Sonnet).
**Owner:** A (design); implementation split proposed below

## Goal

Turn the committed Mars field data (MOLA shape, GMM-3 gravity) into
laterally varying rigidity for the coupled solver, producing the Mars
Love-number *spectrum* — the response modes beyond (2,0) that lateral
structure excites. This is the Mars application of the machinery validated
on the Weber Moon (fig5), and the "proposed work" the NASA proposal figures
point toward.

## Physical chain (five steps)

1. **Fields in.** `sh_data.load_shape` (MarsTopo719) and `load_shadr`
   (GMM-3), already committed and validated (Hellas/Olympus test).

2. **Crustal thickness variation δt(θ,φ).** Two options:
   - **(2a) Airy compensation from topography** (stage-A default): local
     crustal root scales with surface topography, δt = h·ρc/(ρm−ρc) with
     ρc = 2900, ρm = 3400 (both already in `MARS`). Zero new data; the
     approximation and its limits (Tharsis is partly flexurally supported,
     not Airy) stated in the docs.
   - **(2b) Published InSight-anchored crustal-thickness SH model**
     (e.g. Wieczorek et al. 2022 companion data, Zenodo) — better physics,
     one more pinned dataset. Proposed as a follow-up swap, not the first
     implementation.

3. **δt → lateral rigidity of the crust layer.** The reference model has a
   fixed 50 km crust shell (3339.5–3389.5 km). Where the real crust is
   thicker/thinner than 50 km, the shell contains a different crust/mantle
   volume mix. Define the effective shear modulus of the shell pointwise:
   `mu_eff(θ,φ) = f·mu_crust + (1−f)·mu_um`, with f = local crust fraction
   of the shell from δt (clipped to [0,1]). δμ/μ̄ expanded in real SH via
   the existing `mapping`/conversion machinery → complex-SH `mu_variable`
   entries for the crust layer (the same CSH convention validated against
   MATLAB `get_rheology` in the Moon harness).

4. **Truncation.** Feed degrees n_lv ∈ {1..4} (dichotomy = degree 1
   dominant; Tharsis loads 2–3). Coupled mode count at
   perturbation_order=2 stays modest (N ≈ 15–30); the JAX scan path keeps
   sweeps fast. Cutoff sensitivity reported, not assumed.

5. **Coupled solve → Mars Love spectrum.** `get_love` with `mu_variable`
   (NumPy path first — it is the MATLAB-validated one), JAX path for
   sweeps. Outputs: mode amplitudes k_{nm} per input harmonic and
   amplitude, and a laterally-resolved degree-2 response map (fig4
   upgraded).

## Validation plan

- Zero-amplitude reduction: spectrum collapses to the 1D k2 = 0.169
  (machine precision).
- Order-1 modes scale linearly with input amplitude (slope test, as in the
  Qin comparisons).
- **MATLAB cross-check (B, TASK-014 part 2):** export the exact
  `mu_variable` inputs; B runs native LOV3D on the same laterally-varying
  Mars model. Watch the eta0 empty-vs-NaN convention B documented.
- JAX vs NumPy coupled equivalence on one Mars lateral case (existing
  test pattern).

## Deliverables

- `pylov3d/mars_lateral.py` (field → δt → mu_variable pipeline; the
  δt→mu_eff step body-agnostic where practical)
- Tests (reduction, linearity, JAX/NumPy equivalence)
- `docs/MARS_MODEL.md` section + updated fig4-style figure with the
  lateral spectrum
- Export file of `mu_variable` inputs for B's MATLAB run

## Split

- A: steps 2a/3 science design + review gate (Opus, code+science)
- Sonnet: implementation per approved design
- Codex: candidate for the SH-arithmetic utilities if they grow
  (truncate/percent-p2p conversions already exist)
- B: MATLAB coupled cross-check (014 pt 2) + heavy amplitude/degree sweeps

## Open decisions (user)

1. Start with Airy (2a) or fetch the published crustal model (2b) first?
2. Degree cutoff for the first pass (proposal: n_lv ≤ 4)?
3. Should the lateral amplitude be a free MC parameter from day one
   (extends the pocoMC vector), or fixed-amplitude forward runs first
   (proposal: forward runs first)?
