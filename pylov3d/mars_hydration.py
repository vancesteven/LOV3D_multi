"""Hydration-front tidal-signature module (TASK-021).

Quantifies the proposal's core hypothesis: a downward-propagating crustal
hydration (serpentinization) front on Mars produces laterally varying
crust rigidity with a tidal signature. Built on the already-validated
TASK-016 lateral machinery (:mod:`pylov3d.mars_lateral`, Airy/areoid
crustal-thickness field, MATLAB-validated to 3e-13) and the TASK-011 1D
reference model (:mod:`pylov3d.mars`) -- no ``pylov3d`` module modified.

Physical model (full numbers, sources, and every review deviation:
``docs/MARS_MODEL.md``, "Hydration-front tidal signature (TASK-021)")
---------------------------------------------------------------------------
1. **Serpentinite elastic properties (web-verified).** ``mu_serp/mu_crust``,
   ``K_serp/K_crust`` against ``mu_crust``=``LAYER_MU_CRUST``=30 GPa
   (Christensen-Mooney average basalt -- see the reference-crust
   sensitivity note in the doc; this denominator drives the result more
   than the serpentinite bracket) and ``K_crust``=``LAYER_KS[3]``=70 GPa,
   from published Vp/Vs/rho: central 0.48 mu / 0.69 K (Christensen 1966,
   JGR 71); bracket low 0.26 mu / 0.27 K (Falcon-Suarez et al. 2017, GJI
   211); bracket high 0.81 mu / 1.02 K (Christensen 2004, Int. Geol. Rev.
   46 -- the K-high value is that paper's antigorite endmember, at/above
   1.0; the *mu* antigorite endmember, 1.16, is excluded from the mu
   bracket, kept only as a caveat -- an antigorite-dominated real front
   could show much less mu signal than modeled here).

2. **Hydration geometry.** ``t_h(theta,phi) = f_h * t_crust(theta,phi)``,
   ``f_h`` in [0,1], ``t_crust = 50 km + dt`` from the TASK-016
   areoid-referenced Airy field
   (:func:`pylov3d.mars_lateral.crustal_thickness_variation`, reused
   as-is). Rationale: thicker crust hosts proportionally more hydratable
   volume -- a stage-1 coupling. **Clip vs. spectral (deviation):** the
   spec's "clip t_crust to [0,50 km]" and "evaluate spectrally (linear in
   SH coefficients)" are in tension (clipping is nonlinear/pointwise);
   adopted resolution: linear form for the actual mu/K coefficients, clip
   enforced as a **checked bound** instead
   (:func:`hydration_geometry_diagnostics`). Verified: never violated
   over ``f_h`` <= 0.5 (max t_h_linear = 42.1 km < 50 km).

3. **Effective shell rigidity (mean + lateral split).** ``dt`` has zero
   spatial mean, so ``delta_mu/mu_crust = (mu_serp-mu_crust)/mu_crust *
   t_h/50km`` splits cleanly into a mean (degree-0) term
   ``f_h*(mu_serp-mu_crust)/mu_crust`` and a lateral (degree>=1) term
   ``f_h*(mu_serp-mu_crust)/(mu_crust*50km)*dt(theta,phi)`` (same for K).
   The **mean term** -- first-order important, not dropped -- becomes a
   modified crust ``mu0``/``Ks0`` for a fresh 1D solve,
   ``mu0_soft=(1-f_h)*mu_crust+f_h*mu_serp`` (:func:`build_hydrated_mars_model`,
   pinned by ``test_mars_hydration.py::TestMeanShiftConsistency``). The
   **lateral term** feeds ``get_love``'s ``mu_variable``/``K_variable``
   coupled path via :func:`pylov3d.mars_lateral._real_sh_to_complex_mu_variable`
   (reused, not reimplemented); its normalization is relative to the
   *softened* mu0/Ks0, not the original 30/70 GPa -- see
   :func:`hydration_lateral_variables`'s docstring (pinned by
   ``TestMeanShiftConsistency::test_lateral_normalization_pin_at_10_mode``).

4. **K_variable gap in every rheology branch** (not elastic-specific --
   corrected from an earlier draft). ``process_lateral_variations``
   hardcodes ``K_amp=0`` regardless of ``K_variable`` in *both* its
   elastic (``rheology.py`` ~424) and viscoelastic (~480) branches
   (``LateralRheology.K_amp``'s docstring says as much; confirmed by
   ``test_lateral_rheology.py::test_elastic_K_amp_zero``). The low-level
   ``K_amp`` channel *is* wired through the coupled solver (validated by
   ``test_jax_coupled_scan.py::test_nonzero_K_amp_selection``, injecting
   it the same way this module does). :func:`get_love_hydrated` adds the
   missing step by hand -- see the ``K_ROW0_FACTOR`` comment above that
   function (row-0 normalization) and its own docstring (why
   ``K_variable`` must also reach ``process_lateral_variations``, or its
   ``(n,m)`` union silently misses K's modes whenever ``mu_variable`` is
   empty). No ``pylov3d`` module is modified.

5. **Forward + detectability.** :func:`hydration_forward_sweep` runs the
   ``f_h`` grid x ratio bracket, separating mean/lateral/total k2;
   :func:`detectability_summary` compares ``Delta k2(f_h)`` to
   ``sigma_k2``=0.006 and reports the precision that would resolve
   ``f_h=0.1`` (no mission claims). **k2 is blind to WHERE the hydration
   is**, not just how much: the lateral (front-shaped vs. uniform)
   contribution is a small fraction of the total everywhere sampled --
   k2 constrains the globally-averaged hydrated fraction; the front's
   lateral signature lives almost entirely in the off-(2,0) Love
   spectrum (TASK-016 fig6 machinery), whose detectability is future work.

6. **Stated limitations.** Density channel not modeled (solver supports
   mu/K only; companion observable is gravity/MoI, future work). Elastic
   only (``mars.py`` Caveat 1 applies identically). ``f_h*t_crust`` is a
   stage-1 choice (water-table-controlled depth is stage 2). Airy-at-
   Tharsis caveat inherits from TASK-016. Reference-crust/antigorite
   caveats above.

Runtime (grid reduction, spec permits this; full numbers, docs/MARS_MODEL.md section 4)
---------------------------------------------------------------------------
Validated config (``lmax=4``, ``Nrbase=30``) costs ~130-200 s/solve; an
18-point sweep at that cost runs tens of minutes, over the ~5 min guard.
Default sweep keeps ``Nrbase=30``, reduces ``lmax`` 4->2 (~10 s/solve,
degree-1 dominant in ``dt``'s RMS); full sweep ~150-250 s. A 3-point
lmax=4 spot check (``f_h``=0.1/0.3/0.5, central) is reported in the doc:
changes the *lateral* contribution (mean term is lmax-independent) but
not the qualitative conclusion.
"""

from __future__ import annotations

import time

import numpy as np

from .couplings import get_couplings
from .grid import set_boundary_indices
from .love import extract_love_numbers, get_love
from .mars import LAYER_KS, LAYER_MU_CRUST, MARS, MARS_FORCING_TD, build_mars_model
from .mars_lateral import (
    CRUST_LAYER_INDEX,
    _real_sh_to_complex_mu_variable,
    crustal_thickness_variation,
)
from .rheology import get_rheology, process_lateral_variations
from .solver import get_solution
from .types import make_forcing, make_numerics

# Serpentinite elastic properties (web-verified; module docstring, item 1)
K_CRUST = LAYER_KS[CRUST_LAYER_INDEX]  # 70e9 Pa, mars.py TASK-011 crust bulk modulus

MU_SERP_RATIO_CENTRAL = 0.48   # Christensen (1966), JGR 71, 5921-5931
MU_SERP_RATIO_BRACKET = (0.26, 0.81)  # Falcon-Suarez et al. (2017) GJI 211 / Christensen (2004)
K_SERP_RATIO_CENTRAL = 0.69    # Christensen (1966), same sample
K_SERP_RATIO_BRACKET = (0.27, 1.02)   # Falcon-Suarez et al. (2017) / Christensen (2004) antigorite

# (mu_ratio, K_ratio) scenario pairs used by the forward sweep.
RATIO_SCENARIOS: dict[str, tuple[float, float]] = {
    "low": (MU_SERP_RATIO_BRACKET[0], K_SERP_RATIO_BRACKET[0]),
    "central": (MU_SERP_RATIO_CENTRAL, K_SERP_RATIO_CENTRAL),
    "high": (MU_SERP_RATIO_BRACKET[1], K_SERP_RATIO_BRACKET[1]),
}

SIGMA_K2 = MARS["k2_sigma"]  # 0.006, Konopliv, Park & Folkner (2016)

DEFAULT_F_H_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_LMAX = 2       # reduced from TASK-016's 4; see module docstring, "Runtime"
DEFAULT_NRBASE = 30    # held at the validated value


# Step 1/3 (mean term): softened crust moduli + modified 1D model
def mean_softened_crust_moduli(
    f_h: float, mu_ratio: float = MU_SERP_RATIO_CENTRAL, K_ratio: float = K_SERP_RATIO_CENTRAL,
) -> tuple[float, float]:
    """Volume-mixed (mean, degree-0) crust ``(mu0, Ks0)`` [Pa] at hydrated
    fraction ``f_h`` (module docstring, item 3). ``f_h=0`` returns exactly
    ``(LAYER_MU_CRUST, K_CRUST)`` bit-for-bit (needed for the exact
    zero-amplitude reduction test)."""
    mu_serp = mu_ratio * LAYER_MU_CRUST
    K_serp = K_ratio * K_CRUST
    mu0 = (1.0 - f_h) * LAYER_MU_CRUST + f_h * mu_serp
    Ks0 = (1.0 - f_h) * K_CRUST + f_h * K_serp
    return mu0, Ks0


def build_hydrated_mars_model(
    f_h: float,
    mu_ratio: float = MU_SERP_RATIO_CENTRAL,
    K_ratio: float = K_SERP_RATIO_CENTRAL,
    mu_scale: float | None = None,
):
    """TASK-011 Mars model with the crust layer's ``mu0``/``Ks0`` replaced
    by the mean-softened values (module docstring, item 3, mean term).
    Everything else (densities, mantle mu0, geometry) is exactly
    :func:`pylov3d.mars.build_mars_model`'s output -- only the crust
    layer's two moduli are overwritten via ``._replace`` (no
    ``pylov3d.mars`` code is modified)."""
    model = build_mars_model(mu_scale=mu_scale)
    mu0_crust, Ks0_crust = mean_softened_crust_moduli(f_h, mu_ratio, K_ratio)
    mu0_new = model.mu0.at[CRUST_LAYER_INDEX].set(mu0_crust)
    Ks0_new = model.Ks0.at[CRUST_LAYER_INDEX].set(Ks0_crust)
    return model._replace(mu0=mu0_new, Ks0=Ks0_new)


# Reference-crust shear moduli for crust_reference_sensitivity (S1,
# review), web-verified: InSight in-situ crustal Vs, near the lower end of
# the published range (~2.5 km/s @ ~2700 kg/m3 -> ~17 GPa; Knapmeyer-
# Endrun et al. 2021, Science 373, 438-443 -- published estimates span
# ~2.5-3.2 km/s, all below the shipped 30 GPa); shipped default
# (Christensen-Mooney average basalt, 30 GPa); oceanic gabbro (~4.0 km/s
# @ ~2830 kg/m3 -> ~45.3 GPa); unaltered peridotite (~4.5 km/s @ ~3350
# kg/m3 -> ~68 GPa) -- standard crustal/mantle values (Christensen 1966).
CRUST_REFERENCE_MU_GPA: dict[str, float] = {
    "InSight in-situ crust": 17.0, "shipped (Christensen-Mooney basalt)": 30.0,
    "gabbro": 45.3, "peridotite": 68.0,
}


def crust_reference_sensitivity(
    f_h: float = 0.5,
    mu_serp_abs: float = 14.4e9,
    mu_crust_refs_gpa: dict[str, float] = CRUST_REFERENCE_MU_GPA,
    mu_scale: float | None = None,
) -> list[dict]:
    """S1 (review): ``Delta k2(f_h, mu-only)`` across alternative crust
    shear-modulus *references* (:data:`CRUST_REFERENCE_MU_GPA`), holding
    the absolute serpentinite modulus fixed (14.4 GPa, central-ratio
    value) rather than its ratio to whichever reference is in force --
    isolates how much the shipped 30 GPa denominator itself drives the
    result. K stays at its default (unsoftened) -- mu-only, per spec.
    ``mu0_soft=(1-f_h)*mu_crust_ref+f_h*mu_serp_abs``, built by hand on
    :func:`pylov3d.mars.build_mars_model`'s raw output. ``delta_k2`` is
    *self-referenced* per row -- ``k2(mu0_soft) - k2(mu_crust_ref,
    unsoftened)`` -- NOT relative to the single global ``MARS["k2"]``,
    which is tied to the original fitted 30 GPa model only; sharing that
    baseline would conflate the hydration signal with the baseline-k2
    drift from swapping crust references alone (module docstring,
    "mu_crust is not freely adjustable"). Two 1D solves per reference."""
    ref = build_mars_model(mu_scale=mu_scale)
    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=30, perturbation_order=2)
    out = []
    for label, mu_gpa in mu_crust_refs_gpa.items():
        mu_crust_ref = mu_gpa * 1e9
        model_base = ref._replace(mu0=ref.mu0.at[CRUST_LAYER_INDEX].set(mu_crust_ref))
        love_base, _, _ = get_love(model_base, forcing, numerics)
        k2_base = complex(love_base.k[0]).real

        mu0_soft = (1.0 - f_h) * mu_crust_ref + f_h * mu_serp_abs
        model_soft = ref._replace(mu0=ref.mu0.at[CRUST_LAYER_INDEX].set(mu0_soft))
        love_soft, _, _ = get_love(model_soft, forcing, numerics)
        k2_soft = complex(love_soft.k[0]).real

        out.append({
            "label": label, "mu_crust_ref_gpa": mu_gpa,
            "k2_baseline": k2_base, "k2_softened": k2_soft, "delta_k2": k2_soft - k2_base,
        })
    return out


# Step 2/3 (lateral term): SH coefficients, both normalizations
def hydration_dmu_over_mu_bar_real(
    f_h: float, mu_ratio: float = MU_SERP_RATIO_CENTRAL, lmax: int = DEFAULT_LMAX,
) -> dict[tuple[int, int], float]:
    """Real, 4pi-normalized lateral ``delta_mu/mu_crust`` SH coefficients,
    normalized by the *original* (unsoftened) ``LAYER_MU_CRUST`` -- for
    inspection/plotting (fig7), matching
    :func:`pylov3d.mars_lateral.dmu_over_mu_real`'s convention. **Not**
    the normalization used for solver injection (see
    :func:`hydration_lateral_variables`, which normalizes by the softened
    mu0 instead, as ``process_lateral_variations`` requires). Empty dict
    at ``f_h=0``."""
    if f_h == 0.0:
        return {}
    dt = crustal_thickness_variation(lmax=lmax)
    mu_serp = mu_ratio * LAYER_MU_CRUST
    coeff = f_h * (mu_serp - LAYER_MU_CRUST) / (LAYER_MU_CRUST * MARS["crust_thickness"])
    return {nm: coeff * val for nm, val in dt.items()}


def hydration_lateral_variables(
    f_h: float,
    mu_ratio: float = MU_SERP_RATIO_CENTRAL,
    K_ratio: float = K_SERP_RATIO_CENTRAL,
    lmax: int = DEFAULT_LMAX,
) -> tuple[dict[int, list], dict[int, list]]:
    """Crust-layer ``mu_variable``/``K_variable`` entries for the lateral
    (degree>=1) hydration term (module docstring, item 3). Normalization:
    ``process_lateral_variations``'s elastic branch injects
    ``muC_amp = mu_i * mu_map[nm]`` with ``mu_i`` the layer's own
    normalized modulus (=1.0 for the crust, the model's surface layer),
    so the fractional amplitude here is relative to the layer's own
    *softened* ``mu0``, not ``LAYER_MU_CRUST`` -- unlike
    :func:`hydration_dmu_over_mu_bar_real` (plotting only). Same
    reasoning for K via :func:`get_love_hydrated`'s ``K_ROW0_FACTOR *
    Ks_i * K_map[nm]``. Returns ``({}, {})`` at ``f_h=0`` (entries
    filtered to empty by ``_real_sh_to_complex_mu_variable``'s
    ``c != 0.0`` check when the coefficient is exactly 0.0)."""
    if f_h == 0.0:
        return {}, {}
    dt = crustal_thickness_variation(lmax=lmax)
    mu0_soft, Ks0_soft = mean_softened_crust_moduli(f_h, mu_ratio, K_ratio)
    mu_serp = mu_ratio * LAYER_MU_CRUST
    K_serp = K_ratio * K_CRUST
    t0 = MARS["crust_thickness"]

    coeff_mu = f_h * (mu_serp - LAYER_MU_CRUST) / (t0 * mu0_soft)
    coeff_K = f_h * (K_serp - K_CRUST) / (t0 * Ks0_soft)
    real_dmu = {nm: coeff_mu * val for nm, val in dt.items()}
    real_dK = {nm: coeff_K * val for nm, val in dt.items()}

    mu_entries = _real_sh_to_complex_mu_variable(real_dmu)
    K_entries = _real_sh_to_complex_mu_variable(real_dK)
    mu_variable = {CRUST_LAYER_INDEX: mu_entries} if mu_entries else {}
    K_variable = {CRUST_LAYER_INDEX: K_entries} if K_entries else {}
    return mu_variable, K_variable


# Step 4: K_variable-through-elastic-layers workaround

# K_amp row-0 convention (D2 review finding, verified against
# pylov3d.propagator source, not assumed): _a1a2_geometric's row-0
# docstring ("Row 0 uses (3*lambda+2*mu)=1 -> multiply by actual
# (3*lambda+2*mu) for diagonal, or K_nm for coupling") plus
# _coupling_A1_A2's code (row 0: ``A1c[0,:] += K_nm*Cp[0]*A1g[0,:]``, NO
# extra factor -- contrast rows 1-5, where the code itself multiplies
# ``mu_nm`` by 2 to match the diagonal's ``2*mu``) together imply K_nm
# must represent the diagonal's own normalized quantity,
# (3*lambda+2*mu)=3*K (lambda=K-2*mu/3): K_nm = delta(3*lambda+2*mu) =
# 3*delta_K, not plain delta_K. No validated reference pins this (every
# pylov3d rheology branch -- elastic AND viscoelastic, rheology.py ~424
# and ~480 -- zeroes K_amp unconditionally, so nothing upstream exercises
# this row for nonzero K); this factor-of-3 reading is the one
# internally consistent with the diagonal case. Impact is small either
# way: at f_h=0.3, lmax=2, central ratio, four defensible readings span
# only 3.8x (no K_variable: +5.47e-7; plain delta_K, the pre-fix code:
# +3.73e-7; 3*delta_K, adopted here: +1.45e-7; an alternative
# un-rescaled-fractional reading: +4.68e-7) -- the lateral contribution
# is already <=1.5% of total Delta k2 everywhere (docs/MARS_MODEL.md
# section 4-5), so none of these readings change any conclusion here.
K_ROW0_FACTOR = 3.0


def get_love_hydrated(model, forcing, numerics, mu_variable=None, K_variable=None):
    """``get_love()``-equivalent that also threads lateral bulk-modulus
    (K) perturbations through elastic layers (module docstring, item 4).
    Two extra hand-computed steps vs. plain ``get_love``: (1) pass
    ``K_variable`` into ``process_lateral_variations`` too, not just
    ``mu_variable`` -- omitting this let its ``(n,m)`` union silently
    miss K's modes whenever ``mu_variable`` was empty (D3; its own K_amp
    output is still exactly 0 either way, both rheology branches hardcode
    that). (2) Overwrite K_amp with the row-0-consistent
    :data:`K_ROW0_FACTOR` = 3 convention (comment above, D2). Returns
    ``(love, model_out)``."""
    numerics2, model2 = set_boundary_indices(numerics, model)
    model2 = get_rheology(model2, forcing)

    lateral = None
    couplings = None
    if mu_variable or K_variable:
        model2, lateral = process_lateral_variations(
            model2, forcing, mu_variable=mu_variable, K_variable=K_variable,
        )

        if K_variable:
            sorted_nm = [tuple(int(x) for x in row) for row in np.asarray(lateral.variations)]
            K_amp = np.array(lateral.K_amp, dtype=complex)
            for ilayer, entries in K_variable.items():
                if not entries:
                    continue
                Ks_i = float(model2.Ks[ilayer])
                K_map = {(int(n), int(m)): amp for n, m, amp in entries}
                for j, nm in enumerate(sorted_nm):
                    if nm in K_map:
                        K_amp[ilayer, j] = K_ROW0_FACTOR * Ks_i * K_map[nm]
            lateral = lateral._replace(K_amp=K_amp)

        f0 = forcing[0] if isinstance(forcing, list) else forcing
        couplings = get_couplings(
            lateral.variations, f0.n, f0.m, perturbation_order=numerics2.perturbation_order,
        )

    y_sol, _r_grid, _Y, _Aprop_aux = get_solution(
        model2, forcing, numerics2, couplings=couplings, lateral=lateral,
    )
    love = extract_love_numbers(y_sol, model2, forcing, couplings=couplings)
    return love, model2


# Step 5: forward sweep + detectability
def hydration_k2(
    f_h: float,
    mu_ratio: float = MU_SERP_RATIO_CENTRAL,
    K_ratio: float = K_SERP_RATIO_CENTRAL,
    lmax: int = DEFAULT_LMAX,
    Nrbase: int = DEFAULT_NRBASE,
    perturbation_order: int = 2,
    mu_scale: float | None = None,
    include_lateral: bool = True,
) -> dict:
    """Mean, lateral, and total k2 contributions at one ``(f_h, ratio)``
    point (module docstring, items 3-4). At ``f_h=0`` (or
    ``include_lateral=False``) the lateral/total path is skipped and
    ``k2_total = k2_mean`` exactly (single 1D solve, no coupling)."""
    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=4, method="combination", Nrbase=Nrbase, perturbation_order=perturbation_order,
    )
    hydrated_model = build_hydrated_mars_model(f_h, mu_ratio, K_ratio, mu_scale=mu_scale)

    t0 = time.perf_counter()
    love_mean, _, _ = get_love(hydrated_model, forcing, numerics)
    wall_mean = time.perf_counter() - t0
    k2_mean = complex(love_mean.k[0])

    result = {
        "f_h": f_h, "mu_ratio": mu_ratio, "K_ratio": K_ratio,
        "k2_mean": k2_mean, "wall_mean_s": wall_mean,
        "k2_total": k2_mean, "k2_lateral": complex(0.0, 0.0),
        "n_coupled_modes": 1, "wall_total_s": 0.0,
    }

    if include_lateral and f_h > 0.0:
        mu_variable, K_variable = hydration_lateral_variables(f_h, mu_ratio, K_ratio, lmax=lmax)
        t1 = time.perf_counter()
        love_total, _ = get_love_hydrated(
            hydrated_model, forcing, numerics, mu_variable=mu_variable, K_variable=K_variable,
        )
        wall_total = time.perf_counter() - t1
        k_idx = np.where((love_total.n == 2) & (love_total.m == 0))[0][0]
        k2_total = complex(love_total.k[k_idx])
        result.update({
            "k2_total": k2_total,
            "k2_lateral": k2_total - k2_mean,
            "n_coupled_modes": int(len(love_total.n)),
            "wall_total_s": wall_total,
        })

    return result


def hydration_forward_sweep(
    f_h_grid: tuple[float, ...] = DEFAULT_F_H_GRID,
    ratio_scenarios: dict[str, tuple[float, float]] = RATIO_SCENARIOS,
    lmax: int = DEFAULT_LMAX,
    Nrbase: int = DEFAULT_NRBASE,
    include_lateral: bool = True,
) -> list[dict]:
    """Full ``f_h`` x ratio-scenario sweep (module docstring, "Runtime").
    Returns a flat list of :func:`hydration_k2` result dicts, each tagged
    with ``scenario``."""
    rows = []
    for label, (mu_r, K_r) in ratio_scenarios.items():
        for f_h in f_h_grid:
            row = hydration_k2(
                f_h, mu_ratio=mu_r, K_ratio=K_r, lmax=lmax, Nrbase=Nrbase,
                include_lateral=include_lateral,
            )
            row["scenario"] = label
            rows.append(row)
    return rows


def detectability_summary(rows: list[dict], scenario: str = "central") -> dict:
    """Honest detectability statement (module docstring, item 5): the
    smallest sampled ``f_h`` (if any) at which ``|Delta k2(f_h)| >=
    sigma_k2``, and the k2 precision that would resolve ``f_h=0.1`` --
    the actual computed ``|Delta k2|`` at that grid point, not an
    extrapolation. Requires ``f_h=0.1`` and ``f_h=0.0`` in the swept
    grid for the given ``scenario``; raises ``ValueError`` if absent."""
    scen_rows = sorted((r for r in rows if r["scenario"] == scenario), key=lambda r: r["f_h"])
    f_hs = [r["f_h"] for r in scen_rows]
    if 0.0 not in f_hs or 0.1 not in f_hs:
        raise ValueError(
            f"detectability_summary requires f_h=0.0 and f_h=0.1 in the "
            f"'{scenario}' sweep; got f_h grid {f_hs}"
        )
    k2_base = next(r["k2_total"] for r in scen_rows if r["f_h"] == 0.0)

    crossing_f_h = None
    deltas = {}
    for r in scen_rows:
        delta = abs(r["k2_total"] - k2_base)
        deltas[r["f_h"]] = delta
        if r["f_h"] > 0.0 and crossing_f_h is None and delta >= SIGMA_K2:
            crossing_f_h = r["f_h"]

    return {
        "scenario": scenario,
        "sigma_k2": SIGMA_K2,
        "delta_k2_by_f_h": deltas,
        "crossing_f_h": crossing_f_h,  # None => never exceeds sigma_k2 over the sampled grid
        "precision_to_resolve_f_h_0p1": deltas[0.1],
    }


# Geometry-bound diagnostic (module docstring, item 2, the clip discussion)
def hydration_geometry_diagnostics(
    f_h_grid: tuple[float, ...] = DEFAULT_F_H_GRID,
    lmax: int = 4,
    nlat: int = 180,
    nlon: int = 360,
) -> list[dict]:
    """Checks whether the linear/spectral ``t_h = f_h*(50km+dt)`` field
    (actually used for mu_variable/K_variable) ever violates the physical
    "shell-resident" bound ``t_h <= 50 km`` that spec point 2's literal
    clip instruction encodes (module docstring, item 2). Defaults to
    ``lmax=4`` (TASK-016's full field, max|dt|=34.2 km) -- *not*
    :data:`DEFAULT_LMAX` (2, the solver-sweep's default) -- so this
    diagnostic's own default matches the numbers cited in the module
    docstring and ``docs/MARS_MODEL.md``; no coupled solve here, so
    lmax=4 only costs a finer SH synthesis, not a slower solve."""
    from .mapping import sh_to_latlon

    dt = crustal_thickness_variation(lmax=lmax)
    grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon)
    t0 = MARS["crust_thickness"]
    t_crust_grid = t0 + grid.z

    out = []
    for f_h in f_h_grid:
        t_h_linear = f_h * t_crust_grid
        exceed = t_h_linear > t0
        out.append({
            "f_h": f_h,
            "max_t_h_km": float(np.max(t_h_linear)) / 1e3,
            "bound_violation": bool(np.any(exceed)),
            "area_fraction_violating": float(np.mean(exceed)),
        })
    return out
