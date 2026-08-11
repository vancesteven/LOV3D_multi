# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Non-Airy (InSight-calibrated) Mars crustal-thickness substitution (TASK-028).

TASK-027 measured the Airy crust/mantle-density calibration bracket to move
the lateral spectrum by a factor ~7.6 (docs/MARS_MODEL.md section 6,
"Airy-calibration sensitivity"), against a truncation-ladder uncertainty of
only ~1.13x (4-6% per lmax step). That makes the Airy *assumption itself* --
not just its density parameters -- the input most worth checking directly.
This module re-derives the crust-thickness lateral field from five
InSight-calibrated crust-mantle interface (Moho) models
(`data/mars/insight_moho/`, see `SOURCES.md` there for provenance, format,
and selection criterion) instead of computing it from topography via Airy
isostasy, and feeds the result through the *same* unmodified downstream
machinery in :mod:`pylov3d.mars_lateral` (``_dmu_ddt_coeff``,
``_real_sh_to_complex_mu_variable``, ``CRUST_LAYER_INDEX``,
:func:`pylov3d.love.get_love`) -- imported, never edited, so the only thing
that changes between the Airy and InSight results is the input crustal
thickness field itself.

**Headline finding.** Because this module retains C20 (see "The C20
decision" below), it exercises a first-order coupling channel the Airy path
structurally cannot: (2,0) rheology, not just (4,0), couples the (2,0)
forcing mode back to itself at first order (coupling coefficients, verified
against ``pylov3d.couplings.coupling_coefficients``: max\|C\|=0.6389 for
(2,0), 0.8571 for (4,0), identically zero for (1,0),(3,0),(5,0),(6,0) --
parity requires even rheology degree, the triangle inequality caps it at 4,
and order conservation restricts it to m=0 for a degree-2 zonal tide). On
the DWAK field the two channels' signed, amplitude-extrapolated
contributions are (2,0) alone = -1.5901e-5, (4,0) alone = +1.4528e-5,
**both together = -1.3738e-6** -- a 91.4% cancellation, additive to 0.1% as
first order requires. The mechanism is robust (confirmed non-artifact
across eps in [1e-4,1e-1] and perturbation_order in {1,2,3}); its *net*
amplitude is not, because it is a near-total cancellation between two
same-order terms of opposite sign whose relative size is convention-
dependent (the Airy path's own (4,0)-alone figure, ~1.75e-5, is a property
of the Airy field specifically, not a universal constant). See
docs/MARS_MODEL.md, "Non-Airy crustal model substitution (TASK-028)",
section "The headline: (2,0)/(4,0) sign cancellation" for the full
derivation. A direct consequence: the crustal-model sensitivity of the
(2,0) forcing-mode shift is *not* well summarized by a single ratio --
report all three of x1.551 (as-shipped spread across the five models),
x1.726 (including Airy), and x1.379 (C20 suppressed on both sides, the
like-for-like pattern-only comparison) -- see the same docs section, "The
(2,0) forcing-shift statistics need three numbers, not one".

Physical chain
--------------
1. :func:`moho_thickness_variation` -- crustal thickness ``t(theta, phi) =
   R_topo(theta, phi) - R_moho(theta, phi)``, both terms real 4pi-normalized
   SH radius fields in the *same* convention
   (``pylov3d.sh_data.load_shape``), so the SH coefficients of the
   difference are exactly the coefficient-wise difference (linearity of the
   expansion) -- no re-synthesis needed. C00 (mean thickness) is removed
   (not a lateral variation, same as :func:`pylov3d.mars_lateral.crustal_thickness_variation`).
   **C20 is retained** -- see "The C20 decision" below, the subtlest choice
   in this module and a deliberate *departure* from the Airy path.
2. Downstream: the crust-layer rigidity linearization
   (``pylov3d.mars_lateral._dmu_ddt_coeff``) and the real->complex SH
   conversion (``pylov3d.mars_lateral._real_sh_to_complex_mu_variable``) are
   reused unmodified -- both are already validated (round-trip test,
   MATLAB cross-check) against the Airy field, and take a real ``dt`` SH
   dict as their only input, with no Airy-specific assumption baked in.
3. :func:`compare_crustal_models` runs the Airy baseline and all five
   InSight models through :func:`pylov3d.love.get_love` at the project's
   validated ``lmax=4``, ``Nrbase=30`` and reports the (2,0) forcing shift
   plus the tracked off-diagonal modes (3,0), (2,+/-2), (3,+/-1) -- the same
   modes TASK-027's truncation/Airy-sensitivity tables track, so the three
   result sets (truncation ladder, Airy-parameter sweep, this
   crustal-*model* substitution) are directly comparable.

The C20 decision
-----------------
The Airy path (:func:`pylov3d.mars_lateral.crustal_thickness_variation`)
drops the topography's C20 term before applying Airy compensation, because
Mars's C20 is dominated (~93%, per docs/MARS_MODEL.md section 1) by
rotational flattening -- an equilibrium-figure effect that bulges the whole
body (mantle included), not a crustal-density load. Feeding it through the
Airy formula (which assumes *all* topography above a level surface is
locally supported by crustal thickening) would fabricate a large, spurious
crustal root under the equatorial bulge.

That reasoning does not carry over to a *measured* (gravity+seismic
inverted) Moho depth. Here ``dt`` is not derived from an isostatic-load
assumption at all -- it is the literal difference of two independent radius
fields, each already whatever shape it actually has. Zeroing C20 would only
be correct if the Moho, like an idealized fluid level surface, flattened in
lockstep with the topographic surface, making the C20 of their difference a
pure double-counted rotational artifact. It does not: for all five
committed models, the Moho's actual C20 is far smaller in magnitude than the
value predicted by scaling the topographic C20 by ``(R_moho/R_topo)**2``, a
*plausible* (not the only defensible) leading-order comparator for how a
level surface's flattening scales with depth -- see "Comparator choice"
below for why the exponent is not load-bearing here:

| model | Moho C20 [m] | (R_moho/R_topo)^2-scaled prediction [m] | ratio |
|---|---|---|---|
| DWAK | -2057.6 | -5792.1 | 0.36 |
| DWThotCrust1 | -2247.7 | -5792.9 | 0.39 |
| EH45Tcold | -2114.8 | -5790.3 | 0.37 |
| EH45TcoldCrust1r | -2251.0 | -5789.8 | 0.39 |
| Khan2022 | -2745.1 | -5789.9 | 0.47 |

(topo C20 = -5966.2 m; MarsTopo719 C00 = 3389.500 km truncated to lmax=4,
same convention as the Airy path; R_moho/R_topo = 0.9853.) Every model's
Moho is flattened only 0.36-0.47x as much as the scaled prediction --
consistently, not as an outlier of one model -- meaning the Moho's degree-2
shape is measurably decoupled from the surface's rotational bulge. Sign,
stated plainly because an earlier revision of this docstring had it
backwards: retaining the actual (small-magnitude) Moho C20 instead of the
larger passive-geometry value makes the crust **thinner** at the poles, not
thicker -- ``dt``'s C20 term alone evaluates to -8740 m at either pole
(DWAK), against only -389 m from the hypothetical passive-geometry
(``(R_moho/R_topo)**2``-scaled) value, i.e. about 8.4 km of *additional*
thinning at the poles beyond what passive geometry alone would produce
(equivalently, thickening around the equator -- consistent with the
equatorial Tharsis root; zonal-mean ``dt`` confirms this independently:
-24.2 km at 84.5N vs. +12.6 km at 85.5S). This is the direct geometric
analogue of the already-stated fact (docs/MARS_MODEL.md section 1) that ~7%
of Mars's *observed* C20 is a non-hydrostatic, real mass-anomaly
contribution equivalent to a ~2.4 km crustal root -- the same phenomenon,
now visible directly in the two surfaces' relative shapes rather than
inferred from the gravity field's hydrostatic/non-hydrostatic split.

**Comparator choice.** ``(R_moho/R_topo)**2`` is not the general hydrostatic
law -- Clairaut's theorem gives flattening proportional to ``r`` for a
homogeneous body and to ``r**3`` in the centrally-condensed limit; ``r**2``
is intermediate, a plausible but not uniquely-justified interpolation. It
is used here anyway because R_moho/R_topo = 0.9853 is so close to 1 that
the exponent barely matters: DWAK's predicted ratio moves only
0.350/0.355/0.361 for exponent 1/2/3 respectively, and the full
five-model x three-exponent grid spans 0.350-0.481 -- the "measurably
decoupled" conclusion above is insensitive to which power is chosen.

A second, independent argument for retaining C20: the 1D reference model
this feeds (``pylov3d.mars.build_mars_model``) has a perfectly spherical
crust-mantle boundary at every layer -- zero flattening anywhere in the
background geometry. The lateral-rigidity perturbation scheme measures
deviation from *that* spherical background, not from some separately
flattened-but-load-free reference figure. Since there is no flattened
background to avoid double-subtracting against, there is no bookkeeping
reason (independent of the physical Airy-load argument above, which does
not apply here) to remove C20 from a directly-observed thickness field.

Consequently, unlike the Airy path, no areoid (Bruns-relation) referencing
is applied here either: that correction exists in the Airy path specifically
to convert bare topography into "height above the level surface" before
invoking an isostatic-support assumption. There is no isostatic-support step
here to correct the input to -- ``dt`` is already the actual crustal
thickness, not a proxy for it.

Reported numbers (peak |dt|, elastic-positivity bound, degree RMS, and the
comparison-driver spectrum) are all in docs/MARS_MODEL.md, "Non-Airy
crustal model substitution (TASK-028)".
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .love import get_love
from .mapping import fully_normalized_legendre, sh_to_latlon
from .mars import MARS, MARS_FORCING_TD, build_mars_model
from .mars_lateral import (
    CRUST_LAYER_INDEX,
    TOPO_PATH,
    _dmu_ddt_coeff,
    _real_sh_to_complex_mu_variable,
    crustal_thickness_variation,
)
from .sh_data import load_shape, truncate
from .types import make_forcing, make_numerics

# ---------------------------------------------------------------------------
# Paths and model registry
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
MOHO_DIR = _REPO_ROOT / "data" / "mars" / "insight_moho"

# The five committed InSight-calibrated Moho models (data/mars/insight_moho/SOURCES.md).
MOHO_MODELS: dict[str, Path] = {
    "DWAK": MOHO_DIR / "Moho-Mars-DWAK-33-2900.sh",
    "DWThotCrust1": MOHO_DIR / "Moho-Mars-DWThotCrust1-36-2900.sh",
    "EH45Tcold": MOHO_DIR / "Moho-Mars-EH45Tcold-36-2900.sh",
    "EH45TcoldCrust1r": MOHO_DIR / "Moho-Mars-EH45TcoldCrust1r-36-2900.sh",
    "Khan2022": MOHO_DIR / "Moho-Mars-Khan2022-33-2900.sh",
}

# Mean crustal thickness [km] as reported in SOURCES.md ("MarsTopo719 C00 -
# Moho C00", both via pylov3d.sh_data.load_shape) -- an independent
# machine-A figure this module's loader is checked against (test file).
SOURCES_MEAN_THICKNESS_KM: dict[str, float] = {
    "DWAK": 49.81,
    "DWThotCrust1": 49.59,
    "EH45Tcold": 50.35,
    "EH45TcoldCrust1r": 50.48,
    "Khan2022": 50.47,
}

# Off-(2,0) response modes tracked by TASK-027's truncation/Airy tables
# (docs/MARS_MODEL.md section 6) -- kept identical here for a like-for-like
# three-way comparison (truncation ladder / Airy-parameter sweep / this
# crustal-model substitution).
OFF_MODES: tuple[tuple[int, int], ...] = ((3, 0), (2, 2), (2, -2), (3, 1), (3, -1))
FORCING_MODE: tuple[int, int] = (2, 0)


def _resolve_moho_path(model: str | Path) -> Path:
    if isinstance(model, Path):
        return model
    if model in MOHO_MODELS:
        return MOHO_MODELS[model]
    return Path(model)


# ---------------------------------------------------------------------------
# Step 1: Moho-derived crustal thickness variation
# ---------------------------------------------------------------------------

def moho_thickness_variation(
    model: str | Path,
    lmax: int = 4,
    topo_path: Path | str = TOPO_PATH,
) -> dict[tuple[int, int], float]:
    """InSight-calibrated crustal thickness variation, real 4pi-normalized SH.

    ``model`` is a key into :data:`MOHO_MODELS` or an explicit path to a
    Moho ``.sh`` file in the same format. Loads MarsTopo719 and the Moho
    shape, truncates both to ``lmax``, and returns the SH coefficients of
    ``dt = R_topo - R_moho`` [m] with the (0,0) mean removed. C20 is
    **retained** (see module docstring, "The C20 decision") -- the one
    deliberate difference from
    :func:`pylov3d.mars_lateral.crustal_thickness_variation`'s C00/C20
    handling.

    Returns ``dict[(n, m), float]`` in the same convention as
    :func:`pylov3d.mapping.sh_to_latlon` / the Airy loader (``m >= 0``
    cosine, ``(n, -m)`` ``m >= 1`` sine); only nonzero entries included.
    """
    moho_path = _resolve_moho_path(model)
    topo = truncate(load_shape(topo_path), lmax)
    moho = truncate(load_shape(moho_path), lmax)

    clm = topo["clm"] - moho["clm"]
    slm = topo["slm"] - moho["slm"]
    clm[0, 0] = 0.0  # mean thickness: not a lateral variation

    dt: dict[tuple[int, int], float] = {}
    for n in range(lmax + 1):
        for m in range(n + 1):
            c = float(clm[n, m])
            if c != 0.0:
                dt[(n, m)] = c
            if m >= 1:
                s = float(slm[n, m])
                if s != 0.0:
                    dt[(n, -m)] = s
    return dt


def _evaluate_dt_at(
    dt: dict[tuple[int, int], float], lat_deg: float, lon_deg: float,
) -> float:
    """Pointwise real-SH synthesis of ``dt`` at a single (lat, lon) [deg].

    Same convention as :func:`pylov3d.mapping.sh_to_latlon` (which only
    evaluates on a regular grid); used here for the named-location
    geographic-sanity checks (Tharsis/Hellas/Utopia) without needing a full
    grid synthesis.
    """
    if not dt:
        return 0.0
    t = np.array([math.sin(math.radians(lat_deg))])
    n_max = max(n for n, _m in dt)
    P = fully_normalized_legendre(n_max, t)
    total = 0.0
    for (n, m), amp in dt.items():
        mm = abs(m)
        Pnm = float(P[n, mm, 0])
        if m >= 0:
            total += amp * Pnm * math.cos(math.radians(m * lon_deg))
        else:
            total += amp * Pnm * math.sin(math.radians(mm * lon_deg))
    return total


# Named locations for the geographic-sanity check (degrees). Tharsis matches
# the Airy field's own measured peak location (docs/MARS_MODEL.md section 1,
# "lat -8.4, lon -106.4"); Hellas and Utopia are the classic low-degree
# crustal-thinning basins (Wieczorek & Zuber 2004; Neumann et al. 2004).
GEO_LOCATIONS: dict[str, tuple[float, float]] = {
    "Tharsis": (-8.4, -106.4),
    "Hellas": (-42.4, 70.5),
    "Utopia": (45.0, 110.0),
}


def moho_thickness_diagnostics(
    model: str | Path,
    lmax: int = 4,
    topo_path: Path | str = TOPO_PATH,
    mu_scale: float | None = None,
    nlat: int = 180, nlon: int = 360,
) -> dict:
    """Sanity/clipping diagnostics for an InSight-calibrated thickness field.

    Mirrors :func:`pylov3d.mars_lateral.crustal_thickness_diagnostics`'s
    contract (``max_abs_dt_m``, ``max_abs_dt_over_t0``,
    ``max_abs_dmu_over_mubar`` -- hard ``ValueError`` if the shell-residency
    bound is violated, degree_rms_km``), plus ``geo`` -- ``dt`` [m]
    evaluated at :data:`GEO_LOCATIONS`, for the geographic-sanity check
    (thickest crust at Tharsis, thinnest around Hellas/Utopia).
    """
    dt = moho_thickness_variation(model, lmax=lmax, topo_path=topo_path)
    grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon)
    max_abs_dt_m = float(np.max(np.abs(grid.z)))
    t0 = MARS["crust_thickness"]

    if not (max_abs_dt_m < t0):
        raise ValueError(
            f"max|dt| = {max_abs_dt_m / 1e3:.3f} km exceeds the 50 km shell "
            "thickness; the linearized rigidity map is not meaningful "
            "beyond this bound (see module docstring)."
        )

    degree_rms_km: dict[int, float] = {}
    for n in range(1, lmax + 1):
        acc = 0.0
        for m in range(0, n + 1):
            acc += dt.get((n, m), 0.0) ** 2
            if m >= 1:
                acc += dt.get((n, -m), 0.0) ** 2
        degree_rms_km[n] = math.sqrt(acc) / 1e3

    geo = {name: _evaluate_dt_at(dt, lat, lon)
           for name, (lat, lon) in GEO_LOCATIONS.items()}

    return {
        "max_abs_dt_m": max_abs_dt_m,
        "max_abs_dt_over_t0": max_abs_dt_m / t0,
        "max_abs_dmu_over_mubar": abs(_dmu_ddt_coeff(mu_scale)) * max_abs_dt_m,
        "degree_rms_km": degree_rms_km,
        "geo": geo,
    }


# ---------------------------------------------------------------------------
# Step 2/3: crust mu_variable from an arbitrary dt field
# ---------------------------------------------------------------------------

def mu_variable_from_dt(
    dt: dict[tuple[int, int], float], mu_scale: float | None = None,
) -> dict[int, list[tuple[int, int, complex]]]:
    """Crust-layer ``mu_variable`` entries from an arbitrary real ``dt`` SH dict.

    Reuses :func:`pylov3d.mars_lateral._dmu_ddt_coeff` (the linearized
    crust-fraction-in-``dt`` coefficient) and
    :func:`pylov3d.mars_lateral._real_sh_to_complex_mu_variable` (the
    validated real->complex SH conversion) unmodified -- neither makes any
    Airy-specific assumption; both take a real ``dt`` dict as their only
    physical input. ``dt`` may come from :func:`moho_thickness_variation` or
    :func:`pylov3d.mars_lateral.crustal_thickness_variation` (the Airy
    field) -- same downstream path either way, so a difference in the
    resulting Love spectrum is attributable only to ``dt`` itself.
    """
    coeff = _dmu_ddt_coeff(mu_scale)
    real_dmu = {nm: coeff * val for nm, val in dt.items()}
    entries = _real_sh_to_complex_mu_variable(real_dmu)
    return {CRUST_LAYER_INDEX: entries}


# ---------------------------------------------------------------------------
# Step 4: comparison driver -- Airy baseline vs. the five InSight models
# ---------------------------------------------------------------------------

def _extract(love, n: int, m: int) -> complex:
    mask = (love.n == n) & (love.m == m)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return complex("nan")
    return complex(love.k[idx[0]])


def _solve_dt(
    dt: dict[tuple[int, int], float],
    Nrbase: int, mu_scale: float | None,
    method: str, perturbation_order: int, F: complex,
) -> dict:
    mu_variable = mu_variable_from_dt(dt, mu_scale=mu_scale)
    model = build_mars_model(mu_scale=mu_scale)
    forcing = make_forcing(Td=MARS_FORCING_TD, n=FORCING_MODE[0], m=FORCING_MODE[1], F=F)
    numerics = make_numerics(
        n_layers=4, method=method, Nrbase=Nrbase, perturbation_order=perturbation_order,
    )
    t_start = time.perf_counter()
    love, _y_rad, _model_out = get_love(model, forcing, numerics, mu_variable=mu_variable)
    wall_s = time.perf_counter() - t_start

    row = {"n_modes": int(len(love.n)), "wall_s": wall_s,
           "k20": _extract(love, *FORCING_MODE)}
    for nm in OFF_MODES:
        row[f"k_{nm[0]}_{nm[1]}"] = _extract(love, *nm)
    return row


def compare_crustal_models(
    lmax: int = 4,
    Nrbase: int = 30,
    mu_scale: float | None = None,
    method: str = "combination",
    perturbation_order: int = 2,
    F: complex = 1.0,
    models: list[str] | None = None,
) -> dict:
    """Airy baseline + the five InSight Moho models through the coupled solver.

    Solves ``forcing=(2,0)``, ``perturbation_order=2`` (the project's
    validated production settings, ``lmax=4``/``Nrbase=30`` defaults) once
    per model plus one uniform (no-lateral) baseline, and reports the (2,0)
    forcing-mode k2 shift and the tracked :data:`OFF_MODES` amplitudes for
    each -- directly comparable to TASK-027's truncation-ladder and
    Airy-parameter-sweep tables (docs/MARS_MODEL.md section 6), which track
    the identical modes.

    Returns ``{"k2_uniform": float, "rows": {"Airy": {...}, "DWAK": {...},
    ...}}``; each row has ``n_modes``, ``wall_s``, ``k20`` (complex),
    ``shift20`` (complex, ``k20 - k2_uniform``), and ``k_<n>_<m>`` (complex)
    for each tracked off-mode.
    """
    if models is None:
        models = list(MOHO_MODELS)

    uniform_model = build_mars_model(mu_scale=mu_scale)
    uniform_forcing = make_forcing(Td=MARS_FORCING_TD, n=FORCING_MODE[0], m=FORCING_MODE[1], F=F)
    uniform_numerics = make_numerics(
        n_layers=4, method=method, Nrbase=Nrbase, perturbation_order=perturbation_order,
    )
    love_u, _y, _m = get_love(uniform_model, uniform_forcing, uniform_numerics, mu_variable=None)
    k2_uniform = complex(_extract(love_u, *FORCING_MODE))

    rows: dict[str, dict] = {}

    airy_dt = crustal_thickness_variation(lmax=lmax)
    r = _solve_dt(airy_dt, Nrbase, mu_scale, method, perturbation_order, F)
    r["shift20"] = r["k20"] - k2_uniform
    rows["Airy"] = r

    for name in models:
        dt = moho_thickness_variation(name, lmax=lmax)
        r = _solve_dt(dt, Nrbase, mu_scale, method, perturbation_order, F)
        r["shift20"] = r["k20"] - k2_uniform
        rows[name] = r

    return {"k2_uniform": k2_uniform, "rows": rows}
