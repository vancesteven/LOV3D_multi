"""Mars instantiation of the body-agnostic forward-model framework (TASK-012).

Wires the four-layer Mars geometry/constraints already established in
:mod:`pylov3d.mars` (TASK-011's deterministic point fit) into the generic
:mod:`pylov3d.forward` machinery, so the same
``build_model``/``compute_observables``/``log_likelihood`` code that will
later serve the Moon (and, behind a different Stage-2 solver, gas giants)
also serves Mars. Every numeric constant is imported from
:mod:`pylov3d.mars` — this module intentionally re-types nothing (per
TASK-012: "importing every number from pylov3d.mars ... do not re-type
constants").

Free parameters (4) and constraints (4)
--------------------------------------------------------------------------
====================  ===========================  ===============================================
Name                  Bounds                        Meaning
====================  ===========================  ===============================================
``rho_core``           :data:`pylov3d.mars.MARS_    Core density (Stahler et al. 2021 mean-core-
                        CORE_DENSITY_BOUNDS`          density range; same bound already enforced by
                        ([5700, 6300] kg/m^3)         ``pylov3d.mars._solve_densities``).
``rho_lm``              [3400, 4400] kg/m^3           Lower-mantle density.
``mu_scale``            [0.3, 3.0]                    Shared shear-modulus scale multiplying both the
                                                       lower-mantle (100e9 Pa) and upper-mantle
                                                       (70e9 Pa) base shear moduli — a :class:`~pylov3d
                                                       .forward.Scaled` free parameter, demonstrating
                                                       the "one parameter, several layers" mechanism.
``R_core``              core_radius +/- 2*sigma       Core outer radius (Stahler et al. 2021,
                        ([1750, 1910] km)             1830 +/- 40 km) -- widened to +/-2 sigma (not
                                                       +/-1 sigma) so the ``core_radius_km`` Gaussian
                                                       constraint below, not a hard box edge, is what
                                                       actually informs the posterior; see
                                                       "Identifiability" below.
====================  ===========================  ===============================================

Constraints: mass, moi_mean, k2 (see below) plus **core_radius_km** — see
"Identifiability" below for why a 4th constraint is required with 4 free
parameters.

All other layer scalars (crust/upper-mantle density, crust shear modulus,
every bulk modulus, the L1/L2 and surface radii) are FIXED at the values in
``pylov3d.mars.MARS`` / ``LAYER_KS`` / ``LAYER_RADII_KM``.

Note on R_core: with the core radius free, the core (L0) outer radius moves
independently of the fixed L1/L2/surface radii (2340 / 3339.5 / 3389.5 km);
:func:`pylov3d.forward.build_model` handles this automatically since every
layer's radius is resolved independently (no assumption that only the
surface radius can vary).

Identifiability: why core_radius_km is a constraint, not just a free parameter
--------------------------------------------------------------------------
With 4 free parameters (rho_core, rho_lm, mu_scale, R_core) and only the 3
TASK-011 constraints (mass, moi_mean, k2), the Jacobian of those 3
observables with respect to the 4 parameters generically has a 1-dimensional
null space: a combined shift of all four parameters along that direction
leaves mass, moi_mean, and k2 exactly unchanged to first order, so the
posterior is flat along that ridge -- not merely "wide", but genuinely
unidentifiable in that direction (an axis-aligned perturbation of R_core
alone, holding the other three fixed, does *not* lie along this ridge and
does show a posterior decrease, which is why a naive single-axis probe can
miss the degeneracy). Adding a 4th observable/constraint whose Jacobian row
is independent of the other three's span makes the posterior proper. This
module uses the identity observable core_radius_km = model.R0[0] (see
:func:`pylov3d.forward.compute_observables`) with sigma =
:data:`pylov3d.mars.MARS`\\ ``["core_radius_sigma"]`` (40 km, Stahler et al.
2021) -- this is exactly the seismological information that motivated
R_core's bounds in the first place, previously left as a prior-box edge
only and never actually fed to the likelihood. To be precise about what
this buys: the bulk/tidal data (mass, MoI, k2) contribute NO information
about R_core anywhere on the ridge; the R_core marginal IS the (box-
truncated) Stahler Gaussian. The constraint makes the posterior proper and
honest about its provenance -- it does not make R_core data-constrained by
the tidal observables.
"""

from __future__ import annotations

from .forward import (
    BodyParameterization,
    Constraint,
    Free,
    LayerSpec,
    Scaled,
    make_log_posterior,
)
from .mars import (
    LAYER_KS,
    LAYER_MU_CRUST,
    LAYER_RADII_KM,
    MARS,
    MARS_CORE_DENSITY_BOUNDS,
    MARS_FORCING_TD,
    _MU_LM_BASE,
    _MU_UM_BASE,
)
from .types import make_forcing, make_numerics

# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------

#: Fixed (non-free) L1/L2/surface radii, imported from pylov3d.mars.
_R1_KM, _R2_KM, _R3_KM = LAYER_RADII_KM[1], LAYER_RADII_KM[2], LAYER_RADII_KM[3]

MARS_LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec(
        name="core",
        radius_km=Free("R_core"),
        rho=Free("rho_core"),
        mu=0.0,
        Ks=LAYER_KS[0],
        eta=None,
    ),
    LayerSpec(
        name="lower_mantle",
        radius_km=_R1_KM,
        rho=Free("rho_lm"),
        mu=Scaled(_MU_LM_BASE, "mu_scale"),
        Ks=LAYER_KS[1],
        eta=None,
    ),
    LayerSpec(
        name="upper_mantle",
        radius_km=_R2_KM,
        rho=MARS["upper_mantle_density"],
        mu=Scaled(_MU_UM_BASE, "mu_scale"),
        Ks=LAYER_KS[2],
        eta=None,
    ),
    LayerSpec(
        name="crust",
        radius_km=_R3_KM,
        rho=MARS["crust_density"],
        mu=LAYER_MU_CRUST,
        Ks=LAYER_KS[3],
        eta=None,
    ),
)

MARS_FREE_PARAMS: tuple[str, ...] = ("rho_core", "rho_lm", "mu_scale", "R_core")

#: Core radius +/- 2*sigma (Stahler et al. 2021: 1830 +/- 40 km), derived
#: from pylov3d.mars.MARS rather than re-typed -- the *width* (2 sigma, not
#: 1) is deliberate: it keeps the box wide enough that the core_radius_km
#: Gaussian constraint (not the box edge) carries the identifying
#: information (see module docstring, "Identifiability").
_R_CORE_KM = MARS["core_radius"] / 1e3
_R_CORE_SIGMA_KM = MARS["core_radius_sigma"] / 1e3

MARS_BOUNDS: dict[str, tuple[float, float]] = {
    # Single source of truth shared with pylov3d.mars._solve_densities's own
    # guard rail -- see MARS_CORE_DENSITY_BOUNDS's docstring in mars.py.
    "rho_core": MARS_CORE_DENSITY_BOUNDS,
    "rho_lm": (3400.0, 4400.0),
    "mu_scale": (0.3, 3.0),
    "R_core": (_R_CORE_KM - 2 * _R_CORE_SIGMA_KM, _R_CORE_KM + 2 * _R_CORE_SIGMA_KM),
}

MARS_PARAMETERIZATION = BodyParameterization(
    name="mars",
    layers=MARS_LAYERS,
    free_params=MARS_FREE_PARAMS,
    bounds=MARS_BOUNDS,
)

# ---------------------------------------------------------------------------
# Constraints (values/sigmas cited in pylov3d.mars; not restated here)
# ---------------------------------------------------------------------------

MARS_CONSTRAINTS: tuple[Constraint, ...] = (
    # sigma dominated by G's uncertainty (CODATA 2018 relative uncertainty
    # ~2.2e-5), not GM's (MRO120D GM is known to ~1e-6 relative); 0.1% is a
    # conservative round number well above either, so mass barely
    # constrains (rho_core, rho_lm) relative to the much tighter MoI/k2
    # constraints -- intentional, matches TASK-012 spec.
    Constraint(name="mass", value=MARS["GM"] / MARS["G"], sigma=0.001 * (MARS["GM"] / MARS["G"])),
    Constraint(name="moi_mean", value=MARS["MoI_factor"], sigma=MARS["MoI_factor_sigma"]),
    Constraint(name="k2", value=MARS["k2"], sigma=MARS["k2_sigma"]),
    # 4th constraint, required for identifiability with 4 free parameters --
    # see module docstring, "Identifiability". Uses the hitherto-unused
    # MARS["core_radius_sigma"] (Stahler et al. 2021, 1830 +/- 40 km).
    Constraint(name="core_radius_km", value=_R_CORE_KM, sigma=_R_CORE_SIGMA_KM),
)

# ---------------------------------------------------------------------------
# Forcing / numerics (matches pylov3d.mars's fit configuration)
# ---------------------------------------------------------------------------


def mars_forcing():
    """Forcing used for TASK-011's k2 fit (n=2, m=0; frequency-irrelevant
    for this purely elastic parameterization, see ``pylov3d.mars`` docstring)."""
    return make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)


def mars_numerics(Nrbase: int = 100):
    """Numerics config; defaults match ``pylov3d.mars.build_mars_model``'s
    fit discretization (``Nrbase=100``). For this 4-layer elastic model a
    smaller ``Nrbase`` is essentially free in accuracy: measured k2 is
    constant to ~1e-12 from Nrbase=10 through 400 (the small-Nrbase option
    exists to cut per-evaluation wall time in Monte Carlo runs, not because
    accuracy is being traded away). ``method="fixed"`` is rejected by
    ``compute_observables`` (it rewrites layer radii inside the solver)."""
    return make_numerics(n_layers=4, method="combination", Nrbase=Nrbase)


def mars_log_posterior(Nrbase: int = 100, which: tuple[str, ...] | None = None):
    """Convenience: ``make_log_posterior`` wired up for Mars.

    Parameters
    ----------
    Nrbase : forwarded to :func:`mars_numerics` (smaller -> faster, less
        accurate; see that function's docstring).
    which : observables to compute; defaults to every constrained observable
        (mass, moi_mean, k2, core_radius_km).
    """
    return make_log_posterior(
        MARS_PARAMETERIZATION,
        MARS_CONSTRAINTS,
        mars_forcing(),
        mars_numerics(Nrbase=Nrbase),
        which=which,
    )


# ---------------------------------------------------------------------------
# TASK-011 point fit, expressed as a theta vector in MARS_FREE_PARAMS order
# ---------------------------------------------------------------------------


def mars_point_fit_theta() -> tuple[float, float, float, float]:
    """The TASK-011 deterministic point fit, as a theta vector.

    ``mu_scale`` is exactly :data:`pylov3d.mars.MARS_MU_SCALE`; ``rho_core``/
    ``rho_lm`` are exactly :func:`pylov3d.mars._solve_densities`'s fitted
    values; ``R_core`` is the fixed core radius used throughout
    ``pylov3d.mars`` (:data:`pylov3d.mars.MARS` ``["core_radius"]``).
    """
    from .mars import MARS_MU_SCALE, _solve_densities

    rho_core, rho_lm = _solve_densities()
    R_core_km = MARS["core_radius"] / 1e3
    return (rho_core, rho_lm, MARS_MU_SCALE, R_core_km)
