"""Mars 1D radial reference interior model (TASK-011).

Stage-1 elastic reference model for Mars, fit to published bulk geophysical
constraints and solved through the existing pylov3d radial tidal solver
(:func:`pylov3d.love.get_love`).  Four homogeneous layers span core to
surface; the liquid outer core is represented natively by the LOV3D
fluid-core-mantle-boundary (CMB) condition (layer 0 with ``mu0 = 0``, no
``ocean`` flag needed — see ``pylov3d/boundary_conditions.py``, which always
treats layer 0 as the core and never integrates through it; consequently the
core's ``Ks0`` (bulk modulus) is numerically inert — see "Model structure"
below).

Published bulk constraints (cited exactly; do not alter without flagging)
---------------------------------------------------------------------------
=====================================  ========================================  =========================================================
Quantity                                Value                                     Source
=====================================  ========================================  =========================================================
GM                                      42828.375 km^3/s^2                        Konopliv, Park & Folkner (2016), "An improved JPL Mars
                                                                                    gravity field and orientation from Mars orbiter and lander
                                                                                    tracking data" (MRO120D gravity field), Icarus 274, 253-260
Mass M                                  GM/G, G = 6.6743e-11 (CODATA 2018)         derived, ~= 6.4169e23 kg
                                         ~= 6.4169e23 kg
Mean radius R                            3389.5 km                                Seidelmann et al. (2007) IAU report / MOLA
Polar moment-of-inertia factor C/MR^2   0.3644 +/- 0.0005                         Konopliv et al. (2011), Icarus 211, 401-428
Mean moment-of-inertia factor I/MR^2    0.36310 +/- 0.0005                        derived: I/MR^2 = C/MR^2 - (2/3)*J2
                                                                                    = 0.3644 - (2/3)(1.9555e-3) = 0.36310
                                                                                    (J2 = 1.9555e-3, Konopliv et al. 2016, MRO120D)
Tidal k2 (degree-2, solar semidiurnal)  0.169 +/- 0.006                            Konopliv, Park & Folkner (2016), Icarus 274 (MRO120D);
                                                                                    consistent with k2 = 0.174 +/- 0.008 from Chandler-wobble /
                                                                                    MRO120F gravity, Konopliv et al. (2020), GRL 47,
                                                                                    e2020GL090568
Core radius                             1830 +/- 40 km, liquid                     Stahler et al. (2021), Science 373 (InSight seismology);
                                                                                    Le Maistre et al. (2023), Nature 619, 733-737 gives
                                                                                    1835 +/- 55 km (spin-state solution; a lander cannot itself
                                                                                    measure k2, see caveat above)
Crustal thickness (global mean)         ~24-72 km; adopt 50 km                     Knapmeyer-Endrun et al. (2021), Science 373
Crust density                           2900 kg/m^3                                within Knapmeyer-Endrun et al. (2021) range
Mantle seismic properties               Vs ~4.4-5.0 km/s, rho ~3400-4000 kg/m^3     Khan et al. (2021), Science 373 / Stahler et al. (2021)
  NOTE: the fitted lower-mantle density (4136.5) exceeds this range; L1 is
  a mass/moment-balancing shell, not a literal wadsleyite-layer density
  (see docs/MARS_MODEL.md, "Interpretation caveat on the fitted L1
  density").
=====================================  ========================================  =========================================================

IMPORTANT CAVEAT 1 — elastic vs. anelastic k2
---------------------------------------------------------------------------
The observed k2 = 0.169 (Konopliv, Park & Folkner 2016) includes an
anelastic (dissipative) contribution from mantle viscoelastic relaxation.
This stage-1 model is purely elastic (no viscosity), so fitting the elastic
model's k2 to the observed value slightly *underestimates* the true mantle
rigidity (an elastic model needs a softer mantle than an anelastic one to
reach the same k2, because anelasticity itself raises the effective
compliance at tidal frequency). This is an accepted approximation for a
reference model; incorporating anelasticity (e.g. Andrade or Maxwell
rheology) is left for future work.

IMPORTANT CAVEAT 2 — Stahler-family core vs. the basal-melt reinterpretation
---------------------------------------------------------------------------
Khan et al. (2023), Nature 622, 718-723, and Samuel et al. (2023), Nature
622, 712-717, reinterpret the seismic CMB reflector originally identified by
Stahler et al. (2021) as instead marking the top of a ~150 km thick molten
(silicate-melt) basal mantle layer, implying a smaller (~1650-1675 km) and
denser (~6.65 g/cm^3) core than the Stahler-family solution used here. This
stage-1 model deliberately adopts the Stahler et al. (2021) / RISE
1830 km-radius core parameterization; the fitted core density below should
be understood as the large-core, lower-density-branch value corresponding
to that parameterization, not a claim that the basal-melt-layer branch is
disfavored. Revisiting the core radius against the Khan/Samuel (2023) branch
is future work.

Model structure (fixed; layer 0 = core, indices increase outward)
---------------------------------------------------------------------------
    L0 liquid core:   0      -  1830   km, mu0 = 0,        rho = FIT (rho_core)
                       Ks0 = 155e9 Pa but numerically inert: the solver
                       integrates layers 1..n_layers-1 only and applies an
                       analytic fluid-core CMB boundary condition at the
                       core/mantle interface, so the core's Ks0 (and mu0)
                       never enter the radial integration (see
                       ``pylov3d/solver.py``, integration loop over
                       ``range(1, n_layers)``).
    L1 lower mantle:  1830   -  2340   km, mu0 = 100e9*s,   rho = FIT (rho_lm)
    L2 upper mantle:  2340   -  3339.5 km, mu0 = 70e9*s,    rho = 3400 (fixed, Khan et al. 2021)
    L3 crust:         3339.5 -  3389.5 km, mu0 = 30e9,      rho = 2900 (fixed)
where ``s`` (``MARS_MU_SCALE``) is a single shear-modulus scale factor fit so
that the elastic k2 from ``get_love`` matches the observed k2 = 0.169.  The
L1/L2 boundary at 2340 km (~1050 km depth) is the olivine -> wadsleyite
mantle phase transition inferred from InSight seismic data by Khan et al.
(2021), Science 373, and Stahler et al. (2021), Science 373.

Fit procedure (deterministic, no black-box optimizer)
---------------------------------------------------------------------------
1. With rho_um and rho_crust fixed, (rho_core, rho_lm) are the unique
   solution of the exact 2x2 linear system {mass, mean moment of inertia} —
   see :func:`_solve_densities`.  The mean-moment target 0.36310 is derived
   from the *polar* moment C/MR^2 = 0.3644 via I/MR^2 = C/MR^2 - (2/3)*J2
   (a spherically symmetric 1D profile must match the mean moment, not the
   polar moment, which includes the rotational/tidal flattening
   contribution captured by J2).
2. The mantle shear-modulus scale ``s`` is found by bisection
   (:func:`fit_mu_scale`) on k2(s) from ``get_love``; the converged value is
   hardcoded as :data:`MARS_MU_SCALE` so importing this module never runs
   the (multi-second) tidal solve.  ``fit_mu_scale()`` remains the
   reproducibility path.

Discretization note
---------------------------------------------------------------------------
:data:`MARS_MU_SCALE` was fit using
``numerics = make_numerics(n_layers=4, method="combination", Nrbase=100)``;
it is tied to that discretization. Re-solving with
``method="fixed"`` (same ``Nrbase``) shifts k2 by about -1.3e-5 (measured;
still far inside the 0.006 observational uncertainty and the 1e-4 fit
tolerance), so the choice of grid method does not materially affect the
model but is not perfectly interchangeable either.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .constants import G
from .love import get_love
from .types import InteriorModel, make_forcing, make_interior_model, make_numerics

# ---------------------------------------------------------------------------
# Published bulk constraints (cite exactly; see module docstring table)
# ---------------------------------------------------------------------------

MARS: dict[str, float] = {
    # GM: Konopliv, Park & Folkner (2016), "An improved JPL Mars gravity
    # field and orientation from Mars orbiter and lander tracking data"
    # (MRO120D gravity field), Icarus 274, 253-260. km^3/s^2 -> m^3/s^2.
    "GM": 42828.375e9,
    # G: CODATA 2018 (matches pylov3d.constants.G).
    "G": G,
    # Mass: derived, M = GM/G ~= 6.4169e23 kg.
    "M": 42828.375e9 / G,
    # Mean radius: Seidelmann et al. (2007) IAU report / MOLA. km -> m.
    "R": 3389.5e3,
    # Polar moment-of-inertia factor C/MR^2: Konopliv et al. (2011),
    # Icarus 211, 401-428.
    "MoI_polar_factor": 0.3644,
    "MoI_polar_factor_sigma": 0.0005,
    # J2 (dynamical form factor): Konopliv et al. (2016), MRO120D.
    "J2": 1.9555e-3,
    # Mean moment-of-inertia factor I/MR^2 (what a spherically symmetric
    # radial profile must be fit to): I/MR^2 = C/MR^2 - (2/3)*J2
    # = 0.3644 - (2/3)(1.9555e-3) = 0.3630963... -> retarget to 0.36310
    # (derived, rounded per the reviewed spec; sigma propagated from the
    # polar-moment uncertainty, J2's own uncertainty being negligible in
    # comparison). The literal rounded value 0.36310 is used as the actual
    # fit target throughout (not the unrounded 0.3630963...) so that this
    # constant is exactly reproducible from the documented derivation.
    "MoI_factor": 0.36310,
    "MoI_factor_sigma": 0.0005,
    # Tidal k2 (degree-2, solar semidiurnal): Konopliv, Park & Folkner
    # (2016), Icarus 274 (MRO120D); consistent with k2 = 0.174 +/- 0.008
    # from Chandler-wobble / MRO120F gravity, Konopliv et al. (2020),
    # GRL 47, e2020GL090568.
    "k2": 0.169,
    "k2_sigma": 0.006,
    # Core radius: Stahler et al. (2021), Science 373 (InSight seismology);
    # Le Maistre et al. (2023), Nature 619, 733-737 gives 1835 +/- 55 km
    # (spin-state solution). km -> m.
    "core_radius": 1830e3,
    "core_radius_sigma": 40e3,
    # Crustal thickness (global mean): Knapmeyer-Endrun et al. (2021),
    # Science 373; range ~24-72 km, adopt 50 km. km -> m.
    "crust_thickness": 50e3,
    # Crust density: within Knapmeyer-Endrun et al. (2021) range. kg/m^3.
    "crust_density": 2900.0,
    # Upper-mantle density: Khan et al. (2021), Science 373 / Stahler et al.
    # (2021). kg/m^3.
    "upper_mantle_density": 3400.0,
}

# Layer outer radii, core to surface [km] (fixed model geometry).
# L1/L2 boundary at 2340 km (~1050 km depth): olivine -> wadsleyite phase
# transition, Khan et al. (2021) / Stahler et al. (2021), Science 373.
LAYER_RADII_KM: tuple[float, float, float, float] = (
    MARS["core_radius"] / 1e3,       # 1830.0 km, L0 core outer radius
    2340.0,                          # L1 lower-mantle outer radius
    (MARS["R"] - MARS["crust_thickness"]) / 1e3,  # 3339.5 km, L2 outer radius
    MARS["R"] / 1e3,                 # 3389.5 km, surface
)

# Bulk moduli per layer [Pa] (fixed; see module docstring / TASK-011 spec).
# LAYER_KS[0] (core) is numerically inert; see module docstring.
LAYER_KS: tuple[float, float, float, float] = (155e9, 160e9, 115e9, 70e9)

# Unscaled mantle shear moduli [Pa] (crust mu0 is fixed, not scaled).
_MU_LM_BASE = 100e9
_MU_UM_BASE = 70e9
LAYER_MU_CRUST = 30e9  # Pa, fixed (Vs ~ 3.2 km/s)

# Forcing period used for the k2 fit: half a sol (solar semidiurnal tide),
# Td = 88775.244 s (one sol) / 2 = 44387.62 s.
# Note: the model is purely elastic, so the forcing period Td does not
# actually affect k2 (elastic Love numbers are frequency-independent) — it
# is retained only because ``make_forcing`` requires a value.
MARS_FORCING_TD = 44387.62  # s


# ---------------------------------------------------------------------------
# Step 1: exact 2x2 linear solve for (rho_core, rho_lm)
# ---------------------------------------------------------------------------

def _mass_and_moi(radii_km: list[float], rho: list[float]) -> tuple[float, float]:
    """Mass and moment of inertia of a stack of homogeneous spherical shells.

    Shell i spans [R_{i-1}, R_i] (R_{-1} = 0) with uniform density rho_i.

    Mass:   M = sum_i (4*pi/3) * rho_i * (R_i^3 - R_{i-1}^3)
    MoI:    I = sum_i (8*pi/15) * rho_i * (R_i^5 - R_{i-1}^5)

    The 8*pi/15 coefficient is verified as follows: a solid, uniform-density
    sphere of radius R has I = (2/5) M R^2 = (2/5) * (4/3 pi rho R^3) * R^2
    = (8*pi/15) * rho * R^5.  A shell's moment of inertia is the difference
    of the moments of inertia of the two solid spheres bounding it (both
    evaluated at the shell's own density), so the same (8*pi/15) prefactor
    applies to rho_i * (R_i^5 - R_{i-1}^5).

    Parameters
    ----------
    radii_km : outer radius of each shell, core to surface [km].
    rho : density of each shell [kg/m^3].
    """
    boundaries_m = [0.0] + [r * 1e3 for r in radii_km]
    M = 0.0
    I = 0.0
    for i, rho_i in enumerate(rho):
        r_in, r_out = boundaries_m[i], boundaries_m[i + 1]
        M += (4 * math.pi / 3) * rho_i * (r_out**3 - r_in**3)
        I += (8 * math.pi / 15) * rho_i * (r_out**5 - r_in**5)
    return M, I


@lru_cache(maxsize=None)
def _solve_densities(moi_factor: float | None = None) -> tuple[float, float]:
    """Solve the exact 2x2 linear system for (rho_core, rho_lm).

    Deliberately **not** evaluated at import time: this is a plain function
    (memoized with ``lru_cache`` so repeated calls with the same target are
    cheap, not so that it runs eagerly). A bad constant/target (e.g. an
    unphysical MoI factor) therefore raises only when this function is
    actually called, not merely on ``import pylov3d.mars`` — this is what
    makes the guard rails below testable in isolation (see
    ``pylov3d/tests/test_mars.py``).

    With rho_um (upper mantle) and rho_crust fixed, the mass and
    mean-moment-of-inertia constraints are linear in (rho_core, rho_lm):

        M = (4pi/3)[rho_core*V0 + rho_lm*V1 + rho_um*V2 + rho_crust*V3]
        I = (8pi/15)[rho_core*W0 + rho_lm*W1 + rho_um*W2 + rho_crust*W3]
          = moi_factor * M * R^2

    where V_i = R_i^3 - R_{i-1}^3 and W_i = R_i^5 - R_{i-1}^5 for layer i
    (radii in metres), and ``moi_factor`` is the *mean* moment-of-inertia
    factor I/MR^2 (0.36310; see module docstring for its derivation from the
    published polar moment C/MR^2 = 0.3644 via I/MR^2 = C/MR^2 - (2/3)*J2).
    This is a 2-equation, 2-unknown linear system, solved directly (no
    iterative optimizer).

    Parameters
    ----------
    moi_factor : mean moment-of-inertia target I/MR^2. Defaults to
        ``MARS["MoI_factor"]`` (0.36310). Exposed as a parameter mainly so
        tests can exercise the guard rails below with a deliberately bad
        target.

    Returns
    -------
    (rho_core, rho_lm) : fitted densities [kg/m^3].

    Raises
    ------
    ValueError
        If rho_core falls outside the plausible liquid Fe-alloy core-density
        range [5700, 6300] kg/m^3 (mean core density 5.7-6.3 g/cm^3,
        Stahler et al. 2021, Science 373), or if rho_lm <= rho_um
        (unphysical density inversion).
    """
    if moi_factor is None:
        moi_factor = MARS["MoI_factor"]

    R0_km, R1_km, R2_km, R3_km = LAYER_RADII_KM
    boundaries_m = [0.0, R0_km * 1e3, R1_km * 1e3, R2_km * 1e3, R3_km * 1e3]

    V = [boundaries_m[i + 1] ** 3 - boundaries_m[i] ** 3 for i in range(4)]
    W = [boundaries_m[i + 1] ** 5 - boundaries_m[i] ** 5 for i in range(4)]

    rho_um = MARS["upper_mantle_density"]
    rho_crust = MARS["crust_density"]
    M_target = MARS["M"]
    R = MARS["R"]
    I_target = moi_factor * M_target * R**2

    # Mass equation: a11*rho_core + a12*rho_lm = b1
    a11 = (4 * math.pi / 3) * V[0]
    a12 = (4 * math.pi / 3) * V[1]
    b1 = M_target - (4 * math.pi / 3) * (rho_um * V[2] + rho_crust * V[3])

    # MoI equation: a21*rho_core + a22*rho_lm = b2
    a21 = (8 * math.pi / 15) * W[0]
    a22 = (8 * math.pi / 15) * W[1]
    b2 = I_target - (8 * math.pi / 15) * (rho_um * W[2] + rho_crust * W[3])

    A = np.array([[a11, a12], [a21, a22]])
    b = np.array([b1, b2])
    rho_core, rho_lm = np.linalg.solve(A, b)
    rho_core = float(rho_core)
    rho_lm = float(rho_lm)

    if not (5700.0 <= rho_core <= 6300.0):
        raise ValueError(
            f"Fitted core density {rho_core:.1f} kg/m^3 is outside the "
            "plausible liquid Fe-alloy core-density range "
            "[5700, 6300] kg/m^3 (mean core density 5.7-6.3 g/cm^3, "
            "Stahler et al. 2021, Science 373); check MARS constraints."
        )
    if not (rho_lm > rho_um):
        raise ValueError(
            f"Fitted lower-mantle density {rho_lm:.1f} kg/m^3 is not "
            f"greater than the fixed upper-mantle density {rho_um:.1f} "
            "kg/m^3; unphysical density inversion. Check MARS constraints."
        )

    return rho_core, rho_lm


# ---------------------------------------------------------------------------
# Step 2 (mu_scale): bisection fit, hardcoded after convergence
# ---------------------------------------------------------------------------

# Fitted mantle shear-modulus scale factor s, such that k2(s) = 0.169 via
# get_love() on build_mars_model(mu_scale=s), using the densities from
# _solve_densities() (rho_core ~= 6128.076 kg/m^3, rho_lm ~= 4136.504
# kg/m^3) and the 2340 km L1/L2 boundary above.
# Computed by fit_mu_scale(tol=1e-12) (numerics: n_layers=4,
# method="combination", Nrbase=100); bisection converged at:
#   s = 0.964824766102174, |k2 - 0.169| < 1e-12 (residual digits vary by
#   BLAS/environment; do not pin the k2 literal itself)
# Hardcoded here so importing this module never re-runs the tidal solver.
MARS_MU_SCALE = 0.964824766102174


def build_mars_model(mu_scale: float | None = None) -> InteriorModel:
    """Build the 4-layer Mars InteriorModel.

    Parameters
    ----------
    mu_scale : mantle shear-modulus scale factor s (mu_lm = 100e9*s,
        mu_um = 70e9*s). Defaults to the fitted :data:`MARS_MU_SCALE`.

    Returns
    -------
    InteriorModel from :func:`pylov3d.types.make_interior_model`, core to
    surface: liquid core (mu0=0), lower mantle, upper mantle, crust.
    """
    if mu_scale is None:
        mu_scale = MARS_MU_SCALE

    rho_core, rho_lm = _solve_densities()

    R0_km, R1_km, R2_km, R3_km = LAYER_RADII_KM
    Ks0 = LAYER_KS

    return make_interior_model(
        R0_km=[R0_km, R1_km, R2_km, R3_km],
        rho0=[
            rho_core,
            rho_lm,
            MARS["upper_mantle_density"],
            MARS["crust_density"],
        ],
        mu0=[0.0, _MU_LM_BASE * mu_scale, _MU_UM_BASE * mu_scale, LAYER_MU_CRUST],
        Ks0=list(Ks0),
        eta0=[None, None, None, None],  # purely elastic (see caveat above)
    )


def fit_mu_scale(
    target_k2: float | None = None,
    tol: float = 1e-4,
    s_lo: float = 0.3,
    s_hi: float = 3.0,
    max_iter: int = 60,
    Nrbase: int = 100,
) -> float:
    """Bisect for the mantle shear-modulus scale s such that k2(s) = target_k2.

    Reproducibility path for :data:`MARS_MU_SCALE`. k2 decreases
    monotonically with s over [0.3, 3.0] (stiffer mantle -> smaller k2),
    so a plain bisection is deterministic and sufficient (no black-box
    optimizer).

    Uses forcing n=2, m=0, F=1.0, Td = :data:`MARS_FORCING_TD` (44387.62 s,
    half a sol) — Td is irrelevant here because the model is purely elastic
    (elastic Love numbers do not depend on forcing frequency), it is only
    supplied because ``make_forcing`` requires a value. Uses
    ``numerics = make_numerics(n_layers=4, method="combination",
    Nrbase=Nrbase)``.

    Returns
    -------
    float
        Fitted mu_scale s, converged so that |k2(s) - target_k2| < tol.

    Raises
    ------
    RuntimeError
        If the target is not bracketed by ``k2(s_lo)``/``k2(s_hi)``, or if
        bisection does not converge to within ``tol`` within ``max_iter``
        iterations.
    """
    if target_k2 is None:
        target_k2 = MARS["k2"]

    forcing = make_forcing(Td=MARS_FORCING_TD, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=Nrbase)

    def k2_of(s: float) -> float:
        model = build_mars_model(mu_scale=s)
        love, _, _ = get_love(model, forcing, numerics)
        return float(np.real(np.asarray(love.k[0])))

    lo, hi = s_lo, s_hi
    k_lo, k_hi = k2_of(lo), k2_of(hi)
    if not (k_lo > target_k2 > k_hi):
        raise RuntimeError(
            f"k2 target {target_k2} is not bracketed by s in [{s_lo}, "
            f"{s_hi}]: k2({s_lo})={k_lo}, k2({s_hi})={k_hi}"
        )

    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        k_mid = k2_of(mid)
        if abs(k_mid - target_k2) < tol:
            return mid
        if k_mid > target_k2:
            lo = mid
        else:
            hi = mid

    raise RuntimeError(
        f"fit_mu_scale did not converge within {max_iter} iterations: "
        f"last s={mid}, k2={k_mid}, target={target_k2}, tol={tol}"
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def mars_moi_factor(model_kwargs: dict | None = None) -> float:
    """Return I/(M*R^2) of the constructed Mars profile.

    Parameters
    ----------
    model_kwargs : keyword arguments forwarded to :func:`build_mars_model`
        (e.g. ``{"mu_scale": 1.0}``). Density/geometry are unaffected by
        mu_scale, so this is mainly useful for building models with a
        non-default mu_scale while checking mass/MoI consistency.

    Returns
    -------
    float
        Dimensionless moment-of-inertia factor of the built model, using
        the built model's own surface radius (not the ``MARS["R"]``
        constant) so this diagnostic is self-consistent even if the model's
        geometry is ever changed independently of ``MARS``.
    """
    model = build_mars_model(**(model_kwargs or {}))
    n = model.n_layers
    radii_km = [float(x) for x in np.asarray(model.R0[:n])]
    rho = [float(x) for x in np.asarray(model.rho0[:n])]
    M, I = _mass_and_moi(radii_km, rho)
    R_surface_m = radii_km[-1] * 1e3
    return I / (M * R_surface_m**2)
