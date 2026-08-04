"""Mars lateral-variation stage (TASK-016).

Turns the committed MarsTopo719 shape model into a laterally varying crust
rigidity for the TASK-011 4-layer Mars reference model
(:func:`pylov3d.mars.build_mars_model`), and runs it through the coupled
solver (:func:`pylov3d.love.get_love` with ``mu_variable``) to produce the
Mars Love-number *spectrum* -- response modes beyond the (2,0) tide that
lateral crustal structure excites. Mars application of the mode-coupling
machinery already cross-validated against MATLAB on the Weber Moon
(``test_matlab_validation_ocean.py``, ``fig5_validation_pedigree.py``).
Full derivation/numbers/caveats: ``docs/MARS_MODEL.md``, "Lateral
variations (TASK-016)". Approved design: ``docs/tasks/TASK-016-design.md``
(Airy compensation, ``n_lv <= 4``, fixed-amplitude forward runs).

Physical chain (full derivation, numbers, caveats: ``docs/MARS_MODEL.md``)
---------------------------------------------------------------------------
1. :func:`crustal_thickness_variation` -- MarsTopo719 truncated to
   ``lmax``, C00/C20 dropped, referenced to the low-degree GMM-3 *areoid*
   (:func:`_geoid_height_variation`, first-order Bruns relation
   ``N_lm = r0 * C_lm`` -- D2 revision: topography relative to the level
   surface, not the bare C20-corrected sphere; measured effect max|dt|
   40.65->34.2 km), then Airy ``dt = h_corrected * rho_c/(rho_m-rho_c)``
   (factor 5.8).
2. :func:`mu_variable_from_topography` -- linearizes the fixed 50 km crust
   shell's crust fraction in ``dt``:
   ``d(mu)/mu_bar = (mu_crust - mu_um_eff(mu_scale)) * dt / (50 km * mu_bar)``,
   ``mu_bar = LAYER_MU_CRUST`` (the crust *is* the model's surface layer,
   so ``process_lateral_variations``'s elastic-branch normalization
   ``model.mu[3] == 1.0`` exactly). Thicker crust displaces stiffer mantle
   out of the shell -> mu decreases; no ``f in [0,1]`` clipping (deliberate
   stage-1 choice). ``mu_scale`` is threaded through explicitly, not frozen
   at the fitted default (D3: a non-default ``mu_scale`` changes
   ``mu_um_eff``, so pairing a rescaled model with the default-``mu_scale``
   amplitude is a 7.5x error at ``mu_scale=0.5``, inside
   ``pylov3d.mars_mc``'s [0.3, 3.0] prior box).
3. :func:`_real_sh_to_complex_mu_variable` -- real 4pi-normalized
   ``(C_nm, S_nm)`` -> the solver's complex-SH ``mu_variable`` convention,
   generalizing MATLAB ``get_rheology.m``'s peak-to-peak conversion (lines
   ~517-589)::

       amp(n, 0)  = C_n0
       amp(n, +m) = (C_nm - i*S_nm) / sqrt(2)          (m > 0)
       amp(n, -m) = (-1)**m * (C_nm + i*S_nm) / sqrt(2)

   ``S_nm = 0`` is exactly the already-MATLAB-validated
   ``test_matlab_validation_ocean.py::_p2p_to_mu_variable`` (~2e-6
   relative, 5 Weber-Moon cases, purely elastic -- same code path Mars's
   crust uses). Round-trip validated to machine precision against
   :func:`pylov3d.mapping.sh_to_latlon` (:func:`complex_sh_synthesis`,
   ``TestCSHRoundTrip``); the imaginary/``S_nm`` branch (never exercised by
   the Moon harness's pure-cosine inputs) is additionally checked by
   ``TestRotationCovariance``. Different convention from the
   ``sph_harm_y``-based ``rheology._sh_synthesis`` (viscoelastic-only,
   irrelevant here); do not conflate the two.
4. :func:`mars_lateral_love_spectrum` -- ``build_mars_model()`` +
   ``get_love(..., mu_variable=...)``, NumPy coupled path.

Both ``|dt| < 50 km`` and the elastic-positivity bound
``|d(mu)/mu_bar| < 1`` are checked by :func:`crustal_thickness_diagnostics`
(the first raises ``ValueError``; measured max|dt|=34.2 km,
max|d(mu)/mu_bar|=0.857 post-D2, no violation). The areoid is itself
partly produced by the compensated roots computed here; a rigorous
treatment iterates -- this is the non-iterated first correction (full
caveat: ``docs/MARS_MODEL.md``).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .love import get_love
from .mapping import fully_normalized_legendre, sh_to_latlon
from .mars import (
    LAYER_MU_CRUST,
    MARS,
    MARS_FORCING_TD,
    MARS_MU_SCALE,
    _MU_UM_BASE,
    build_mars_model,
)
from .sh_data import load_shadr, load_shape, truncate
from .types import make_forcing, make_numerics

# ---------------------------------------------------------------------------
# Paths and fixed model constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data" / "mars"
TOPO_PATH = DATA_DIR / "MarsTopo719.shape.gz"
GRAVITY_PATH = DATA_DIR / "gmm3_120_sha.tab"
EXPORT_PATH = DATA_DIR / "mars_mu_variable_lateral.npz"

# Crust layer index in build_mars_model()'s 4-layer stack (0=core, 1=lower
# mantle, 2=upper mantle, 3=crust). The crust IS the surface layer
# (n_layers - 1 = 3) -- see module docstring step 2 (mu_bar).
CRUST_LAYER_INDEX = 3

# Airy isostatic factor rho_c / (rho_m - rho_c) = 2900 / (3400 - 2900) = 5.8.
AIRY_FACTOR = MARS["crust_density"] / (MARS["upper_mantle_density"] - MARS["crust_density"])


def _mu_um_eff(mu_scale: float | None = None) -> float:
    """Effective upper-mantle shear modulus at ``mu_scale`` (default: the
    fitted :data:`pylov3d.mars.MARS_MU_SCALE`)."""
    if mu_scale is None:
        mu_scale = MARS_MU_SCALE
    return _MU_UM_BASE * mu_scale

def _dmu_ddt_coeff(mu_scale: float | None = None) -> float:
    """``d(mu/mu_bar)/d(dt)`` [1/m] (module docstring step 2); ``mu_bar`` is
    the fixed ``LAYER_MU_CRUST``, only the displaced effective upper-mantle
    modulus depends on ``mu_scale`` (D3)."""
    return (LAYER_MU_CRUST - _mu_um_eff(mu_scale)) / (MARS["crust_thickness"] * LAYER_MU_CRUST)


# ---------------------------------------------------------------------------
# Step 1: crustal thickness variation (Airy, areoid-referenced)
# ---------------------------------------------------------------------------

def _geoid_height_variation(
    lmax: int = 4, gravity_path: Path | str = GRAVITY_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-degree areoid (geoid) height, real 4pi-normalized SH, meters.

    First-order Bruns relation (module docstring, step 1): the SH
    coefficient of the geoid undulation N is ``r0 * C_lm`` (``r0 * S_lm``
    for sine terms), C00/C20 dropped to match the topography side. Returns
    ``(n_clm, n_slm)``, ``(lmax+1, lmax+1)`` arrays.
    """
    gravity = truncate(load_shadr(gravity_path), lmax)
    g_clm = gravity["clm"].copy()
    g_slm = gravity["slm"].copy()
    g_clm[0, 0] = 0.0
    g_clm[2, 0] = 0.0
    r0 = gravity["r0_m"]
    return r0 * g_clm, r0 * g_slm

def crustal_thickness_variation(
    lmax: int = 4,
    topo_path: Path | str = TOPO_PATH,
    gravity_path: Path | str = GRAVITY_PATH,
) -> dict[tuple[int, int], float]:
    """Airy-compensated crustal thickness variation, real 4pi-normalized SH.

    Loads MarsTopo719, truncates to ``lmax``, drops C00/C20, subtracts the
    low-degree GMM-3 areoid height (:func:`_geoid_height_variation` --
    module docstring, step 1) so the residual is topography relative to
    the level surface, not the bare sphere, then applies
    ``dt = h_corrected * AIRY_FACTOR``.

    Returns ``dict[(n, m), float]``: real, 4pi-normalized SH coefficients
    of ``dt`` [m], :func:`pylov3d.mapping.sh_to_latlon` convention (``m>=0``
    cosine, ``(n,-m)`` ``m>=1`` sine). Only nonzero entries included.
    """
    shape = truncate(load_shape(topo_path), lmax)
    clm = shape["clm"].copy()
    slm = shape["slm"].copy()
    clm[0, 0] = 0.0  # mean radius: not a lateral variation
    clm[2, 0] = 0.0  # rotational flattening: not a crustal load

    n_clm, n_slm = _geoid_height_variation(lmax=lmax, gravity_path=gravity_path)
    clm = clm - n_clm  # relative to the low-degree areoid (D2), not the bare sphere
    slm = slm - n_slm

    dt: dict[tuple[int, int], float] = {}
    for n in range(lmax + 1):
        for m in range(n + 1):
            c = float(clm[n, m]) * AIRY_FACTOR
            if c != 0.0:
                dt[(n, m)] = c
            if m >= 1:
                s = float(slm[n, m]) * AIRY_FACTOR
                if s != 0.0:
                    dt[(n, -m)] = s
    return dt

def crustal_thickness_diagnostics(
    lmax: int = 4,
    topo_path: Path | str = TOPO_PATH,
    gravity_path: Path | str = GRAVITY_PATH,
    mu_scale: float | None = None,
    nlat: int = 180, nlon: int = 360,
) -> dict:
    """Sanity/clipping diagnostics for the Airy crustal-thickness field.

    Returns a dict (all reported; only the ``|dt| < 50 km`` bound is a hard
    ``ValueError`` -- an explicit raise, not a bare ``assert``, so it
    survives ``python -O``): ``max_abs_dt_m`` (peak |dt|, fine grid [m]),
    ``max_abs_dt_over_t0`` (peak / 50 km), ``max_abs_dmu_over_mubar`` (peak
    |delta_mu/mu_bar| at the given ``mu_scale``; > 1 => linearized mu_eff
    would go non-positive -- reported, not clipped), ``degree_rms_km``
    (dict[n] -> RMS dt at that degree [km], the degree-1-vs-2-3 dichotomy).
    """
    dt = crustal_thickness_variation(lmax=lmax, topo_path=topo_path, gravity_path=gravity_path)
    grid = sh_to_latlon(dt, nlat=nlat, nlon=nlon)
    max_abs_dt_m = float(np.max(np.abs(grid.z)))
    t0 = MARS["crust_thickness"]

    if not (max_abs_dt_m < t0):
        raise ValueError(
            f"max|dt| = {max_abs_dt_m / 1e3:.3f} km exceeds the 50 km shell "
            "thickness; the linearized Airy-to-rigidity map is not "
            "meaningful beyond this bound (see module docstring)."
        )

    degree_rms_km: dict[int, float] = {}
    for n in range(1, lmax + 1):
        acc = 0.0
        for m in range(0, n + 1):
            acc += dt.get((n, m), 0.0) ** 2
            if m >= 1:
                acc += dt.get((n, -m), 0.0) ** 2
        degree_rms_km[n] = math.sqrt(acc) / 1e3

    return {
        "max_abs_dt_m": max_abs_dt_m,
        "max_abs_dt_over_t0": max_abs_dt_m / t0,
        "max_abs_dmu_over_mubar": abs(_dmu_ddt_coeff(mu_scale)) * max_abs_dt_m,
        "degree_rms_km": degree_rms_km,
    }


# ---------------------------------------------------------------------------
# Step 2/3: real dt SH -> complex mu_variable
# ---------------------------------------------------------------------------

def _real_sh_to_complex_mu_variable(
    real_coeffs: dict[tuple[int, int], float],
) -> list[tuple[int, int, complex]]:
    """Real, 4pi-normalized SH amplitudes -> complex-SH ``mu_variable`` entries.

    Generalizes MATLAB ``get_rheology.m``'s peak-to-peak real->complex
    conversion (lines ~517-589: ``m == 0`` / ``m > 0`` / ``m < 0`` branches)
    from a single mode to arbitrary ``(C_nm, S_nm)`` pairs::

        amp(n, 0)  = C_n0
        amp(n, +m) = (C_nm - i*S_nm) / sqrt(2)      (m > 0)
        amp(n, -m) = (-1)**m * (C_nm + i*S_nm) / sqrt(2)

    With ``S_nm = 0`` this is exactly the already end-to-end MATLAB
    validated ``test_matlab_validation_ocean.py::_p2p_to_mu_variable``
    (~2e-6 relative, 5 Weber-Moon cases). See :func:`complex_sh_synthesis`
    for the implicit complex ``Y_n^m`` and its round-trip validation.

    ``real_coeffs``: ``m >= 0`` keys hold cosine coefficients, ``m < 0``
    (``|m| >= 1``) hold sine coefficients (same convention as
    :func:`crustal_thickness_variation`'s return value).

    Returns conjugate-paired ``[(n, m, complex), ...]`` (every ``m != 0``
    mode has both signs), ready to use as a ``mu_variable`` layer's list.
    """
    degrees_orders = sorted({(n, abs(m)) for (n, m) in real_coeffs})
    entries: list[tuple[int, int, complex]] = []
    for n, m in degrees_orders:
        if m == 0:
            c = real_coeffs.get((n, 0), 0.0)
            if c != 0.0:
                entries.append((n, 0, complex(c, 0.0)))
        else:
            c = real_coeffs.get((n, m), 0.0)
            s = real_coeffs.get((n, -m), 0.0)
            if c == 0.0 and s == 0.0:
                continue
            amp_pos = (c - 1j * s) / math.sqrt(2.0)
            amp_neg = ((-1) ** m) * (c + 1j * s) / math.sqrt(2.0)
            entries.append((n, m, amp_pos))
            entries.append((n, -m, amp_neg))
    return entries


def complex_sh_synthesis(
    entries: list[tuple[int, int, complex]],
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    """Evaluate ``mu_variable``-convention complex-SH entries on a lat/lon grid.

    The "solver's own convention" synthesizer used to validate
    :func:`_real_sh_to_complex_mu_variable`
    (``test_mars_lateral.py::TestCSHRoundTrip``). *Derived*, not assumed:
    self-consistency between the ``S_nm=0`` case (validated
    ``_p2p_to_mu_variable``) and its ``C_nm=0`` sibling (``get_rheology.m``'s
    ``m < 0`` branch) against the known real fields
    ``C_nm*Pbar_n^m*cos(m*phi)`` / ``S_nm*Pbar_n^m*sin(m*phi)`` uniquely
    fixes::

        Y_n^0    =        Pbar_n^0(sin lat)
        Y_n^{+m} =        Pbar_n^m(sin lat) * exp(+i*m*lon) / sqrt(2)   (m>0)
        Y_n^{-m} = (-1)^m Pbar_n^m(sin lat) * exp(-i*m*lon) / sqrt(2)

    using the same no-Condon-Shortley 4pi-normalized ``Pbar_n^m`` as
    :func:`pylov3d.mapping.fully_normalized_legendre` -- **not**
    ``scipy.special.sph_harm_y`` (used only by ``rheology._sh_synthesis``,
    an unrelated viscoelastic-only path).

    Returns (len(lat_deg), len(lon_deg)) complex128; for conjugate-paired
    input the imaginary part is zero to machine precision.
    """
    if not entries:
        return np.zeros((len(lat_deg), len(lon_deg)), dtype=complex)

    t = np.sin(np.radians(lat_deg))
    phi = np.radians(lon_deg)
    n_max = max(n for n, _m, _a in entries)
    P = fully_normalized_legendre(n_max, t)  # (n_max+1, n_max+1, nlat)

    field = np.zeros((len(lat_deg), len(lon_deg)), dtype=complex)
    for n, m, amp in entries:
        mm = abs(m)
        Pnm = P[n, mm, :]
        if m == 0:
            ang = np.ones_like(phi, dtype=complex)
            scale = 1.0
        else:
            ang = np.exp(1j * m * phi)
            scale = (((-1) ** mm) if m < 0 else 1.0) / math.sqrt(2.0)
        field += amp * scale * (Pnm[:, None] * ang[None, :])
    return field


def dmu_over_mu_real(
    lmax: int = 4,
    topo_path: Path | str = TOPO_PATH,
    gravity_path: Path | str = GRAVITY_PATH,
    mu_scale: float | None = None,
) -> dict[tuple[int, int], float]:
    """Real, 4pi-normalized ``d(mu)/mu_bar`` SH coefficients of the crust
    shell (step 2, pre real->complex conversion) -- for inspection/plotting
    (fig6). ``mu_scale`` defaults to the fitted ``MARS_MU_SCALE``; pass the
    same value used to build the interior model (D3)."""
    dt = crustal_thickness_variation(lmax=lmax, topo_path=topo_path, gravity_path=gravity_path)
    coeff = _dmu_ddt_coeff(mu_scale)
    return {nm: coeff * val for nm, val in dt.items()}

def mu_variable_from_topography(
    lmax: int = 4,
    topo_path: Path | str = TOPO_PATH,
    gravity_path: Path | str = GRAVITY_PATH,
    mu_scale: float | None = None,
) -> dict[int, list[tuple[int, int, complex]]]:
    """Crust-layer ``mu_variable`` entries from Airy-compensated topography.

    Linearizes the crust shell's rigidity (module docstring, steps 1-3)
    and converts to the solver's complex-SH convention. ``mu_scale`` must
    match the value used to build the interior model this feeds (D3) --
    both default to the fitted :data:`pylov3d.mars.MARS_MU_SCALE`.

    Returns ``{CRUST_LAYER_INDEX: entries}``, usable directly as the
    ``mu_variable`` argument to :func:`pylov3d.love.get_love`.
    """
    real_dmu = dmu_over_mu_real(
        lmax=lmax, topo_path=topo_path, gravity_path=gravity_path, mu_scale=mu_scale,
    )
    entries = _real_sh_to_complex_mu_variable(real_dmu)
    return {CRUST_LAYER_INDEX: entries}


# ---------------------------------------------------------------------------
# Step 4: coupled Love spectrum
# ---------------------------------------------------------------------------

def mars_lateral_love_spectrum(
    lmax: int = 4,
    forcing: tuple[int, int] = (2, 0),
    perturbation_order: int = 2,
    Nrbase: int = 30,
    method: str = "combination",
    mu_scale: float | None = None,
    F: complex = 1.0,
) -> dict:
    """Mars coupled Love-number spectrum from Airy-compensated lateral rigidity.

    Real MarsTopo719 at lmax=4 activates N=115 coupled modes at
    perturbation_order=2 (design doc's N~15-30 rule-of-thumb assumed fewer
    active modes than real cos+sin topography gives). Default ``Nrbase=30``
    converges to ~1e-11 relative in k2 vs. ``Nrbase=15``; wall time is
    machine-dependent (~130-140 s measured, inside the 5-minute guard).
    ``mu_scale`` is forwarded to *both* :func:`mu_variable_from_topography`
    and :func:`pylov3d.mars.build_mars_model` (D3) -- never pair them with
    different values. NumPy coupled path (``pylov3d.jax_coupled`` is the
    JAX equivalent, cross-checked in ``TestJaxEquivalence``).

    Returns a dict: ``love``, ``wall_s``, ``mu_variable``, ``model``,
    ``forcing_obj``, ``numerics``.
    """
    n_f, m_f = forcing
    mu_variable = mu_variable_from_topography(lmax=lmax, mu_scale=mu_scale)
    model = build_mars_model(mu_scale=mu_scale)
    forcing_obj = make_forcing(Td=MARS_FORCING_TD, n=n_f, m=m_f, F=F)
    numerics = make_numerics(
        n_layers=4, method=method, Nrbase=Nrbase,
        perturbation_order=perturbation_order,
    )

    t_start = time.perf_counter()
    love, y_rad, model_out = get_love(
        model, forcing_obj, numerics, mu_variable=mu_variable,
    )
    wall_s = time.perf_counter() - t_start

    return {
        "love": love,
        "wall_s": wall_s,
        "mu_variable": mu_variable,
        "model": model_out,
        "forcing_obj": forcing_obj,
        "numerics": numerics,
    }


# ---------------------------------------------------------------------------
# Step 5: export for MATLAB cross-check (B, TASK-014 pt 2)
# ---------------------------------------------------------------------------

_EXPORT_README = """\
TASK-016 Mars lateral rigidity export -- for the native-MATLAB coupled
cross-check (TASK-014 part 2).

Contents
--------
layer_idx, n, m, amp_real, amp_imag : the crust-layer mu_variable entries
    (complex amp = amp_real + 1j*amp_imag), in the SAME complex-SH
    convention as the already-validated Weber-Moon p2p harness
    (pylov3d/tests/test_matlab_validation_ocean.py::_p2p_to_mu_variable);
    see pylov3d/mars_lateral.py module docstring for the exact formula.
    layer_idx is 0-based (0=core, 1=lower mantle, 2=upper mantle, 3=crust);
    all entries here have layer_idx == crust_layer_index. MATLAB layer
    index = python index + 1 (Interior_Model(ilayer) is 1-based) -- i.e.
    crust_layer_index 3 here is Interior_Model(4) in MATLAB.
dt_n, dt_m, dt_val_m : the real, 4pi-normalized Airy crustal-thickness SH
    coefficients [m] this was derived from (m >= 0: cosine C_nm; m < 0:
    sine S_n|m|) -- pylov3d.mapping.sh_to_latlon convention. Referenced to
    the low-degree GMM-3 areoid, not the bare sphere (see module docstring,
    step 1).
rho_crust, rho_upper_mantle [kg/m^3], airy_factor [-], mu_crust_pa,
    mu_um_eff_pa [Pa], crust_thickness_m [m], mars_mu_scale [-] (the
    mu_scale actually used, may differ from the fitted default),
    crust_layer_index, lmax : provenance / reproducibility constants.

eta0 convention warning (B, please read before running)
---------------------------------------------------------------------------
This model is purely elastic on all 4 layers. Python's make_interior_model
uses eta0 = NaN to mean "elastic". MATLAB's get_rheology.m instead requires
eta0 to be EMPTY/absent for a layer to be treated as elastic -- eta0 = NaN
in MATLAB wrongly falls into the viscoelastic branch and NaN-poisons the
solve (documented TASK-014 part 1 porting gotcha, docs/HANDOFF.md). Leave
eta0 empty ([]) for all 4 layers in the MATLAB Interior_Model struct, do
not set it to NaN.
"""


def export_mu_variable_lateral(
    path: Path | str = EXPORT_PATH, lmax: int = 4, mu_scale: float | None = None,
) -> Path:
    """Write the crust ``mu_variable`` entries + provenance to an .npz file.

    For machine B's native-MATLAB coupled cross-check (TASK-014 part 2).
    See :data:`_EXPORT_README` (embedded as the ``readme`` array) for the
    eta0 empty-vs-NaN convention warning. Returns the path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dt = crustal_thickness_variation(lmax=lmax)
    mu_variable = mu_variable_from_topography(lmax=lmax, mu_scale=mu_scale)
    entries = mu_variable[CRUST_LAYER_INDEX]

    layer_idx = np.full(len(entries), CRUST_LAYER_INDEX, dtype=int)
    n_arr = np.array([e[0] for e in entries], dtype=int)
    m_arr = np.array([e[1] for e in entries], dtype=int)
    amp_arr = np.array([e[2] for e in entries], dtype=complex)

    dt_items = sorted(dt.items())
    dt_n = np.array([k[0] for k, _v in dt_items], dtype=int)
    dt_m = np.array([k[1] for k, _v in dt_items], dtype=int)
    dt_val_m = np.array([v for _k, v in dt_items], dtype=float)

    np.savez(
        path,
        layer_idx=layer_idx, n=n_arr, m=m_arr,
        amp_real=amp_arr.real, amp_imag=amp_arr.imag,
        dt_n=dt_n, dt_m=dt_m, dt_val_m=dt_val_m,
        rho_crust=MARS["crust_density"],
        rho_upper_mantle=MARS["upper_mantle_density"],
        airy_factor=AIRY_FACTOR,
        mu_crust_pa=LAYER_MU_CRUST,
        mu_um_eff_pa=_mu_um_eff(mu_scale),
        crust_thickness_m=MARS["crust_thickness"],
        mars_mu_scale=(MARS_MU_SCALE if mu_scale is None else mu_scale),
        crust_layer_index=CRUST_LAYER_INDEX,
        lmax=lmax,
        readme=_EXPORT_README,
    )
    return path
