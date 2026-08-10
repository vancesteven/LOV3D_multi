# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

r"""Detectability of the off-(2,0) tidal Love-number spectrum (TASK-026).

Closes the question TASK-021 explicitly left open: k2 is blind to WHERE a
hydration front sits; that location information lives in the off-(2,0)
Love-number spectrum TASK-016 computed and MATLAB-validated
(``pylov3d.mars_lateral``, ``data/tests/mars/mars_lateral_cross_check.mat``).
This module asks whether that spectrum is *measurable*: what gravity
precision would be required, and how does it compare to what has actually
been achieved. Parameterized over an arbitrary ``(n, m, k)`` mode list (not
Mars-specific) so the same machinery serves the Moon later. No
``pylov3d`` solver module is modified. A companion module,
:mod:`pylov3d.mars_detectability_k2m`, covers a related but distinct
observable (the diagonal k2m order-splitting benchmarked against the
MaQuIs mission concept and GRAIL) -- see that module's docstring, and
sec. 5 below, for why it is a separate quantity and a separate file.

1. The observable: |k_nm| -> |ΔC_nm|/|ΔS_nm| (derivation)
---------------------------------------------------------------------------
A nonzero tidal Love number k_n means the tide-raising potential induces an
*additional* gravitational potential at the body's own surface, equal to
k_n times the tide-raising potential itself (the textbook definition of a
potential Love number). For an external body (here, the Sun) at distance
d, the tide-raising potential at the response body's surface (radius R),
expanded in the response body's own real, 4pi-fully-normalized,
no-Condon-Shortley spherical harmonics (the exact convention of
``pylov3d.sh_data``/``pylov3d.mapping``, stated explicitly per the
project's normalization-slip warning), is, via the classical addition
theorem for Legendre polynomials written in *normalized* form (derived
below, not assumed)::

    P_n(cos psi) = (1/(2n+1)) * sum_m Pbar_nm(sin phi) Pbar_nm(sin phi')
                   * [cos(m lam) cos(m lam') + sin(m lam) sin(m lam')]

(phi, lam: response-body-fixed colatitude/longitude of the field point;
phi', lam': the sub-external-body point; psi: angular separation). This
follows directly from Ferrers/geodesy-normalized P_nm via
N_nm^2 = (2n+1)(2-delta_0m)(n-m)!/(n+m)!, substituted into the standard
*unnormalized* addition theorem
(P_n(cos psi) = P_n0 P_n0' + 2 sum_{m=1}^n [(n-m)!/(n+m)!] P_nm P_nm' cos(m
Delta lam)) and simplified -- the (2-delta_0m) factors cancel the leading
2 for m>=1, leaving one uniform (1/(2n+1)) prefactor across all m.

The degree-n tide-raising potential at radius r is W_n(r) = -(GM_ext/d)
(r/d)^n P_n(cos psi) (external-body multipole expansion, r < d). Setting
r=R and expanding P_n(cos psi) via the identity above, then equating with
the standard potential-coefficient expansion convention
V(R,phi,lam) = (GM_body/R) sum_m Pbar_nm(sin phi) [C_nm cos(m lam) + S_nm
sin(m lam)] for the *response* potential Delta V(R) = k_n W_n(R), the
(GM_body/R) prefactors cancel on both sides and matching term-by-term in
the Pbar_nm(sin phi) cos/sin(m lam) basis gives, for the classical
same-degree (diagonal) case::

    Delta C_nm = k_n * (GM_ext/GM_body) * (R/d)^(n+1) * (1/(2n+1))
                 * Pbar_nm(sin phi') cos(m lam')
    Delta S_nm = k_n * (GM_ext/GM_body) * (R/d)^(n+1) * (1/(2n+1))
                 * Pbar_nm(sin phi') sin(m lam')

**Cross-check against a published formula (retrieved this session).**
Genova, A., Goossens, S., Lemoine, F. G., et al. (2016), "Seasonal and
static gravity field of Mars from MGS, Mars Odyssey and MRO radio
science," Icarus, 272, 228-245, eq. (5) (their k2, degree-2 tidal
potential felt by an orbiting spacecraft at radius r due to a perturber of
mass GM_p at r_p): ``U = k2 (GM_p/R) (R^6/(r^3 r_p^3)) [3/2(r-hat . rp-hat
- )^2 - 1/2]`` = ``k2 GM_p R^5/(r^3 r_p^3) P2(cos psi)``. Evaluating the
present derivation's degree-n response potential (before restricting to
the surface, r=R) the same way, ``k_n (GM_ext/d) (R/d)^n (R/r)^(n+1)
P_n(cos psi)`` at n=2 gives exactly ``k2 GM_ext R^5/(r^3 d^3) P2(cos
psi)`` -- an *independent* published formula reproducing this derivation's
functional form exactly (R^5 = R^6/R, matching Genova eq. 5 verbatim), and
Genova's own eq. (1)-(2) (retrieved this session, same page) define
``Pbar_lm`` with the same normalization convention used throughout
(``[(l-m)!(2l+1)(2-delta_0m)/(l+m)!]^(1/2)``, i.e. 4pi-full, no
Condon-Shortley -- consistent with ``pylov3d.sh_data``/``pylov3d.mapping``).

**Generalization to the off-diagonal (coupled) case, from the solver's own
normalization.** ``pylov3d.love.extract_love_numbers`` defines, for a
coupled solve forced at (n_f, m_f) with unit amplitude (``F=1.0``, the
convention every caller in this project uses): the forced mode's Love
number is ``k_(n_f,m_f) = Phi_surf - 1`` (subtracting the forcing's own
unit contribution), while every *other* coupled mode's Love number is
``k_(n,m) = Phi_surf`` directly -- no subtraction, because there is no
direct forcing there. Both are read from the *same* normalized potential
state variable, nondimensionalized by one degree-independent constant
throughout the solve (never an (R/r)^n-type factor baked in per mode --
that scaling enters only through the boundary condition at the forced
degree). Because the whole coupled boundary-value problem is linear in
the forcing amplitude, scaling the unit forcing by any real physical
amplitude scales every mode's response by that same factor. Consequently
the *physical* potential perturbation at any coupled mode (n, m) is::

    Delta Phi_phys(n, m) = k_(n,m) [code, F=1] * W_(n_f,m_f)^tidal(R)

i.e. the code's dimensionless k_(n,m) times the *actual* physical
tide-raising potential amplitude at the forcing degree/order (n_f, m_f)
-- **not** a degree-n-specific tidal amplitude (there is none; nothing
forces the body at degree n directly). Matching this against the
potential-coefficient expansion exactly as in the diagonal derivation
above (now with the response degree n on the left, but the *forcing*
degree/order (n_f, m_f) fixing the right-hand amplitude) gives the
generalized relation this module implements::

    Delta C_nm = k_(n,m) * (GM_ext/GM_body) * (R/d)^(n_f+1) * (1/(2n_f+1))
                 * Pbar_(n_f,m_f)(sin phi') cos(m_f lam')          (*)

-- the SAME (GM_ext/GM_body)(R/d)^(n_f+1)/(2n_f+1) prefactor for *every*
response mode n (:func:`solar_tide_amplitude_parameter`), because it is
set by the forcing degree, not the response degree. Setting n=n_f, m=m_f
collapses (*) to the diagonal formula above exactly -- the hand-checkable
degree-2 case pinned by ``test_mars_detectability.py`` and
:data:`DEGREE2_HAND_CHECK_DC20`.

2. Mars's real solar tide, forcing-order scope, and what this cannot resolve
---------------------------------------------------------------------------
Mars's real degree-2 solar tidal potential has power at m=0 (zonal,
varying on the ~343-day half-orbital/seasonal timescale as the sub-solar
latitude cycles through Mars's 25.19-degree obliquity), m=1 (diurnal, one
sol), and m=2 (sectoral, semidiurnal -- the ~44,387.62 s period
Konopliv, Park & Folkner (2016) actually measure k2=0.169 at, and the one
:data:`pylov3d.mars.MARS_FORCING_TD` names). The TASK-016 N=115 spectrum
(``data/tests/mars/mars_lateral_cross_check.mat``, MATLAB-cross-validated)
was computed with a **unit (2,0) forcing** -- a deliberate, documented
convenience in this project (``docs/MARS_MODEL.md``, "Lateral variations"):
because this Mars model is purely elastic, the *diagonal* k2 does not
depend on which m is forced (elastic Love numbers are frequency- and
order-independent). The *coupled, off-diagonal* spectrum does **not**
share that invariance: ``pylov3d.couplings.next_coupling`` sets
``m_new = m0 + m1`` (an additive selection rule on the real, spatially
fixed MarsTopo719 crustal pattern), so forcing at (2,0) vs (2,2) excites a
**different set** of (n, m) response modes at different amplitudes -- both
theoretically (the selection rule itself) and empirically: a reduced-grid
spot check (``lmax=2, Nrbase=30``, see :func:`forcing_order_robustness_check`,
not MATLAB-cross-validated) finds the (2,2)-forced spectrum's largest mode
((3,+2), |k|=5.41e-5) within 30% of the (2,0)-forced spectrum's largest
mode ((3,0), |k|=7.23e-5) at the *same* truncation -- comparable overall
scale, different (n, m) identities. **This document uses the given
(2,0)-forced, MATLAB-validated N=115 spectrum as its primary data (as the
task specifies), and the physically dominant semidiurnal component of the
real tide is (2,2), not (2,0)** -- so the required-precision numbers below
should be read as an order-of-magnitude measurement requirement for the
off-forcing-mode spectrum as a class, not a mode-by-mode-exact prediction
of the true semidiurnal-frequency response. This is the central scope
caveat of the whole analysis, stated once here and referenced throughout.

3. What has been achieved (retrieved this session, with sources)
---------------------------------------------------------------------------
- **Mars, current orbiter tracking, closest real analogue** -- recovered
  *seasonal* (CO2 mass-exchange) low-degree gravity, the only real,
  time-varying, low-degree Mars gravity signal anyone has actually
  measured: Genova, A., Goossens, S., Lemoine, F. G., Mazarico, E.,
  Neumann, G. A., Smith, D. E., & Zuber, M. T. (2016), "Seasonal and
  static gravity field of Mars from MGS, Mars Odyssey and MRO radio
  science," Icarus, 272, 228-245 (open access, CC-BY 4.0, retrieved
  directly this session via Zenodo record 894840), Table 3: formal
  1-sigma uncertainty on each fitted annual/semi-annual/tri-annual
  Cbar_20 and Cbar_30 amplitude term, sigma(Cbar_20)=0.016e-9=1.6e-11,
  sigma(Cbar_30)=0.011e-9=1.1e-11 (:data:`MARS_SIGMA_C20_SEASONAL`,
  :data:`MARS_SIGMA_C30_SEASONAL`). **This project's own spec named
  Konopliv et al. (2016, Icarus 274) and Konopliv et al. (2020) for this
  number; both were tried this session (ScienceDirect/Wiley abstract
  pages, ADS) and returned only paywalled/empty responses -- no seasonal
  C20/C30 uncertainty table from either was retrieved.** Genova et al.
  (2016) is the same generation of MGS/Odyssey/MRO tracking data (in fact
  the source of ``data/mars/gmm3_120_sha.tab``, GMM-3, already used
  elsewhere in this repository for the TASK-016 areoid correction) and is
  the actual, retrieved source for the achieved-precision numbers used
  below; the substitution is recorded here rather than left silent.
- **Dedicated-mission benchmark: GRAIL, lunar degree-2/3 tidal Love-number
  recovery.** Konopliv, A. S., Park, R. S., Yuan, D.-N., et al. (2013),
  "The JPL lunar gravity field to spherical harmonic degree 660 from the
  GRAIL Primary Mission," J. Geophys. Res. Planets, 118, 1415-1434 (PDF
  retrieved directly this session), Table 4 and text: "GRAIL has now
  determined the Love number to better than 1%, k2=0.02405+/-0.00018" and
  "The degree-3 Love number is determined to about 25% with the formal
  errors scaled by 40" -- Table 4 itself lists k3=0.0089+/-0.0021 (GRAIL
  Primary Mission). :data:`GRAIL_SIGMA_K2`, :data:`GRAIL_K3`,
  :data:`GRAIL_SIGMA_K3` are these directly-retrieved numbers. A
  frequently cited *later*, more refined GRAIL k3 (~0.0163+/-0.0007, often
  attributed to Williams et al. 2014) turned up in web-search summaries
  but every direct-fetch attempt on that paper (Wiley agupubs) returned
  HTTP 402 (paywalled) -- **not verified in this session and not used
  here**; only the directly-retrieved Konopliv et al. (2013) Primary
  Mission number is used, which is the more conservative (larger,
  easier-to-beat) of the two, appropriate for a "what dedicated missions
  have actually demonstrated" floor.
- Both are compared in **Love-number space** for the GRAIL benchmark
  (GRAIL's sigma_k3 is directly comparable to a required |k_(n,m)|,
  dimensionless in both cases, without any further unit conversion), and
  in **Stokes-coefficient space** for the Mars-orbiter benchmark (via
  eq. (*) above).

4. Frequency separation (point 4 of the task spec)
---------------------------------------------------------------------------
The tidal signal is periodic at the solar semidiurnal period,
:data:`pylov3d.mars.MARS_FORCING_TD` = 44,387.62 s (verified TASK-025a).
The CO2 seasonal signal (section 3 above) is periodic at the Mars orbital
period, T=686.98 days (Genova et al. 2016, eq. 3, retrieved this
session) = :data:`MARS_ORBITAL_PERIOD_S`. :func:`frequency_separation_factor`
returns their ratio, ~1338x. This quantifies *only* that the two signals
sit in well-separated Fourier bins over a multi-year tracking baseline (no
aliasing concern either: spacecraft orbital periods, ~2 h, are far shorter
than the 44,387.62 s tidal period, so it is not undersampled) -- i.e. the
achieved seasonal-band precision is not degraded by confusion with the
tidal signal, or vice versa. **It does not, and cannot, be read as a
statement about achieved precision AT the semidiurnal frequency itself**
for degree ell>=3: no published degree>=3 Mars gravity recovery at
44,387.62 s period was found in this session. That gap is stated
explicitly, not filled by assumption.

5. The diagonal k2m order-splitting benchmark (MaQuIs) -- a companion,
   *different* module
---------------------------------------------------------------------------
:mod:`pylov3d.mars_detectability_k2m` (split out to keep both files under
this repo's 500-line-per-file convention -- same reason
``pylov3d/anelastic.py``/``anelastic_moon.py`` are split) computes a
*third*, distinct observable, benchmarked against Wörner, L., Root, B.
C., Bouyer, P., et al. (2023), "MaQuIs -- Concept for a Mars Quantum
Gravity Mission," Planetary and Space Science, 239, 105800 (retrieved
directly this session): whether the SAME-degree, SAME-order response
coefficient k_2m (forcing degree 2, order m; response degree 2, order the
*same* m) differs across m=0,1,2, because a laterally heterogeneous body
is not perfectly spherically symmetric. **This is a diagonal entry of the
generalized Love-number tensor (forcing (2,m), response (2,m)) -- not the
same quantity as the off-diagonal (2,+/-1)/(2,+/-2) entries of the
off-(2,0) N=115 spectrum used in sections 1-4 above** (forcing (2,0),
response (2,+/-m)); conflating the two would repeat exactly the kind of
normalization error this module's derivation exists to avoid. See that
module's docstring for the full derivation, sources, and numbers.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .mapping import fully_normalized_legendre
from .mars import MARS, MARS_FORCING_TD

# ---------------------------------------------------------------------------
# Fundamental constants (retrieved this session; see module docstring)
# ---------------------------------------------------------------------------

# IAU (2015) Resolution B3 nominal solar mass parameter -- exact by
# definition (Prsa, A., et al. (2016), "Nominal values for selected solar
# and planetary quantities: IAU 2015 Resolution B3," AJ, 152, 41;
# arXiv:1510.07674, retrieved directly this session).
GM_SUN = 1.3271244e20  # m^3/s^2

# IAU (2012) Resolution B2 astronomical unit -- exact by definition,
# retrieved directly this session (same arXiv:1510.07674, endnote 4).
AU_M = 149_597_870_700.0  # m

# Mars orbital semi-major axis / eccentricity / obliquity: widely tabulated
# planetary constants (NASA NSSDC Mars Fact Sheet; a=227,939,366 km, e and
# obliquity cross-checked via Wikipedia's Mars infobox, retrieved this
# session, which itself cites the NASA Fact Sheet and Allen (2000),
# "Astrophysical Quantities"). Direct fetch of nssdc.gsfc.nasa.gov itself
# redirected to a generic landing page in this session; not independently
# re-verified beyond the aggregator cross-check. MARS_OBLIQUITY_DEG is
# recorded for context only -- it is not used in any computation below,
# because the peak-amplitude bound (Section "characteristic amplitude")
# is evaluated at the sub-solar-point equinox (sin(phi')=0), which is
# reached once per Mars year for any nonzero obliquity.
MARS_SEMIMAJOR_AXIS_M = 227_939_366_000.0  # m (1.52368055 AU)
MARS_ECCENTRICITY = 0.0934
MARS_OBLIQUITY_DEG = 25.19  # context only, see above

MARS_PERIHELION_M = MARS_SEMIMAJOR_AXIS_M * (1.0 - MARS_ECCENTRICITY)

# Mars orbital period -- Genova, A., et al. (2016), Icarus 272, 228-245,
# eq. (3): "T = 686.98 days is the orbital period of Mars" (retrieved
# directly this session).
MARS_ORBITAL_PERIOD_S = 686.98 * 86400.0

# ---------------------------------------------------------------------------
# Achieved precision (retrieved this session; see module docstring, sec. 3)
# ---------------------------------------------------------------------------

# Genova et al. (2016), Icarus 272, 228-245, Table 3 (annual/semi-annual/
# tri-annual Cbar_20, Cbar_30 fit, formal 1-sigma per amplitude term).
MARS_SIGMA_C20_SEASONAL = 0.016e-9  # 1.6e-11
MARS_SIGMA_C30_SEASONAL = 0.011e-9  # 1.1e-11

# Konopliv, Park, Yuan, et al. (2013), JGR Planets, 118, 1415-1434, Table 4
# (GRAIL Primary Mission).
GRAIL_K2 = 0.02405
GRAIL_SIGMA_K2 = 0.00018
GRAIL_K3 = 0.0089
GRAIL_SIGMA_K3 = 0.0021

# ---------------------------------------------------------------------------
# Core relation (eq. * in the module docstring) -- body-agnostic
# ---------------------------------------------------------------------------

def solar_tide_amplitude_parameter(
    GM_ext: float, GM_body: float, R_body: float, d: float, n_forcing: int = 2,
) -> float:
    """``(GM_ext/GM_body) * (R_body/d)**(n_forcing+1) / (2*n_forcing+1)``.

    The forcing-degree-only prefactor in eq. (*) (module docstring, sec.
    1); shared by every response mode regardless of its own degree.
    """
    return (GM_ext / GM_body) * (R_body / d) ** (n_forcing + 1) / (2 * n_forcing + 1)


def peak_legendre_factor(n_forcing: int, m_forcing: int) -> float:
    """``max_phi |Pbar_(n_forcing,m_forcing)(sin phi)|``, attained at the
    equinox sub-point (``sin(phi)=0``) for every ``(n, m)`` used in this
    module (verified directly: ``d/d(sin phi) Pbar_20`` and ``Pbar_22`` are
    both zero and each is a local extremum at ``sin(phi)=0`` over
    ``[-1, 1]``). Uses :func:`pylov3d.mapping.fully_normalized_legendre`
    (the same 4pi-normalized, no-Condon-Shortley convention as everything
    else in this module) rather than a fresh, unvalidated formula.
    """
    P = fully_normalized_legendre(n_forcing, np.array([0.0]))
    return abs(float(P[n_forcing, m_forcing, 0]))


def required_stokes_amplitude(
    k_nm: complex | float,
    GM_ext: float, GM_body: float, R_body: float, d: float,
    n_forcing: int, m_forcing: int,
) -> float:
    """Peak |Delta C_nm| (equivalently |Delta S_nm|) implied by a coupled
    Love number ``k_nm``, via eq. (*) (module docstring, sec. 1), evaluated
    at the sub-forcing-body equinox (the amplitude bound, not a full time
    series -- module docstring, sec. "characteristic amplitude"; this is
    an *upper* bound on the achievable signal, i.e. the most optimistic
    case for detectability, not an average or worst case).
    """
    xi = solar_tide_amplitude_parameter(GM_ext, GM_body, R_body, d, n_forcing)
    p = peak_legendre_factor(n_forcing, m_forcing)
    return abs(k_nm) * xi * p


# Hand-checkable degree-2 pin (module docstring, sec. 1): k2=0.169 (the
# real, measured Mars value, pylov3d.mars.MARS["k2"]), forcing=(2,0),
# mean Mars-Sun distance, equinox. Independently computed by hand
# (calculator arithmetic, not by calling this module's functions) as
# 0.169 * (GM_SUN/MARS["GM"]) * (MARS["R"]/MARS_SEMIMAJOR_AXIS_M)**3 / 5
# * sqrt(5)/2 = 3.8503539180526744e-10 (using this module's own numeric
# constants for GM_SUN/MARS_SEMIMAJOR_AXIS_M and pylov3d.mars.MARS's GM/R)
# -- pinned by test_mars_detectability.py::test_degree2_hand_check.
DEGREE2_HAND_CHECK_DC20 = 3.8503539180526744e-10


# ---------------------------------------------------------------------------
# Mars convenience wrappers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECTRUM_PATH = _REPO_ROOT / "data" / "tests" / "mars" / "mars_lateral_cross_check.mat"


def load_mars_lateral_spectrum(path: Path | str = DEFAULT_SPECTRUM_PATH) -> dict:
    """Load the MATLAB-cross-validated N=115 TASK-016 coupled spectrum.

    ``scipy.io.loadmat`` per the task spec. Returns a dict with ``modes``
    (list of ``{"n", "m", "k", "is_forcing"}``, sorted by descending
    |k| among non-forcing modes first) and the scalar provenance fields
    (``forcing_n``, ``forcing_m``, ``k2_uniform``, ``k2_forcing``,
    ``k2_shift``, ``Nrbase``, ``perturbation_order``).
    """
    import scipy.io as sio

    d = sio.loadmat(path)
    n_arr = d["n"].ravel().astype(int)
    m_arr = d["m"].ravel().astype(int)
    k_arr = d["k"].ravel().astype(complex)
    forcing_n = int(d["forcing_n"].ravel()[0])
    forcing_m = int(d["forcing_m"].ravel()[0])

    modes = [
        {
            "n": int(ni), "m": int(mi), "k": complex(ki),
            "is_forcing": (int(ni) == forcing_n and int(mi) == forcing_m),
        }
        for ni, mi, ki in zip(n_arr, m_arr, k_arr)
    ]
    modes.sort(key=lambda row: (row["is_forcing"], -abs(row["k"])))

    return {
        "modes": modes,
        "forcing_n": forcing_n,
        "forcing_m": forcing_m,
        "k2_uniform": float(d["k2_uniform"].ravel()[0]),
        "k2_forcing": float(d["k2_forcing"].ravel()[0]),
        "k2_shift": float(d["k2_shift"].ravel()[0]),
        "Nrbase": int(d["Nrbase"].ravel()[0]),
        "perturbation_order": int(d["perturbation_order"].ravel()[0]),
    }


def mars_required_precision(k_nm: complex | float, bound: str = "optimistic") -> float:
    """Peak |Delta C_nm| for a Mars off-forcing mode, forcing degree n_f=2.

    Two bounding conventions (module docstring, sec. 2 -- the real tide's
    dominant azimuthal order is (2,2), not the (2,0) the given spectrum
    was computed with; this brackets that ambiguity together with the
    orbital-distance choice rather than picking one number silently):

    - ``"optimistic"``: perihelion distance (largest forcing amplitude)
      and ``m_forcing=2`` (the larger of Pbar_20/Pbar_22 at equinox) --
      the largest, most detection-favorable required-precision bound (a
      bigger real signal means a coarser instrument still detects it, so
      the "required precision" bar is loosest here; correspondingly
      gives the *smallest* achieved/required ratio in
      :func:`mars_off20_detectability_table`).
    - ``"conservative"``: mean Sun distance and ``m_forcing=0`` (matching
      the given spectrum's own forcing order exactly) -- about 2.3x
      smaller (tighter/harder bar) than the optimistic bound.
    """
    if bound == "optimistic":
        d, m_forcing = MARS_PERIHELION_M, 2
    elif bound == "conservative":
        d, m_forcing = MARS_SEMIMAJOR_AXIS_M, 0
    else:
        raise ValueError(f"bound must be 'optimistic' or 'conservative', got {bound!r}")
    return required_stokes_amplitude(
        k_nm, GM_SUN, MARS["GM"], MARS["R"], d, n_forcing=2, m_forcing=m_forcing,
    )


def mars_off20_detectability_table(spectrum: dict | None = None) -> list[dict]:
    """Per-mode required-precision table for every non-forcing mode.

    Each row: ``n``, ``m``, ``k_abs``, ``required_dC_optimistic``,
    ``required_dC_conservative`` (:func:`mars_required_precision`, both
    bounds), ``ratio_orbiter`` (:data:`MARS_SIGMA_C30_SEASONAL` /
    required, optimistic bound -- how many times *too coarse* current
    Mars-orbiter seasonal-gravity precision is; <1 would mean detectable),
    ``ratio_grail`` (:data:`GRAIL_SIGMA_K3` / ``k_abs`` directly, Love-number
    space, no unit conversion needed). Sorted by descending |k|.
    """
    spectrum = spectrum or load_mars_lateral_spectrum()
    rows = []
    for mode in spectrum["modes"]:
        if mode["is_forcing"]:
            continue
        k_abs = abs(mode["k"])
        req_opt = mars_required_precision(k_abs, bound="optimistic")
        req_cons = mars_required_precision(k_abs, bound="conservative")
        rows.append({
            "n": mode["n"], "m": mode["m"], "k_abs": k_abs,
            "required_dC_optimistic": req_opt,
            "required_dC_conservative": req_cons,
            "ratio_orbiter": MARS_SIGMA_C30_SEASONAL / req_opt,
            "ratio_grail": GRAIL_SIGMA_K3 / k_abs,
        })
    return rows


def frequency_separation_factor() -> float:
    """Mars-year / semidiurnal-tidal-period ratio (module docstring, sec.
    4): how many semidiurnal cycles fit in one seasonal (CO2) cycle."""
    return MARS_ORBITAL_PERIOD_S / MARS_FORCING_TD


def forcing_order_robustness_check(lmax: int = 2, Nrbase: int = 30) -> dict:
    """(2,0)- vs (2,2)-forced coupled spectra at reduced resolution.

    Not MATLAB-cross-validated (unlike the N=115 (2,0) spectrum); a cheap
    (~10-20 s total) order-of-magnitude check supporting the module
    docstring's forcing-order scope caveat (sec. 2). Returns the top-|k|
    non-forcing mode from each run and their amplitude ratio.
    """
    from .mars_lateral import mars_lateral_love_spectrum

    out = {}
    for fm in (0, 2):
        result = mars_lateral_love_spectrum(lmax=lmax, forcing=(2, fm), Nrbase=Nrbase)
        love = result["love"]
        best_idx, best_abs = None, -1.0
        for idx in range(len(love.n)):
            if int(love.n[idx]) == 2 and int(love.m[idx]) == fm:
                continue
            a = abs(complex(love.k[idx]))
            if a > best_abs:
                best_idx, best_abs = idx, a
        out[fm] = {
            "n": int(love.n[best_idx]), "m": int(love.m[best_idx]),
            "k_abs": best_abs, "n_modes": len(love.n),
        }
    out["ratio"] = out[2]["k_abs"] / out[0]["k_abs"]
    return out
