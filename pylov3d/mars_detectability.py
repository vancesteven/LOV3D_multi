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

    Delta C_nm = -k_n * (GM_ext/GM_body) * (R/d)^(n+1) * (1/(2n+1))
                 * Pbar_nm(sin phi') cos(m lam')
    Delta S_nm = -k_n * (GM_ext/GM_body) * (R/d)^(n+1) * (1/(2n+1))
                 * Pbar_nm(sin phi') sin(m lam')

(the leading minus carried through from ``W_n``'s own sign convention
above; every quantity this module actually computes is ``|Delta C_nm|``
-- :func:`required_stokes_amplitude` returns ``abs(...)`` unconditionally
-- so the sign is immaterial to every number this module reports and is
restored here only so the displayed algebra is internally consistent, a
sign that had previously been dropped silently in this derivation).

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

**Scope of the two validation checks above.** Both the by-hand degree-2
pin (:data:`DEGREE2_HAND_CHECK_DC20`) and the Genova eq. (5) cross-check
set n=n_f, m=m_f -- the diagonal case -- by construction: the general
relation (*) below collapses to the classical same-degree formula exactly
when response and forcing degree/order coincide, so passing either check
is a **necessary but not sufficient** validation of (*)'s off-diagonal
generalization. Genova's eq. (5) additionally uses the *unnormalized*
``P2(cos psi)``, so matching it validates only the ``GM``/radial-distance
scaling of the relation -- not the ``1/(2n+1)`` factor, the ``Pbar``
normalization convention, or (most importantly for this module) anything
about a response mode with ``m != m_f``. Neither check exercises the
basis-normalization correction introduced below, which affects only
``m != m_f`` response modes; that correction is validated separately (see
"Basis normalization" below), not by either of these two checks.

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
generalized relation, *before* the basis-normalization correction below::

    Delta C_nm = -k_(n,m) * (GM_ext/GM_body) * (R/d)^(n_f+1) * (1/(2n_f+1))
                 * Pbar_(n_f,m_f)(sin phi') cos(m_f lam')          (*)

-- the SAME (GM_ext/GM_body)(R/d)^(n_f+1)/(2n_f+1) prefactor for *every*
response mode n (:func:`solar_tide_amplitude_parameter`), because it is
set by the forcing degree, not the response degree. Setting n=n_f, m=m_f
collapses (*) to the diagonal formula above exactly -- the hand-checkable
degree-2 case pinned by ``test_mars_detectability.py`` and
:data:`DEGREE2_HAND_CHECK_DC20` -- but see "Scope of the two validation
checks above": that collapse is *why* neither check can validate the
basis-normalization correction below, since the correction is identically
1 whenever m=m_f.

**Basis normalization (the correction (*) above is missing).** ``k_(n,m)
= Phi_surf(n,m) / F(n_f,m_f)`` (module docstring above) is a ratio of two
coefficients of the solver's own complex spherical-harmonic basis, and
that basis is **not uniformly normalized across m**. The synthesis
routine that produces both the forcing field and the solution potential
(LOV3D's ``src/get_map.m``, lines ~196-201, the ``SPH(i).Y`` construction
inside the ``m==0`` / ``m>0`` / ``m<0`` branches of its ``n,m`` loop) and
its Python port :func:`pylov3d.mars_lateral.complex_sh_synthesis` (derived
independently there from the solver's own real-to-complex conversion,
:func:`pylov3d.mars_lateral._real_sh_to_complex_mu_variable`) both state
the same basis::

    Y_n^0    =        Pbar_n^0(sin phi)                      (norm 1)
    Y_n^{+m} =        Pbar_n^m(sin phi) * exp(+i*m*lam) / sqrt(2)  (m>0)
    Y_n^{-m} = (-1)^m Pbar_n^m(sin phi) * exp(-i*m*lam) / sqrt(2)

i.e. every m=0 mode carries norm 1 relative to the real, 4pi-normalized
``Pbar_nm`` cos/sin basis used throughout this derivation, while every
m!=0 mode carries norm 1/sqrt(2). ``k_(n,m) = Phi_surf(n,m)/F(n_f,m_f)``
is therefore a ratio of coefficients in *differently normed* basis
elements whenever the response order m and the forcing order m_f differ
in whether they are zero -- which happens for every off-diagonal mode of
the shipped (2,0)-forced spectrum, since m_f=0 for all of them. The
correction is the *ratio* of the two basis norms, not a hardcoded
sqrt(2) (:func:`sh_basis_norm`, ``c_{n_f,m_f}/c_{n,m}`` below), so the
same relation stays correct for any other forcing order a future caller
uses. Eq. (*) above, corrected, is what :func:`required_stokes_amplitude`
actually implements::

    Delta C_nm = -k_(n,m) * (c_{n_f,m_f}/c_{n,m})
                 * (GM_ext/GM_body) * (R/d)^(n_f+1) * (1/(2n_f+1))
                 * Pbar_(n_f,m_f)(sin phi') cos(m_f lam')         (*')

with ``c_{n,0}=1``, ``c_{n,m}=1/sqrt(2)`` for m!=0 (:func:`sh_basis_norm`,
independent of n). For the shipped (2,0)-forced spectrum, ``c_{n_f,m_f} =
c_{2,0} = 1``, so the correction is exactly ``sqrt(2)`` for every m!=0
response mode and exactly 1 for m=0 -- the (3,0) headline mode is
unaffected; every (2,+/-2), (3,+/-1), (3,+/-3), (4,+/-2), (2,+/-1) mode's
required precision (and therefore its detectability ratio) is sqrt(2)
times tighter than an uncorrected calculation would report. In (*'),
``(n_f, m_f)`` is the order the coupled solve was *actually forced at*
(always (2,0) for the shipped spectrum) -- **not** necessarily the same
``m_forcing`` an amplitude-bound calculation elsewhere in this module
chooses to assume for ``xi``/the Legendre-peak factor
(:func:`mars_required_precision`'s "optimistic" bound uses a hypothetical
``m_forcing=2`` geometry on top of the actual (2,0)-forced ``k_(n,m)``
values -- see that function's docstring); :func:`required_stokes_amplitude`
keeps the two separate (``m_forcing`` vs ``m_forcing_solve``) for exactly
this reason.

Verified two independent ways before shipping (both reproducible from
this module's own dependencies, not asserted): (a) synthesizing
``amp(n,+m)=k, amp(n,-m)=(-1)^m*conj(k)`` with
:func:`pylov3d.mars_lateral.complex_sh_synthesis` and projecting the
result onto the real, 4pi-normalized ``Pbar_nm`` basis by direct
numerical quadrature over a lat/lon grid gives
``sqrt(C_nm^2+S_nm^2)/|k| = sqrt(2) = 1.41421356`` for (2,2), (3,1),
(3,3), (4,2) and exactly 1.0 for (3,0) (to 1e-5 relative, quadrature
grid 361x720); (b) reciprocity of the shipped N=115 spectrum against an
independent (2,2)-forced coupled solve at the same resolution
(``lmax=4, Nrbase=30``): the *raw*, uncorrected code ratio
``k[(2,+2)<-(2,0)]=3.807096e-5`` and ``k[(2,0)<-(2,2)]=3.806694e-5``
agree to 1e-4 relative, so if the physically meaningful, real-basis
admittance is self-consistent between the two directions (as a linear
response coupling ought to be), it is the ``sqrt(2)``-corrected value,
``~5.384e-5``, not the raw ``~3.807e-5``, that is that self-consistent
number. Both checks are numerical facts reproduced in this project's own
review process, not claims taken on faith.

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
  Primary Mission, solution "GL0660B"). :data:`GRAIL_SIGMA_K2`,
  :data:`GRAIL_K3`, :data:`GRAIL_SIGMA_K3` are these directly-retrieved
  numbers. **Williams, J. G., Konopliv, A. S., Boggs, D. H., et al.
  (2014), "Lunar interior properties from the GRAIL mission," JGR
  Planets, 119, 1546-1578, WAS retrieved and its Table 4 read directly
  this session (an open-mirror PDF, not the paywalled Wiley ``agupubs``
  page that returns HTTP 402 on every attempt) -- an earlier version of
  this docstring claimed it could not be retrieved and substituted an
  unverified ``k3~0.0163+/-0.0007`` figure sourced only from web-search
  summaries; that figure does not appear anywhere in the retrieved paper
  and has been deleted.** Williams et al. (2014) Table 4 instead
  gives k3=0.0089+/-0.0021 for the GL0660B (JPL) solution -- the *same*
  number as Konopliv et al. (2013) above, which Williams et al. (2014)'s
  own text notes is "from a separate related solution" rather than the
  published GL0660B fit -- and k30=0.00734+/-0.00375 for the independent
  GRGM660PRIM (GSFC) solution (Lemoine et al., 2013); no degree-3 GRAIL
  number sharper than these two exists in that table. This module still
  uses only the directly-retrieved Konopliv et al. (2013) k3 above; see
  :mod:`pylov3d.mars_detectability_k2m` for where Williams et al. (2014)'s
  degree-2 k2m Table 4 entries (both solutions) are used directly.
- **The off-(2,0) spectrum's |k_(n,m)| is an off-diagonal quantity (the
  response at (n,m) when Mars is forced at (2,0)); GRAIL's sigma_k3 is a
  diagonal admittance uncertainty (driven by the Moon's OWN degree-3
  tide).** An earlier version of this module compared the two directly
  in Love-number space, asserting no unit conversion was needed -- the
  same diagonal/off-diagonal conflation this module's own derivation
  exists to prevent, just committed in the opposite direction (compared
  a Mars off-diagonal response to a Moon diagonal uncertainty as if they
  were the same kind of quantity). That comparison, and the
  ``ratio_grail`` table column it produced, have been removed rather
  than patched with an ad hoc Stokes-space conversion factor (the
  driving potentials differ by the ratio of the Moon's degree-3 lunar
  tide amplitude to Mars's degree-2 solar tide amplitude, a conversion
  this module does not attempt); :func:`mars_off20_detectability_table`
  now reports only the Mars-orbiter (Stokes-coefficient-space, via eq.
  (*') above) comparison, at both the "optimistic" and "conservative"
  bounds (module docstring, sec. 2).

4. Frequency separation (point 4 of the task spec)
---------------------------------------------------------------------------
The tidal signal is periodic at the solar semidiurnal period,
:data:`pylov3d.mars.MARS_FORCING_TD` = 44,387.62 s (verified TASK-025a).
The CO2 seasonal signal (section 3 above) is periodic at the Mars orbital
period, T=686.98 days (Genova et al. 2016, eq. 3, retrieved this
session) = :data:`MARS_ORBITAL_PERIOD_S`. :func:`frequency_separation_factor`
returns their ratio, ~1337x (1337.20, module docstring value corrected
from an earlier ~1338x rounding). This quantifies *only* that the two
signals sit in well-separated Fourier bins over a multi-year tracking
baseline (no aliasing concern either: spacecraft orbital periods, ~2 h,
are far shorter than the 44,387.62 s tidal period, so it is not
undersampled) -- i.e. the achieved seasonal-band precision is not
degraded by confusion with the tidal signal, or vice versa. **It does
not, and cannot, be read as a statement about achieved precision AT the
semidiurnal frequency itself** for degree ell>=3: no published degree>=3
Mars gravity recovery at 44,387.62 s period was found in this session.
That gap is stated explicitly, not filled by assumption.

**This separation argument holds only for the m=2 (sectoral, semidiurnal)
component of the tide.** The (2,0) (zonal) component of the *real* solar
tide is not semidiurnal at all -- it varies on the annual/semi-annual
timescale set by the sub-solar latitude's yearly excursion through Mars's
obliquity (module docstring, sec. 2) -- which is exactly the band Genova
et al. (2016) fit and attribute to CO2 mass exchange. So under the
self-consistent reading that matches the shipped spectrum's own forcing
order (m_f=0, the "conservative" bound), the tidal signal this module's
required-precision numbers describe is *degenerate* with the seasonal
benchmark's own signal in frequency space, not separated from it by
~1337x or by any factor; that separation factor is a property of the
m=2 physical tide only, not of the (2,0)-forced spectrum this module's
tier-2 table actually uses.

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
# re-verified beyond the aggregator cross-check. MARS_OBLIQUITY_DEG DOES
# enter a computation: :func:`peak_legendre_factor` searches for the peak
# |Pbar_nm| only over sub-solar latitudes actually reachable
# (|phi'| <= MARS_OBLIQUITY_DEG), not over the full [-90, 90] deg range
# (an earlier version of this module claimed obliquity "enters no
# computation" -- wrong; see that function's docstring).
MARS_SEMIMAJOR_AXIS_M = 227_939_366_000.0  # m (1.52368055 AU)
MARS_ECCENTRICITY = 0.0934
MARS_OBLIQUITY_DEG = 25.19

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

# Williams, Konopliv, Boggs, et al. (2014), JGR Planets, 119, 1546-1578,
# Table 4 (retrieved directly this session; module docstring sec. 3) --
# recorded for provenance/context only, not used in mars_off20_detectability_table
# (module docstring sec. 3's diagonal/off-diagonal category-error fix).
WILLIAMS2014_K3_GL0660B = 0.0089        # same figure as GRAIL_K3 above
WILLIAMS2014_SIGMA_K3_GL0660B = 0.0021  # "from a separate related solution"
WILLIAMS2014_K30_GRGM660PRIM = 0.00734
WILLIAMS2014_SIGMA_K30_GRGM660PRIM = 0.00375

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
    """``max |Pbar_(n_forcing,m_forcing)(sin phi')|`` over sub-forcing-body
    latitudes ``phi'`` actually reachable, ``|phi'| <= MARS_OBLIQUITY_DEG``
    -- **not** a max over the full ``phi in [-90, 90]`` range (an earlier
    version of this docstring claimed the latter and that obliquity
    "enters no computation"; both were wrong: the true global maximum of
    ``|Pbar_20|`` is 2.236 at the poles, unreachable by any sub-solar
    point on a body with obliquity < 90 deg, and MARS_OBLIQUITY_DEG now
    bounds the search domain below).

    For ``(n_forcing, m_forcing) = (2, 0)`` and ``(2, 2)``, the only pairs
    this module's ``bound="conservative"``/``"optimistic"`` choices use,
    the obliquity-constrained peak still falls at the equinox
    (``sin(phi')=0``) -- verified numerically (fine grid over
    ``[-obliquity, +obliquity]``), not assumed, because Mars's obliquity
    (25.19 deg) sits inside both functions' first monotonic lobe (out to
    35.26 deg for ``m=0``, 90 deg for ``m=2``). ``(2, 1)`` does **not**
    share that property: ``Pbar_21(sin(0))=0`` (an earlier version of this
    function returned that silently, with no guard), and its
    obliquity-constrained peak instead falls at the solstice,
    ``phi'=+/-MARS_OBLIQUITY_DEG``. Uses
    :func:`pylov3d.mapping.fully_normalized_legendre` (the same
    4pi-normalized, no-Condon-Shortley convention as everything else in
    this module) rather than a fresh, unvalidated formula.
    """
    phi_deg = np.linspace(-MARS_OBLIQUITY_DEG, MARS_OBLIQUITY_DEG, 2001)
    P = fully_normalized_legendre(n_forcing, np.sin(np.radians(phi_deg)))
    return float(np.max(np.abs(P[n_forcing, m_forcing, :])))


def sh_basis_norm(m: int) -> float:
    """Norm of the solver's complex spherical-harmonic basis element
    ``Y_n^m`` relative to the real, 4pi-fully-normalized ``Pbar_n^m``
    cos/sin basis used throughout this module's derivation: 1.0 for
    ``m=0``, ``1/sqrt(2)`` for ``m!=0`` -- independent of ``n``. Module
    docstring, sec. 1, "Basis normalization": derived from
    ``src/get_map.m`` lines ~196-201 and
    :func:`pylov3d.mars_lateral.complex_sh_synthesis`, both of which
    define ``Y_n^0 = Pbar_n^0`` (no ``1/sqrt(2)``) but
    ``Y_n^{+/-m} = (...) Pbar_n^m exp(+/-i m lam) / sqrt(2)`` for ``m!=0``.
    """
    return 1.0 if m == 0 else 1.0 / math.sqrt(2.0)


def required_stokes_amplitude(
    k_nm: complex | float,
    GM_ext: float, GM_body: float, R_body: float, d: float,
    n_forcing: int, m_forcing: int, m_response: int,
    m_forcing_solve: int | None = None,
) -> float:
    """Peak |Delta C_nm| (equivalently |Delta S_nm|) implied by a coupled
    Love number ``k_nm``, via eq. (*') (module docstring, sec. 1),
    evaluated at the sub-forcing-body equinox (the amplitude bound, not a
    full time series -- module docstring, sec. "characteristic
    amplitude"; this is an *upper* bound on the achievable signal, i.e.
    the most optimistic case for detectability, not an average or worst
    case).

    ``m_forcing`` sets the *amplitude-bound geometry* (``xi`` via
    :func:`solar_tide_amplitude_parameter`, and the Legendre peak via
    :func:`peak_legendre_factor`) -- i.e. which physical tide component's
    forcing amplitude to assume for the required-precision estimate.
    ``m_forcing_solve`` is the *actual* forcing order the coupled solve
    used to produce ``k_nm`` -- what the basis-normalization correction
    (module docstring, "Basis normalization") must be computed against --
    and defaults to ``m_forcing`` when not given (the ordinary case where
    the two coincide, e.g. the diagonal hand check, or the "conservative"
    bound which is defined to match the spectrum's own forcing order).
    They deliberately differ for the "optimistic" bound
    (:func:`mars_required_precision`): its ``m_forcing=2`` is a
    *hypothetical* geometry assumption ("as if the dominant real tide
    component set the amplitude"), but the ``k_nm`` values it is applied
    to were actually produced by a (2,0)-forced solve, so the basis
    correction must still use ``m_forcing_solve=0``, not 2 -- conflating
    the two would make the (3,0) mode (m_response=0) spuriously *change*
    under the "optimistic" bound, when the fix's own invariant is that
    m=0 response modes are unaffected by this correction at any bound.

    ``m_response`` is the response mode's azimuthal order (``abs(m)``;
    sign does not affect :func:`sh_basis_norm`) -- **required**, because
    the basis-normalization correction depends on both the actual solve
    forcing order and the response order, not on the bound's geometry
    assumption. For the diagonal case (``m_response == m_forcing_solve``,
    e.g. the degree-2 hand check), the correction is identically 1 and
    this reduces to the uncorrected relation exactly.
    """
    xi = solar_tide_amplitude_parameter(GM_ext, GM_body, R_body, d, n_forcing)
    p = peak_legendre_factor(n_forcing, m_forcing)
    m_fs = m_forcing if m_forcing_solve is None else m_forcing_solve
    basis_correction = sh_basis_norm(m_fs) / sh_basis_norm(m_response)
    return abs(k_nm) * xi * p * basis_correction


# Hand-checkable degree-2 pin (module docstring, sec. 1): k2=0.169 (the
# real, measured Mars value, pylov3d.mars.MARS["k2"]), forcing=(2,0),
# mean Mars-Sun distance, equinox, m_response=m_forcing=0 (the diagonal
# case, for which sh_basis_norm(0)/sh_basis_norm(0)=1 -- the
# basis-normalization correction below is identically absent here, by
# construction; see module docstring, "Scope of the two validation checks
# above"). Independently computed by hand (calculator arithmetic, not by
# calling this module's functions) as
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


def mars_required_precision(
    k_nm: complex | float, m_response: int, bound: str = "optimistic",
    m_forcing_solve: int = 0,
) -> float:
    """Peak |Delta C_nm| for a Mars off-forcing mode, forcing degree n_f=2.

    ``m_response`` is the response mode's azimuthal order (``abs(m)``),
    forwarded to :func:`required_stokes_amplitude` for the
    basis-normalization correction (module docstring, sec. 1, "Basis
    normalization") -- **required** as of the fix for the sqrt(2) error
    described there; every caller must now say which response mode it
    means, not just which bound. ``m_forcing_solve`` is the order the
    coupled solve was *actually* forced at (default 0, matching the
    shipped N=115 spectrum -- named explicitly, not silently assumed, in
    case a future caller uses a differently-forced spectrum); see
    :func:`required_stokes_amplitude` for why this is kept separate from
    the bound's own ``m_forcing`` geometry choice below.

    Two bounding conventions (module docstring, sec. 2 -- the real tide's
    dominant azimuthal order is (2,2), not the (2,0) the given spectrum
    was computed with; this brackets that ambiguity together with the
    orbital-distance choice rather than picking one number silently):

    - ``"optimistic"``: perihelion distance (largest forcing amplitude)
      and ``m_forcing=2`` (the larger of Pbar_20/Pbar_22 within Mars's
      reachable sub-solar latitude range) -- the largest, most
      detection-favorable required-precision bound (a bigger real signal
      means a coarser instrument still detects it, so the "required
      precision" bar is loosest here; correspondingly gives the
      *smallest* achieved/required ratio in
      :func:`mars_off20_detectability_table`). This is a *hybrid*: the
      geometry (``xi``, Legendre peak) assumes an (2,2) forcing, but the
      ``k_nm`` values it is applied to are (2,0)-forced
      (``m_forcing_solve=0`` still governs the basis correction).
    - ``"conservative"``: mean Sun distance and ``m_forcing=0`` (matching
      the given spectrum's own forcing order exactly, and so the
      *self-consistent*, non-hybrid bound for a (2,0)-forced spectrum) --
      about 2.3x smaller (tighter/harder bar) than the optimistic bound;
      for the top mode (3,0) this is the bound that gives the 66.2x ratio
      quoted alongside the optimistic bound's 28.5x (module docstring,
      sec. 2; docs/MARS_MODEL.md sec. 4).
    """
    if bound == "optimistic":
        d, m_forcing = MARS_PERIHELION_M, 2
    elif bound == "conservative":
        d, m_forcing = MARS_SEMIMAJOR_AXIS_M, 0
    else:
        raise ValueError(f"bound must be 'optimistic' or 'conservative', got {bound!r}")
    return required_stokes_amplitude(
        k_nm, GM_SUN, MARS["GM"], MARS["R"], d,
        n_forcing=2, m_forcing=m_forcing, m_response=m_response,
        m_forcing_solve=m_forcing_solve,
    )


def mars_off20_detectability_table(spectrum: dict | None = None) -> list[dict]:
    """Per-mode required-precision table for every non-forcing mode.

    Each row: ``n``, ``m``, ``k_abs``, ``required_dC_optimistic``,
    ``required_dC_conservative`` (:func:`mars_required_precision`, both
    bounds, basis-normalization-corrected -- module docstring sec. 1),
    ``ratio_orbiter_optimistic`` / ``ratio_orbiter_conservative``
    (:data:`MARS_SIGMA_C30_SEASONAL` / required, at each bound -- how many
    times *too coarse* current Mars-orbiter seasonal-gravity precision is;
    <1 would mean detectable). Sorted by descending |k|.

    There is **no GRAIL-based column here.** An earlier version of this
    function returned ``ratio_grail = GRAIL_SIGMA_K3 / k_abs`` directly,
    comparing this table's off-diagonal |k_(n,m)| (response at (n,m) to a
    (2,0) forcing) against GRAIL's diagonal sigma(k3) (the Moon's OWN
    degree-3 tidal admittance uncertainty) as if the two were the same
    kind of quantity -- a diagonal/off-diagonal conflation (module
    docstring, sec. 3). It has been removed rather than patched with an
    unvalidated Stokes-space conversion factor; see
    :mod:`pylov3d.mars_detectability_k2m` for the module's one
    GRAIL-benchmarked comparison, which *is* diagonal-vs-diagonal and
    does not have this problem.
    """
    spectrum = spectrum or load_mars_lateral_spectrum()
    m_forcing_solve = spectrum["forcing_m"]
    rows = []
    for mode in spectrum["modes"]:
        if mode["is_forcing"]:
            continue
        k_abs = abs(mode["k"])
        m_response = abs(mode["m"])
        req_opt = mars_required_precision(
            k_abs, m_response, bound="optimistic", m_forcing_solve=m_forcing_solve,
        )
        req_cons = mars_required_precision(
            k_abs, m_response, bound="conservative", m_forcing_solve=m_forcing_solve,
        )
        rows.append({
            "n": mode["n"], "m": mode["m"], "k_abs": k_abs,
            "required_dC_optimistic": req_opt,
            "required_dC_conservative": req_cons,
            "ratio_orbiter_optimistic": MARS_SIGMA_C30_SEASONAL / req_opt,
            "ratio_orbiter_conservative": MARS_SIGMA_C30_SEASONAL / req_cons,
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
