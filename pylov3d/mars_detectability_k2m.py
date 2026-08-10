# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

r"""Diagonal k2m order-splitting: the MaQuIs/GRAIL benchmark (TASK-026).

Companion to :mod:`pylov3d.mars_detectability` (split out to keep both
files under this repo's 500-line-per-file convention -- the same reason
``pylov3d/anelastic.py``/``anelastic_moon.py`` are split). That module's
off-(2,0) spectrum answers "how big is the coupled response at (n,m) when
Mars is forced at (2,0)" -- an *off-diagonal* entry of the generalized
Love-number tensor. This module answers a *different*, diagonal question:
does the ordinary degree-2 tidal Love number itself, k_2m, differ across
azimuthal order m=0,1,2 (forcing (2,m), response the *same* (2,m))? A
laterally heterogeneous body need not respond identically at every order,
even though a spherically symmetric one would (k2 alone, order-independent
by construction). **These are not the same observable and must not be
conflated** -- see :mod:`pylov3d.mars_detectability` module docstring,
sec. 5, for the explicit distinction; this module docstring restates it
briefly, then gives the numbers.

Why this benchmark, and the source (retrieved directly this session)
---------------------------------------------------------------------------
Wörner, L., Root, B. C., Bouyer, P., Braxmaier, C., Dirkx, D., Encarnação,
J., Hauber, E., Hussmann, H., Karatekin, Ö., Koch, A., Kumanchik, L.,
Migliaccio, F., Reguzzoni, M., Ritter, B., Schilling, M., Schubert, C.,
Thieulot, C., v. Klitzing, W., & Witasse, O. (2023), "MaQuIs -- Concept
for a Mars Quantum Gravity Mission," Planetary and Space Science, 239,
105800, doi:10.1016/j.pss.2023.105800 (open access; full text fetched
directly this session from the TU Delft repository,
``repository.tudelft.nl``, record uuid 5261722a-a969-4aff-a728-
a8317c76ccbf, file ``1_s2.0_S0032063323001691_main.pdf``; page numbers
below refer to the fetched PDF's own printed page numbers). Quoted
verbatim from the retrieved text (p. 5, second column, the paragraph
following the k2 discussion): "For the Moon, separate values of k20, k21
and k22 have been determined using GRAIL data, providing further interior
constraints (Williams et al., 2014). Although these values of k2m at
different orders m proved to be almost equal to one another, their small
differences may be relevant in processing of high-accuracy data proposed
here. Past attempts to measure these separate coefficients using Doppler
tracking proved unsuccessful for Mars."

This is the closest published statement to a direct measurement
*requirement* for exactly the class of observable this project's lateral
model predicts: k2m order-splitting is driven by the same lateral crustal
rigidity heterogeneity (:mod:`pylov3d.mars_lateral`) that produces the
whole off-(2,0) spectrum, evaluated at the diagonal (2,m)-(2,m) entries
instead of the off-diagonal (2,0)-(n,m) ones.

Predicted splitting (Python-pipeline, validated config, not itself
MATLAB-checked)
---------------------------------------------------------------------------
:data:`MARS_K21_FORCING` and :data:`MARS_K22_FORCING` were computed this
session by calling :func:`pylov3d.mars_lateral.mars_lateral_love_spectrum`
with ``forcing=(2, 1)`` and ``forcing=(2, 2)`` respectively, at
``lmax=4, Nrbase=30, perturbation_order=2, method="combination"`` -- the
*identical* numerical configuration MATLAB-cross-validated for
``forcing=(2, 0)`` (TASK-014 part 2, ``data/tests/mars/
mars_lateral_cross_check.mat``). Only the (2,0) run has that independent
MATLAB anchor; (2,1) and (2,2) inherit confidence from running the same
already-validated code path and numerical grid, not from a fresh
cross-check of their own. Both diagonal Love numbers came out real to
~1e-15 (``forcing=(2,1)``: 1.80e-15i; ``forcing=(2,2)``: 1.73e-15i),
consistent with this being a purely elastic model (as expected -- the
same reality-check :mod:`pylov3d.mars` already documents for k2 itself).
:func:`recompute_k2m_diagonal_shift` is the reproducibility path
(~130-140 s per call, matches the (2,0) run's own measured wall time).

The predicted splitting, relative to the uniform-body baseline
``MARS["k2"]`` = 0.169 (all three would equal 0.169 exactly for a
spherically symmetric Mars):

============  ===================  ========================
m             k_2m (predicted)     Delta = k_2m - 0.169
============  ===================  ========================
0             0.16905517           +5.517e-05  (TASK-016, MATLAB-validated)
1             0.16902091           +2.091e-05  (this session)
2             0.16903400           +3.400e-05  (this session)
============  ===================  ========================

For context (not used in any comparison below, since it is a different
observable): the (2,0)-forced spectrum's own off-diagonal (2,+/-2) mode
has |k|=3.81e-5 (``docs/tasks/TASK-026-off-forcing-detectability.md``) --
close in *order of magnitude* to the diagonal Delta_k22=3.40e-5 above,
consistent with both being driven by the same underlying crustal
heterogeneity amplitude, but they are not the same quantity and this
proximity is not a validation of either against the other.

Achieved precision: GRAIL, lunar k20/k21/k22 (retrieved directly this
session, same PDF as :mod:`pylov3d.mars_detectability`'s section 3)
---------------------------------------------------------------------------
Konopliv, A. S., Park, R. S., Yuan, D.-N., et al. (2013), "The JPL lunar
gravity field to spherical harmonic degree 660 from the GRAIL Primary
Mission," J. Geophys. Res. Planets, 118, 1415-1434, Table 4 (GRAIL Primary
Mission, individually constrained solutions): k20 = 0.02408 +/- 0.00045,
k21 = 0.02414 +/- 0.00025, k22 = 0.02394 +/- 0.00028
(:data:`GRAIL_K2M_SIGMA`). This is the likely numerical basis (same GRAIL
Primary Mission dataset, shared co-author J. G. Williams) for the
"separate values... determined" statement Wörner et al. (2023) attribute
to Williams et al. (2014); Williams et al. (2014) itself was not
independently retrieved in this session (Wiley agupubs, HTTP 402 both
times tried), so this module does not assert the two papers report
identical numbers -- only that Konopliv et al. (2013) is a
directly-verified, same-mission source for individual-order k2m
precision, and the more conservative of the two to use here (unlike
section 3's k3 comparison, no more-refined later number was found or
substituted).

**No achieved Mars number exists to compare against at all** -- per the
verbatim quote above, "past attempts to measure these separate
coefficients using Doppler tracking proved unsuccessful for Mars." The
comparison below therefore uses GRAIL's lunar precision as the best
available "a dedicated, dual-spacecraft mission has demonstrated this
kind of measurement to X precision" floor, exactly as the task's default
scoping asked for -- with the caveat, stated plainly, that GRAIL's
precision was achieved at the Moon (a smaller body, closer to Earth,
under an inter-satellite-ranging mission architecture) and is not
automatically achievable at Mars; MaQuIs is the concept aiming to test
that.

MaQuIs's own stated numbers (retrieved directly this session, same PDF)
---------------------------------------------------------------------------
Current Mars gravity field resolution: "the global resolution of the
Martian gravity fields is known up to wavelengths of approximately
115 km (90 d/o)" (p. 6) -- consistent with "up to the order of 90-100
degree and order" restated in Section 2.5's bullet list
(:data:`MAQUIS_CURRENT_RESOLUTION_DO`). MaQuIs's own target: "the
resolution of the gravitational field shall be improved above the
spherical harmonic (SH) degree 90 (approximately 115 km) for better
flexural modelling of the Martian Lithosphere... the mission should aim
for a global gravity field resolution up to 360 d/o" (Section 2.5,
:data:`MAQUIS_TARGET_RESOLUTION_DO`). Temporal signal context (Section
3.1): "The temporal gravity field variations seem to be dominated by CO2
exchange between the northern and southern hemisphere and the tidal
potential from the Martian moons, Phobos and Deimos[.] ... the gravity
signal at an orbit height of 150 km to 200 km is in the order of
230 microGal" (:data:`MAQUIS_SEASONAL_SIGNAL_UGAL` -- note this is the
CO2-plus-Phobos/Deimos-tide signal, *not* the solar semidiurnal tide this
project's k2/off-(2,0) spectrum concerns; a different forcing body,
mentioned here only for scale). Section 2.3's "Temporal secular" bullet:
"We propose that MaQuIs should be designed to be able to observe 0.01
microGal per year global changes of the gravity field to have significant
science return" (:data:`MAQUIS_SECULAR_GOAL_UGAL_PER_YR`). No microGal-
space conversion of the k2m splitting is attempted here (that requires a
further, non-trivial degree-2-Stokes-coefficient-to-orbit-altitude-
gravity-anomaly step, a different geodetic calculation than this module's
Love-number-space relation); the splitting-vs-GRAIL comparison below
stays entirely in Love-number space, where the two are already directly
comparable with no unit conversion at all. These MaQuIs numbers are
recorded for scientific context (per the PI redirect) rather than
combined algebraically with the k2m comparison.

The alignment already in this project's Mars anelasticity work (brief,
not over-claimed)
---------------------------------------------------------------------------
Wörner et al. (2023) flag k2/Q frequency-dependence as an interior
constraint, citing Pou et al. (2022) and Dirkx et al. (2014). TASK-025a
(``docs/MARS_MODEL.md``, "Anelasticity") already establishes the same
point independently: Mars's only two published anelastic constraints (k2
at the ~44,388 s solar-semidiurnal period; Q at the ~19,991 s Phobos-tide
period) sit at different forcing frequencies and are not directly
comparable without a frequency-dependence model -- exactly the gap
Wörner et al. (2023) note "could provide further constraints... provided
their signature in the data could be decoupled." No new claim is made
here; this is only a pointer between the two already-independent lines
of work.
"""

from __future__ import annotations

from .mars import MARS
from .mars_detectability import load_mars_lateral_spectrum

# Konopliv, Park, Yuan, et al. (2013), JGR Planets, 118, 1415-1434, Table 4
# (GRAIL Primary Mission, k20/k21/k22 constrained independently).
GRAIL_K2M = {0: 0.02408, 1: 0.02414, 2: 0.02394}
GRAIL_K2M_SIGMA = {0: 0.00045, 1: 0.00025, 2: 0.00028}

# Wörner et al. (2023), Planetary and Space Science, 239, 105800.
MAQUIS_CURRENT_RESOLUTION_DO = 90       # p. 6 / Sec. 2.5 bullet list
MAQUIS_TARGET_RESOLUTION_DO = 360       # Sec. 2.5
MAQUIS_SEASONAL_SIGNAL_UGAL = 230.0     # Sec. 3.1, 150-200 km altitude
MAQUIS_SECULAR_GOAL_UGAL_PER_YR = 0.01  # Sec. 2.3, "Temporal secular"

# Diagonal forcing-mode Love numbers at (2,1) and (2,2), computed this
# session at the MATLAB-cross-validated (2,0) run's own numerical config
# (lmax=4, Nrbase=30, perturbation_order=2, method="combination") -- see
# module docstring for wall-time and imaginary-part sanity checks.
MARS_K21_FORCING = 0.16902091346458947
MARS_K22_FORCING = 0.16903399541441133


def recompute_k2m_diagonal_shift(m_forcing: int, lmax: int = 4, Nrbase: int = 30) -> complex:
    """Reproducibility path for :data:`MARS_K21_FORCING`/:data:`MARS_K22_FORCING`
    (``m_forcing in (1, 2)``): reruns the coupled solve and returns the
    forced mode's own (complex) diagonal Love number. Costs ~130-140 s
    (module docstring)."""
    from .mars_lateral import mars_lateral_love_spectrum

    result = mars_lateral_love_spectrum(lmax=lmax, forcing=(2, m_forcing), Nrbase=Nrbase)
    love = result["love"]
    idx = next(
        i for i in range(len(love.n))
        if int(love.n[i]) == 2 and int(love.m[i]) == m_forcing
    )
    return complex(love.k[idx])


def mars_diagonal_k2m_table(spectrum: dict | None = None) -> list[dict]:
    """The predicted k20/k21/k22 order-splitting, and its comparison
    against GRAIL's own individual-order lunar precision (module
    docstring) -- the direct analogue of what Wörner et al. (2023)
    describe MaQuIs as targeting, and what "proved unsuccessful for Mars"
    with current Doppler tracking. Each row: ``m``, ``k2m`` (predicted,
    real part), ``delta`` (``k2m - MARS["k2"]``, the uniform-body
    baseline), ``grail_k2m``/``grail_sigma`` (Konopliv et al. 2013, same
    order), ``ratio_grail`` (``grail_sigma / |delta|`` -- how many times
    too coarse GRAIL's own demonstrated, lunar, individual-order
    precision is for this Mars-predicted splitting; >1 means not
    detectable even at GRAIL's own demonstrated precision).
    """
    spectrum = spectrum or load_mars_lateral_spectrum()
    k2m = {
        0: MARS["k2"] + spectrum["k2_shift"],
        1: MARS_K21_FORCING,
        2: MARS_K22_FORCING,
    }
    rows = []
    for m in (0, 1, 2):
        delta = k2m[m] - MARS["k2"]
        sigma = GRAIL_K2M_SIGMA[m]
        rows.append({
            "m": m, "k2m": k2m[m], "delta": delta,
            "grail_k2m": GRAIL_K2M[m], "grail_sigma": sigma,
            "ratio_grail": sigma / abs(delta),
        })
    return rows
