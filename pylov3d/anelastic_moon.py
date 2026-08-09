# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Anelastic (viscoelastic) tidal response for the Moon (TASK-025a).

Moon-specific companion to :mod:`pylov3d.anelastic` (split out to keep
both files under the project's 500-line-per-file convention; the shared
solver-capability audit, the three PyALMA3 findings this task surfaced,
and the Mars-side calculations live there — read that module's docstring
first). This module imports its shared machinery
(:func:`pylov3d.anelastic.implied_Q`, ``_with_eta``,
``_bisect_log_decreasing``, ``_import_alma``, ``ALMA_T0_S``) rather than
duplicating it.

Why the Moon needs its own module, structurally: the Weber Moon profile
has an internal ocean (the fluid outer core, layer index 2, sandwiched
between the solid inner core and the solid mantle) that PyALMA3 cannot
represent (its propagator supports a fluid layer only at the center —
see :mod:`pylov3d.anelastic` module docstring, "third finding"). Mars's
core, by contrast, is a simple fluid layer at the center, so its
PyALMA3 comparisons need no workaround. This module therefore carries
:func:`moon_simplified_body` — a fluid-core + viscoelastic-mantle +
elastic-crust analogue PyALMA3 *can* represent — used for both the
Maxwell validation and the (necessarily external-tool) Andrade estimate,
alongside :func:`moon_maxwell_k2` / :func:`fit_moon_maxwell_gap`, which
run on the **real** 10-layer structure directly (pylov3d's own
ocean-aware solver handles the internal ocean natively; no PyALMA3
needed for that headline number).

Stage a / b boundary: same as :mod:`pylov3d.anelastic` — every public
function here takes a plain scalar free parameter and returns a complex
Love number; nothing here reads or writes MCMC/pocoMC state (that is
TASK-025b, not built here).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import moon as _moon_mod
from .anelastic import ALMA_T0_S, _bisect_log_decreasing, _import_alma, _with_eta, implied_Q
from .love import get_love
from .types import make_forcing, make_interior_model, make_numerics

# ---------------------------------------------------------------------------
# Forcing period and mantle-layer conventions
# ---------------------------------------------------------------------------

#: Draconic (nodical) month, 27.212 days — the period of the Moon's
#: argument of latitude F (node-crossing period), NOT the anomalistic
#: month (27.55455 d, perigee-to-perigee). Williams & Boggs (2015), JGR
#: Planets 120, 689-724, doi:10.1002/2014JE004755, report the lunar
#: monthly tidal dissipation quality factor Q = 38 +/- 4 at their
#: reference period Pref = 27.212 days, described there (per web search
#: of the primary source, Section 4/Table) as the period of the largest
#: latitude libration, due to the tilt of the equator plane to the orbit
#: plane -- i.e. the argument-of-latitude period, not the anomalistic
#: month. This is corroborated by Williams, Konopliv, Boggs, et al.
#: (2014), JGR Planets 119, 1546-1578 (the GRAIL interior-properties
#: paper cited throughout this document for k2 and the fluid-core
#: radius), which (per web search of the primary source) writes its
#: monthly (k2/Q)_F relation over a 27.212-day period and defines, in
#: its own notation table, "P_F" as the 27.212-day period of argument of
#: latitude. The 27.55455 d anomalistic period does appear in this
#: literature, but attached to the DeltaJ2/DeltaC22/DeltaS22 tide
#: amplitude rather than to Q = 38 +/- 4 -- verified verbatim in
#: Williams et al. (2014), not in Williams & Boggs (2015) as an earlier
#: draft of this comment stated. Corroborated directly by Briaud et al.
#: (2023), "Constraints on the lunar core viscosity from tidal
#: deformation," *Icarus* (arXiv:2301.04035), which states that per
#: Williams & Boggs (2015) the major periods of interest are F = 27.212
#: days and l_0 = 365.260 days; and by Walterova, Behounkova &
#: Efroimsky (2023). See
#: docs/MOON_MODEL.md, "Anelasticity (TASK-025a)", "Forcing-period
#: provenance" for the full citation trail and why this differs from
#: ``pylov3d.moon.MOON_FORCING_TD``, which uses the sidereal month and
#: is frequency-irrelevant there only because that model is purely
#: elastic. (The 27.55455 d anomalistic month itself, where it is still
#: legitimately mentioned in this repo's docs as context, is an
#: Astronomical Almanac / NASA value, not a Williams & Boggs 2015 one.)
MOON_DRACONIC_MONTH_DAYS: float = 27.212
MOON_DRACONIC_MONTH_TD: float = MOON_DRACONIC_MONTH_DAYS * 86400.0  # s

#: Mantle layer indices in ``pylov3d.moon.build_moon_model()`` (partial
#: melt zone + 5 mantle shells) — the same 6 layers
#: ``pylov3d.moon_mc``'s ``mu_scale`` parameter scales. Layers 0/1/2
#: (artificial core / inner core / fluid outer core) and layer 9 (crust)
#: stay elastic.
MOON_MANTLE_LAYERS: tuple[int, ...] = (3, 4, 5, 6, 7, 8)


# ---------------------------------------------------------------------------
# Real 10-layer Weber structure — native pylov3d Maxwell (no PyALMA3)
# ---------------------------------------------------------------------------

def moon_maxwell_k2(
    eta_mantle: float,
    *,
    Td: float | None = None,
    Nrbase: int | None = None,
    mantle_layers: Sequence[int] = MOON_MANTLE_LAYERS,
) -> complex:
    """Complex degree-2 k2 of the as-built 10-layer Weber Moon model with
    a Maxwell-viscoelastic mantle, computed by pylov3d's own solver on
    the **real** structure (including the internal fluid outer core —
    pylov3d's ocean-aware boundary conditions handle it natively; no
    simplification needed here, unlike the PyALMA3 path below).

    All layers in ``mantle_layers`` get the same ``eta_mantle`` [Pa s].
    Default forcing period is :data:`MOON_DRACONIC_MONTH_TD`.
    """
    if Td is None:
        Td = MOON_DRACONIC_MONTH_TD
    if Nrbase is None:
        Nrbase = _moon_mod.MOON_NUMERICS_NRBASE

    model = _moon_mod.build_moon_model()
    model = _with_eta(model, {i: eta_mantle for i in mantle_layers})

    forcing = make_forcing(Td=Td, n=2, m=0, F=1.0)
    numerics = make_numerics(
        n_layers=model.n_layers, method=_moon_mod.MOON_NUMERICS_METHOD, Nrbase=Nrbase,
    )
    love, _, _ = get_love(model, forcing, numerics)
    return complex(love.k[0])


def fit_moon_maxwell_gap(
    target_k2: float | None = None,
    *,
    tol: float = 1e-6,
    eta_lo: float = 1e10,
    eta_hi: float = 1e20,
    Nrbase: int | None = None,
    max_iter: int = 60,
) -> tuple[float, complex, float]:
    """Bisect the uniform mantle Maxwell viscosity [Pa s] that raises the
    as-built (elastic) Moon k2 up to ``target_k2`` (default:
    ``pylov3d.moon.MOON["k2"]`` = 0.02422, the GRAIL-observed value; the
    as-built elastic baseline is
    ``pylov3d.moon.WEBER_K2_UNIFORM`` = 0.023159..., a 4.6% gap — see
    docs/MOON_MODEL.md "As-built residuals").

    Returns ``(eta_mantle, k2_complex, Q_implied)`` where
    ``Q_implied = implied_Q(k2_complex)``. This is the headline
    Maxwell-side Moon calculation (TASK-025a) — see docs/MOON_MODEL.md,
    "Anelasticity (TASK-025a)" for the number and its Q-consistency
    verdict against Williams & Boggs (2015) Q = 38 +/- 4.
    """
    if target_k2 is None:
        target_k2 = _moon_mod.MOON["k2"]

    def f(eta: float) -> float:
        return moon_maxwell_k2(eta, Nrbase=Nrbase).real

    eta_fit, _ = _bisect_log_decreasing(f, eta_lo, eta_hi, target_k2, tol=tol, max_iter=max_iter)
    k2 = moon_maxwell_k2(eta_fit, Nrbase=Nrbase)
    return eta_fit, k2, implied_Q(k2)


# ---------------------------------------------------------------------------
# Simplified body — the only structure PyALMA3 can represent
# ---------------------------------------------------------------------------

def moon_simplified_body() -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    """A fluid-core + viscoelastic-mantle + elastic-crust analogue of the
    Weber Moon, structurally comparable to what PyALMA3 can represent
    (see ``pylov3d.anelastic`` module docstring, "third finding": PyALMA3
    supports a fluid layer only at the center, not the Weber profile's
    internal ocean).

    Construction: the Weber profile's three innermost layers (artificial
    core 0-50 km, solid inner core 50-240 km, fluid outer core
    240-330 km) are merged into a single fluid layer spanning 0-330 km,
    density = the merged layer's mean density, i.e. combined mass over
    combined volume of the three (``pylov3d.moon.LAYER_RHO[:3]`` weighted
    by their respective shell volumes, 6215.55 kg/m^3 -- an earlier draft
    of this docstring called this a "mass-weighted average," which is not
    the correct term for total-mass-over-total-volume; corrected here).
    Layers 3-9 (partial melt zone through crust) are kept exactly as in
    ``pylov3d.moon.build_moon_model()``.

    This body is used for (1) the pylov3d-vs-PyALMA3 Maxwell validation
    at Moon-relevant radii/densities/rigidities/period (cross-checked
    against the real-structure :func:`moon_maxwell_k2` result: both give
    the same gap-closing Maxwell viscosity and the same catastrophically
    low implied Q to 3 significant figures — see docs/MOON_MODEL.md,
    "Anelasticity (TASK-025a)"), and (2) the PyALMA3 Andrade estimate
    (:func:`moon_alma_k2` with ``rheology="andrade"``), since pylov3d has
    no Andrade path and PyALMA3 cannot represent the real internal ocean.

    Returns
    -------
    radii_km, rho, mu, Ks, mantle_layers
        Per-layer arrays (core to surface, 8 layers) and the list of
        mantle-layer indices (1-6) within this simplified body.
    """
    names = _moon_mod.LAYER_NAMES
    radii = _moon_mod.LAYER_RADII_KM
    rho = _moon_mod.LAYER_RHO
    mu = _moon_mod.LAYER_MU
    ks = _moon_mod.LAYER_KS
    assert names[0] == "artificial_core" and names[2] == "outer_core"

    volumes = [radii[0] ** 3, radii[1] ** 3 - radii[0] ** 3, radii[2] ** 3 - radii[1] ** 3]
    masses = [rho[i] * volumes[i] for i in range(3)]
    core_rho = sum(masses) / sum(volumes)

    simp_radii = [radii[2]] + list(radii[3:])
    simp_rho = [core_rho] + list(rho[3:])
    simp_mu = [0.0] + list(mu[3:])
    simp_ks = [0.0] + list(ks[3:])
    n_simp = len(simp_radii)
    mantle_layers = list(range(1, n_simp - 1))  # exclude core (0) and crust (last)
    return simp_radii, simp_rho, simp_mu, simp_ks, mantle_layers


def moon_simplified_maxwell_k2(
    eta_mantle: float,
    *,
    Td: float | None = None,
    Nrbase: int = 50,
    method: str = "variable",
    incompressible: bool = True,
) -> complex:
    """pylov3d's own Maxwell k2 for :func:`moon_simplified_body`.

    ``incompressible=True`` (default) drives Ks to the incompressible
    limit to match PyALMA3's propagator exactly (see
    ``pylov3d.anelastic`` module docstring) — use this for the
    pylov3d-vs-PyALMA3 cross-check. Set False to see the (structurally
    simplified, but still compressible) body's own Maxwell k2 on its own
    terms.
    """
    if Td is None:
        Td = MOON_DRACONIC_MONTH_TD
    radii, rho, mu, ks, mantle_layers = moon_simplified_body()
    n = len(radii)
    Ks0 = [1e7 * mu[-1]] * n if incompressible else ks
    model = make_interior_model(R0_km=radii, rho0=rho, mu0=mu, Ks0=Ks0, eta0=[None] * n)
    model = _with_eta(model, {i: eta_mantle for i in mantle_layers})

    forcing = make_forcing(Td=Td, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=n, method=method, Nrbase=Nrbase)
    love, _, _ = get_love(model, forcing, numerics)
    return complex(love.k[0])


def moon_alma_k2(
    eta_mantle: float,
    *,
    rheology: str = "maxwell",
    alpha: float | None = None,
    Td: float | None = None,
    ndigits: int = 64,
    order: int = 8,
) -> complex:
    """PyALMA3 (external reference) complex k2 for
    :func:`moon_simplified_body`, with Maxwell or Andrade mantle layers.

    See ``pylov3d.anelastic`` module docstring for why the simplified
    (not the real 10-layer Weber) structure is used here.
    """
    alma = _import_alma()
    if rheology not in ("maxwell", "andrade"):
        raise ValueError(f"Unknown rheology {rheology!r}; use 'maxwell' or 'andrade'")
    if rheology == "andrade" and alpha is None:
        raise ValueError("alpha is required for rheology='andrade'")
    if Td is None:
        Td = MOON_DRACONIC_MONTH_TD

    radii, rho, mu, _ks, mantle_layers = moon_simplified_body()
    n = len(radii)
    r_in = [x * 1e3 for x in radii]

    eta_in = [0.0] * n
    rheol = ["fluid"]
    params = np.zeros((n, 2))
    for i in range(1, n):
        if i in mantle_layers:
            eta_in[i] = eta_mantle
            rheol.append(rheology)
            if rheology == "andrade":
                params[i, 0] = alpha
        else:
            rheol.append("elastic")

    alma_model = alma.build_model(
        r_in, rho, mu, eta_in, rheol, params,
        ndigits=ndigits, verbose=False, parallel=False,
    )
    Td_norm = Td / ALMA_T0_S
    _h, _l, k = alma.love_numbers(
        [2], [Td_norm], "tidal", "step", 0,
        alma_model, "complex", order=order, verbose=False, parallel=False,
    )
    return complex(k[0, 0])


def moon_simplified_elastic_k2_alma(*, Td: float | None = None, ndigits: int = 64, order: int = 8) -> complex:
    """Exact elastic k2 of :func:`moon_simplified_body`, via PyALMA3 with
    actual ``rheology="elastic"`` mantle layers (not a large-eta
    Maxwell/Andrade limit).

    This matters specifically for :func:`fit_moon_andrade_gap`'s
    automatic fractional-gap target: Andrade's complex modulus approaches
    the real elastic value only as ``eta -> infinity``, and at small
    alpha that approach is slow (the residual term decays as
    ``(s*eta/mu)^(-alpha)``, e.g. still ~0.8% off true elastic at
    ``eta = 1e30`` for ``alpha = 0.15``). Using ``moon_alma_k2(huge_eta,
    rheology="andrade", alpha=alpha)`` as the elastic anchor therefore
    silently biases the gap target by an alpha-dependent amount (caught
    while building this module's tests — see docs/MOON_MODEL.md,
    "Anelasticity (TASK-025a)" headline table note). This function
    sidesteps the issue entirely by asking PyALMA3 for the actual
    elastic rheology code (alpha-independent, exact, no large-eta
    assumption).
    """
    alma = _import_alma()
    if Td is None:
        Td = MOON_DRACONIC_MONTH_TD
    radii, rho, mu, _ks, mantle_layers = moon_simplified_body()
    n = len(radii)
    r_in = [x * 1e3 for x in radii]
    eta_in = [0.0] * n
    rheol = ["fluid"] + ["elastic"] * len(mantle_layers) + ["elastic"]
    params = np.zeros((n, 2))
    alma_model = alma.build_model(
        r_in, rho, mu, eta_in, rheol, params,
        ndigits=ndigits, verbose=False, parallel=False,
    )
    Td_norm = Td / ALMA_T0_S
    _h, _l, k = alma.love_numbers(
        [2], [Td_norm], "tidal", "step", 0,
        alma_model, "complex", order=order, verbose=False, parallel=False,
    )
    return complex(k[0, 0])


def fit_moon_andrade_gap(
    alpha: float,
    *,
    target_k2: float | None = None,
    eta_lo: float = 1e14,
    eta_hi: float = 1e26,
    tol: float = 1e-6,
    max_iter: int = 60,
    ndigits: int = 64,
    order: int = 8,
) -> tuple[float, complex, float]:
    """Bisect the simplified-body Andrade mantle viscosity [Pa s] (at
    given ``alpha``) that closes the SAME fractional k2 gap as
    :func:`fit_moon_maxwell_gap`, i.e. the same ~4.6% enhancement over
    the simplified body's own elastic k2 (not the absolute
    ``MOON["k2"]``, since the simplified body's elastic baseline differs
    slightly from the real 10-layer structure's — see
    :func:`moon_simplified_body`).

    ``target_k2``, if given, overrides the automatic fractional-gap
    target with an absolute Re(k2) value.

    Returns ``(eta_mantle, k2_complex, Q_implied)``. This is the headline
    Andrade-side Moon calculation (TASK-025a) — see docs/MOON_MODEL.md,
    "Anelasticity (TASK-025a)".
    """
    if target_k2 is None:
        elastic_k2 = moon_simplified_elastic_k2_alma(ndigits=ndigits, order=order).real
        gap_ratio = _moon_mod.MOON["k2"] / _moon_mod.WEBER_K2_UNIFORM
        target_k2 = elastic_k2 * gap_ratio

    def f(eta: float) -> float:
        return moon_alma_k2(eta, rheology="andrade", alpha=alpha, ndigits=ndigits, order=order).real

    eta_fit, _ = _bisect_log_decreasing(f, eta_lo, eta_hi, target_k2, tol=tol, max_iter=max_iter)
    k2 = moon_alma_k2(eta_fit, rheology="andrade", alpha=alpha, ndigits=ndigits, order=order)
    return eta_fit, k2, implied_Q(k2)


# ---------------------------------------------------------------------------
# Literature-sourced parameter ranges for stage b (TASK-025b) priors
# ---------------------------------------------------------------------------
# Full citation trail: docs/MOON_MODEL.md, "Anelasticity (TASK-025a)" ->
# "Literature parameter ranges for stage b". Repeated here only as plain
# constants for reuse; the docs are the citation record of provenance.
# Andrade-alpha ranges (body-agnostic) live in pylov3d.anelastic
# (ANDRADE_ALPHA_RANGE, ANDRADE_ALPHA_COMMON_RANGE).

#: Williams & Boggs (2015) monthly Q = 38 +/- 4 (draconic month, 27.212
#: d); see :data:`MOON_DRACONIC_MONTH_TD` above for the full citation.
#: Used as the stage-b Andrade/Maxwell likelihood target.
MOON_MONTHLY_Q: float = 38.0
MOON_MONTHLY_Q_SIGMA: float = 4.0
