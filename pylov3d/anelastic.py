# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Anelastic (viscoelastic) tidal response for Mars and Moon (TASK-025a).

Solver capability audit (read this first)
---------------------------------------------------------------------------
``pylov3d`` implements **Maxwell** viscoelasticity natively: a per-layer
viscosity ``eta0`` (:func:`pylov3d.types.make_interior_model`) feeds
:func:`pylov3d.rheology.compute_complex_rheology`, which computes

    muC = mu / (1 - i / MaxTime),   MaxTime = 2*pi*eta0 / (mu0 * Td)

for every layer with a finite ``eta0`` (``NaN`` => elastic, ``muC = mu``
real). This is the *only* rheology implemented anywhere in this Python
port. A repo-wide grep (``grep -rn -i andrade pylov3d/`` and the same
across the vendored MATLAB source under ``src/``) turns up **no** Andrade
(or Burgers/Kelvin) implementation in either codebase checked into this
repository -- only prose caveats noting it as future work (``pylov3d/mars.py``,
``pylov3d/moon.py``, ``pylov3d/moon_mc.py``). The Maxwell path itself is
exercised end-to-end by the ordinary ``get_love`` pipeline (grid ->
rheology -> propagator -> Love numbers) exactly as the elastic path is --
an elastic layer is just ``eta0 = NaN`` -- and already has dedicated unit
tests (``pylov3d/tests/test_rheology.py``) and a toy-body PyALMA3 cross-
check (``pylov3d/tests/test_benchmark_pyalma3.py``,
``TestViscoelasticMaxwellBenchmark``, agreement < 0.1% relative).

Where this module needs **Andrade** (the Moon Q-consistency headline
below), it calls **PyALMA3** (``alma``, optional dependency, installed in
this venv -- see ``pylov3d/tests/test_benchmark_pyalma3.py`` for the
existing convention/unit-mapping notes) directly as an *external*
reference implementation. Those calls are clearly separated from
pylov3d's own solver calls throughout this module and are named
``*_alma_*``; they are never routed through ``pylov3d.love.get_love``.

A second, load-bearing finding from building the validation harness
(``pylov3d/tests/test_anelastic.py``): **PyALMA3's propagator is
incompressible** -- ``alma.build_model`` takes no bulk-modulus/lambda
argument at all, and neither ``direct_matrix`` nor ``complex_rigidity`` in
``alma/__init__.py`` ever reference one. pylov3d's Mars/Moon models are
compressible (real ``Ks0`` values comparable in magnitude to ``mu0``), so
a naive comparison at the as-built ``Ks0`` disagrees by several percent on
Re(k2) -- a compressibility artifact, not a rheology bug (confirmed by
re-running the comparison with ``Ks0`` driven to the incompressible limit,
``1e7 * mu0_surface`` -- the same convention the pre-existing toy-body
benchmark already uses -- which recovers < 1e-4 relative agreement, see
"Validation" in ``docs/MARS_MODEL.md`` / ``docs/MOON_MODEL.md``,
"Anelasticity (TASK-025a)"). The Maxwell complex-modulus computation
itself (``compute_complex_rheology``) does not depend on ``Ks0`` -- bulk
modulus only enters via the Lame parameter ``lam = Ks - 2/3*muC`` in the
radial ODE -- so the incompressible-limit comparison isolates and
validates exactly the rheology mechanics, while the compressible elastic
terms are independently validated to ~1e-12 against native MATLAB
(TASK-014/TASK-020, ``data/tests/mars/``).

A third finding: **PyALMA3 supports a fluid layer only at the center**
(``rheol[0] == 0`` is special-cased in ``love_numbers_sampler`` via
``fluid_core_bc``; ``complex_rigidity`` has no branch for rheology code 0
at all, so an internal fluid layer elsewhere in the stack raises). The
Moon's as-built Weber profile has an internal ocean (the fluid outer
core, layer index 2, sandwiched between the solid inner core and the
solid mantle) -- exactly the feature pylov3d's own dedicated ocean/coupled
solver (TASK-005/006/007) was built to add, and exactly what PyALMA3
cannot represent. See :mod:`pylov3d.anelastic_moon` (this module's
companion, split out to keep both files under the project's
500-line-per-file convention) for how the Moon's validation/Andrade
calculations work around that (Mars's core, by contrast, is a simple
layer-0 fluid core -- no simplification is needed there; the real
4-layer Mars model, handled entirely in this module, is used directly
against PyALMA3).

Stage a / b boundary
---------------------------------------------------------------------------
This module (and :mod:`pylov3d.anelastic_moon`) compute **forward**
anelastic Love numbers for point parameter values (a mantle viscosity,
an Andrade alpha) -- TASK-025a. Neither threads anelasticity into the
Monte Carlo (``pylov3d.mars_mc`` / ``pylov3d.moon_mc``) -- that is
TASK-025b. Every public function takes plain scalar free parameter(s)
and returns a complex Love number, the same call shape a future stage-b
``log_likelihood`` would need:
``mars_maxwell_k2(eta_mantle, ...) -> complex``,
``pylov3d.anelastic_moon.moon_maxwell_k2(eta_mantle, ...) -> complex``.
Nothing here reads or writes MCMC/pocoMC state.
"""

from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from . import mars as _mars_mod
from .love import get_love
from .types import InteriorModel, make_forcing, make_interior_model, make_numerics

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: PyALMA3 normalizes time internally by t0 = 1000 yr (see
#: ``alma/__init__.py:build_model``, and
#: ``pylov3d/tests/test_benchmark_pyalma3.py``'s module docstring, which
#: first documented this). Every ``timesteps`` argument to
#: ``alma.love_numbers()`` must be divided by this value.
ALMA_T0_S = 1000.0 * 365.25 * 86400.0  # s (3.1558e10 s)


def _import_alma():
    try:
        import alma
    except ImportError as exc:  # pragma: no cover - exercised via importorskip in tests
        raise ImportError(
            "PyALMA3 ('alma') is required for this calculation. pylov3d "
            "itself implements only Maxwell viscoelasticity (see "
            "pylov3d.anelastic module docstring); Andrade calculations "
            "route through PyALMA3 as an external reference. Install the "
            "'compat' extra (pyalma3)."
        ) from exc
    return alma


def implied_Q(k2: complex) -> float:
    """Tidal quality factor implied by a complex degree-2 Love number.

    Uses the standard small-lag relation for a Love number with phase lag
    delta, k2* = |k2| * exp(-i*delta), tan(delta) ~= 1/Q, and the
    dissipative sign convention Im(k2) < 0 (pylov3d and PyALMA3 both use
    this convention -- see
    ``pylov3d/tests/test_benchmark_pyalma3.py::TestViscoelasticMaxwellBenchmark
    .test_imaginary_sign_convention_matches``):

        Q ~= -Re(k2) / Im(k2)

    This is the same relation used in the literature this module cites to
    infer Q from a measured/predicted complex Love number or tidal phase
    lag (e.g. Williams & Boggs 2015; the cot(2*gamma) relation in Bills et
    al. 2005 is the loading/orbital-decay analogue of the same small-angle
    approximation).

    Returns ``math.inf`` for a purely real (non-dissipative) k2.
    """
    if k2.imag == 0:
        return math.inf
    return -k2.real / k2.imag


def _with_eta(model: InteriorModel, layer_etas: dict[int, float | None]) -> InteriorModel:
    """Return a copy of ``model`` with ``eta0`` overridden at given layers.

    ``layer_etas``: ``{layer_index: eta_Pa_s}``; a value of ``None`` marks
    that layer elastic (NaN, pylov3d's convention). All other layers
    (including layer 0, always numerically inert / fluid-core) are left
    exactly as in ``model``. This is the mechanism every ``*_maxwell_k2``
    function below uses to add viscoelasticity to an existing as-built
    elastic model without duplicating its construction.
    """
    eta0 = np.array(model.eta0, dtype=float)
    for idx, eta in layer_etas.items():
        eta0[idx] = float("nan") if eta is None else float(eta)
    return model._replace(eta0=jnp.array(eta0, dtype=jnp.float64))


def _bisect_log_decreasing(
    f,
    lo: float,
    hi: float,
    target: float,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> tuple[float, float]:
    """Bisect for ``x`` such that ``f(x) == target``, ``x`` in ``[lo, hi]``.

    Assumes ``f`` is monotonically *decreasing* in ``log10(x)`` over the
    bracket -- true in the broad, practically-relevant sense for every
    anelastic-softening Re(k2) vs. viscosity curve in this module: larger
    viscosity => more elastic => smaller Re(k2); smaller viscosity =>
    more relaxed => larger Re(k2). This is not an exact, everywhere-
    monotonic statement: on the real 10-layer Moon body specifically,
    Re(k2) has been observed to rise slightly (by ~1e-4, i.e. well under
    this function's default ``tol``) across eta ~ 5.6e11-1.8e12 Pa s
    before resuming its overall decrease -- harmless for every bracket
    actually used in this module (that wiggle sits two decades below the
    gap-closing viscosities of interest, ~1e16-1e17 Pa s), but the
    monotonicity assumption is therefore a good approximation over the
    brackets this module uses, not a proven exact property of the
    underlying physics. Search is log-spaced because viscosity/eta
    ranges of interest here span many decades.

    Returns ``(x, f(x))`` with ``abs(f(x) - target) < tol``.
    """
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    f_lo, f_hi = f(10.0**log_lo), f(10.0**log_hi)
    if not (f_lo > target > f_hi):
        raise RuntimeError(
            f"target {target} is not bracketed by f(lo)={f_lo}, f(hi)={f_hi} "
            f"over x in [{lo}, {hi}]"
        )
    mid = log_lo
    f_mid = f_lo
    for _ in range(max_iter):
        mid = 0.5 * (log_lo + log_hi)
        f_mid = f(10.0**mid)
        if abs(f_mid - target) < tol:
            return 10.0**mid, f_mid
        if f_mid > target:
            log_lo = mid
        else:
            log_hi = mid
    raise RuntimeError(
        f"_bisect_log_decreasing did not converge within {max_iter} "
        f"iterations: last x={10.0**mid}, f(x)={f_mid}, target={target}"
    )


# ---------------------------------------------------------------------------
# Mars
# ---------------------------------------------------------------------------

#: Mantle layer indices in ``pylov3d.mars.build_mars_model()`` (lower
#: mantle, upper mantle) -- the layers this module makes viscoelastic.
#: Layer 0 (fluid core) and layer 3 (crust) stay elastic.
MARS_MANTLE_LAYERS: tuple[int, int] = (1, 2)


def mars_maxwell_k2(
    eta_mantle: float,
    *,
    mu_scale: float | None = None,
    Td: float | None = None,
    Nrbase: int = 100,
    method: str = "combination",
) -> complex:
    """Complex degree-2 k2 of the as-built 4-layer Mars model with a
    Maxwell-viscoelastic mantle, computed by pylov3d's own solver.

    Both mantle layers (:data:`MARS_MANTLE_LAYERS`) get the same
    ``eta_mantle`` [Pa s]; the fluid core and elastic crust are untouched.
    Default forcing period is :data:`pylov3d.mars.MARS_FORCING_TD`
    (44387.622 s, half a sol) -- the solar-semidiurnal period Konopliv,
    Park & Folkner (2016) measure k2 = 0.169 +/- 0.006 at (see
    "Forcing-period provenance" in docs/MARS_MODEL.md, TASK-025a section,
    and the module-level note in ``pylov3d/mars.py`` -- checked against
    web search of the primary literature, not assumed: this is a
    *different* tidal frequency than the Phobos-orbital-decay tide Bills
    et al. (2005) measure Q = 85.58 +/- 0.37 at (~19,991 s, degree 2; an
    earlier draft of this docstring said "Q~80" and "~19,980 s" -- both
    wrong, see docs/MARS_MODEL.md for the corrected values and citation
    trail), so the two published Mars anelastic numbers are not directly
    comparable -- see docs for the full caveat).
    """
    if mu_scale is None:
        mu_scale = _mars_mod.MARS_MU_SCALE
    if Td is None:
        Td = _mars_mod.MARS_FORCING_TD

    model = _mars_mod.build_mars_model(mu_scale=mu_scale)
    model = _with_eta(model, {i: eta_mantle for i in MARS_MANTLE_LAYERS})

    forcing = make_forcing(Td=Td, n=2, m=0, F=1.0)
    numerics = make_numerics(n_layers=model.n_layers, method=method, Nrbase=Nrbase)
    love, _, _ = get_love(model, forcing, numerics)
    return complex(love.k[0])


def mars_alma_k2(
    eta_mantle: float,
    *,
    rheology: str = "maxwell",
    alpha: float | None = None,
    mu_scale: float | None = None,
    Td: float | None = None,
    ndigits: int = 64,
    order: int = 8,
) -> complex:
    """Complex degree-2 k2 of the as-built 4-layer Mars model via PyALMA3.

    External reference implementation (see module docstring) -- NOT
    routed through pylov3d's own solver. Mars needs no structural
    simplification (its core is a simple fluid layer at the center, the
    only case PyALMA3 supports -- see module docstring), so this uses the
    exact same 4-layer radii/densities/rigidities as
    :func:`mars_maxwell_k2`.

    Parameters
    ----------
    rheology : ``"maxwell"`` or ``"andrade"``.
    alpha : Andrade alpha parameter (required if ``rheology="andrade"``).

    Note: PyALMA3 never asks for a bulk modulus/Ks at all (its propagator
    is incompressible throughout -- see module docstring, "second
    finding"), so there is no compressibility knob to expose here. This
    function's result should therefore be compared only against other
    incompressible-limit calculations (e.g. pylov3d run with Ks driven to
    ``1e7 * mu0_surface``, see :func:`mars_maxwell_incompressible_model`),
    never directly against pylov3d's compressible as-built k2.
    """
    alma = _import_alma()
    if rheology not in ("maxwell", "andrade"):
        raise ValueError(f"Unknown rheology {rheology!r}; use 'maxwell' or 'andrade'")
    if rheology == "andrade" and alpha is None:
        raise ValueError("alpha is required for rheology='andrade'")

    if mu_scale is None:
        mu_scale = _mars_mod.MARS_MU_SCALE
    if Td is None:
        Td = _mars_mod.MARS_FORCING_TD

    model = _mars_mod.build_mars_model(mu_scale=mu_scale)
    n = model.n_layers
    r_in = [float(x) * 1e3 for x in np.asarray(model.R0[:n])]
    rho_in = [float(x) for x in np.asarray(model.rho0[:n])]
    mu_in = [float(x) for x in np.asarray(model.mu0[:n])]

    eta_in = [0.0] * n
    rheol = ["fluid"]
    params = np.zeros((n, 2))
    for i in range(1, n):
        if i in MARS_MANTLE_LAYERS:
            eta_in[i] = eta_mantle
            rheol.append(rheology)
            if rheology == "andrade":
                params[i, 0] = alpha
        else:
            rheol.append("elastic")

    alma_model = alma.build_model(
        r_in, rho_in, mu_in, eta_in, rheol, params,
        ndigits=ndigits, verbose=False, parallel=False,
    )
    Td_norm = Td / ALMA_T0_S
    _h, _l, k = alma.love_numbers(
        [2], [Td_norm], "tidal", "step", 0,
        alma_model, "complex", order=order, verbose=False, parallel=False,
    )
    return complex(k[0, 0])


def mars_maxwell_incompressible_model(mu_scale: float | None = None) -> InteriorModel:
    """As-built 4-layer Mars model with Ks driven to the incompressible
    limit (``1e7 * mu0_surface``), for pylov3d-vs-PyALMA3 comparisons
    (see module docstring, "second finding"). All other parameters
    (radii, densities, mu, forcing) identical to
    :func:`pylov3d.mars.build_mars_model`.
    """
    if mu_scale is None:
        mu_scale = _mars_mod.MARS_MU_SCALE
    rho_core, rho_lm = _mars_mod._solve_densities()
    R0_km, R1_km, R2_km, R3_km = _mars_mod.LAYER_RADII_KM
    mu0_list = [
        0.0,
        _mars_mod._MU_LM_BASE * mu_scale,
        _mars_mod._MU_UM_BASE * mu_scale,
        _mars_mod.LAYER_MU_CRUST,
    ]
    Ks_incompr = 1e7 * mu0_list[-1]
    return make_interior_model(
        R0_km=[R0_km, R1_km, R2_km, R3_km],
        rho0=[rho_core, rho_lm, _mars_mod.MARS["upper_mantle_density"], _mars_mod.MARS["crust_density"]],
        mu0=mu0_list,
        Ks0=[Ks_incompr] * 4,
        eta0=[None, None, None, None],
    )


# ---------------------------------------------------------------------------
# Literature-sourced parameter ranges for stage b (TASK-025b) priors
# ---------------------------------------------------------------------------
# Full citation trail: docs/MARS_MODEL.md / docs/MOON_MODEL.md,
# "Anelasticity (TASK-025a)" -> "Literature parameter ranges for stage b".
# Repeated here only as plain constants for reuse; the docs are the
# citation record of provenance.

#: Mars mantle Andrade viscosity range [Pa s], reported by Bagheri et al.
#: (2019), JGR Planets 124(11), 2703-2727 (attributing the estimate to
#: Castillo-Rogez & Banerdt 2012) -- the tidal-frequency-relevant range,
#: NOT the long-timescale (glacial-isostatic-adjustment) present-day deep
#: mantle viscosity of Broquet et al. (2025), Nature 639, 109-113
#: (2-6e22 Pa s for depths > 500 km, a much longer-timescale, different
#: physical regime -- see docs for why the two are not interchangeable).
MARS_MANTLE_ETA_ANDRADE_RANGE: tuple[float, float] = (1e19, 1e22)

#: Andrade alpha range for silicate mantles. Walterova, Behounkova &
#: Efroimsky (2023), JGR Planets, e2022JE007652 (arXiv:2301.02476v2,
#: retrieved and quoted directly in this session -- the module's earlier
#: draft attributed a different, non-existent quotation to this paper;
#: corrected here), state: alpha "typically lies in the interval
#: 0.2-0.4, although values outside this range have also been observed
#: ... Geodetic measurements performed on the Earth favour a narrower
#: interval of 0.14-0.2, and the currently accepted model of tides in
#: the solid Earth, presented in the IERS Conventions on Earth
#: Rotation, employs the value of alpha = 0.15." ANDRADE_ALPHA_RANGE
#: below is the 0.2-0.4 lab figure; the narrower 0.14-0.2 geodetic
#: figure and the IERS alpha=0.15 point value are documentation-only
#: context (see docs/MOON_MODEL.md / docs/MARS_MODEL.md, "Literature
#: parameter ranges").
#:
#: A commonly-adopted narrower range, 0.2-0.3, is attributed here to
#: Efroimsky (2012), Celestial Mechanics and Dynamical Astronomy 112,
#: 283-330 (arXiv:1105.6086) -- retrieved and paraphrased in this
#: session, Section 5.3: lab-measured alpha for minerals (including
#: ices) spans 0.14-0.4, "more often" through 0.3, i.e. the commonly-
#: quoted range is the lower part of the full lab range. (An earlier
#: draft cited Castillo-Rogez et al. (2011) for this narrower range;
#: that citation could not be verified in this session and is dropped.)
#: Dumoulin et al. (2017), JGR Planets 122(6), 1338-1352
#: (doi:10.1002/2016JE005249), retrieved and paraphrased in this
#: session: they *adopt* alpha in 0.2-0.3 for their prospective Venus
#: interior models "which frames the typical value required to explain
#: the Q factor of the Earth's mantle" (their own wording, Section 2.2)
#: -- i.e. the range is imported from Earth, not derived from or shown
#: to explain any Venus tidal measurement; the paper's own conclusion is
#: that a future mission (EnVision) is needed before Venus tidal data
#: can constrain mantle rheology at all. (An earlier draft of this
#: module claimed Dumoulin et al. "find" this range "explains Venus's
#: short-period tidal response" -- overstated, corrected here.)
ANDRADE_ALPHA_RANGE: tuple[float, float] = (0.2, 0.4)
ANDRADE_ALPHA_COMMON_RANGE: tuple[float, float] = (0.2, 0.3)

# Moon-specific constants (MOON_DRACONIC_MONTH_TD, MOON_MANTLE_LAYERS,
# MOON_MONTHLY_Q, MOON_MONTHLY_Q_SIGMA) live in pylov3d.anelastic_moon,
# this module's companion — see that module's docstring for why the Moon
# needs its own file.
