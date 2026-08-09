# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Body-agnostic forward-model + Monte Carlo framework for interior inference.

Three separable stages (TASK-012):

Stage 1 (parameters -> model): :class:`LayerSpec` / :class:`BodyParameterization`
describe an ordered stack of layers whose scalars are each either FIXED (a
plain float) or FREE (resolved from a named entry of a flat ``theta`` vector,
optionally as a multiplicative scale on a fixed base value). :func:`build_model`
turns ``(parameterization, theta)`` into a :class:`~pylov3d.types.InteriorModel`
via :func:`~pylov3d.types.make_interior_model`. This stage has no knowledge of
any specific body; :mod:`pylov3d.mars_mc` instantiates it for Mars, and it is
also intended to be replaced (upstream of this module, not within it) by a
petrological model — :mod:`pylov3d.compat` already converts PlanetProfile
structs to :class:`~pylov3d.types.InteriorModel` for that purpose.

Stage 2 (model -> observables): :func:`compute_observables` accepts **any**
``InteriorModel`` (built by :func:`build_model` or by
:func:`pylov3d.compat.planetstruct_to_interior_model`) and returns a dict of
named scalar observables: "mass", "moi_mean", and "core_radius_km" computed
analytically from the layered density profile (cheap, no tidal solve), plus
"k2"/"h2"/"l2" from :func:`pylov3d.love.get_love`. ``numerics.method ==
"fixed"`` is rejected (raises ``NotImplementedError``): that grid method
silently rewrites the solver's layer radii, which would decouple the
analytic observables (computed on the caller's original ``model.R0``) from
the Love numbers (computed on the solver's rewritten radii).

    Gas-giant / fluid-dominated-body caveat: this stage assumes LOV3D's
    existing solid-body-with-fluid-core formulation (the boundary-condition
    machinery in ``pylov3d/boundary_conditions.py`` always treats layer 0 as
    a fluid core and integrates layers 1..n-1). A fluid-dominated body (e.g.
    Uranus/other gas giants) will need a different Stage-2 implementation
    (a different equation-of-motion / boundary-condition solve) behind this
    same ``compute_observables(model, ...) -> dict`` interface; nothing here
    should need to change for Stage 1 or Stage 3 when that lands.

Stage 3 (likelihood/posterior): :class:`Constraint` + :func:`log_likelihood`
(Gaussian) + :func:`log_prior` (uniform box over the FREE-parameter bounds)
combine into :func:`make_log_posterior`, a callable(theta) -> float usable as
the ``likelihood`` argument of a ``pocomc.Sampler`` (paired with a separate
``pocomc.Prior`` built from the same bounds for the properly normalized
uniform density and the SMC proposal — see :mod:`pylov3d.mars_mc` /
``scripts/mars_pocomc.py``). It guards prior-box violations and any solver
exception (e.g. ``numpy.linalg.LinAlgError``) by returning ``-inf``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import numpy as np

from .love import get_love
from .types import Forcing, InteriorModel, NumericsConfig, make_interior_model

# ---------------------------------------------------------------------------
# Stage 1: parameters -> model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Free:
    """A layer scalar equal directly to ``theta[name]``."""

    name: str


@dataclass(frozen=True)
class Scaled:
    """A layer scalar equal to ``base * theta[name]`` (a shared multiplicative
    parameter, e.g. one ``mu_scale`` applied to several layers' shear moduli).
    """

    base: float
    name: str


#: A per-layer scalar is a plain float (FIXED) or a :class:`Free`/:class:`Scaled`
#: reference into the flat ``theta`` vector (FREE).
ParamSpec = Union[float, Free, Scaled]


def _param_name(spec: ParamSpec) -> str | None:
    if isinstance(spec, (Free, Scaled)):
        return spec.name
    return None


def _resolve(spec: ParamSpec, theta_map: Mapping[str, float]) -> float:
    if isinstance(spec, Free):
        return float(theta_map[spec.name])
    if isinstance(spec, Scaled):
        return float(spec.base) * float(theta_map[spec.name])
    return float(spec)


@dataclass(frozen=True)
class LayerSpec:
    """One homogeneous layer, core (index 0) to surface.

    ``eta`` is ``None`` for a purely elastic layer (matches
    :func:`~pylov3d.types.make_interior_model`'s ``eta0=None`` -> NaN
    convention); otherwise it is a :data:`ParamSpec` in Pa s.
    """

    name: str
    radius_km: ParamSpec
    rho: ParamSpec
    mu: ParamSpec
    Ks: ParamSpec
    eta: ParamSpec | None = None
    ocean: int = 0


@dataclass(frozen=True)
class BodyParameterization:
    """An ordered layer stack plus the FREE-parameter vector it depends on.

    Parameters
    ----------
    name : label (e.g. "mars", "toy_body").
    layers : layer stack, core to surface.
    free_params : names of the FREE parameters, in the order a flat ``theta``
        vector must supply them. Must be exactly the set of parameter names
        referenced anywhere in ``layers`` (validated in ``__post_init__``).
    bounds : ``{name: (lo, hi)}`` prior box per free parameter; keys must
        match ``free_params`` exactly.
    core_layer_index : which layer's outer radius :func:`compute_observables`
        reads for the ``"core_radius_km"`` observable (default 0, i.e. LOV3D
        layer 0 — correct whenever the physically free "core" being tracked
        *is* the LOV3D-solver core layer, e.g. Mars's 4-layer model). Set
        this to a different layer index for a parameterization where layer 0
        is an inert placeholder rather than the physically free core (e.g.
        an artificial numerically-inert buffer layer ahead of a real free
        fluid-core boundary a few layers out — see :mod:`pylov3d.moon_mc`,
        which sets this to 2).
    """

    name: str
    layers: tuple[LayerSpec, ...]
    free_params: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    core_layer_index: int = 0

    def __post_init__(self) -> None:
        referenced: set[str] = set()
        for layer in self.layers:
            for spec in (layer.radius_km, layer.rho, layer.mu, layer.Ks, layer.eta):
                nm = _param_name(spec)
                if nm is not None:
                    referenced.add(nm)

        free_set = set(self.free_params)
        if referenced != free_set:
            raise ValueError(
                "BodyParameterization.free_params must exactly match the "
                f"parameter names referenced in layers: referenced={sorted(referenced)}, "
                f"declared free_params={sorted(free_set)}"
            )
        if set(self.bounds.keys()) != free_set:
            raise ValueError(
                "BodyParameterization.bounds keys must exactly match "
                f"free_params: bounds={sorted(self.bounds.keys())}, "
                f"free_params={sorted(free_set)}"
            )
        for nm, (lo, hi) in self.bounds.items():
            if not (lo < hi):
                raise ValueError(f"bounds for {nm!r} must have lo < hi, got ({lo}, {hi})")
        if not (0 <= self.core_layer_index < len(self.layers)):
            raise ValueError(
                f"core_layer_index={self.core_layer_index} out of range for "
                f"{len(self.layers)} layers"
            )

    @property
    def n_free(self) -> int:
        return len(self.free_params)

    @property
    def bounds_array(self) -> np.ndarray:
        """``(n_free, 2)`` array of (lo, hi) in ``free_params`` order."""
        return np.array([self.bounds[nm] for nm in self.free_params], dtype=float)


def build_model(parameterization: BodyParameterization, theta: Sequence[float]) -> InteriorModel:
    """Map a flat ``theta`` vector to an :class:`~pylov3d.types.InteriorModel`.

    No body-specific knowledge: purely mechanical resolution of each layer's
    FIXED/FREE/Scaled scalars followed by
    :func:`~pylov3d.types.make_interior_model`. Free layer radii (e.g. a free
    core radius) are handled automatically — every layer's outer radius is
    resolved independently, there is no assumption that only the surface or
    only relative offsets can vary.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (parameterization.n_free,):
        raise ValueError(
            f"theta has shape {theta.shape}, expected ({parameterization.n_free},) "
            f"for free_params={parameterization.free_params}"
        )
    theta_map = dict(zip(parameterization.free_params, theta))

    R0_km, rho0, mu0, Ks0, eta0, ocean = [], [], [], [], [], []
    for layer in parameterization.layers:
        R0_km.append(_resolve(layer.radius_km, theta_map))
        rho0.append(_resolve(layer.rho, theta_map))
        mu0.append(_resolve(layer.mu, theta_map))
        Ks0.append(_resolve(layer.Ks, theta_map))
        eta0.append(None if layer.eta is None else _resolve(layer.eta, theta_map))
        ocean.append(layer.ocean)

    if not np.all(np.diff(np.asarray(R0_km, dtype=float)) > 0):
        raise ValueError(
            f"layer outer radii must be strictly increasing core to surface, "
            f"got R0_km={R0_km} at theta={np.asarray(theta).tolist()}; a free "
            "radius parameter has produced a zero- or negative-volume shell "
            "(e.g. a free core radius exceeding the next fixed layer's "
            "radius). Narrow that parameter's bounds or add a constraint "
            "that penalizes this region instead of letting it silently "
            "reach the solver."
        )

    return make_interior_model(
        R0_km=R0_km, rho0=rho0, mu0=mu0, Ks0=Ks0, eta0=eta0, ocean=ocean,
    )


# ---------------------------------------------------------------------------
# Stage 2: model -> observables
# ---------------------------------------------------------------------------

#: Observables computed by default when ``which`` is not given explicitly.
DEFAULT_OBSERVABLES: tuple[str, ...] = (
    "mass", "moi_mean", "core_radius_km", "k2", "h2", "l2",
)

_ANALYTIC = frozenset({"mass", "moi_mean", "core_radius_km"})
_LOVE = frozenset({"k2", "h2", "l2"})
_LOVE_ATTR = {"k2": "k", "h2": "h", "l2": "l"}


def analytic_mass_moi(model: InteriorModel) -> tuple[float, float]:
    """Mass [kg] and mean moment of inertia [kg m^2] of a shell-stack profile.

    Shell i spans [R_{i-1}, R_i] (R_{-1} = 0) with uniform density rho_i:

        M = sum_i (4*pi/3) * rho_i * (R_i^3 - R_{i-1}^3)
        I = sum_i (8*pi/15) * rho_i * (R_i^5 - R_{i-1}^5)

    (mirrors the shell formulas in ``pylov3d/mars.py:_mass_and_moi``; a solid
    uniform sphere has I = (2/5) M R^2 = (8*pi/15) rho R^5, and a shell's
    moment is the difference of the two solid-sphere moments at its own
    density). Works for any layer count/ordering, not just Mars's four.
    """
    n = model.n_layers
    radii_m = np.asarray(model.R0[:n], dtype=float) * 1e3
    rho = np.asarray(model.rho0[:n], dtype=float)
    boundaries_m = np.concatenate([[0.0], radii_m])

    M = 0.0
    I = 0.0
    for i in range(n):
        r_in, r_out = boundaries_m[i], boundaries_m[i + 1]
        M += (4.0 * math.pi / 3.0) * rho[i] * (r_out**3 - r_in**3)
        I += (8.0 * math.pi / 15.0) * rho[i] * (r_out**5 - r_in**5)
    return float(M), float(I)


def _is_elastic(model: InteriorModel) -> bool:
    """True if every active layer's eta0 is NaN (purely elastic input model)."""
    n = model.n_layers
    eta0 = np.asarray(model.eta0[:n], dtype=float)
    return bool(np.all(np.isnan(eta0)))


def compute_observables(
    model: InteriorModel,
    forcing: Forcing,
    numerics: NumericsConfig,
    which: Sequence[str] = DEFAULT_OBSERVABLES,
    core_layer_index: int = 0,
) -> dict[str, float | complex]:
    """Compute named observables from any ``InteriorModel``.

    Parameters
    ----------
    model : any InteriorModel (parametric, from :func:`build_model`, or from
        e.g. :func:`pylov3d.compat.planetstruct_to_interior_model`).
    forcing, numerics : as for :func:`pylov3d.love.get_love`; only used if
        ``which`` requests a Love number. ``numerics.method == "fixed"`` is
        rejected outright (see below), independent of ``which``.
    which : subset of ``{"mass", "moi_mean", "core_radius_km", "k2", "h2",
        "l2"}``.
    core_layer_index : which layer's outer radius ``"core_radius_km"`` reads
        (default 0). Pass ``parameterization.core_layer_index`` when calling
        this on a :class:`BodyParameterization`-built model where the
        physically free core layer is not LOV3D layer 0 (see that field's
        docstring).

    Returns
    -------
    dict mapping each requested name to a float (elastic model: Love numbers
    are cast to their real part) or complex (viscoelastic model: Love
    numbers are returned complex, propagating the dissipative phase lag).
    "mass" is in kg; "moi_mean" is the dimensionless factor I/(M R_surface^2);
    "core_radius_km" is layer ``core_layer_index``'s outer radius in km, read
    directly off the model (no solve) — a cheap, generic (not Mars-specific)
    identifiability handle for any parameterization with a free core radius.

    Raises
    ------
    NotImplementedError
        If ``numerics.method == "fixed"``: that grid method
        (``pylov3d.grid.set_boundary_indices``) snaps the *solver's copy* of
        the layer radii onto a fixed-spacing grid before ``get_love`` ever
        sees them, without mutating the ``model`` this function was handed.
        The analytic mass/MoI/core-radius observables above are computed
        from the caller's unmodified ``model.R0``, so combining them with
        Love numbers computed from ``method="fixed"``'s silently-rewritten
        radii would describe two slightly different bodies sharing one
        ``theta`` (measured: ~2.2 sigma mass inconsistency at Mars's own
        Nrbase). Use ``method="combination"`` (the default throughout
        ``pylov3d.mars``) or ``"variable"`` instead.
    """
    if numerics.method == "fixed":
        raise NotImplementedError(
            "compute_observables does not support numerics.method='fixed': "
            "pylov3d.grid.set_boundary_indices snaps the solver's layer "
            "radii onto a fixed-spacing grid, decoupling them from the "
            "model.R0 this function reads for the analytic mass/moi_mean/"
            "core_radius_km observables -- the two would describe slightly "
            "different bodies from the same theta. Use method='combination' "
            "or 'variable'."
        )

    which = tuple(which)
    unknown = set(which) - (_ANALYTIC | _LOVE)
    if unknown:
        raise ValueError(f"unknown observable(s): {sorted(unknown)}")

    out: dict[str, float | complex] = {}

    if _ANALYTIC & set(which):
        if {"mass", "moi_mean"} & set(which):
            M, I = analytic_mass_moi(model)
            if "mass" in which:
                out["mass"] = M
            if "moi_mean" in which:
                n = model.n_layers
                R_surface_m = float(np.asarray(model.R0[n - 1])) * 1e3
                out["moi_mean"] = I / (M * R_surface_m**2)
        if "core_radius_km" in which:
            out["core_radius_km"] = float(np.asarray(model.R0[core_layer_index]))

    love_wanted = [k for k in which if k in _LOVE]
    if love_wanted:
        love, _y_rad, _norm_model = get_love(model, forcing, numerics)
        elastic = _is_elastic(model)
        for key in love_wanted:
            arr = getattr(love, _LOVE_ATTR[key])
            val = complex(arr[0])
            out[key] = val.real if elastic else val

    return out


# ---------------------------------------------------------------------------
# Stage 3: likelihood / prior / posterior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    """A single Gaussian observational constraint on a named observable."""

    name: str
    value: float
    sigma: float


def log_likelihood(
    observables: Mapping[str, float | complex], constraints: Sequence[Constraint]
) -> float:
    """Gaussian log-likelihood: sum of -0.5*((obs-value)/sigma)^2 - log(sigma*sqrt(2pi)).

    Complex observables (viscoelastic Love numbers, see
    :func:`compute_observables`) are compared against ``constraint.value``
    by their **real part only** — this function has no notion of a
    constraint on the imaginary/dissipative part. An anelastic fit that
    wants to also constrain a phase lag or a Q/dissipation observable needs
    that handled explicitly (either as its own real-valued observable name
    computed upstream, e.g. ``"k2_imag"`` or ``"phase_lag"``, or by a
    dedicated complex-aware likelihood term) — silently dropping the
    imaginary part here is intentional for the real-valued constraints this
    module currently ships (mass, moi_mean, core_radius_km, and elastic k2),
    not a general anelastic-fit capability.
    """
    ll = 0.0
    for c in constraints:
        if c.name not in observables:
            raise KeyError(
                f"constraint {c.name!r} not found in observables "
                f"{sorted(observables.keys())}; pass a matching `which` to "
                "compute_observables()"
            )
        obs = observables[c.name]
        val = obs.real if isinstance(obs, complex) else float(obs)
        resid = (val - c.value) / c.sigma
        ll += -0.5 * resid * resid - math.log(c.sigma * math.sqrt(2.0 * math.pi))
    return ll


def log_prior(theta: Sequence[float], parameterization: BodyParameterization) -> float:
    """Uniform-box bounds gate: 0.0 inside every free-parameter's bounds, else -inf.

    This is intentionally *not* the fully normalized uniform log-density
    (which would subtract sum(log(hi-lo)) inside the box); it is meant to be
    combined with a separate properly-normalized prior object (e.g.
    ``pocomc.Prior`` built from scipy.stats.uniform over the same bounds — see
    ``scripts/mars_pocomc.py``), so that combination is correct with no
    double-counting of the normalization constant while this function still
    does its job of gating :func:`make_log_posterior` against evaluating the
    (possibly expensive, possibly ill-posed) forward model outside the
    physically valid parameter box.
    """
    theta = np.asarray(theta, dtype=float)
    bounds = parameterization.bounds_array
    if theta.shape != (parameterization.n_free,):
        raise ValueError(
            f"theta has shape {theta.shape}, expected ({parameterization.n_free},)"
        )
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        return -np.inf
    return 0.0


def make_log_posterior(
    parameterization: BodyParameterization,
    constraints: Sequence[Constraint],
    forcing: Forcing,
    numerics: NumericsConfig,
    which: Sequence[str] | None = None,
):
    """Build a ``callable(theta) -> float`` suitable as pocomc's ``likelihood``.

    Returns -inf for any theta outside the prior box (:func:`log_prior`), and
    -inf (with a warning, counted on the returned callable's
    ``.warn_count`` mutable dict) if the forward model raises anything while
    building the model or solving for Love numbers (e.g.
    ``numpy.linalg.LinAlgError`` from an ill-conditioned/unphysical layer
    stack).

    Parameters
    ----------
    which : observables to compute; defaults to the union of all
        ``constraints`` names (only what's needed, in first-seen order).
    """
    if which is None:
        seen: list[str] = []
        for c in constraints:
            if c.name not in seen:
                seen.append(c.name)
        which = tuple(seen)

    warn_count = {"n": 0}

    def log_posterior(theta: Sequence[float]) -> float:
        lp = log_prior(theta, parameterization)
        if not np.isfinite(lp):
            return -np.inf
        try:
            model = build_model(parameterization, theta)
            obs = compute_observables(
                model, forcing, numerics, which=which,
                core_layer_index=parameterization.core_layer_index,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any
            # solver failure (LinAlgError, bracketing failures, NaNs, ...)
            # on an unphysical theta must degrade to -inf, not crash a sampler.
            warn_count["n"] += 1
            warnings.warn(
                f"forward model failed at theta={np.asarray(theta).tolist()} "
                f"({exc.__class__.__name__}: {exc}); returning -inf "
                f"(failure #{warn_count['n']})",
                RuntimeWarning,
                stacklevel=2,
            )
            return -np.inf
        return lp + log_likelihood(obs, constraints)

    log_posterior.warn_count = warn_count  # type: ignore[attr-defined]
    return log_posterior
