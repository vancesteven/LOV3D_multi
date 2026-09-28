# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

r"""Pointwise effective-medium hydration fields for the higher-degree study.

Extends the proposal's Figure-3 experiment (mean-only Voigt/Hill/Reuss
mixing of the hydrated crust, ``scripts/
mars_serpentinite_connectivity_sensitivity.py``) to the *lateral* problem:
the mixing law is applied pointwise to the spatial hydration field, the
resulting rigidity/bulk-modulus fields are re-expanded in spherical
harmonics, and the full coupled tidal response — the ``k_{2m}`` splitting
and the degree >= 3 modes — is evaluated with
:func:`pylov3d.extended_love.extended_love_tensor`. This is the
"nonlinear lateral connectivity experiment" of
``PROPOSAL_COMPLETION_STATUS.md``.

Field definition (same geometry as the validated linear pipeline,
``pylov3d.mars_hydration``):

* local hydrated fraction ``f_loc(th, ph) = f_h * t(th, ph) / t0``
  with ``t = t0 + dt`` the Airy-compensated crustal thickness
  (:func:`pylov3d.mars_lateral.crustal_thickness_variation`), clipped to
  ``[0, 1]`` (the clipped area fraction is reported);
* pointwise mixing ``mu_eff = law(f_loc, mu_dry, mu_serp)`` and likewise
  for ``K``, law in ``{voigt, hill, reuss}`` (identical formulas to the
  mean-only script);
* mean part: the ``(0,0)`` Stokes coefficient of the mixed field (its
  area mean) replaces the crust layer's modulus;
* lateral part: ``d mu / mu_bar`` expanded to ``lmax_out`` and injected
  as ``mu_variable`` (and ``dK / K_bar`` as ``K_variable``) in the
  canonical :func:`pylov3d.mars_lateral._real_sh_to_complex_mu_variable`
  convention.

Exact reductions (tested): for the linear Voigt law the pointwise mean
equals :func:`pylov3d.mars_hydration.mean_softened_crust_moduli`
bit-for-bit-level (quadrature roundoff), and the lateral coefficients
equal :func:`pylov3d.mars_hydration.hydration_lateral_variables`'s.

Synthesis/analysis conventions: the field is built and re-expanded with
the exactness-tested MATLAB-faithful pair
(:func:`pylov3d.matlab_sph.stokes_to_grid` /
:func:`~pylov3d.matlab_sph.grid_to_stokes`), with two corrections so the
coefficients in and out follow the no-CS
:func:`pylov3d.mapping.sh_to_latlon` convention used everywhere else in
the Mars pipeline: the pair's internal ``(-1)^m`` is converted at both
ends, and the analysis direction's deterministic m-dependent half-cell
longitude phase is removed with
:func:`pylov3d.mars_alteration_gravity._remove_matlab_half_cell_phase`
(exactly as the physical 3D gravity path does). The corrected round
trip is exact to ~4e-16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mars import MARS, build_mars_model
from .mars_hydration import K_CRUST
from .mars import LAYER_MU_CRUST
from .mars_lateral import (
    CRUST_LAYER_INDEX,
    _real_sh_to_complex_mu_variable,
    crustal_thickness_variation,
)
from .mars_alteration_gravity import _remove_matlab_half_cell_phase
from .matlab_sph import grid_to_stokes, stokes_to_grid

MIXING_LAWS = ("voigt", "hill", "reuss")


def mix_pointwise(f, dry: float, wet: float, law: str):
    """Voigt/Hill/Reuss mixing, vectorized over the local fraction ``f``.

    Same formulas as ``scripts/mars_serpentinite_connectivity_sensitivity
    .py`` (arithmetic / harmonic / their mean), applied elementwise.
    """
    f = np.asarray(f, dtype=float)
    v = (1.0 - f) * dry + f * wet
    if law == "voigt":
        return v
    if dry <= 0 or wet <= 0:
        raise ValueError("Reuss/Hill mixing requires positive moduli")
    r = 1.0 / ((1.0 - f) / dry + f / wet)
    if law == "reuss":
        return r
    if law == "hill":
        return 0.5 * (v + r)
    raise ValueError(f"unknown mixing law: {law}")


def _coeff_dict_to_arrays(coeffs, lmax: int):
    """Mapping-convention {(n, m): a} -> MATLAB-convention (clm, slm)."""
    clm = np.zeros((lmax + 1, lmax + 1))
    slm = np.zeros_like(clm)
    for (n, m), a in coeffs.items():
        if n > lmax:
            continue
        phase = (-1.0) ** abs(m)  # undo stokes_to_grid's internal phase
        if m >= 0:
            clm[n, m] = phase * a
        else:
            slm[n, -m] = phase * a
    return clm, slm


def _arrays_to_coeff_dict(clm, slm, lmax: int):
    """MATLAB-convention (clm, slm) -> mapping-convention {(n, m): a}."""
    out = {}
    for n in range(1, lmax + 1):  # degree 0 (the mean) handled separately
        for m in range(n + 1):
            phase = (-1.0) ** m
            if clm[n, m] != 0.0:
                out[(n, m)] = phase * clm[n, m]
            if m > 0 and slm[n, m] != 0.0:
                out[(n, -m)] = phase * slm[n, m]
    return out


@dataclass(frozen=True)
class PointwiseHydratedCrust:
    """Mixed crust for one (f_h, scenario, law) combination."""

    f_h: float
    law: str
    mu_bar: float          # area-mean crust shear modulus [Pa]
    Ks_bar: float          # area-mean crust bulk modulus [Pa]
    mu_variable: dict      # {CRUST_LAYER_INDEX: [(n, m, complex), ...]} or {}
    K_variable: dict
    clip_fraction: float   # area fraction where f_loc was clipped to [0, 1]
    model: object          # InteriorModel with the crust moduli replaced


def pointwise_hydrated_crust(
    f_h: float,
    mu_ratio: float,
    K_ratio: float,
    law: str,
    *,
    lmax_field: int = 4,
    lmax_out: int = 2,
    grid_lmax: int = 32,
    mu_scale: float | None = None,
) -> PointwiseHydratedCrust:
    """Build the pointwise-mixed crust and its solver inputs.

    ``lmax_field`` truncates the crustal-thickness field defining the
    hydration geometry; ``lmax_out`` truncates the lateral modulus
    expansion passed to the solver (nonlinear laws generate content
    beyond ``lmax_field``); ``grid_lmax`` sets the analysis grid
    (``2*grid_lmax x 4*grid_lmax``), which must comfortably exceed
    ``lmax_out`` to keep aliasing negligible.
    """
    if law not in MIXING_LAWS:
        raise ValueError(f"unknown mixing law: {law}")
    if not 0.0 <= f_h <= 1.0:
        raise ValueError("f_h must lie in [0, 1]")
    if grid_lmax < 2 * max(lmax_field, lmax_out):
        raise ValueError("grid_lmax should be >= 2*max(lmax_field, lmax_out)")

    mu_dry, K_dry = float(LAYER_MU_CRUST), float(K_CRUST)
    mu_wet, K_wet = mu_ratio * mu_dry, K_ratio * K_dry
    t0 = float(MARS["crust_thickness"])

    if f_h == 0.0:
        model = build_mars_model(mu_scale=mu_scale)
        return PointwiseHydratedCrust(f_h, law, mu_dry, K_dry, {}, {}, 0.0, model)

    dt = crustal_thickness_variation(lmax=lmax_field)
    clm, slm = _coeff_dict_to_arrays(dt, grid_lmax)
    _, _, dt_grid = stokes_to_grid(clm, slm, grid_lmax)

    f_loc = f_h * (1.0 + dt_grid / t0)
    clipped = (f_loc < 0.0) | (f_loc > 1.0)
    f_loc = np.clip(f_loc, 0.0, 1.0)

    mu_grid = mix_pointwise(f_loc, mu_dry, mu_wet, law)
    K_grid = mix_pointwise(f_loc, K_dry, K_wet, law)

    # quadrature-weighted clipped-area fraction (same weights as analysis)
    from .matlab_sph import _dh_weights

    w = _dh_weights(grid_lmax)[:, None]
    clip_fraction = float((clipped * w).sum() / (np.ones_like(f_loc) * w).sum())

    mu_c, mu_s = _remove_matlab_half_cell_phase(*grid_to_stokes(mu_grid, grid_lmax), grid_lmax)
    K_c, K_s = _remove_matlab_half_cell_phase(*grid_to_stokes(K_grid, grid_lmax), grid_lmax)
    mu_bar, Ks_bar = float(mu_c[0, 0]), float(K_c[0, 0])

    real_dmu = {
        nm: a / mu_bar for nm, a in _arrays_to_coeff_dict(mu_c, mu_s, lmax_out).items()
    }
    real_dK = {
        nm: a / Ks_bar for nm, a in _arrays_to_coeff_dict(K_c, K_s, lmax_out).items()
    }
    mu_entries = _real_sh_to_complex_mu_variable(real_dmu)
    K_entries = _real_sh_to_complex_mu_variable(real_dK)

    base = build_mars_model(mu_scale=mu_scale)
    model = base._replace(
        mu0=base.mu0.at[CRUST_LAYER_INDEX].set(mu_bar),
        Ks0=base.Ks0.at[CRUST_LAYER_INDEX].set(Ks_bar),
    )
    return PointwiseHydratedCrust(
        f_h=f_h, law=law, mu_bar=mu_bar, Ks_bar=Ks_bar,
        mu_variable={CRUST_LAYER_INDEX: mu_entries} if mu_entries else {},
        K_variable={CRUST_LAYER_INDEX: K_entries} if K_entries else {},
        clip_fraction=clip_fraction, model=model,
    )
