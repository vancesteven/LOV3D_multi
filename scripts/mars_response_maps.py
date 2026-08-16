#!/usr/bin/env python
# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Mars lateral tidal-response spatial maps (TASK-041 part 1).

Everything TASK-016/TASK-026 produced for Mars 3D was spectral -- lists
of k_nm coefficients. This synthesizes the actual spatial field the
off-forcing spectrum implies: the amplitude of the periodic tidal
gravity/displacement anomaly over Mars's surface, in measurable units,
so the spatial-detectability question (where does the signal peak --
Tharsis? the dichotomy boundary? Hellas?) can be answered directly rather
than read off a list of coefficients.

Scope caveat (TASK-026, carried verbatim, not re-litigated here): this
response is computed to a **unit (2,0) forcing**. Mars's real solar tidal
potential has power at m=0 (zonal, seasonal), m=1 (diurnal), and m=2
(sectoral, semidiurnal -- the physically dominant component, and the one
Konopliv, Park & Folkner (2016) actually measure k2=0.169 at) -- three
distinct frequencies, not one. See
``pylov3d/mars_detectability.py`` module docstring, section "2. Mars's
real solar tide, forcing-order scope, and what this cannot resolve", for
the full argument: the coupled off-diagonal spectrum is *not*
forcing-order invariant (elastic k_n itself is, but which (n, m) modes a
given forcing order excites is set by the additive coupling selection
rule ``m_new = m0 + m1``), so these maps should be read as an
order-of-magnitude spatial pattern for the off-(2,0) response as a class,
not a mode-by-mode-exact prediction of the true semidiurnal-frequency
response.

Physical scaling reuses (does not re-derive) ``pylov3d.mars_detectability``:
``S = solar_tide_amplitude_parameter(GM_SUN, MARS["GM"], MARS["R"],
MARS_SEMIMAJOR_AXIS_M) * peak_legendre_factor(2, 0)`` -- the dimensionless
amplitude of the (2,0) solar tide potential coefficient (in units of
g*R), evaluated at the obliquity-constrained sub-solar peak, exactly the
convention :func:`pylov3d.mars_detectability.required_stokes_amplitude`
uses (``xi * p``, no basis-normalization correction -- that correction
only applies when converting a *response mode's* k_nm into a required
Stokes-coefficient precision for a *different* assumed forcing geometry,
not here).

Only the lateral part of the response is mapped: every off-(2,0) mode
as-is, plus the forcing mode (2,0) itself with the *uniform* (spherically
symmetric) response subtracted -- ``delta_k2 = k20 - k2_uniform``,
``delta_h2 = h20 - h_uniform`` -- so the map shows the anomaly a
laterally uniform Mars would not show, not the (much larger) uniform
tidal response itself.

Observables (amplitude of the periodic signal, |complex field|, via
:func:`pylov3d.mars_lateral.complex_sh_synthesis` -- the solver's own
validated complex-SH convention, never a hand-rolled basis conversion):

1. Gravity anomaly: ``(n+1) * k_nm`` per mode (the (n+1)/R
   surface-gravity-from-potential factor is degree-dependent, so it must
   be applied to each mode's coefficient *before* synthesis -- it cannot
   be factored out and applied to the synthesized field), then
   ``field * S * g_mars`` [m/s^2], ``g_mars = GM_mars/R^2``; reported in
   uGal (1 uGal = 1e-8 m/s^2).
2. Radial displacement: ``h_nm`` per mode, then ``field * S * R_mars``
   [m]; reported in mm.

Usage
-----
    venvLOV3Dconv/bin/python scripts/mars_response_maps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.mapping import latlon_grid
from pylov3d.mars import MARS
from pylov3d.mars_detectability import (
    GM_SUN,
    MARS_SEMIMAJOR_AXIS_M,
    peak_legendre_factor,
    solar_tide_amplitude_parameter,
)
from pylov3d.mars_lateral import complex_sh_synthesis

SPECTRUM_PATH = REPO_ROOT / "docs" / "figures" / "proposal" / "mars_lateral_spectrum.npz"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "proposal" / "mars_response_maps.npz"

NLAT = 181
NLON = 360

# Named sites, (lat deg N, lon deg East). Dichotomy boundary point is
# explicitly representative (the boundary is a curve, not a point).
SITES: dict[str, tuple[float, float]] = {
    "Tharsis": (0.0, 250.0),
    "Dichotomy boundary (representative)": (30.0, 340.0),
    "Hellas": (-42.0, 70.0),
    "InSight": (4.5, 135.6),
}


def _lon_east_to_grid(lon_east: float) -> float:
    """0-360 East -> the grid's [-180, 180) convention."""
    return ((lon_east + 180.0) % 360.0) - 180.0


def _lon_grid_to_east(lon_grid: float) -> float:
    """[-180, 180) -> 0-360 East, for reporting."""
    return lon_grid % 360.0


def _nearest_index(grid: np.ndarray, value: float, period: float | None = None) -> int:
    diff = grid - value
    if period is not None:
        diff = (diff + period / 2.0) % period - period / 2.0
    return int(np.argmin(np.abs(diff)))


def _build_entries(
    n: np.ndarray, m: np.ndarray, amp: np.ndarray,
    forcing_n: int, forcing_m: int, forcing_shift: complex,
    weight_fn,
) -> list[tuple[int, int, complex]]:
    """Off-forcing modes as-is (weighted), forcing mode replaced by its
    lateral shift (weighted)."""
    entries: list[tuple[int, int, complex]] = []
    for ni, mi, ai in zip(n, m, amp):
        ni, mi = int(ni), int(mi)
        if ni == forcing_n and mi == forcing_m:
            entries.append((ni, mi, weight_fn(ni, forcing_shift)))
        else:
            entries.append((ni, mi, weight_fn(ni, ai)))
    return entries


def _area_weighted_mean_sq(field: np.ndarray, lat_deg: np.ndarray) -> float:
    """Spherical (cos-latitude weighted) mean of |field|^2 over the grid."""
    w = np.cos(np.radians(lat_deg))
    w2 = np.broadcast_to(w[:, None], field.shape)
    return float(np.sum(w2 * np.abs(field) ** 2) / np.sum(w2))


def variance_by_degree(
    n: np.ndarray, m: np.ndarray, amp: np.ndarray,
    forcing_n: int, forcing_m: int, forcing_shift: complex,
    weight_fn, lat_deg: np.ndarray, lon_deg: np.ndarray,
) -> dict[int, float]:
    """Fraction of total (cos-lat-weighted) map variance contributed by
    each spherical-harmonic degree n, via orthogonality of distinct-degree
    spherical harmonics over the sphere: mean(|sum_n field_n|^2) =
    sum_n mean(|field_n|^2) exactly (cross terms integrate to zero), so
    the per-degree variance can be computed by synthesizing each degree's
    modes alone rather than by any basis-normalization bookkeeping."""
    degrees = sorted(set(int(ni) for ni in n))
    per_degree_var: dict[int, float] = {}
    for deg in degrees:
        mask = (n.astype(int) == deg)
        entries = _build_entries(
            n[mask], m[mask], amp[mask], forcing_n, forcing_m, forcing_shift, weight_fn,
        )
        field_n = complex_sh_synthesis(entries, lat_deg, lon_deg)
        per_degree_var[deg] = _area_weighted_mean_sq(field_n, lat_deg)
    total = sum(per_degree_var.values())
    return {deg: (val / total if total > 0 else float("nan")) for deg, val in per_degree_var.items()}


def main() -> None:
    d = np.load(SPECTRUM_PATH, allow_pickle=True)
    n = d["n"].astype(int)
    m = d["m"].astype(int)
    k = d["k"].astype(complex)
    h = d["h"].astype(complex)
    forcing_n = int(d["forcing_n"])
    forcing_m = int(d["forcing_m"])
    delta_k2 = complex(d["delta_k2"])
    delta_h2 = complex(d["delta_h2"])

    R = MARS["R"]
    GM_mars = MARS["GM"]
    g_mars = GM_mars / R**2

    xi = solar_tide_amplitude_parameter(GM_SUN, GM_mars, R, MARS_SEMIMAJOR_AXIS_M)
    p = peak_legendre_factor(2, 0)
    S = xi * p

    lat, lon = latlon_grid(nlat=NLAT, nlon=NLON)

    # ---- gravity anomaly: (n+1)*k_nm per mode, forcing mode -> shift ----
    grav_entries = _build_entries(
        n, m, k, forcing_n, forcing_m, delta_k2,
        weight_fn=lambda deg, amp: (deg + 1) * amp,
    )
    grav_field = complex_sh_synthesis(grav_entries, lat, lon)
    grav_amp_m_s2 = np.abs(grav_field) * S * g_mars
    grav_amp_ugal = grav_amp_m_s2 / 1e-8

    # ---- radial displacement: h_nm per mode, forcing mode -> shift ----
    disp_entries = _build_entries(
        n, m, h, forcing_n, forcing_m, delta_h2,
        weight_fn=lambda deg, amp: amp,
    )
    disp_field = complex_sh_synthesis(disp_entries, lat, lon)
    disp_amp_m = np.abs(disp_field) * S * R
    disp_amp_mm = disp_amp_m * 1e3

    # ---- peaks ----
    grav_peak_idx = np.unravel_index(np.argmax(grav_amp_ugal), grav_amp_ugal.shape)
    disp_peak_idx = np.unravel_index(np.argmax(disp_amp_mm), disp_amp_mm.shape)
    grav_peak = float(grav_amp_ugal[grav_peak_idx])
    grav_peak_lat = float(lat[grav_peak_idx[0]])
    grav_peak_lon_e = _lon_grid_to_east(float(lon[grav_peak_idx[1]]))
    disp_peak = float(disp_amp_mm[disp_peak_idx])
    disp_peak_lat = float(lat[disp_peak_idx[0]])
    disp_peak_lon_e = _lon_grid_to_east(float(lon[disp_peak_idx[1]]))

    # ---- named sites ----
    site_values = {}
    for name, (site_lat, site_lon_e) in SITES.items():
        site_lon_grid = _lon_east_to_grid(site_lon_e)
        i_lat = _nearest_index(lat, site_lat)
        i_lon = _nearest_index(lon, site_lon_grid, period=360.0)
        site_values[name] = {
            "lat": site_lat, "lon_east": site_lon_e,
            "grid_lat": float(lat[i_lat]), "grid_lon_east": _lon_grid_to_east(float(lon[i_lon])),
            "gravity_ugal": float(grav_amp_ugal[i_lat, i_lon]),
            "displacement_mm": float(disp_amp_mm[i_lat, i_lon]),
        }

    # ---- variance by degree ----
    grav_var_frac = variance_by_degree(
        n, m, k, forcing_n, forcing_m, delta_k2,
        weight_fn=lambda deg, amp: (deg + 1) * amp, lat_deg=lat, lon_deg=lon,
    )
    disp_var_frac = variance_by_degree(
        n, m, h, forcing_n, forcing_m, delta_h2,
        weight_fn=lambda deg, amp: amp, lat_deg=lat, lon_deg=lon,
    )

    # ---- save ----
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    degrees_sorted = sorted(grav_var_frac.keys())
    np.savez(
        DEFAULT_OUTPUT,
        lat=lat, lon=lon,
        gravity_ugal=grav_amp_ugal, displacement_mm=disp_amp_mm,
        gravity_peak_ugal=grav_peak, gravity_peak_lat=grav_peak_lat, gravity_peak_lon_east=grav_peak_lon_e,
        displacement_peak_mm=disp_peak, displacement_peak_lat=disp_peak_lat,
        displacement_peak_lon_east=disp_peak_lon_e,
        S=S, xi=xi, p=p, GM_sun=GM_SUN, GM_mars=GM_mars, R_mars=R,
        mars_sun_distance_m=MARS_SEMIMAJOR_AXIS_M, g_mars=g_mars,
        forcing_n=forcing_n, forcing_m=forcing_m, delta_k2=delta_k2, delta_h2=delta_h2,
        degree=np.asarray(degrees_sorted, dtype=int),
        gravity_variance_fraction_by_degree=np.asarray([grav_var_frac[dgr] for dgr in degrees_sorted]),
        displacement_variance_fraction_by_degree=np.asarray([disp_var_frac[dgr] for dgr in degrees_sorted]),
        site_names=np.asarray(list(SITES.keys())),
        site_lat=np.asarray([SITES[name][0] for name in SITES]),
        site_lon_east=np.asarray([SITES[name][1] for name in SITES]),
        site_gravity_ugal=np.asarray([site_values[name]["gravity_ugal"] for name in SITES]),
        site_displacement_mm=np.asarray([site_values[name]["displacement_mm"] for name in SITES]),
        provenance=(
            "TASK-041 part 1. Synthesized from docs/figures/proposal/"
            "mars_lateral_spectrum.npz (unit (2,0) forcing, N=115 coupled "
            "modes, method='variable', Nrbase=30, perturbation_order=2). "
            "Lateral-only: off-forcing modes as-is, forcing mode (2,0) uses "
            "the lateral shift (delta_k2/delta_h2) only. Gravity: (n+1)*k_nm "
            "per mode before synthesis, field*S*g_mars, uGal. Displacement: "
            "h_nm per mode, field*S*R_mars, mm. Scope caveat: unit (2,0) "
            "forcing only -- see pylov3d/mars_detectability.py module "
            "docstring section 2 (real solar tide has m=0/1/2 at distinct "
            "frequencies; TASK-026 scope, carried verbatim)."
        ),
    )

    # ---- report ----
    print(f"S = xi*p = {xi:.6e} * {p:.6f} = {S:.6e}")
    print(f"g_mars = {g_mars:.6f} m/s^2, R_mars = {R:.1f} m")
    print()
    print(f"gravity anomaly peak: {grav_peak:.4f} uGal at lat={grav_peak_lat:+.2f}, "
          f"lon={grav_peak_lon_e:.2f} E")
    print(f"displacement peak: {disp_peak:.6f} mm at lat={disp_peak_lat:+.2f}, "
          f"lon={disp_peak_lon_e:.2f} E")
    print()
    print("named sites:")
    for name, v in site_values.items():
        print(f"  {name:34s} (lat={v['lat']:+.1f}, lon={v['lon_east']:.1f}E; "
              f"grid lat={v['grid_lat']:+.2f}, lon={v['grid_lon_east']:.2f}E): "
              f"gravity={v['gravity_ugal']:.4f} uGal, displacement={v['displacement_mm']:.6f} mm")
    print()
    print("gravity variance fraction by degree:")
    for dgr in degrees_sorted:
        print(f"  n={dgr}: {grav_var_frac[dgr]:.4%}")
    print("displacement variance fraction by degree:")
    for dgr in degrees_sorted:
        print(f"  n={dgr}: {disp_var_frac[dgr]:.4%}")
    print()
    print(f"saved {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
