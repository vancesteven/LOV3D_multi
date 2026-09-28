# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Berne et al. 2026 assumed 1D reference interior model for Mars (item B2).

Extended Data Table 2 of Berne et al. 2026 ("Solar tidal tomography of
Mars", the LOV3D-based annual degree-3 study) defines the 1D reference
model their lateral inversion perturbs. The table is image-only in the
paper and absent from the Zenodo archives; the values below were
transcribed on 2026-09-28 from the page-20 table image of the paper PDF
(caption: "Assumed 1D reference interior model for Mars. ... The model
is constrained by Mars's degree-2 tidal deformation, moment of inertia,
mean density and seismic wave arrival times.").

======================  ==================  ===================  ====================
Radius (km)             Density (kg/m^3)    Bulk modulus (GPa)   Shear modulus (GPa)
======================  ==================  ===================  ====================
0 -- 1830 (liquid core) 6200                142.8                0.0
1830 -- 3340 (mantle)   3600                130.2                79.5
3340 -- 3390 (crust)    3300                104.3                61.0
======================  ==================  ===================  ====================

Derived bulk properties of the transcribed model (analytic, pinned in
``test_berne2026.py`` as transcription guards):

* mean density 3995.92 kg/m^3 (observed 3933.96: the model is 1.6% heavy);
* mean moment factor I/MR^2 = 0.370148 (observed mean 0.3631: 1.9% high);
* elastic k2 = 0.176316 (pylov3d, 3-layer 'variable' grid), within 1.2
  sigma of the InSight-era 0.169 +/- 0.006 used in :mod:`pylov3d.mars`.

The mass/MoI misfits are a property of Berne's deliberately coarse
3-layer model, not of the transcription; the caption's "constrained by"
is loose. Treat this model as *their* reference for parity work (B2),
not as a pylov3d Mars model — :func:`pylov3d.mars.build_mars_model`
remains the fitted one.
"""

from __future__ import annotations

from .types import InteriorModel, make_interior_model

#: (outer radius [km], density [kg/m^3], bulk modulus [Pa], shear modulus [Pa])
BERNE2026_ED_TABLE2 = (
    (1830.0, 6200.0, 142.8e9, 0.0),
    (3340.0, 3600.0, 130.2e9, 79.5e9),
    (3390.0, 3300.0, 104.3e9, 61.0e9),
)

#: Mars heliocentric orbital period [s]: the annual tidal forcing period
#: used throughout Berne et al. 2026 (686.98 d, as in the Zenodo scripts).
BERNE2026_ANNUAL_TD = 686.98 * 86400.0


def build_berne2026_reference_model() -> InteriorModel:
    """The Extended Data Table 2 model as a pylov3d ``InteriorModel`` (elastic)."""
    return make_interior_model(
        R0_km=[r for r, _, _, _ in BERNE2026_ED_TABLE2],
        rho0=[rho for _, rho, _, _ in BERNE2026_ED_TABLE2],
        mu0=[mu for _, _, _, mu in BERNE2026_ED_TABLE2],
        Ks0=[K for _, _, K, _ in BERNE2026_ED_TABLE2],
    )


# ---------------------------------------------------------------------------
# Zenodo example (doi:10.5281/zenodo.20823473) — cross-solver parity anchor
# ---------------------------------------------------------------------------

#: Non-dimensional 2-layer "Mars-like" model of the archive's
#: ``Scripts/run_forward_shear.m`` / ``example.m``.
ZENODO_R_RATIO = 0.54
ZENODO_RHO_RATIO = 1.57
ZENODO_KS_ND = 3.0
ZENODO_MU_EFF = 1.72


def build_zenodo_example_model(
    R_m: float = 3390e3, rho_mantle: float = 3000.0
) -> InteriorModel:
    """Dimensional realization of the archive's non-dimensional model.

    Love numbers of an elastic self-gravitating body depend only on the
    non-dimensional ratios (r_core/R, rho_core/rho_mantle, Ks/mu, and
    mu_eff = mu_mantle/(rho_av g_s R), the meaning of the archive's
    ``Gg = 3/(4 pi mu_eff rho_av^2)``), so any (R, rho_mantle) scale
    reproduces the MATLAB results exactly.
    """
    import math

    from .constants import G

    rho_av = rho_mantle * (
        ZENODO_RHO_RATIO * ZENODO_R_RATIO**3 + (1 - ZENODO_R_RATIO**3)
    )
    g_s = (4 * math.pi / 3) * G * rho_av * R_m
    mu = ZENODO_MU_EFF * rho_av * g_s * R_m
    return make_interior_model(
        R0_km=[ZENODO_R_RATIO * R_m / 1e3, R_m / 1e3],
        rho0=[ZENODO_RHO_RATIO * rho_mantle, rho_mantle],
        mu0=[0.0, mu],
        Ks0=[ZENODO_KS_ND * mu, ZENODO_KS_ND * mu],
    )


def read_zenodo_shear_coefficients(path) -> list[tuple[int, int, float]]:
    """Parse ``shear_modulus_coefficients.txt``: (degree, order, percent)."""
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            n, m, p = line.split()
            rows.append((int(n), int(m), float(p)))
    return rows


def zenodo_mu_variable(
    rows, *, convention: str = "script", grid_lmax: int = 59
) -> dict[int, list[tuple[int, int, complex]]]:
    """Percent-peak-to-peak real-SH rows -> complex ``mu_variable`` (layer 1).

    ``convention='script'`` is ``run_forward_shear.m`` lines ~103-112, the
    map that demonstrably produced the archive's shipped
    ``k2_responses.txt`` (pylov3d matches it to 1.0e-12 worst relative over
    365 modes x 5 forcings; the ``'canonical'`` ``get_rheology.m`` /
    :func:`pylov3d.mars_lateral._real_sh_to_complex_mu_variable` map
    misses by 7.1e-2). The two differ exactly by ``S_nm -> (-1)^m S_nm``
    on sine terms — an input-interpretation difference, not a solver one.
    ``grid_lmax=59`` matches the archive's ``Y.lmax = 2*30 - 1`` grid for
    the peak-to-peak normalization ``Delta_nm`` (sign-invariant, so the
    port's internal ``(-1)^m`` synthesis phase drops out).
    """
    import numpy as np

    from .matlab_sph import stokes_to_grid

    if convention not in ("script", "canonical"):
        raise ValueError("convention must be 'script' or 'canonical'")
    delta_cache: dict[tuple[int, int], float] = {}

    def delta(n: int, M: int) -> float:
        if (n, M) not in delta_cache:
            clm = np.zeros((grid_lmax + 1, grid_lmax + 1))
            slm = np.zeros_like(clm)
            clm[n, M] = 1.0
            _, _, z = stokes_to_grid(clm, slm, grid_lmax)
            delta_cache[(n, M)] = float(z.max() - z.min())
        return delta_cache[(n, M)]

    s2 = 2**0.5 / 2
    c: dict[tuple[int, int], complex] = {}

    def add(n: int, m: int, v: complex) -> None:
        c[(n, m)] = c.get((n, m), 0j) + v

    for n, m, p in rows:
        A = (p / 100.0) / delta(n, abs(m))
        if m == 0:
            add(n, 0, A)
        elif m > 0:
            add(n, m, s2 * A)
            add(n, -m, ((-1) ** m) * s2 * A)
        else:
            M = -m
            if convention == "script":
                add(n, M, -1j * ((-1) ** M) * s2 * A)
                add(n, -M, 1j * s2 * A)
            else:
                add(n, M, -1j * s2 * A)
                add(n, -M, 1j * ((-1) ** M) * s2 * A)
    return {1: [(n, m, v) for (n, m), v in sorted(c.items())]}
