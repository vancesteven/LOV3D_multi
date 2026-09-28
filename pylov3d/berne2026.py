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
