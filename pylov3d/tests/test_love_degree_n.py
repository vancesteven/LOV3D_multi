# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Degree-n (n = 2, 3, 4) forcing of a 1D body against the analytic
homogeneous incompressible elastic sphere (Berne et al. 2026 item B4).

    k_n = (3 / (2 (n-1))) / (1 + (2n^2 + 4n + 3) mu / (n rho g R))

(Love 1911. For n = 2 it reduces to k2 = (3/2) / (1 + 19 mu/(2 rho g R)),
the form used in test_analytical.py.)

The solver needs a fluid core and a finite bulk modulus. The test body
approximates the homogeneous incompressible limit as follows:

* Fluid core radius 1 km in a 1000 km body, with uniform density. A core
  of 1% R biases k2 by ~5e-6 relative. At 0.1% R the bias falls below the
  solver's noise floor.
* Ks = 1e7 mu (``make_interior_model``'s own incompressible default
  ratio). Measured sweep: Ks/mu = 1e6 gives ~6e-9 relative error, and
  1e7 gives <=1.7e-9. At 1e10 the conditioning degrades to ~3e-8, and at
  1e13 to ~3e-4.

Measured relative errors at the test settings (Nrbase=500, 'variable'):
n=2 1.6e-9, n=3 6.8e-10, n=4 6.3e-10. Asserted: <= 1e-8.
"""

from __future__ import annotations

import math

import pytest

from pylov3d.constants import G
from pylov3d.love import get_love
from pylov3d.types import make_forcing, make_interior_model, make_numerics

R_KM = 1000.0
RHO = 3000.0
MU = 1e10
K_OVER_MU = 1e7
CORE_KM = 1.0


def k_n_analytic(n: int) -> float:
    R = R_KM * 1e3
    g = G * 4.0 / 3.0 * math.pi * RHO * R
    return (3.0 / (2.0 * (n - 1))) / (
        1.0 + (2 * n * n + 4 * n + 3) * MU / (n * RHO * g * R)
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_k_n_homogeneous_incompressible_sphere(n):
    model = make_interior_model(
        R0_km=[CORE_KM, R_KM],
        rho0=[RHO, RHO],
        mu0=[0.0, MU],
        Ks0=[K_OVER_MU * MU, K_OVER_MU * MU],
    )
    numerics = make_numerics(n_layers=2, method="variable", Nrbase=500)
    love, _, _ = get_love(model, make_forcing(86400.0, n, 0, 1.0), numerics)

    assert int(love.n[0]) == n
    k = complex(love.k[0])
    expected = k_n_analytic(n)
    assert abs(k.imag) <= 1e-12
    assert abs(k.real / expected - 1.0) <= 1e-8, (n, k.real, expected)


def test_analytic_n2_matches_kelvin_form():
    """Control: at n=2 the formula equals the k2 = 3 h2 / 5 form of
    test_analytical.py::TestUniformElasticSphere."""
    R = R_KM * 1e3
    g = G * 4.0 / 3.0 * math.pi * RHO * R
    h2 = 5.0 / (2.0 * (1.0 + 19.0 * MU / (2.0 * RHO * g * R)))
    assert k_n_analytic(2) == pytest.approx(3.0 * h2 / 5.0, rel=1e-15)
