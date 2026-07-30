"""Tests for the JAX 1D propagator (jax_propagator.py).

Correctness is anchored on the Kelvin/Love analytical solution for a
uniform elastic sphere:

    k2_analytic = 3 h2 / 5
    h2 = 5 / (2 (1 + 19 mu / (2 rho g R)))

For R=1000 km, rho=3000 kg/m^3, mu=1e10 Pa:
    k2_analytic ≈ 0.038704

Two tests:
    1. JAX k2 matches the analytic value to rel 1e-3.
    2. JAX k2 matches the NumPy get_love k2 to rel 1e-5.
"""

import math

import pytest

import jax
jax.config.update("jax_enable_x64", True)

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.constants import G as G_phys
from pylov3d.love import get_love
from pylov3d.rheology import get_rheology
from pylov3d.grid import set_boundary_indices
from pylov3d.jax_propagator import jax_get_love_k2


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_uniform_sphere():
    """Return (model, forcing, params) for a uniform elastic sphere.

    Uses a 10 km fluid core + 1000 km elastic mantle to approximate a
    homogeneous sphere while satisfying the fluid-core BC.
    """
    R_km = 1000.0
    R_m  = R_km * 1e3
    rho  = 3000.0
    mu   = 1e10

    g_surf = G_phys * (4.0 / 3.0) * math.pi * rho * R_m
    h2_an  = 5.0 / (2.0 * (1.0 + 19.0 * mu / (2.0 * rho * g_surf * R_m)))
    k2_an  = 3.0 * h2_an / 5.0

    model = make_interior_model(
        R0_km=[10.0, R_km],
        rho0=[rho, rho],
        mu0=[0.0, mu],
        eta0=[None, None],
    )
    forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
    return model, forcing, k2_an


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJaxPropagator:
    """JAX propagator correctness vs analytic and NumPy references."""

    def test_k2_analytic(self):
        """JAX k2 must match the Kelvin/Love analytic value to rel 1e-3."""
        model, forcing, k2_an = _make_uniform_sphere()
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=500)
        numerics, model_norm = set_boundary_indices(numerics, model)
        model_norm = get_rheology(model_norm, forcing)

        k2_jax = jax_get_love_k2(model_norm, forcing, numerics)

        assert abs(k2_jax.real - k2_an) / k2_an < 1e-3, (
            f"JAX k2={k2_jax.real:.8f} vs analytic k2={k2_an:.8f} "
            f"(rel err={abs(k2_jax.real - k2_an) / k2_an:.2e})"
        )
        assert abs(k2_jax.real) > 0, "k2 should be positive"
        assert abs(k2_jax.imag) < 1e-10, "Elastic: k2 should be real"

    def test_k2_matches_numpy(self):
        """JAX k2 must agree with NumPy get_love k2 to rel 1e-5."""
        model, forcing, _ = _make_uniform_sphere()
        Nrbase = 500

        # NumPy reference
        numerics_np = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
        love_np, _, _ = get_love(model, forcing, numerics_np)
        k2_np = complex(love_np.k[0])

        # JAX result (uses the same Nrbase for direct comparison)
        numerics_jax = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
        numerics_jax, model_norm = set_boundary_indices(numerics_jax, model)
        model_norm = get_rheology(model_norm, forcing)
        k2_jax = jax_get_love_k2(model_norm, forcing, numerics_jax)

        rel_err = abs(k2_jax - k2_np) / abs(k2_np)
        assert rel_err < 1e-5, (
            f"JAX k2={k2_jax:.10f} vs NumPy k2={k2_np:.10f} "
            f"(rel err={rel_err:.2e})"
        )

    def test_k2_positive_real(self):
        """k2 should be real and positive for the elastic sphere."""
        model, forcing, _ = _make_uniform_sphere()
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=300)
        numerics, model_norm = set_boundary_indices(numerics, model)
        model_norm = get_rheology(model_norm, forcing)

        k2_jax = jax_get_love_k2(model_norm, forcing, numerics)

        assert k2_jax.real > 0
        assert abs(k2_jax.imag) < 1e-9

    def test_n_ne_2_raises(self):
        """Non-n=2 forcing should raise NotImplementedError."""
        model, _, _ = _make_uniform_sphere()
        forcing_n3 = make_forcing(Td=86400.0, n=3, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=100)
        numerics, model_norm = set_boundary_indices(numerics, model)
        model_norm = get_rheology(model_norm, forcing_n3)

        with pytest.raises(NotImplementedError):
            jax_get_love_k2(model_norm, forcing_n3, numerics)
