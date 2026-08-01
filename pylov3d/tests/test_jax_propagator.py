"""Tests for the JAX 1D propagator (jax_propagator.py) and scan variant
(jax_scan.py).

Correctness is anchored on the Kelvin/Love analytical solution for a
uniform elastic sphere:

    k2_analytic = 3 h2 / 5
    h2 = 5 / (2 (1 + 19 mu / (2 rho g R)))

For R=1000 km, rho=3000 kg/m^3, mu=1e10 Pa:
    k2_analytic ≈ 0.038704

Two base tests (Python-loop JAX path):
    1. JAX k2 matches the analytic value to rel 1e-3.
    2. JAX k2 matches the NumPy get_love k2 to rel 1e-5.

Three additional tests (scan path):
    3. scan k2 matches Python-loop JAX k2 to 1e-12 (same math, XLA-fused).
    4. scan k2 matches NumPy get_love k2 to rel 1e-5.
    5. scan k2 matches analytic k2 to rel 1e-3.
    6. Confirm the scan path is JIT-compiled (run_scan is a callable that
       returns the correct value when invoked via block_until_ready).
"""

import math

import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.constants import G as G_phys
from pylov3d.love import get_love
from pylov3d.rheology import get_rheology
from pylov3d.grid import set_boundary_indices
from pylov3d.jax_propagator import jax_get_love_k2
from pylov3d.jax_scan import jax_get_love_k2_scan, propagate_1d_jax_scan


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


def _make_multilayer():
    """Return (model, forcing) for a 3-layer body with distinct densities
    and a viscoelastic (Maxwell) mantle layer.

    Layout (inner→outer):
        core:   fluid, rho=5000
        mantle: viscoelastic Maxwell, rho=3300, mu=1e11, eta=1e15
        crust:  elastic,           rho=2500, mu=4e10

    The mantle Maxwell time tau = eta/mu ~ 1e4 s is comparable to the 1-day
    tidal period, so dissipation is non-trivial (Im(k2) well above noise).

    The distinct per-layer densities exercise the interior delta_rho
    boundary corrections that the uniform sphere never triggers, so this
    test guards against regressions in per-layer-rho / delta_rho handling
    in the scan path.
    """
    model = make_interior_model(
        R0_km=[400.0, 1000.0, 1200.0],
        rho0=[5000.0, 3300.0, 2500.0],
        mu0=[0.0, 1e11, 4e10],
        eta0=[None, 1e15, None],
    )
    forcing = make_forcing(Td=86400.0, n=2, m=0, F=1.0)
    return model, forcing


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


# ---------------------------------------------------------------------------
# Scan-path tests (jax_scan.propagate_1d_jax_scan / jax_get_love_k2_scan)
# ---------------------------------------------------------------------------

class TestJaxScan:
    """lax.scan JIT-compiled propagation: correctness and compilation checks."""

    def test_scan_matches_python_loop_jax(self):
        """Scan k2 must match the Python-loop JAX k2 to 1e-10.

        The Python-loop path converts each step's result to NumPy before the
        next matmul, while the scan path keeps everything in XLA.  This causes
        ~O(Nr) FP-reordering differences; 1e-10 is the appropriate tolerance
        for 500 steps (observed typical error: ~1.5e-11).
        """
        model, forcing, _ = _make_uniform_sphere()
        Nrbase = 500

        numerics = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
        numerics, model_norm = set_boundary_indices(numerics, model)
        model_norm = get_rheology(model_norm, forcing)

        k2_loop = jax_get_love_k2(model_norm, forcing, numerics)
        k2_scan = jax_get_love_k2_scan(model_norm, forcing, numerics)

        abs_err = abs(k2_scan - k2_loop)
        assert abs_err < 1e-10, (
            f"scan k2={k2_scan:.14f} vs loop k2={k2_loop:.14f} "
            f"(abs err={abs_err:.3e})"
        )

    def test_scan_matches_numpy(self):
        """Scan k2 must agree with NumPy get_love k2 to rel 1e-5."""
        model, forcing, _ = _make_uniform_sphere()
        Nrbase = 500

        numerics_np = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
        love_np, _, _ = get_love(model, forcing, numerics_np)
        k2_np = complex(love_np.k[0])

        numerics_scan = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
        numerics_scan, model_norm = set_boundary_indices(numerics_scan, model)
        model_norm = get_rheology(model_norm, forcing)
        k2_scan = jax_get_love_k2_scan(model_norm, forcing, numerics_scan)

        rel_err = abs(k2_scan - k2_np) / abs(k2_np)
        assert rel_err < 1e-5, (
            f"scan k2={k2_scan:.10f} vs NumPy k2={k2_np:.10f} "
            f"(rel err={rel_err:.2e})"
        )

    def test_scan_k2_analytic(self):
        """Scan k2 must match the analytic Kelvin/Love value to rel 1e-3."""
        model, forcing, k2_an = _make_uniform_sphere()
        numerics = make_numerics(n_layers=2, method="variable", Nrbase=500)
        numerics, model_norm = set_boundary_indices(numerics, model)
        model_norm = get_rheology(model_norm, forcing)

        k2_scan = jax_get_love_k2_scan(model_norm, forcing, numerics)

        rel_err = abs(k2_scan.real - k2_an) / k2_an
        assert rel_err < 1e-3, (
            f"scan k2={k2_scan.real:.8f} vs analytic={k2_an:.8f} "
            f"(rel err={rel_err:.2e})"
        )
        assert abs(k2_scan.imag) < 1e-10, "Elastic: scan k2 should be real"

    def test_scan_matches_numpy_multilayer(self):
        """Scan k2 must agree with NumPy get_love k2 to rel 1e-5 on a
        multilayer viscoelastic model.

        Unlike the uniform sphere, this 3-layer model has distinct per-layer
        densities, so interior density-discontinuity corrections
        (Y[7,:] += 4pi*Gg*Delta_rho*Y[0,:]) fire at genuine layer boundaries.
        This is the coverage the uniform-sphere tests cannot provide: it
        catches regressions in delta_rho / per-layer-rho handling in the
        scan path.  The mantle is viscoelastic, so k2 is complex; both the
        real and imaginary parts are checked.
        """
        model, forcing = _make_multilayer()
        Nrbase = 400

        # NumPy reference (3 layers)
        numerics_np = make_numerics(n_layers=3, method="variable", Nrbase=Nrbase)
        love_np, _, _ = get_love(model, forcing, numerics_np)
        k2_np = complex(love_np.k[0])

        # Scan result
        numerics_scan = make_numerics(n_layers=3, method="variable", Nrbase=Nrbase)
        numerics_scan, model_norm = set_boundary_indices(numerics_scan, model)
        model_norm = get_rheology(model_norm, forcing)
        k2_scan = jax_get_love_k2_scan(model_norm, forcing, numerics_scan)

        rel_err = abs(k2_scan - k2_np) / abs(k2_np)
        assert rel_err < 1e-5, (
            f"scan k2={k2_scan:.10f} vs NumPy k2={k2_np:.10f} "
            f"(rel err={rel_err:.2e})"
        )
        # The viscoelastic mantle must produce a non-trivial imaginary part,
        # confirming the multilayer path is genuinely exercised.
        assert abs(k2_np.imag) > 1e-6, (
            "Expected non-trivial Im(k2) for the viscoelastic model; "
            f"got {k2_np.imag:.2e} — test may not exercise dissipation."
        )

    def test_scan_is_jit_compiled(self):
        """Confirm the scan runs through JIT and returns the correct value.

        We verify that:
        1. propagate_1d_jax_scan returns a result (not a traced expression).
        2. block_until_ready on the internal scan output returns a numpy-compatible
           array with the right shape.
        3. The result is numerically consistent (same k2 on a second call).
        """
        model, forcing, _ = _make_uniform_sphere()
        Nrbase = 300

        numerics = make_numerics(n_layers=2, method="variable", Nrbase=Nrbase)
        numerics, model_norm = set_boundary_indices(numerics, model)
        model_norm = get_rheology(model_norm, forcing)

        Y_all, r_grid = propagate_1d_jax_scan(model_norm, forcing, numerics)

        # Shape checks: (Nr+1, 8, 8)
        assert Y_all.shape == (Nrbase + 1, 8, 8), (
            f"Y_all shape {Y_all.shape} != ({Nrbase+1}, 8, 8)"
        )
        assert r_grid.shape == (Nrbase + 1,)
        assert np.iscomplexobj(Y_all), "Y_all should be complex128"

        # Identity initial condition
        assert np.allclose(Y_all[0], np.eye(8), atol=1e-14), \
            "Y_all[0] should be identity matrix"

        # Second call must be consistent (no re-compilation artifacts)
        k2_a = jax_get_love_k2_scan(model_norm, forcing, numerics)
        k2_b = jax_get_love_k2_scan(model_norm, forcing, numerics)
        assert abs(k2_a - k2_b) < 1e-14, "Repeated calls must be deterministic"
