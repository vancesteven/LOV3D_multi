import numpy as np
import pytest

from pylov3d.mars_alteration_state import AlterationEndmembers
from pylov3d.mars_common_state import build_common_alteration_forward_inputs
from pylov3d.matlab_sph import stokes_to_grid


ENDMEMBERS = AlterationEndmembers(
    rho_dry_kg_m3=3200.0,
    rho_hydrated_kg_m3=2600.0,
    mu_dry_Pa=60e9,
    mu_hydrated_Pa=20e9,
    K_dry_Pa=100e9,
    K_hydrated_Pa=50e9,
)


def test_uniform_alteration_changes_mean_properties_but_no_lateral_gravity():
    lmax = 6
    nz = 2
    fh = np.full((2 * lmax, 4 * lmax, nz), 0.5)
    r = np.array([3.34e6, 3.365e6, 3.39e6])
    out = build_common_alteration_forward_inputs(
        fh,
        0.4,
        r,
        lmax,
        ENDMEMBERS,
        mixing_law="hill",
    )
    # Degree zero mass change is intentionally outside the lateral gravity arrays.
    np.testing.assert_allclose(out.gravity_q_cos, 0.0, rtol=0, atol=2e-18)
    np.testing.assert_allclose(out.gravity_q_sin, 0.0, rtol=0, atol=2e-18)
    assert out.tidal_layer.mean_mu_Pa < ENDMEMBERS.mu_dry_Pa
    assert out.tidal_layer.mean_K_Pa < ENDMEMBERS.K_dry_Pa
    assert out.tidal_layer.mu_variable == {}
    assert out.tidal_layer.K_variable == {}


def test_same_nonuniform_alteration_field_drives_gravity_and_rigidity_modes():
    lmax = 8
    c = np.zeros((lmax + 1, lmax + 1))
    s = np.zeros_like(c)
    c[0, 0] = 0.5
    c[2, 0] = 0.10
    _, _, fh2d = stokes_to_grid(c, s, lmax)
    assert fh2d.min() >= 0.0 and fh2d.max() <= 1.0
    fh = fh2d[:, :, None]
    r = np.array([3.34e6, 3.39e6])

    out = build_common_alteration_forward_inputs(
        fh,
        0.5,
        r,
        lmax,
        ENDMEMBERS,
        mixing_law="voigt",
    )
    assert abs(out.gravity_q_cos[2, 0]) > 0
    mu_modes = {tuple(row[:2]) for row in out.tidal_layer.mu_variable[3]}
    K_modes = {tuple(row[:2]) for row in out.tidal_layer.K_variable[3]}
    assert (2, 0) in mu_modes
    assert (2, 0) in K_modes


def test_invalid_hydration_field_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        build_common_alteration_forward_inputs(
            np.zeros((3, 4, 1)),
            1.0,
            np.array([3.3e6, 3.39e6]),
            2,
            ENDMEMBERS,
        )
