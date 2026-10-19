import math

import pytest

from pylov3d.mars_gravity_normalization import (
    GMM3_REFERENCE_RADIUS_M,
    PYLOV3D_MARS_RADIUS_M,
    SQRT_4PI,
    gmm3_conservative_snr,
    gmm3_normalized_to_orthonormal,
    orthonormal_to_gmm3_normalized,
)


def test_same_radius_bridge_is_sqrt_4pi():
    q = 2.5e-7
    got = orthonormal_to_gmm3_normalized(q, 8, source_radius_m=1.0, gmm3_radius_m=1.0)
    assert got == pytest.approx(q / math.sqrt(4.0 * math.pi), rel=1e-15)


def test_radius_rescaling_matches_external_potential():
    q = 1.2e-6
    degree = 21
    c = orthonormal_to_gmm3_normalized(q, degree)
    # After removing the basis-normalization factor, both coefficient/radius
    # pairs must produce the same R^l * coefficient factor in the potential.
    left = q * PYLOV3D_MARS_RADIUS_M**degree
    right = c * SQRT_4PI * GMM3_REFERENCE_RADIUS_M**degree
    assert right == pytest.approx(left, rel=3e-15)


def test_roundtrip():
    for degree in (0, 5, 21, 43, 85):
        q = (-1.0) ** degree * 3.7e-8
        c = orthonormal_to_gmm3_normalized(q, degree)
        back = gmm3_normalized_to_orthonormal(c, degree)
        assert back == pytest.approx(q, rel=2e-15)


def test_reference_radius_factor_is_small_but_degree_dependent():
    q = 1.0
    c5 = orthonormal_to_gmm3_normalized(q, 5) * SQRT_4PI
    c85 = orthonormal_to_gmm3_normalized(q, 85) * SQRT_4PI
    assert 0.98 < c5 < 1.0
    assert 0.8 < c85 < c5


def test_conservative_snr():
    assert gmm3_conservative_snr(6e-9, 1e-9) == pytest.approx(2.0)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        orthonormal_to_gmm3_normalized(1.0, -1)
    with pytest.raises(ValueError):
        gmm3_conservative_snr(1.0, 0.0)
