from __future__ import annotations

from pylov3d.mars_gmm3 import (
    coefficients_at_degree,
    formal_sigmas_at_degree,
    parse_coefficient,
    parse_header,
)


def test_parse_gmm3_header_public_product():
    line = (
        "3.3960000000000000e+03, 4.2828372854187757e+04, "
        "1.6202815226760665e-05,  120,  120,    1, "
        "0.0000000000000000e+00, 0.0000000000000000e+00"
    )
    h = parse_header(line)
    assert h.reference_radius_km == 3396.0
    assert h.max_degree == 120
    assert h.max_order == 120
    assert h.normalization_state == 1


def test_parse_gmm3_degree2_order0_public_row():
    line = (
        "2,    0,-8.7502113235452894e-04,0.0000000000000000e+00,"
        "1.2523111777280104e-11,0.0000000000000000e+00"
    )
    row = parse_coefficient(line)
    assert row.degree == 2
    assert row.order == 0
    assert row.c < 0.0
    assert row.s == 0.0
    assert row.sigma_c > 0.0
    assert row.sigma_s == 0.0


def test_degree_selection_and_zero_s_sigma_filter():
    rows = [
        parse_coefficient(
            "5,0,-1.7266823981571990e-06,0,8.8093357857947103e-12,0"
        ),
        parse_coefficient(
            "5,1,4.8379878369827899e-07,2.1232209244940802e-06,"
            "7.5954061775403315e-12,7.7190540246310785e-12"
        ),
        parse_coefficient("6,0,1e-6,0,9e-12,0"),
    ]
    d5 = coefficients_at_degree(rows, 5)
    assert len(d5) == 2
    sigmas = formal_sigmas_at_degree(rows, 5)
    assert len(sigmas) == 3
    assert all(x > 0 for x in sigmas)
