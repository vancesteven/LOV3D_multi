import numpy as np
import pytest

from pylov3d.bulk_lateral import (
    bulk_constitutive_amplitude,
    inject_bulk_modulus_variations,
    process_lateral_variations_with_bulk,
)
from pylov3d.propagator import _coupling_A1_A2, build_A1_A2
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.types import Forcing, make_interior_model


def test_factor_three_matches_uniform_constitutive_deltaK_exactly():
    n = 4
    mu = 0.7 + 0.03j
    K = 2.1
    lam = K - (2.0 / 3.0) * mu
    delta_K = 0.17

    A1_0, A2_0 = build_A1_A2(n, mu, lam)
    A1_1, A2_1 = build_A1_A2(n, mu, lam + delta_K)

    Cp = np.zeros(26)
    Cp[0] = 1.0
    A1_c, A2_c = _coupling_A1_A2(
        n,
        bulk_constitutive_amplitude(1.0, delta_K),
        0.0j,
        Cp,
    )

    # Varying K at fixed mu changes only the scalar-stress row. The coupling
    # kernel with Cp[0]=1 must equal that constitutive difference to floating-
    # point roundoff. Keep an absolute-only tolerance because the expected
    # matrix contains exact zeros.
    np.testing.assert_allclose(A1_1 - A1_0, A1_c, rtol=0, atol=5e-15)
    np.testing.assert_allclose(A2_1 - A2_0, A2_c, rtol=0, atol=5e-15)


def _elastic_model_and_forcing():
    model = make_interior_model(
        [1700.0, 3000.0, 3390.0],
        [7000.0, 3500.0, 2900.0],
        [0.0, 70e9, 30e9],
        Ks0=[1e12, 120e9, 70e9],
        eta0=[None, None, None],
    )
    forcing = Forcing(Td=88775.0, n=2, m=2, F=1.0 + 0j)
    return get_rheology(model, forcing), forcing


def test_K_only_mode_survives_even_when_shear_path_is_uniform():
    model, forcing = _elastic_model_and_forcing()
    model0, lateral0 = process_lateral_variations(model, forcing)
    assert np.all(lateral0.uniform[1:model.n_layers])

    frac = 0.08
    lateral = inject_bulk_modulus_variations(
        model0,
        lateral0,
        {2: [(2, 0, frac)]},
    )
    np.testing.assert_array_equal(lateral.variations, [[2, 0]])
    assert np.allclose(lateral.muC_amp, 0.0)
    assert lateral.K_amp[2, 0] == pytest.approx(3.0 * float(model0.Ks[2]) * frac)
    assert not lateral.uniform[2]


def test_bulk_wrapper_preserves_shear_modes_and_unions_K_modes():
    model, forcing = _elastic_model_and_forcing()
    model_out, lateral = process_lateral_variations_with_bulk(
        model,
        forcing,
        mu_variable={2: [(2, 0, 0.05)]},
        K_variable={2: [(3, 0, -0.04)]},
    )
    modes = [tuple(row) for row in lateral.variations.tolist()]
    assert (2, 0) in modes
    assert (3, 0) in modes
    i_mu = modes.index((2, 0))
    i_K = modes.index((3, 0))
    assert lateral.muC_amp[2, i_mu] != 0
    assert lateral.K_amp[2, i_K] == pytest.approx(
        3.0 * float(model_out.Ks[2]) * -0.04
    )


def test_bulk_conjugate_partner_is_added_for_real_field():
    model, forcing = _elastic_model_and_forcing()
    _, lateral = process_lateral_variations_with_bulk(
        model,
        forcing,
        K_variable={2: [(3, 2, 0.02 + 0.01j)]},
    )
    modes = [tuple(row) for row in lateral.variations.tolist()]
    assert (3, 2) in modes
    assert (3, -2) in modes
    ip = modes.index((3, 2))
    im = modes.index((3, -2))
    assert lateral.K_amp[2, im] == pytest.approx(np.conj(lateral.K_amp[2, ip]))
