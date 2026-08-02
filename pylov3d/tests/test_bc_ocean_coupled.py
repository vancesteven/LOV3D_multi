"""Tests for coupled subsurface-ocean boundary-condition assembly."""

import numpy as np
import pytest

from pylov3d.bc_ocean_coupled import assemble_bc_ocean_coupled
from pylov3d.boundary_conditions import assemble_bc_ocean
from pylov3d.types import make_forcing


SCALARS = (
    1.4,   # gc
    0.35,  # Rc
    2.1,   # rho2
    0.8,   # rhoK_surface
    0.05,  # Gg
    1.9,   # rho1
    1.2,   # gO
    1.1,   # gI
    0.9,   # rhoO
    2.3,   # rho_below_ocean
    0.7,   # rho_above_ocean
)


def _forcing(n, m):
    return make_forcing(Td=86400.0, n=n, m=m, F=1.0)


def _random_matrices(N, seed):
    """Return four fixed-seed, complex, well-conditioned 8N matrices."""
    rng = np.random.default_rng(seed)
    size = 8 * N
    result = []
    for _ in range(4):
        # Dense random (not near-identity): a sharper discriminator for
        # row/column mix-ups and magnitude-sensitive errors.
        dense = rng.normal(size=(size, size)) + 1j * rng.normal(
            size=(size, size),
        )
        result.append(np.eye(size) + dense)
    return result


@pytest.mark.parametrize(
    ("n", "m", "forcing"),
    [
        (2, 0, _forcing(2, 0)),
        (3, 1, _forcing(3, 1)),
        (3, 1, [_forcing(2, 0), _forcing(3, 1)]),
        # Low degrees exercise every n==0 / n==1 / n<2 special branch
        # against the verified 1D reference (BC3/4/6/7/8/10/11/21/22).
        (0, 0, _forcing(2, 0)),
        (1, 0, _forcing(2, 0)),
        (1, 1, _forcing(1, 1)),
    ],
)
def test_single_mode_exact_reduction(n, m, forcing):
    """N=1 coupled assembly is entry-for-entry equal to the 1D assembler."""
    matrices = _random_matrices(1, seed=6100 + n)
    expected_B, expected_B2 = assemble_bc_ocean(
        *matrices, n, m, *SCALARS, forcing,
    )
    actual_B, actual_B2 = assemble_bc_ocean_coupled(
        *matrices, np.array([n]), np.array([m]), *SCALARS, forcing,
    )
    assert np.max(np.abs(actual_B - expected_B)) < 1e-14
    assert np.max(np.abs(actual_B2 - expected_B2)) < 1e-14


def test_multimode_structure_and_separation():
    """Each 24-row block reads only its mode's grouped state rows."""
    n_s = np.array([2, 4])
    m_s = np.array([0, 0])
    matrices = _random_matrices(2, seed=6200)
    B, B2 = assemble_bc_ocean_coupled(
        *matrices, n_s, m_s, *SCALARS, _forcing(2, 0),
    )
    assert B.shape == (48, 48)
    assert B2.shape == (48,)
    np.testing.assert_array_equal(np.flatnonzero(B2), np.array([7]))
    assert B2[7] == 5

    N, N3, N6, N8 = 2, 6, 12, 16
    for k in range(N):
        components = np.array([
            3 * k, 3 * k + 1, 3 * k + 2,
            N3 + 3 * k, N3 + 3 * k + 1, N3 + 3 * k + 2,
            N6 + 2 * k, N6 + 2 * k + 1,
        ])
        masked = []
        for matrix in matrices:
            selected = np.zeros_like(matrix)
            selected[components, :] = matrix[components, :]
            masked.append(selected)
        B_masked, _ = assemble_bc_ocean_coupled(
            *masked, n_s, m_s, *SCALARS, _forcing(2, 0),
        )
        rows = slice(24 * k, 24 * k + 24)
        np.testing.assert_array_equal(B_masked[rows], B[rows])

        ocean_components = components[:6]
        for offset, component in enumerate(ocean_components):
            row = B[24 * k + 13 + offset]
            nonzero = np.flatnonzero(row)
            np.testing.assert_array_equal(nonzero, np.array([N8 + component]))
            assert row[N8 + component] == 1.0


def test_degree_one_is_finite():
    """Degree-1 toroidal special handling introduces no NaN or infinity."""
    matrices = _random_matrices(2, seed=6300)
    B, B2 = assemble_bc_ocean_coupled(
        *matrices,
        np.array([1, 2]),
        np.array([0, 0]),
        *SCALARS,
        _forcing(2, 0),
    )
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(B2))
