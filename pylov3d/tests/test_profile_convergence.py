import numpy as np
import pytest

from pylov3d.profile_convergence import love_number_reduction_convergence
from pylov3d.profile_io import RadialArtifactShells
from pylov3d.types import make_forcing


def _homogeneous_shells(n=8):
    r = np.linspace(0.4e6, 3.39e6, n)
    return RadialArtifactShells(
        outer_radius_m=r,
        rho_kgm3=np.full(n, 3500.0),
        K_Pa=np.full(n, 120e9),
        mu_Pa=np.full(n, 70e9),
        metadata={},
    )


def test_homogeneous_body_is_invariant_to_shell_reduction():
    forcing = make_forcing(Td=44387.62, n=2, m=0, F=1.0)
    out = love_number_reduction_convergence(
        _homogeneous_shells(),
        [4, 6, 8],
        forcing,
        Nrbase=40,
    )
    assert out.reference_layers == 8
    assert [p.actual_layers for p in out.points] == [4, 6, 8]

    # Internal boundaries between identical material are physically invisible.
    # Any change is therefore numerical integration noise, not reduction error.
    for p in out.points:
        assert p.rel_dk < 1e-4
        assert p.rel_dh < 1e-4
        assert p.rel_dl < 1e-4

    ref = out.points[-1]
    assert ref.abs_dk == pytest.approx(0.0, abs=0.0)
    assert ref.abs_dh == pytest.approx(0.0, abs=0.0)
    assert ref.abs_dl == pytest.approx(0.0, abs=0.0)


def test_bad_layer_requests_are_rejected():
    forcing = make_forcing(Td=44387.62, n=2, m=0, F=1.0)
    shells = _homogeneous_shells()
    with pytest.raises(ValueError, match="at least one"):
        love_number_reduction_convergence(shells, [], forcing)
    with pytest.raises(ValueError, match=">= 2"):
        love_number_reduction_convergence(shells, [1, 4], forcing)
