# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Cross-solver parity against the Berne et al. 2026 Zenodo example.

The anchor files in ``data/tests/berne_zenodo/`` (CC-BY-4.0,
doi:10.5281/zenodo.20823473) hold the shipped native-MATLAB LOV3D_open
responses for the archive's 2-layer model with degree-1..3 lateral shear
structure at perturbation order 2. The full five-forcing comparison
(2026-09-28, Nr=200): pylov3d matches to a worst relative difference of
1.0e-12 over 365 modes under the 'script' real->complex sine convention
of ``run_forward_shear.m``; the 'canonical' ``get_rheology.m`` map
misses by 7.1e-2. That settles coordination/open-questions.md 2026-09-28
item 2: the shipped outputs use the script convention (S_nm scaled by
(-1)^m relative to canonical).

The slow test re-runs one decisive forcing column, (2,-1) — the column
where the canonical convention erred worst (7.1e-2 at mode (3,2)).
"""

from pathlib import Path

import pytest

from pylov3d.berne2026 import (
    build_zenodo_example_model,
    read_zenodo_shear_coefficients,
    zenodo_mu_variable,
)
from pylov3d.love import get_love
from pylov3d.types import make_forcing, make_numerics

DATA = Path(__file__).resolve().parents[2] / "data" / "tests" / "berne_zenodo"
TD = 686.98 * 86400.0


def _rows():
    return read_zenodo_shear_coefficients(DATA / "shear_modulus_coefficients.txt")


def _shipped():
    import re

    ship: dict[int, dict[tuple[int, int], complex]] = {}
    for line in open(DATA / "k2_responses.txt"):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        mp = int(re.match(r"\((-?\d+),(-?\d+)\)", t[0]).group(2))
        ship.setdefault(mp, {})[(int(t[1]), int(t[2]))] = float(t[3]) + 1j * float(t[4])
    return ship


class TestConventionStructure:
    """Fast structural guards (no solver run)."""

    def test_conventions_differ_by_parity_sign_on_sine_terms(self):
        rows = _rows()
        a = dict(((n, m), v) for n, m, v in zenodo_mu_variable(rows, convention="script")[1])
        b = dict(((n, m), v) for n, m, v in zenodo_mu_variable(rows, convention="canonical")[1])
        differing = sorted(k for k in a if abs(a[k] - b[k]) > 1e-15)
        # exactly the odd-m modes with sine content, both signed partners
        assert differing == [(1, -1), (1, 1), (2, -1), (2, 1), (3, -3), (3, -1), (3, 1), (3, 3)]
        # real parts agree everywhere; imaginary parts flip on those modes
        for k in differing:
            assert a[k].real == pytest.approx(b[k].real, abs=1e-15)
            assert a[k].imag == pytest.approx(-b[k].imag, abs=1e-15)

    def test_delta_normalization_pins(self):
        """Peak-to-peak of unit harmonics on the archive's lmax=59 grid.
        Guards the MATLAB-faithful synthesis path the amplitudes rely on."""
        from pylov3d.berne2026 import zenodo_mu_variable  # noqa: F401
        import numpy as np

        from pylov3d.matlab_sph import stokes_to_grid

        def delta(n, M):
            clm = np.zeros((60, 60))
            clm[n, M] = 1.0
            return float(np.ptp(stokes_to_grid(clm, np.zeros_like(clm), 59)[2]))

        assert delta(1, 1) == pytest.approx(3.463488, abs=2e-6)
        assert delta(3, 0) == pytest.approx(5.288690, abs=2e-6)
        assert delta(3, 3) == pytest.approx(4.181818, abs=2e-6)


@pytest.mark.slow
def test_forcing_2_minus1_column_matches_shipped_matlab():
    ship = _shipped()[-1]
    love, _, _ = get_love(
        build_zenodo_example_model(),
        make_forcing(TD, 2, -1, 1.0),
        make_numerics(n_layers=2, method="variable", Nrbase=200,
                      perturbation_order=2, rheology_cutoff=2.0),
        mu_variable=zenodo_mu_variable(_rows(), convention="script"),
    )
    got = {(int(n), int(m)): complex(k) for n, m, k in zip(love.n, love.m, love.k)}
    assert set(got) == set(ship)
    kf = abs(ship[(2, -1)])
    worst = max(abs(got[mode] - ship[mode]) / kf for mode in ship)
    assert worst <= 1e-10, worst
