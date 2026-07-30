"""Tests that save .npz reference data for future regression testing.

Run with --save-output to persist data to pylov3d/tests/output/.
"""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.love import get_love
from pylov3d.energy import get_energy


class TestSaveReference:

    def test_save_io_1d_reference(self, output_dir):
        """Save Io 4-layer 1D reference data for all 3 forcing components."""
        raw_model = make_interior_model(
            R0_km=[965.0, 1591.6, 1791.6, 1821.6],
            rho0=[5150.0, 3244.0, 3244.0, 3244.0],
            mu0=[0.0, 6e10, 7.8e5, 6.5e10],
            Ks0=[0.0, 200e16, 200e16, 200e16],
            eta0=[None, 1e20, 1e11, 1e23],
            Delta_rho0=[5150.0 - 3244.0, 5150.0 - 3244.0, 0.0, 0.0],
        )
        numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)

        omega0 = 4.1086e-05
        Td = 2 * math.pi / omega0
        forcings = [
            make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5)),
            make_forcing(Td=Td, n=2, m=-2, F=-7 / 8 * math.sqrt(6 / 5)),
            make_forcing(Td=Td, n=2, m=2, F=1 / 8 * math.sqrt(6 / 5)),
        ]

        k_arr = np.zeros(3, dtype=complex)
        h_arr = np.zeros(3, dtype=complex)
        l_arr = np.zeros(3, dtype=complex)

        for i, f in enumerate(forcings):
            love, y_rad, model = get_love(raw_model, f, numerics)
            k_arr[i] = love.k[0]
            h_arr[i] = love.h[0]
            l_arr[i] = love.l[0]

        # Energy for first forcing (using the full solver path)
        numerics2, model2 = set_boundary_indices(
            make_numerics(n_layers=4, method="combination", Nrbase=200),
            raw_model,
        )
        model2 = get_rheology(model2, forcings[0])
        y_sol, r_grid, Y, Aprop_aux = get_solution(model2, forcings[0], numerics2)
        energy = get_energy(y_sol, r_grid, Aprop_aux, model2, forcings[0], numerics2)

        np.savez(
            output_dir / "io_1d_reference.npz",
            k=k_arr, h=h_arr, l=l_arr,
            forcing_nm=np.array([(f.n, f.m) for f in forcings]),
            forcing_F=np.array([f.F for f in forcings]),
            r_grid=r_grid,
            y_sol_f0_real=y_sol.real, y_sol_f0_imag=y_sol.imag,
            energy_integral=energy.energy_integral,
            energy_profile=energy.energy_profile,
        )

        # Sanity checks
        assert np.all(np.isfinite(k_arr))
        assert np.all(np.isfinite(h_arr))
        assert np.all(np.isfinite(l_arr))
        assert energy.energy_integral[0] != 0
        # k2 should be the same for all forcings (1D, axisymmetric)
        assert k_arr[0] == pytest.approx(k_arr[1], rel=1e-6)

    def test_save_io_coupled_reference(self, output_dir):
        """Save Io 3-layer coupled reference data."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=100)

        love, y_rad, model = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        np.savez(
            output_dir / "io_coupled_reference.npz",
            k=love.k, h=love.h, l=love.l,
            n_s=love.n, m_s=love.m,
            nf=love.nf, mf=love.mf,
            r_grid=y_rad.r,
            y_sol_real=y_rad.y.real, y_sol_imag=y_rad.y.imag,
        )

        assert np.all(np.isfinite(love.k))
        assert len(love.k) > 1  # Multi-mode
        assert 2 in love.n  # Forcing mode present
