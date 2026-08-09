# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Cross-validation against MATLAB LOV3D reference data.

Compares pylov3d results against pre-computed MATLAB outputs stored in
data/tests/ for key benchmark cases from the published literature.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.love import get_love


# ---------------------------------------------------------------------------
# Enceladus 2-layer benchmark (Rovira-Navarro et al. 2024, Figure 2)
# ---------------------------------------------------------------------------

@pytest.fixture
def matlab_data_path():
    """Path to MATLAB reference data directory."""
    # Relative to pylov3d/tests/
    return Path(__file__).parent.parent.parent / "data" / "tests" / "enceladus"


class TestEnceladusBenchmark:
    """Enceladus 2-layer elastic ice shell with lateral variations.

    Compares against MATLAB LOV3D results from:
    - Rovira-Navarro et al. (2024) PSJ
    - Cross-validated with Berne et al. (2023) FEM
    - Cross-validated with Qin et al. (2014) perturbation theory

    Model: rigid core + ocean (layer 1) + elastic ice shell (layer 2)
    with lateral variations in shear modulus.
    """

    @pytest.fixture
    def enceladus_params(self):
        """Enceladus model parameters (normalized units)."""
        # Non-dimensional parameters from Test_Enceladus_Two_Layer_Lateral_Variations.mlx
        G = 6.67e-11
        r_ratio = 0.91  # R_core / R_surface
        rho_ratio = 1610 / 1000  # rho_avg / rho_ice
        Ks_nd = 100  # bulk modulus / shear modulus
        mu_eff = 3.3e9 / (4/3 * math.pi * G * (252.1e3)**2 * 1610**2)

        # Compute derived quantities
        rho_r = (rho_ratio - 1 + r_ratio**3) / r_ratio**3
        rho_av = rho_r * r_ratio**3 + (1 - r_ratio**3)
        Gg = 3 / (4 * math.pi) / mu_eff / rho_av**2

        return {
            'r_ratio': r_ratio,
            'rho_r': rho_r,
            'Ks_nd': Ks_nd,
            'Gg': Gg,
            'mu_eff': mu_eff,
        }

    def _build_enceladus_model(self, params):
        """Build Enceladus model in normalized units."""
        # MATLAB uses normalized coordinates: R_surface = 1.0
        # Layer radii in dimensional units [m], then normalize
        R_core = params['r_ratio'] * 252.1e3
        R_surface = 252.1e3

        # MATLAB sets Interior_Model(1).Delta_rho0 = 0 for the ice-ocean
        # interface (see Test_Enceladus_Two_Layer_Lateral_Variations.mlx).  The
        # core-boundary density contrast must be forced to 0 rather than
        # auto-computed as rho_core - rho_ice, or the CMB potential BC (rho2)
        # is wrong and k2 comes out with the wrong sign/magnitude.
        rho_core = params['rho_r'] * 1000
        model = make_interior_model(
            R0_km=[R_core / 1e3, R_surface / 1e3],
            rho0=[rho_core, 1000],  # Normalize to ice density
            mu0=[0.0, 3.3e9],  # Core rigid, ice elastic
            Ks0=[1e20, params['Ks_nd'] * 3.3e9],  # Core incompressible
            Delta_rho0=[0.0, rho_core - 1000],
        )
        return model

    def _load_matlab_reference(self, matlab_data_path, n_lv, m_lv):
        """Load MATLAB reference data for given (n, m) rheology mode."""
        filename = matlab_data_path / f"Q_{n_lv}{m_lv}.mat"
        if not filename.exists():
            pytest.skip(f"MATLAB reference data not found: {filename}")

        mat = scipy.io.loadmat(filename)

        # Parse structure:
        # k_Q shape: (N_modes, N_amplitudes+3)
        # Columns 0-2: [degree, order, perturbation_order]
        # Columns 3+: Love numbers for each amplitude in 'amp'
        k_Q = mat['k_Q']
        amp = mat['amp'].flatten()
        k2_uniform = float(mat['k2_Q'][0, 0])

        # Extract mode indices
        n_modes = k_Q[1:, 0].astype(int)  # Skip first row (amplitude values)
        m_modes = k_Q[1:, 1].astype(int)
        order = k_Q[1:, 2].astype(int)

        # Love number spectra (skip first 3 columns)
        k_spectra = k_Q[1:, 3:]

        return {
            'n': n_modes,
            'm': m_modes,
            'order': order,
            'k': k_spectra,
            'amp': amp,
            'k2_uniform': k2_uniform,
        }

    @pytest.mark.parametrize("n_lv,m_lv", [(1, 0), (1, 1), (2, 0)])
    def test_uniform_k2_matches(self, enceladus_params, matlab_data_path, n_lv, m_lv):
        """Uniform (no lateral variation) k2 should match MATLAB."""
        model = self._build_enceladus_model(enceladus_params)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="fixed", Nrbase=100)

        love, _, _ = get_love(model, forcing, numerics)

        # Load MATLAB reference
        ref = self._load_matlab_reference(matlab_data_path, n_lv, m_lv)

        # Compare uniform k2 (forcing mode)
        k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
        k2_pylov3d = love.k[k_idx]
        k2_matlab = ref['k2_uniform']

        # Relative error should be < 0.1%
        rel_error = abs(k2_pylov3d - k2_matlab) / abs(k2_matlab)
        assert rel_error < 1e-3, f"k2 mismatch: pylov3d={k2_pylov3d:.6e}, MATLAB={k2_matlab:.6e}"

    @pytest.mark.parametrize("n_lv,m_lv,idx_amp", [
        (2, 0, 4),    # n=2, m=0 at grid node ~5.3% (index 4)
        (2, 0, 9),    # n=2, m=0 at grid node ~10.6% (index 9)
        (1, 1, 4),    # n=1, m=1 at grid node ~5.1% (index 4)
    ])
    def test_lateral_love_spectra(self, enceladus_params, matlab_data_path, n_lv, m_lv, idx_amp):
        """Love number spectra with lateral variations match MATLAB.

        Drives the comparison off the actual MATLAB amplitude-grid node
        (``ref['amp'][idx_amp]``) rather than a hardcoded percentage — the grid
        steps by ~1.05%/node, so exact 5%/10% points do not exist.  The complex
        spherical-harmonic amplitude is amp/sqrt(4*pi) for m=0 and
        amp/sqrt(2)/sqrt(4*pi) for m!=0 (verified <0.25% across all coupled
        modes; see ISSUE_matlab_validation.md).
        """
        model = self._build_enceladus_model(enceladus_params)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=2,
            method="fixed",
            Nrbase=100,
            perturbation_order=2,
        )

        # Load MATLAB reference and use the actual grid-node amplitude.
        ref = self._load_matlab_reference(matlab_data_path, n_lv, m_lv)
        amp_sph = float(ref['amp'][idx_amp])

        # Convert amplitude to complex spherical harmonic amplitude
        # MATLAB uses: amp / sqrt(4*pi) for m=0, amp / sqrt(2) / sqrt(4*pi) for m≠0
        if m_lv == 0:
            amp_c = amp_sph / math.sqrt(4 * math.pi)
        else:
            amp_c = amp_sph / math.sqrt(2) / math.sqrt(4 * math.pi)

        # Build lateral variation dict
        mu_variable = {1: [(n_lv, m_lv, amp_c)]}
        if m_lv != 0:
            # Add conjugate mode for m < 0
            mu_variable[1].append((n_lv, -m_lv, (-1)**m_lv * amp_c))

        love, _, _ = get_love(model, forcing, numerics, mu_variable=mu_variable)

        # Compare Love number spectrum
        # Match modes by (n, m)
        for i, (n_m, m_m) in enumerate(zip(ref['n'], ref['m'])):
            # Find this mode in pylov3d output
            idx_py = np.where((love.n == n_m) & (love.m == m_m))[0]
            if len(idx_py) == 0:
                continue  # Mode not excited in pylov3d

            k_pylov3d = love.k[idx_py[0]]
            k_matlab = ref['k'][i, idx_amp]

            # Skip modes with very small amplitude (numerical noise)
            if abs(k_matlab) < 1e-8:
                continue

            rel_error = abs(k_pylov3d - k_matlab) / abs(k_matlab)

            # Tolerance depends on perturbation order
            order_m = ref['order'][i]
            if order_m == 1:
                tol = 0.01  # 1% for first order
            elif order_m == 2:
                tol = 0.05  # 5% for second order
            else:
                tol = 0.10  # 10% for higher order

            assert rel_error < tol, (
                f"Mode (n={n_m}, m={m_m}): "
                f"pylov3d={k_pylov3d:.6e}, MATLAB={k_matlab:.6e}, "
                f"rel_error={rel_error:.2%} > {tol:.2%}"
            )

    def test_amplitude_sweep(self, enceladus_params, matlab_data_path):
        """Love numbers scale correctly with amplitude (n=2, m=0 case)."""
        model = self._build_enceladus_model(enceladus_params)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=2, method="fixed", Nrbase=100, perturbation_order=2)

        # Load MATLAB reference for n=2, m=0
        ref = self._load_matlab_reference(matlab_data_path, 2, 0)

        # Test at 3 amplitudes: 5%, 10%, 25%
        amp_test = [5.0, 10.0, 25.0]

        for amp_pct in amp_test:
            amp_c = (amp_pct / 100.0) / math.sqrt(4 * math.pi)
            mu_variable = {1: [(2, 0, amp_c)]}

            love, _, _ = get_love(model, forcing, numerics, mu_variable=mu_variable)

            # Find matching amplitude in MATLAB
            amp_matlab_pct = ref['amp'] * 100
            idx_amp = np.argmin(np.abs(amp_matlab_pct - amp_pct))

            # Compare forcing mode (n=2, m=0)
            k_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
            k_pylov3d = love.k[k_idx]

            # The reference has TWO (2,0) rows: order=0 is the constant uniform
            # value (k_M(2,:) in MATLAB), order=1 is the amplitude-dependent
            # response.  Select the order=1 row — comparing against the
            # amplitude-independent order=0 row spuriously grows the error with
            # amplitude.
            idx_forcing = np.where(
                (ref['n'] == 2) & (ref['m'] == 0) & (ref['order'] == 1)
            )[0][0]
            k_matlab = ref['k'][idx_forcing, idx_amp]

            rel_error = abs(k_pylov3d - k_matlab) / abs(k_matlab)
            assert rel_error < 0.01, f"Amplitude {amp_pct}%: rel_error={rel_error:.2%}"
