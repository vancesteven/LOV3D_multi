"""Tests that produce convergence study plots.

Run with --save-output to persist plots to pylov3d/tests/output/.
"""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.love import get_love


# ---------------------------------------------------------------------------
# Grid convergence
# ---------------------------------------------------------------------------

class TestGridConvergence:

    def test_grid_convergence_1d(self, output_dir, mpl):
        """k2 convergence vs grid resolution (1D Io 4-layer)."""
        raw_model = make_interior_model(
            R0_km=[965.0, 1591.6, 1791.6, 1821.6],
            rho0=[5150.0, 3244.0, 3244.0, 3244.0],
            mu0=[0.0, 6e10, 7.8e5, 6.5e10],
            Ks0=[0.0, 200e16, 200e16, 200e16],
            eta0=[None, 1e20, 1e11, 1e23],
            Delta_rho0=[5150.0 - 3244.0, 5150.0 - 3244.0, 0.0, 0.0],
        )
        omega0 = 4.1086e-05
        Td = 2 * math.pi / omega0
        forcing = make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5))

        nrbase_vals = [20, 50, 100, 200, 500]
        k2_vals = []

        for nr in nrbase_vals:
            numerics = make_numerics(n_layers=4, method="combination", Nrbase=nr)
            love, _, _ = get_love(raw_model, forcing, numerics)
            k2_vals.append(love.k[0])

        k2_ref = k2_vals[-1]
        diffs = [abs(k - k2_ref) for k in k2_vals[:-1]]

        fig, ax = mpl.subplots(figsize=(8, 5))
        ax.loglog(nrbase_vals[:-1], diffs, "bo-", markersize=8, linewidth=1.5)
        ax.set_xlabel("Nrbase")
        ax.set_ylabel("|k2 - k2_ref|")
        ax.set_title("Io 4-layer: grid convergence (1D)")
        ax.grid(True, alpha=0.3, which="both")

        # Add reference slope lines
        x0, y0 = nrbase_vals[0], diffs[0]
        x_ref = np.array([x0, nrbase_vals[-2]])
        ax.loglog(x_ref, y0 * (x0 / x_ref) ** 2, "k--", alpha=0.4, label="O(h²)")
        ax.loglog(x_ref, y0 * (x0 / x_ref) ** 4, "k:", alpha=0.4, label="O(h⁴)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / "io_convergence_1d.png", dpi=150)
        mpl.close(fig)

        np.savez(
            output_dir / "io_convergence_1d.npz",
            nrbase=nrbase_vals, k2_real=np.real(k2_vals), k2_imag=np.imag(k2_vals),
        )

        # Overall convergence trend: coarsest → finest should decrease
        assert diffs[1] < diffs[0]
        # Final resolution close to reference (within 1% of k2)
        assert abs(k2_vals[-2] - k2_ref) / abs(k2_ref) < 0.01

    def test_grid_convergence_coupled(self, output_dir, mpl):
        """k convergence vs grid resolution (coupled 3-layer)."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)

        nrbase_vals = [10, 20, 50, 100, 200]
        k_forcing = []
        k_degree4 = []

        for nr in nrbase_vals:
            numerics = make_numerics(n_layers=3, method="combination", Nrbase=nr)
            love, _, _ = get_love(
                raw_model, forcing, numerics,
                mu_variable={1: [(2, 0, 0.1)]},
            )
            k_f_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
            k_forcing.append(love.k[k_f_idx])

            k4_idx = np.where((love.n == 4) & (love.m == 0))[0]
            k_degree4.append(love.k[k4_idx[0]] if len(k4_idx) > 0 else 0.0)

        k_ref_f = k_forcing[-1]
        k_ref_4 = k_degree4[-1]
        diffs_f = [abs(k - k_ref_f) for k in k_forcing[:-1]]
        diffs_4 = [abs(k - k_ref_4) for k in k_degree4[:-1]]

        fig, ax = mpl.subplots(figsize=(8, 5))
        ax.loglog(
            nrbase_vals[:-1], diffs_f, "bo-", markersize=8, lw=1.5,
            label="k(2,0) forcing mode",
        )
        ax.loglog(
            nrbase_vals[:-1], diffs_4, "rs-", markersize=8, lw=1.5,
            label="k(4,0) coupled mode",
        )
        ax.set_xlabel("Nrbase")
        ax.set_ylabel("|k - k_ref|")
        ax.set_title("3-layer coupled: grid convergence")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig(output_dir / "io_convergence_coupled.png", dpi=150)
        mpl.close(fig)

        # Monotonic convergence for forcing mode
        assert diffs_f[1] < diffs_f[0]
        assert diffs_f[2] < diffs_f[1]


# ---------------------------------------------------------------------------
# Perturbation amplitude sweep
# ---------------------------------------------------------------------------

class TestPerturbationSweep:

    def test_perturbation_amplitude_sweep(self, output_dir, mpl):
        """Love number shift vs lateral variation amplitude."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=100)

        love_1d, _, _ = get_love(raw_model, forcing, numerics)
        k_1d = love_1d.k[0]

        amps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
        dk_forcing = []
        h_degree4 = []

        for amp in amps:
            love, _, _ = get_love(
                raw_model, forcing, numerics,
                mu_variable={1: [(2, 0, amp)]},
            )
            k_f_idx = np.where((love.n == 2) & (love.m == 0))[0][0]
            dk_forcing.append(abs(love.k[k_f_idx] - k_1d))

            k4_idx = np.where((love.n == 4) & (love.m == 0))[0]
            h_degree4.append(abs(love.h[k4_idx[0]]) if len(k4_idx) > 0 else 0.0)

        fig, axes = mpl.subplots(1, 2, figsize=(12, 5))

        # Forcing mode shift
        ax = axes[0]
        ax.loglog(amps, dk_forcing, "bo-", markersize=7, lw=1.5, label="|Δk(2,0)|")
        # Reference: amp^2 scaling
        a0, d0 = amps[0], dk_forcing[0]
        a_ref = np.array(amps)
        ax.loglog(a_ref, d0 * (a_ref / a0) ** 2, "k--", alpha=0.4, label="∝ amp²")
        ax.set_xlabel("Lateral variation amplitude")
        ax.set_ylabel("|k_coupled - k_1d|")
        ax.set_title("Forcing mode shift vs amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        # Degree-4 response
        ax = axes[1]
        nonzero = [(a, h) for a, h in zip(amps, h_degree4) if h > 0]
        if nonzero:
            a_nz, h_nz = zip(*nonzero)
            ax.loglog(a_nz, h_nz, "rs-", markersize=7, lw=1.5, label="|h(4,0)|")
            # Reference: amp^1 scaling
            a0, h0 = a_nz[0], h_nz[0]
            a_ref2 = np.array(a_nz)
            ax.loglog(a_ref2, h0 * (a_ref2 / a0), "k--", alpha=0.4, label="∝ amp")
        ax.set_xlabel("Lateral variation amplitude")
        ax.set_ylabel("|h(4,0)|")
        ax.set_title("Degree-4 response vs amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        fig.savefig(output_dir / "io_perturbation_sweep.png", dpi=150)
        mpl.close(fig)

        np.savez(
            output_dir / "io_perturbation_sweep.npz",
            amps=amps, dk_forcing=dk_forcing, h_degree4=h_degree4,
        )

        # Larger amplitude → larger shift
        assert dk_forcing[-1] > dk_forcing[0]
