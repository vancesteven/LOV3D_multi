"""Tests that produce radial profile plots — y-functions, stress/strain, energy.

Run with --save-output to persist plots to pylov3d/tests/output/.
"""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.grid import set_boundary_indices
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution
from pylov3d.love import get_love
from pylov3d.energy import compute_stress_strain, get_energy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Y_LABELS_1D = ["U", "V", "W", "R", "S", "T", r"$\Phi$", r"$d\Phi/dr$"]


def _layer_spans(model):
    """Return list of (r_inner, r_outer, label) for each layer."""
    spans = []
    r_prev = 0.0
    for i in range(model.n_layers):
        r_out = float(model.R[i])
        label = "core" if i == 0 else f"layer {i}"
        spans.append((r_prev, r_out, label))
        r_prev = r_out
    return spans


def _shade_layers(ax, spans, alpha=0.08):
    """Add alternating pastel bands for layer boundaries."""
    colors = ["#4477AA", "#EE6677"]
    for idx, (r_in, r_out, _) in enumerate(spans):
        ax.axvspan(r_in, r_out, alpha=alpha, color=colors[idx % 2])


def _io_4layer():
    """Build and solve Io 4-layer model (1D)."""
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
    numerics = make_numerics(n_layers=4, method="combination", Nrbase=200)
    numerics, raw_model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(raw_model, forcing)
    y_sol, r_grid, Y, Aprop_aux = get_solution(model, forcing, numerics)
    return model, forcing, numerics, y_sol, r_grid, Aprop_aux


# ---------------------------------------------------------------------------
# 1D radial profiles
# ---------------------------------------------------------------------------

class TestRadialProfiles1D:

    def test_io_radial_profiles_1d(self, output_dir, mpl):
        """Plot all 8 y-function components vs radius for Io (1D)."""
        model, forcing, numerics, y_sol, r_grid, _ = _io_4layer()
        spans = _layer_spans(model)

        fig, axes = mpl.subplots(4, 2, figsize=(12, 14))
        fig.suptitle("Io 4-layer: radial y-functions (1D, n=2 m=0)", fontsize=14)

        for idx, ax in enumerate(axes.flat):
            re = y_sol[:, idx].real
            im = y_sol[:, idx].imag
            _shade_layers(ax, spans)
            ax.plot(r_grid, re, "b-", linewidth=1.2, label="Re")
            ax.plot(r_grid, im, "r--", linewidth=1.0, label="Im")
            ax.set_ylabel(_Y_LABELS_1D[idx])
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("r (normalized)")
        axes[-1, 1].set_xlabel("r (normalized)")

        fig.tight_layout()
        fig.savefig(output_dir / "io_radial_y_1d.png", dpi=150)
        mpl.close(fig)

        np.savez(
            output_dir / "io_radial_y_1d.npz",
            r=r_grid, y_real=y_sol.real, y_imag=y_sol.imag,
        )

        assert np.all(np.isfinite(y_sol))

    def test_io_stress_strain_profiles(self, output_dir, mpl):
        """Plot GSH stress and strain components vs radius."""
        model, forcing, numerics, y_sol, r_grid, Aprop_aux = _io_4layer()
        u_gsh, stress, strain = compute_stress_strain(
            y_sol, r_grid, Aprop_aux, model, forcing, numerics,
        )
        spans = _layer_spans(model)

        labels = [
            r"$\sigma_{n,n,0}$", r"$\sigma_{n,n-2,2}$", r"$\sigma_{n,n-1,2}$",
            r"$\sigma_{n,n,2}$", r"$\sigma_{n,n+1,2}$", r"$\sigma_{n,n+2,2}$",
        ]
        fig, axes = mpl.subplots(2, 6, figsize=(20, 8))
        fig.suptitle("Io 4-layer: GSH stress (top) and strain (bottom)", fontsize=14)

        for j in range(6):
            ax_s = axes[0, j]
            ax_e = axes[1, j]
            _shade_layers(ax_s, spans)
            _shade_layers(ax_e, spans)
            ax_s.plot(r_grid, stress[:, j].real, "b-", lw=1)
            ax_s.plot(r_grid, stress[:, j].imag, "r--", lw=0.8)
            ax_s.set_title(labels[j], fontsize=10)
            ax_e.plot(r_grid, strain[:, j].real, "b-", lw=1)
            ax_e.plot(r_grid, strain[:, j].imag, "r--", lw=0.8)
            if j == 0:
                ax_s.set_ylabel("stress")
                ax_e.set_ylabel("strain")
            ax_e.set_xlabel("r")
            ax_s.grid(True, alpha=0.3)
            ax_e.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "io_stress_strain_1d.png", dpi=150)
        mpl.close(fig)

        # Stress should be nonzero in solid layers
        solid_mask = r_grid > float(model.R[0])
        assert np.any(np.abs(stress[solid_mask]) > 0)

    def test_io_energy_profile_1d(self, output_dir, mpl):
        """Plot tidal dissipation density vs radius."""
        model, forcing, numerics, y_sol, r_grid, Aprop_aux = _io_4layer()
        energy = get_energy(y_sol, r_grid, Aprop_aux, model, forcing, numerics)
        spans = _layer_spans(model)

        fig, ax = mpl.subplots(figsize=(8, 5))
        _shade_layers(ax, spans, alpha=0.12)
        ax.plot(r_grid, energy.energy_profile[:, 0], "k-", linewidth=1.5)
        ax.set_xlabel("r (normalized)")
        ax.set_ylabel("Dissipation density (Im(σ*:ε))")
        ax.set_title("Io 4-layer: radial dissipation profile (1D)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "io_energy_profile_1d.png", dpi=150)
        mpl.close(fig)

        # Nonzero dissipation in the asthenosphere (layer 2: R[1] to R[2])
        r1 = float(model.R[1])
        r2 = float(model.R[2])
        asth_mask = (r_grid >= r1) & (r_grid <= r2)
        assert np.any(np.abs(energy.energy_profile[asth_mask, 0]) > 0)


# ---------------------------------------------------------------------------
# Coupled radial profiles
# ---------------------------------------------------------------------------

class TestRadialProfilesCoupled:

    def _make_coupled(self):
        """Build 3-layer Io model with lateral variations."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=50)
        return raw_model, forcing, numerics

    def test_io_radial_profiles_coupled(self, output_dir, mpl):
        """Plot U, V, Φ per coupled mode, color-coded by (n,m)."""
        raw_model, forcing, numerics = self._make_coupled()
        love, y_rad, model = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        r = y_rad.r
        N = len(y_rad.n_s)
        N3 = 3 * N
        N6 = 6 * N

        fig, axes = mpl.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Io 3-layer coupled: radial profiles ({N} modes)", fontsize=14,
        )
        comp_names = ["U (radial disp.)", "V (tangential disp.)", r"$\Phi$ (potential)"]
        colors = mpl.cm.tab10(np.linspace(0, 1, max(N, 3)))

        for k in range(N):
            n_k = int(y_rad.n_s[k])
            m_k = int(y_rad.m_s[k])
            label = f"n={n_k}, m={m_k}"
            U_k = y_rad.y[:, 3 * k]
            V_k = y_rad.y[:, 3 * k + 1]
            Phi_k = y_rad.y[:, N6 + 2 * k]

            for ax_idx, data in enumerate([U_k, V_k, Phi_k]):
                axes[ax_idx].plot(
                    r, np.abs(data), color=colors[k], label=label, linewidth=1.2,
                )

        for ax_idx in range(3):
            axes[ax_idx].set_xlabel("r (normalized)")
            axes[ax_idx].set_ylabel(f"|{comp_names[ax_idx]}|")
            axes[ax_idx].set_title(comp_names[ax_idx])
            axes[ax_idx].legend(fontsize=8)
            axes[ax_idx].set_yscale("log")
            axes[ax_idx].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "io_radial_y_coupled.png", dpi=150)
        mpl.close(fig)

        # Forcing mode should have the largest |U| at surface
        k_f = np.where((y_rad.n_s == 2) & (y_rad.m_s == 0))[0][0]
        U_f_surf = abs(y_rad.y[-1, 3 * k_f])
        for k in range(N):
            if k != k_f:
                assert abs(y_rad.y[-1, 3 * k]) <= U_f_surf

    def test_uniform_vs_coupled_overlay(self, output_dir, mpl):
        """Overlay 1D and coupled forcing-mode profiles."""
        raw_model, forcing, numerics = self._make_coupled()

        love_1d, y_1d, _ = get_love(raw_model, forcing, numerics)
        love_3d, y_3d, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.05)]},
        )

        N = len(y_3d.n_s)
        N6 = 6 * N
        k_f = np.where((y_3d.n_s == 2) & (y_3d.m_s == 0))[0][0]

        fig, axes = mpl.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("1D vs coupled forcing mode (5% perturbation)", fontsize=14)

        labels_comp = ["U", "V", r"$\Phi$"]
        for ax_idx, (lbl, idx_1d, idx_3d) in enumerate([
            ("U", 0, 3 * k_f),
            ("V", 1, 3 * k_f + 1),
            (r"$\Phi$", 6, N6 + 2 * k_f),
        ]):
            ax = axes[ax_idx]
            ax.plot(y_1d.r, y_1d.y[:, idx_1d].real, "k-", lw=2, label="1D Re")
            ax.plot(y_1d.r, y_1d.y[:, idx_1d].imag, "k--", lw=2, label="1D Im")
            ax.plot(y_3d.r, y_3d.y[:, idx_3d].real, "r-", lw=1, label="3D Re")
            ax.plot(y_3d.r, y_3d.y[:, idx_3d].imag, "r--", lw=1, label="3D Im")
            ax.set_xlabel("r (normalized)")
            ax.set_ylabel(lbl)
            ax.set_title(lbl)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "io_1d_vs_coupled_overlay.png", dpi=150)
        mpl.close(fig)

        # Coupled forcing mode should be close to 1D
        k_1d = love_1d.k[0]
        k_3d = love_3d.k[np.where((love_3d.n == 2) & (love_3d.m == 0))[0][0]]
        assert abs(k_3d - k_1d) / abs(k_1d) < 0.15
