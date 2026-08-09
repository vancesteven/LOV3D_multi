# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests that produce Love number spectra plots.

Run with --save-output to persist plots to pylov3d/tests/output/.
"""

import math

import numpy as np
import pytest

from pylov3d.types import make_interior_model, make_forcing, make_numerics
from pylov3d.love import get_love


# ---------------------------------------------------------------------------
# Love number spectra — coupled
# ---------------------------------------------------------------------------

class TestLoveSpectraPlots:

    def test_love_spectra_bar_chart(self, output_dir, mpl):
        """Grouped bar chart of |k|, |h|, |l| per coupled mode."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        forcing = make_forcing(Td=1.769 * 86400, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=50)

        love, _, _ = get_love(
            raw_model, forcing, numerics,
            mu_variable={1: [(2, 0, 0.1)]},
        )

        N = len(love.k)
        mode_labels = [f"({int(love.n[i])},{int(love.m[i])})" for i in range(N)]
        x = np.arange(N)
        w = 0.25

        fig, ax = mpl.subplots(figsize=(10, 5))
        ax.bar(x - w, np.abs(love.k), w, label="|k|", color="#4477AA")
        ax.bar(x, np.abs(love.h), w, label="|h|", color="#EE6677")
        ax.bar(x + w, np.abs(love.l), w, label="|l|", color="#228833")
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.set_xlabel("Mode (n, m)")
        ax.set_ylabel("Magnitude")
        ax.set_yscale("log")
        ax.set_title("Love number spectra — coupled modes (10% lateral variation)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(output_dir / "io_love_spectra_bar.png", dpi=150)
        mpl.close(fig)

        np.savez(
            output_dir / "io_love_spectra_bar.npz",
            n=love.n, m=love.m, k=love.k, h=love.h, l=love.l,
        )

        assert N > 1
        # Forcing mode should have the largest |h|
        k_f = np.where((love.n == 2) & (love.m == 0))[0][0]
        assert np.abs(love.h[k_f]) == np.max(np.abs(love.h))

    def test_love_spectra_heatmap(self, output_dir, mpl):
        """Heatmap of |k| across forcings and response modes."""
        raw_model = make_interior_model(
            R0_km=[800.0, 1600.0, 1821.6],
            rho0=[5150.0, 3300.0, 3000.0],
            mu0=[0.0, 60e9, 65e9],
            eta0=[None, 1e19, None],
        )
        numerics = make_numerics(n_layers=3, method="combination", Nrbase=50)

        omega0 = 4.1086e-05
        Td = 2 * math.pi / omega0
        forcings = [
            make_forcing(Td=Td, n=2, m=0, F=3 / 4 * math.sqrt(1 / 5)),
            make_forcing(Td=Td, n=2, m=-2, F=-7 / 8 * math.sqrt(6 / 5)),
            make_forcing(Td=Td, n=2, m=2, F=1 / 8 * math.sqrt(6 / 5)),
        ]

        # Collect all response modes from all forcings
        all_love = []
        for f in forcings:
            love, _, _ = get_love(
                raw_model, f, numerics,
                mu_variable={1: [(2, 0, 0.1)]},
            )
            all_love.append(love)

        # Build unified mode set (union of all response modes)
        all_modes = set()
        for love in all_love:
            for i in range(len(love.n)):
                all_modes.add((int(love.n[i]), int(love.m[i])))
        modes_sorted = sorted(all_modes)

        # Build matrix: rows = forcings, cols = response modes
        n_forcings = len(forcings)
        n_modes = len(modes_sorted)
        k_matrix = np.full((n_forcings, n_modes), np.nan)

        for fi, love in enumerate(all_love):
            for mi, (n_m, m_m) in enumerate(modes_sorted):
                idx = np.where((love.n == n_m) & (love.m == m_m))[0]
                if len(idx) > 0:
                    k_matrix[fi, mi] = np.log10(max(np.abs(love.k[idx[0]]), 1e-20))

        fig, ax = mpl.subplots(figsize=(12, 4))
        im = ax.pcolormesh(
            k_matrix, cmap="RdBu_r", vmin=-8, vmax=0,
            edgecolors="gray", linewidth=0.5,
        )
        ax.set_yticks(np.arange(n_forcings) + 0.5)
        ax.set_yticklabels([f"({f.n},{f.m})" for f in forcings])
        ax.set_xticks(np.arange(n_modes) + 0.5)
        ax.set_xticklabels([f"({n},{m})" for n, m in modes_sorted], rotation=45, ha="right")
        ax.set_xlabel("Response mode (n, m)")
        ax.set_ylabel("Forcing (n, m)")
        ax.set_title("log10(|k|) — Love number coupling heatmap")
        fig.colorbar(im, ax=ax, label="log10(|k|)")

        fig.tight_layout()
        fig.savefig(output_dir / "io_love_spectra_heatmap.png", dpi=150)
        mpl.close(fig)

        np.savez(
            output_dir / "io_love_spectra_heatmap.npz",
            k_matrix=k_matrix,
            modes=np.array(modes_sorted),
            forcing_nm=np.array([(f.n, f.m) for f in forcings]),
        )

        # All forcing modes should be present in the heatmap
        for fi, f in enumerate(forcings):
            mi = modes_sorted.index((f.n, f.m))
            assert not np.isnan(k_matrix[fi, mi])

    def test_love_1d_three_forcings(self, output_dir, mpl):
        """Bar chart of k2, h2, l2 for each Io eccentricity forcing (1D)."""
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

        k_vals, h_vals, l_vals = [], [], []
        for f in forcings:
            love, _, _ = get_love(raw_model, f, numerics)
            k_vals.append(love.k[0])
            h_vals.append(love.h[0])
            l_vals.append(love.l[0])

        labels = [f"(2,{f.m})" for f in forcings]
        x = np.arange(len(forcings))
        w = 0.25

        fig, axes = mpl.subplots(1, 2, figsize=(12, 5))

        # Real parts
        ax = axes[0]
        ax.bar(x - w, [v.real for v in k_vals], w, label="Re(k)", color="#4477AA")
        ax.bar(x, [v.real for v in h_vals], w, label="Re(h)", color="#EE6677")
        ax.bar(x + w, [v.real for v in l_vals], w, label="Re(l)", color="#228833")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Real part")
        ax.set_title("Io 4-layer: Love numbers (real)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Imaginary parts
        ax = axes[1]
        ax.bar(x - w, [v.imag for v in k_vals], w, label="Im(k)", color="#4477AA")
        ax.bar(x, [v.imag for v in h_vals], w, label="Im(h)", color="#EE6677")
        ax.bar(x + w, [v.imag for v in l_vals], w, label="Im(l)", color="#228833")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Imaginary part")
        ax.set_title("Io 4-layer: Love numbers (imaginary)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(output_dir / "io_love_1d_three_forcings.png", dpi=150)
        mpl.close(fig)

        # All Love numbers should be the same for this model (axisymmetric)
        # because k2 does not depend on m for the 1D problem
        assert k_vals[0] == pytest.approx(k_vals[1], rel=1e-6)
        assert k_vals[0] == pytest.approx(k_vals[2], rel=1e-6)
