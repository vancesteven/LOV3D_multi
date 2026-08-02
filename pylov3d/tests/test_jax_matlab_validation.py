"""Direct cross-validation of the coupled JAX path against MATLAB (TASK-003).

The coupled JAX scan (``jax_coupled.jax_get_solution_coupled_scan``) is pinned
to the NumPy coupled solver to ~1.5e-15 by ``test_jax_coupled_scan.py``, and the
NumPy coupled path is MATLAB-validated by ``test_matlab_validation.py``.  This
module closes the loop *directly*: it runs the Enceladus lateral-variation
benchmark through the JAX solver and asserts the resulting Love-number spectrum
matches the published MATLAB LOV3D reference data, with no NumPy solver in the
path.  It is therefore a standalone regression guard that survives even if the
NumPy reference is later refactored.

The model, MATLAB reference loader, amplitude→spherical-harmonic conversion, and
per-mode tolerance logic are shared with ``test_matlab_validation.py`` (the
NumPy MATLAB test) so the only difference between the two is the solver:

    NumPy:  get_love(...)                        -> love.k
    JAX:    jax_get_solution_coupled_scan(...)    -> y_sol
            extract_love_numbers(y_sol, ...)      -> love.k

See ``ISSUE_matlab_validation.md`` for the amplitude-convention derivation.
"""

import math

import numpy as np
import pytest

from pylov3d.couplings import get_couplings
from pylov3d.grid import set_boundary_indices
from pylov3d.jax_coupled import jax_get_solution_coupled_scan
from pylov3d.love import extract_love_numbers
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.types import make_forcing, make_numerics

# Reuse the NumPy MATLAB test's Enceladus fixtures/helpers verbatim so the two
# validation paths differ only in the solver, not in the model or reference.
# Import under a non-``Test`` alias so pytest does not re-collect the NumPy
# class inside this module (which would fail: its fixtures live in that module).
from pylov3d.tests.test_matlab_validation import (
    TestEnceladusBenchmark as _EnceladusBench,
)


class TestEnceladusBenchmarkJax:
    """Enceladus 2-layer elastic ice shell with lateral variations, solved via
    the coupled JAX scan and validated against MATLAB LOV3D reference data.

    Mirrors ``TestEnceladusBenchmark`` (NumPy) but routes the radial
    integration through ``jax_get_solution_coupled_scan``.
    """

    # ---- Shared helpers, borrowed from the NumPy MATLAB test ------------
    _bench = _EnceladusBench()

    @pytest.fixture
    def enceladus_params(self):
        # Delegate to the NumPy test's fixture body (a plain method returning a
        # dict — no pytest magic needed to call it directly).
        return _EnceladusBench.enceladus_params.__wrapped__(self._bench)

    def _build_enceladus_model(self, params):
        return self._bench._build_enceladus_model(params)

    def _load_matlab_reference(self, matlab_data_path, n_lv, m_lv):
        return self._bench._load_matlab_reference(matlab_data_path, n_lv, m_lv)

    @pytest.fixture
    def matlab_data_path(self):
        from pathlib import Path
        return Path(__file__).parent.parent.parent / "data" / "tests" / "enceladus"

    # ---- The JAX cross-validation test ----------------------------------
    @pytest.mark.parametrize("n_lv,m_lv,idx_amp", [
        (2, 0, 4),    # n=2, m=0 at grid node ~5.3% (index 4)
        (2, 0, 9),    # n=2, m=0 at grid node ~10.6% (index 9)
        (1, 1, 4),    # n=1, m=1 at grid node ~5.1% (index 4)
    ])
    def test_lateral_love_spectra_jax(
        self, enceladus_params, matlab_data_path, n_lv, m_lv, idx_amp
    ):
        """Coupled JAX Love-number spectrum matches MATLAB.

        Identical case setup and tolerances to
        ``TestEnceladusBenchmark.test_lateral_love_spectra`` — the only change
        is that the solver is the JAX scan.
        """
        raw_model = self._build_enceladus_model(enceladus_params)
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=2,
            method="fixed",
            Nrbase=100,
            perturbation_order=2,
        )

        # MATLAB reference and the actual grid-node amplitude.
        ref = self._load_matlab_reference(matlab_data_path, n_lv, m_lv)
        amp_sph = float(ref['amp'][idx_amp])

        # amp / sqrt(4*pi) for m=0, amp / sqrt(2) / sqrt(4*pi) for m!=0.
        if m_lv == 0:
            amp_c = amp_sph / math.sqrt(4 * math.pi)
        else:
            amp_c = amp_sph / math.sqrt(2) / math.sqrt(4 * math.pi)

        mu_variable = {1: [(n_lv, m_lv, amp_c)]}
        if m_lv != 0:
            mu_variable[1].append((n_lv, -m_lv, (-1) ** m_lv * amp_c))

        # Build the coupled problem exactly as get_love does internally, then
        # solve with the JAX scan instead of the NumPy solver.
        numerics, model = set_boundary_indices(numerics, raw_model)
        model = get_rheology(model, forcing)
        model, lateral = process_lateral_variations(
            model, forcing, mu_variable=mu_variable,
        )
        couplings = get_couplings(
            lateral.variations, forcing.n, forcing.m,
            perturbation_order=numerics.perturbation_order,
        )

        y_sol, _r_grid, _Y, _aux = jax_get_solution_coupled_scan(
            model, forcing, numerics, couplings, lateral,
        )
        love = extract_love_numbers(y_sol, model, forcing, couplings=couplings)

        # Compare Love-number spectrum, matching modes by (n, m).
        compared = 0
        for i, (n_m, m_m) in enumerate(zip(ref['n'], ref['m'])):
            idx_py = np.where((love.n == n_m) & (love.m == m_m))[0]
            if len(idx_py) == 0:
                continue  # Mode not excited in pylov3d

            k_jax = love.k[idx_py[0]]
            k_matlab = ref['k'][i, idx_amp]

            if abs(k_matlab) < 1e-8:
                continue  # numerical noise

            rel_error = abs(k_jax - k_matlab) / abs(k_matlab)

            order_m = ref['order'][i]
            if order_m == 1:
                tol = 0.01
            elif order_m == 2:
                tol = 0.05
            else:
                tol = 0.10

            assert rel_error < tol, (
                f"JAX mode (n={n_m}, m={m_m}): "
                f"jax={k_jax:.6e}, MATLAB={k_matlab:.6e}, "
                f"rel_error={rel_error:.2%} > {tol:.2%}"
            )
            compared += 1

        assert compared > 0, (
            "No modes were compared — the JAX/MATLAB mode-matching found no "
            "overlapping (n, m) with non-trivial amplitude; the test would "
            "otherwise pass vacuously."
        )
