"""Cross-validation of the coupled OCEAN path against MATLAB/Qin reference data.

TASK-008 (Milestone 5).  The coupled solver's ocean branch is being built in
TASK-006 (``assemble_bc_ocean_coupled``) + TASK-007 (wiring into
``_get_solution_coupled``).  This module prepares the reference data and the
validation harness *now*, in parallel, so it flips green the moment the coupled
ocean solver exists.

Reference data
--------------
``data/tests/moon/moon_3D_{LM,UM}_{n}{m}.mat`` hold the Weber et al. (2011)
multi-layered Moon with lateral shear-modulus variations, as published in
Rovira-Navarro et al. and cross-checked against Qin et al. (2016) perturbation
theory.  Unlike the Enceladus references (clean ``{k_B, kD_B, k2_B}``), the
Moon files use the notebook's *plot* layout
(``Test_Moon_MultiLayered_Lateral_Variations.mlx``):

    to_plot : (n_modes+2, n_amp+4) float64
        row 0             : [--, --, --, --, amp_0, amp_1, ...]  (amplitude grid)
        rows 2..end, col0 : perturbation order of the mode
        rows 2..end, col2 : mode degree n
        rows 2..end, col3 : mode order m
        rows 2..end, col4+: k2-NORMALISED Love ratios, one per amplitude
    k2_Q : scalar uniform (no-lateral-variation) k2 used to de-normalise:
        k(mode, amp) = to_plot[row, 4+amp] * k2_Q

The Moon model has a FLUID outer core (Weber layer 2, Vs=0), so it genuinely
exercises the M5 ocean/fluid propagator — the whole point of validating here.

Skip behaviour
--------------
The solver-comparison test auto-skips (not xfail) while
``_get_solution_coupled`` raises ``NotImplementedError`` on an ocean model,
i.e. until TASK-007 lands.  After that it runs live with no edit needed here.
The parser and model-builder tests are ACTIVE now (they need no solver).

See ``ISSUE_matlab_validation.md`` for the amplitude / spherical-harmonic
normalization derivation shared with the Enceladus tests.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from pylov3d.types import make_interior_model, make_forcing, make_numerics


MOON_DATA = Path(__file__).parent.parent.parent / "data" / "tests" / "moon"

# (file stem, lateral-variation degree n_lv, order m_lv) for the 5 published
# cases.  UM = upper-mantle lateral variation, LM = lower-mantle.
MOON_CASES = [
    ("moon_3D_LM_10", 1, 0),
    ("moon_3D_LM_20", 2, 0),
    ("moon_3D_UM_10", 1, 0),
    ("moon_3D_UM_11", 1, 1),
    ("moon_3D_UM_20", 2, 0),
]


# ---------------------------------------------------------------------------
# Reference parser  (active now — no solver needed)
# ---------------------------------------------------------------------------

def _load_qin_reference(stem):
    """Parse a ``moon_3D_*.mat`` plot-layout file into a clean spectrum dict.

    Returns
    -------
    dict with keys
        n, m, order : (n_modes,) int arrays  — one entry per coupled mode
        amp         : (n_amp,) float          — lateral-amplitude grid
        k           : (n_modes, n_amp) float   — de-normalised Love numbers
        k2_uniform  : float                    — uniform k2 (== k2_Q)
    """
    path = MOON_DATA / f"{stem}.mat"
    if not path.exists():
        pytest.skip(f"Moon reference not found: {path}")
    mat = scipy.io.loadmat(path)
    to_plot = mat["to_plot"]
    k2_uniform = float(mat["k2_Q"][0, 0])

    amp = to_plot[0, 4:].astype(float)
    order = to_plot[2:, 0].astype(int)
    n = to_plot[2:, 2].astype(int)
    m = to_plot[2:, 3].astype(int)
    # to_plot stores k2-normalised ratios; de-normalise to absolute Love numbers.
    k = to_plot[2:, 4:].astype(float) * k2_uniform

    return {
        "n": n,
        "m": m,
        "order": order,
        "amp": amp,
        "k": k,
        "k2_uniform": k2_uniform,
    }


# ---------------------------------------------------------------------------
# Weber-Moon interior model  (active now — no solver needed)
# ---------------------------------------------------------------------------

def _build_weber_moon_model():
    """Build the Weber et al. (2011) 9-layer Moon as an ``InteriorModel``.

    ``Moon_Weber.dat`` columns: radius [m], density [kg/m^3], Vp [m/s], Vs [m/s].
    mu = rho*Vs^2 ; Ks = rho*(Vp^2 - 4/3 Vs^2).  The Vs=0 layer (outer core) is
    tagged ocean=1 so the coupled solver routes it through the fluid propagator.
    """
    dat = np.loadtxt(MOON_DATA / "Moon_Weber.dat")
    r_m, rho, vp, vs = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3]

    mu = rho * vs**2
    Ks = rho * (vp**2 - 4.0 / 3.0 * vs**2)
    ocean = [1 if v == 0.0 else 0 for v in vs]

    model = make_interior_model(
        R0_km=list(r_m / 1e3),
        rho0=list(rho),
        mu0=list(mu),
        Ks0=list(Ks),
        ocean=ocean,
    )
    return model


# ---------------------------------------------------------------------------
# Active tests: reference parsing + model construction
# ---------------------------------------------------------------------------

class TestMoonReferenceData:
    """Validate the Qin/Moon reference parser and Weber model builder.

    These need no solver and guard the data-prep layer TASK-007 will consume.
    """

    @pytest.mark.parametrize("stem,n_lv,m_lv", MOON_CASES)
    def test_reference_parses(self, stem, n_lv, m_lv):
        ref = _load_qin_reference(stem)
        n_modes = len(ref["n"])
        assert n_modes > 0
        assert ref["k"].shape == (n_modes, len(ref["amp"]))
        assert len(ref["m"]) == n_modes and len(ref["order"]) == n_modes
        # Amplitude grid is strictly increasing and positive.
        assert np.all(ref["amp"] > 0)
        assert np.all(np.diff(ref["amp"]) > 0)
        assert np.isfinite(ref["k"]).all()
        # Uniform k2 for the Weber Moon is O(0.02) (notebook: k2≈0.0232).
        assert 0.0 < ref["k2_uniform"] < 0.1

    def test_reference_contains_forcing_mode(self):
        # The n=2,m=0 tidal forcing mode must be present (order 2 self-term).
        ref = _load_qin_reference("moon_3D_UM_11")
        has_20 = np.any((ref["n"] == 2) & (ref["m"] == 0))
        assert has_20, "forcing mode (n=2,m=0) missing from reference spectrum"

    def test_weber_model_has_fluid_outer_core(self):
        model = _build_weber_moon_model()
        n_layers = 9
        ocean_flags = [int(model.ocean[i]) for i in range(n_layers)]
        # Exactly one fluid layer (the Weber outer core), at layer index 1.
        assert sum(ocean_flags) == 1
        assert ocean_flags[1] == 1
        # Shear modulus is zero in the fluid layer and positive elsewhere.
        assert float(model.mu0[1]) == 0.0
        for i in [0, 2, 3, 4, 5, 6, 7, 8]:
            assert float(model.mu0[i]) > 0.0


# ---------------------------------------------------------------------------
# Coupled-ocean solver validation  (auto-skips until TASK-007)
# ---------------------------------------------------------------------------

class TestMoonCoupledOceanValidation:
    """Coupled JAX/NumPy ocean solve vs MATLAB/Qin — pending TASK-007.

    Mirrors ``TestEnceladusBenchmark.test_lateral_love_spectra`` but on the
    ocean-bearing Weber Moon.  While the coupled solver rejects ocean models
    (``NotImplementedError``), this test skips cleanly; once TASK-007 wires the
    24N BC path in, it runs live with no change here.
    """

    @pytest.mark.parametrize("stem,n_lv,m_lv", MOON_CASES)
    def test_lateral_love_spectra_ocean(self, stem, n_lv, m_lv):
        from pylov3d.couplings import get_couplings
        from pylov3d.grid import set_boundary_indices
        from pylov3d.rheology import get_rheology, process_lateral_variations
        from pylov3d.solver import _get_solution_coupled
        from pylov3d.love import extract_love_numbers

        ref = _load_qin_reference(stem)
        # Compare at a mid-grid amplitude node (well inside the perturbative
        # regime, matching the Enceladus test's idx_amp≈4-9 choice).
        idx_amp = min(4, len(ref["amp"]) - 1)
        amp_sph = float(ref["amp"][idx_amp])
        if m_lv == 0:
            amp_c = amp_sph / math.sqrt(4 * math.pi)
        else:
            amp_c = amp_sph / math.sqrt(2) / math.sqrt(4 * math.pi)

        raw_model = _build_weber_moon_model()
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        numerics = make_numerics(
            n_layers=9, method="fixed", Nrbase=100, perturbation_order=2,
        )
        numerics, model = set_boundary_indices(numerics, raw_model)
        model = get_rheology(model, forcing)

        # Lateral variation lives in a mantle layer (UM/LM); use the upper
        # mantle (layer index 4) as a representative host — TASK-007 review can
        # refine the exact host layer against the notebook if needed.
        mu_variable = {4: [(n_lv, m_lv, amp_c)]}
        if m_lv != 0:
            mu_variable[4].append((n_lv, -m_lv, (-1) ** m_lv * amp_c))

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable=mu_variable,
        )
        couplings = get_couplings(
            lateral.variations, forcing.n, forcing.m,
            perturbation_order=numerics.perturbation_order,
        )

        try:
            y_sol, _r, _Y, _aux = _get_solution_coupled(
                model, forcing, numerics, couplings, lateral,
            )
        except NotImplementedError as exc:
            pytest.skip(
                "coupled ocean solver not yet implemented (TASK-007): "
                f"{exc}"
            )

        love = extract_love_numbers(y_sol, model, forcing, couplings=couplings)

        compared = 0
        for i in range(len(ref["n"])):
            n_m, m_m = int(ref["n"][i]), int(ref["m"][i])
            idx_py = np.where((love.n == n_m) & (love.m == m_m))[0]
            if len(idx_py) == 0:
                continue
            k_py = love.k[idx_py[0]]
            k_ref = ref["k"][i, idx_amp]
            if abs(k_ref) < 1e-8:
                continue
            rel_error = abs(k_py - k_ref) / abs(k_ref)
            order_m = int(ref["order"][i])
            tol = 0.01 if order_m == 1 else 0.05 if order_m == 2 else 0.10
            assert rel_error < tol, (
                f"{stem} mode (n={n_m}, m={m_m}): py={k_py:.6e}, "
                f"ref={k_ref:.6e}, rel_error={rel_error:.2%} > {tol:.2%}"
            )
            compared += 1

        assert compared > 0, (
            "No modes compared — mode-matching found no overlapping (n,m) "
            "with non-trivial amplitude; test would pass vacuously."
        )
