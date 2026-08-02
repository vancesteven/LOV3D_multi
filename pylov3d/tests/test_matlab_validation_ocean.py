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

Status
------
TASK-007 landed the coupled ocean solver; all tests here run live.  The
solver-comparison cases validate the coupled ocean path end to end against
the Qin-method reference spectra (order-1 modes agree to ~5e-6 relative,
the forcing-mode deviation to ~0.3%).

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
    """Build the Weber Moon exactly as the MATLAB notebook does (10 layers).

    ``Test_Moon_MultiLayered_Lateral_Variations.mlx`` does NOT use the raw
    9-layer Weber profile: it prepends an artificial 50 km, 8000 kg/m^3
    core (so the fluid outer core is not directly above the LOV3D core —
    that configuration is unsupported in MATLAB and here), and rigidifies
    the solid inner core (mu, Ks x1000).  The Vs=0 layer (outer core) is
    tagged ocean=1; it lands at layer index 2.

    ``Moon_Weber.dat`` columns: radius [m], density [kg/m^3], Vp [m/s], Vs [m/s].
    mu = rho*Vs^2 ; Ks = rho*(Vp^2 - 4/3 Vs^2).
    """
    dat = np.loadtxt(MOON_DATA / "Moon_Weber.dat")
    r_m, rho, vp, vs = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3]

    mu = list(rho * vs**2)
    Ks = list(rho * (vp**2 - 4.0 / 3.0 * vs**2))
    mu[0] *= 1000.0   # notebook: "let's make the inner core rigid"
    Ks[0] *= 1000.0

    R0 = [50.0] + list(r_m / 1e3)
    rho0 = [8000.0] + list(rho)
    mu0 = [0.0] + mu
    Ks0 = [0.0] + Ks
    ocean = [1 if m_i == 0.0 else 0 for m_i in mu0]
    ocean[0] = 0  # layer 0 is the LOV3D core, not a subsurface ocean

    model = make_interior_model(
        R0_km=R0,
        rho0=rho0,
        mu0=mu0,
        Ks0=Ks0,
        ocean=ocean,
    )
    return model


# Host layers for the lateral variation, 0-based (notebook, 1-based: UM=6:9,
# LM=4:5).
_HOST_LAYERS = {"UM": [5, 6, 7, 8], "LM": [3, 4]}


def _delta_unit_map(n, m, l_max=30):
    """Peak-to-peak of the unit-coefficient real SH map, MATLAB-style.

    Replicates ``SPH_LatLon`` (4pi-normalized Legendre, half-cell-offset
    lat/lon grid with resolution 180/(2*lmax), lmax = 2*l_max-1) so the
    peak-to-peak normalization matches ``get_rheology.m:520-560`` exactly.
    """
    from scipy.special import lpmv
    from math import factorial

    lmax = 2 * l_max - 1
    d = 180.0 / (2 * lmax)
    lat = np.arange(-90.0 + d / 2.0, 90.0 - d / 2.0 + 1e-9, d)
    lon = np.arange(-180.0 + d / 2.0, 180.0 - d / 2.0 + 1e-9, d)
    t = np.sin(np.radians(lat))

    # 4pi-normalized associated Legendre (no Condon-Shortley; scipy's lpmv
    # includes (-1)^m, cancelled by the notebook's explicit (-1)^m factor).
    norm = math.sqrt((2 - (1 if m == 0 else 0)) * (2 * n + 1)
                     * factorial(n - m) / factorial(n + m))
    P = norm * lpmv(m, n, t)
    z = P[:, None] * np.cos(m * np.radians(lon))[None, :]
    return float(z.max() - z.min())


def _p2p_to_mu_variable(n_lv, m_lv, p2p_percent):
    """Convert a peak-to-peak percent amplitude to CSH mu_variable entries.

    Mirrors ``get_rheology.m:527-560``: amp = (p2p/100)/Delta for m=0;
    for m>0 a conjugate pair with sqrt(2)/2 and Condon-Shortley factors.
    """
    if m_lv < 0:
        # MATLAB's m<0 branch carries +/- i factors not replicated here.
        raise ValueError("pass the positive order; the conjugate is added")
    base = (p2p_percent / 100.0) / _delta_unit_map(n_lv, m_lv)
    if m_lv == 0:
        return [(n_lv, 0, base)]
    a = math.sqrt(2.0) / 2.0 * base
    return [(n_lv, m_lv, a), (n_lv, -m_lv, (-1) ** m_lv * a)]


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
        n_layers = 10  # artificial core + 9 Weber layers (notebook config)
        ocean_flags = [int(model.ocean[i]) for i in range(n_layers)]
        # Exactly one fluid layer (the Weber outer core), at layer index 2
        # (index 1 is the rigidified solid inner core).
        assert sum(ocean_flags) == 1
        assert ocean_flags[2] == 1
        assert float(model.mu0[2]) == 0.0
        for i in [1, 3, 4, 5, 6, 7, 8, 9]:
            assert float(model.mu0[i]) > 0.0

    def test_uniform_k2_matches_matlab(self):
        """1D ocean path on the notebook model reproduces MATLAB's k2_Q.

        Strongest available anchor: measured agreement is ~2e-9 relative
        (0.02315914222851756 vs k2_Q = 0.023159142178491576).
        """
        from pylov3d.love import get_love

        ref = _load_qin_reference("moon_3D_UM_20")
        model = _build_weber_moon_model()
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        numerics = make_numerics(n_layers=10, method="variable", Nrbase=50)
        love, _, _ = get_love(model, forcing, numerics)
        k2 = complex(love.k[0])
        assert abs(k2.imag) < 1e-10
        assert abs(k2.real - ref["k2_uniform"]) / ref["k2_uniform"] < 1e-6


# ---------------------------------------------------------------------------
# Coupled-ocean solver validation  (auto-skips until TASK-007)
# ---------------------------------------------------------------------------

class TestMoonCoupledOceanValidation:
    """Coupled NumPy ocean solve vs MATLAB/Qin reference spectra.

    Mirrors ``TestEnceladusBenchmark.test_lateral_love_spectra`` but on the
    ocean-bearing Weber Moon, configured exactly as the MATLAB notebook
    (``Test_Moon_MultiLayered_Lateral_Variations.mlx``).
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
        # regime).  The amp grid is peak-to-peak percent (notebook x-axis:
        # "100 |dmu/mu0| [%]").
        idx_amp = min(4, len(ref["amp"]) - 1)
        p2p_percent = float(ref["amp"][idx_amp])
        entries = _p2p_to_mu_variable(n_lv, m_lv, p2p_percent)

        raw_model = _build_weber_moon_model()
        # Td=1.0 vs the notebook's lunar period is a verified no-op: the
        # model is elastic (no eta), so the response is frequency-independent.
        forcing = make_forcing(Td=1.0, n=2, m=0, F=1.0)
        # Notebook numerics: variable grid, Nrbase=50.  perturbation_order=2
        # (notebook: 3) keeps runtime sane; order-3 reference modes simply
        # find no (n, m) match and are skipped, and order<=2 mode values are
        # perturbatively insensitive to the truncation at these amplitudes.
        numerics = make_numerics(
            n_layers=10, method="variable", Nrbase=50, perturbation_order=2,
        )
        numerics, model = set_boundary_indices(numerics, raw_model)
        model = get_rheology(model, forcing)

        # The lateral variation spans the full mantle region (notebook:
        # UM = layers 6:9, LM = 4:5, 1-based).
        host = "UM" if "_UM_" in stem else "LM"
        mu_variable = {lay: list(entries) for lay in _HOST_LAYERS[host]}

        model, lateral = process_lateral_variations(
            model, forcing, mu_variable=mu_variable,
        )
        couplings = get_couplings(
            lateral.variations, forcing.n, forcing.m,
            perturbation_order=numerics.perturbation_order,
        )

        y_sol, _r, _Y, _aux = _get_solution_coupled(
            model, forcing, numerics, couplings, lateral,
        )

        love = extract_love_numbers(y_sol, model, forcing, couplings=couplings)

        compared = 0
        for i in range(len(ref["n"])):
            n_m, m_m = int(ref["n"][i]), int(ref["m"][i])
            if m_m < 0:
                continue  # notebook drops negative-m reference rows
            idx_py = np.where((love.n == n_m) & (love.m == m_m))[0]
            if len(idx_py) == 0:
                continue
            k_py = love.k[idx_py[0]]
            k_ref = ref["k"][i, idx_amp]
            # Real -> complex SH basis for Qin's data (notebook Qin-transform
            # cell): zonal forcing divides m>0 reference modes by sqrt(2);
            # non-zonal forcing divides m==0 modes instead.
            if forcing.m == 0 and m_m > 0:
                k_ref /= math.sqrt(2.0)
            elif forcing.m != 0 and m_m == 0:
                k_ref /= math.sqrt(2.0)
            if n_m == forcing.n and m_m == forcing.m:
                # The reference stores the forcing mode as the DEVIATION
                # (k - k2_uniform) — notebook: k2ST(iforcing) = k - k2_uniform.
                k_py = k_py - ref["k2_uniform"]
            if abs(k_ref) < 1e-7 * ref["k2_uniform"]:
                continue  # below the reference's own noise floor
            rel_error = abs(k_py - k_ref) / abs(k_ref)
            order_m = int(ref["order"][i])
            # Measured errors: order-1 modes agree to 2e-6..5e-6 relative and
            # the forcing (order-2-behaved) deviation to ~0.3%; tolerances
            # keep ~100x/15x headroom rather than the siblings' 1%/5%.
            tol = 0.001 if order_m == 1 else 0.05 if order_m == 2 else 0.10
            assert rel_error < tol, (
                f"{stem} mode (n={n_m}, m={m_m}): py={k_py:.6e}, "
                f"ref={k_ref:.6e}, rel_error={rel_error:.2%} > {tol:.2%}"
            )
            compared += 1

        assert compared > 0, (
            "No modes compared — mode-matching found no overlapping (n,m) "
            "with non-trivial amplitude; test would pass vacuously."
        )
