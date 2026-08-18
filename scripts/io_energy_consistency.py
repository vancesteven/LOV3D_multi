# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""TASK-046 Gates A/B -- Io viscoelastic lateral-rheology energy consistency.

Reproduces the upstream MATLAB ``tests/Consistency_test_Energy.m`` Io
four-layer model at a short radial-refinement ladder (default
``Nrbase in {5, 10, 20, 50}``; ``--full-ladder`` adds ``{100, 200}``),
solving all three (2,0)/(2,-2)/(2,2) eccentricity-tide forcing components
for both the spherically-symmetric ("uniform") and the laterally
mu+eta-varying ("lateral") viscoelastic model, and checks:

- Gate A: finite complex Love spectra; the lateral model excites
  non-forcing coupled modes; the forcing-mode Im(k) is nonzero with a
  physically consistent (dissipative) sign; and reducing the lateral field
  amplitude toward zero converges the forcing-mode k toward the uniform
  value, with a reported (not assumed) convergence order.
- Gate B: direct stress-strain dissipation (``pylov3d.energy.get_energy`` /
  ``get_energy_coupled``) versus a Love-number-derived estimate built from
  the same cross-forcing double sum as the MATLAB test
  (``E_k = -sum_i sum_j F_i F_j Im k_j(n_f_i, m_f_i)``), for both models,
  at every rung.

Ks0 audit (TASK-046 spec caution)
---------------------------------------------------------------------------
``pylov3d/tests/test_energy.py``'s ``io_model`` fixture (and its copies in
``conftest.py``/``test_output_*.py``/``test_solver.py``/``test_rheology.py``/
``test_love.py``/``test_analytical.py``) use ``Ks0 = 200e16`` Pa. The
upstream ``tests/Consistency_test_Energy.m`` specifies ``Ks0 = 200e12`` Pa.
``git log --follow -p -- pylov3d/tests/test_energy.py`` shows the fixture
was introduced verbatim in the initial port commit (``76a54c3``, "Add
pylov3d: Python/NumPy port of LOV3D 1D solver (Milestone 1)") and has never
been touched since (only later SPDX-header and dead-code-cleanup commits
touch the file, not this line). ``200e16`` Pa does *not* appear anywhere in
``tests/Consistency_test_Energy.m``; it *does* appear, verbatim, in a
*different* upstream MATLAB script with an otherwise-identical Io model,
``scripts/multiple_layers_example.m`` ("SCRIPT USED TO TEST GET_LOVE",
lines 43/49/58: same four radii, same four densities, same four mu0, same
four eta0 as ``Consistency_test_Energy.m``, but ``Ks0 = 200e16``). This is
strong circumstantial evidence that the Python fixture was transcribed from
``multiple_layers_example.m`` rather than from
``Consistency_test_Energy.m``, and that ``200e16`` Pa (10^7x a real silicate
bulk modulus) is a typo *in that older MATLAB example script* that the
Python port then faithfully copied -- not a deliberate unit-convention
choice. This script uses the MATLAB ``Consistency_test_Energy.m`` value,
``Ks0 = 200e12`` Pa (:data:`pylov3d.io_lateral.IO_KS0`), and does **not**
touch the existing fixtures.

Direct-energy cross-term derivation (uniform model)
---------------------------------------------------------------------------
``pylov3d.energy.get_energy`` operates on a single forcing's 1-D (uncoupled)
solution; there is no ``get_energy_coupled``-style combiner for the
uncoupled 3-forcing case. This script instead sums the three forcings'
independent dissipations weighted by ``F_i^2``:
``e_direct_uniform = sum_i F_i^2 * get_energy(y_sol_i, ...).energy_integral[0]``.
This is not an approximation: for an uncoupled spherically-symmetric model,
forcing ``i``'s solution has angular support *only* at its own ``(n_f_i,
m_f_i)``, and the total-dissipation (monopole, ``n_en=m_en=0``) angular
contraction used throughout this codebase (``energy_couplings.py``'s Wigner
selection rule) only picks up cross terms between mode pairs whose orders
sum to zero -- i.e. between a mode and *its own* conjugate. Two *different*,
non-conjugate ``m``'s (0, -2, +2 are pairwise distinct here) therefore
contribute exactly zero cross term to the monopole energy; the sum over
independent per-forcing dissipations is exact, not an approximation. The
same cross-term selection rule is what makes the MATLAB Love-number double
sum ``E_k_Uni`` reduce to its diagonal (see ``love_energy_estimate`` below)
purely mechanically, without any special-casing -- both reductions are the
same physics, checked two different ways.

Normalization derivation (Gate B)
---------------------------------------------------------------------------
See ``_normalization_prefactor``'s docstring for the full derivation. In
short: writing the actual (dimensional) forcing potential for component
``i`` as ``F_i`` times the standard degree-``n`` reference amplitude
``gs*R/(2n+1)`` (the codebase's own ``pylov3d.energy.global_dissipation``
convention) and combining with ``pylov3d.rheology.normalize``'s definition
of the model's own normalized gravitational constant ``Gg`` gives a
*normalized*-units total physical dissipation
``e_love = C * E_k`` with ``C = gs_norm(surface)^2 / (2 * Gg)`` --
structurally the same ``1/Gg`` dependence as MATLAB's
``2*pi*10/(4*pi*Gg)``, with the ``2*pi`` (period-normalization) exactly
cancelling the ``4*pi`` and the physical assumption ``omega_norm = 2*pi``
following directly from ``Td`` being defined as one forcing period. The
derivation could not be closed against MATLAB's literal ``x10`` factor
without reading the (out-of-scope) MATLAB ``get_energy.m`` source; this
script reports both ``C``-normalized mismatches *and* the raw
(un-prefactored) ``e_direct / E_k`` ratio at every rung so a
Nrbase-independent-but-nonzero raw ratio remains a reportable, honest
fact per the spec's explicit fallback.

No pylov3d module is modified; this is a pure consumer of the existing
public API (``get_rheology``, ``process_lateral_variations``,
``get_couplings``, ``get_solution``, ``extract_love_numbers``,
``get_energy``, ``get_energy_coupled``), used directly (bypassing
``get_love``'s per-call ``get_couplings``) so all three forcings of the
lateral model share one coupled mode set -- the same requirement
``get_energy_coupled``'s own API (a single ``n_s``/``m_s`` shared across
all ``y_solutions``) already encodes.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.couplings import get_couplings
from pylov3d.energy import get_energy, get_energy_coupled
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import (
    IO_ASTHENOSPHERE_LAYER_INDEX,
    IO_FORCING_COMPONENTS,
    IO_FORCING_TD,
    build_io_forcings,
    build_io_model,
    io_default_numerics,
    io_mu_eta_variable,
)
from pylov3d.love import extract_love_numbers
from pylov3d.rheology import get_rheology, process_lateral_variations
from pylov3d.solver import get_solution
from pylov3d.types import InteriorModel, NumericsConfig

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "figures" / "proposal" / "io_energy_consistency.npz"
GATE_C_EXPORT = REPO_ROOT / "data" / "io" / "io_mu_eta_variable.npz"
DEFAULT_NRBASE_LADDER = (5, 10, 20, 50)
FULL_NRBASE_LADDER = (5, 10, 20, 50, 100, 200)


# ---------------------------------------------------------------------------
# Low-level solve helpers (bypass get_love to share couplings across forcings)
# ---------------------------------------------------------------------------

def _solve_uniform(model_raw: InteriorModel, forcings, numerics: NumericsConfig):
    numerics2, model2 = set_boundary_indices(numerics, model_raw)
    model2 = get_rheology(model2, forcings)
    results = []
    for f in forcings:
        y_sol, r_grid, _Y, Aprop_aux = get_solution(model2, f, numerics2)
        love = extract_love_numbers(y_sol, model2, f)
        results.append({"y_sol": y_sol, "r_grid": r_grid, "Aprop_aux": Aprop_aux, "love": love})
    return model2, numerics2, results


def _solve_lateral(
    model_raw: InteriorModel, forcings, numerics: NumericsConfig,
    mu_variable: dict, eta_variable: dict,
):
    numerics2, model2 = set_boundary_indices(numerics, model_raw)
    model2 = get_rheology(model2, forcings)
    model2, lateral = process_lateral_variations(
        model2, forcings, mu_variable=mu_variable, eta_variable=eta_variable,
        rheology_cutoff=numerics2.rheology_cutoff,
    )
    # Couplings built once from the (2,0) forcing's closure and reused for
    # all three forcings -- required so get_energy_coupled's single shared
    # n_s/m_s is valid for every y_solution passed to it.
    couplings = get_couplings(
        lateral.variations, 2, 0, perturbation_order=numerics2.perturbation_order,
    )
    results = []
    for f in forcings:
        y_sol, r_grid, _Y, Aprop_aux = get_solution(
            model2, f, numerics2, couplings=couplings, lateral=lateral,
        )
        love = extract_love_numbers(y_sol, model2, f, couplings=couplings)
        results.append({"y_sol": y_sol, "r_grid": r_grid, "Aprop_aux": Aprop_aux, "love": love})
    return model2, numerics2, couplings, lateral, results


# ---------------------------------------------------------------------------
# Gate A / B metrics
# ---------------------------------------------------------------------------

def love_energy_estimate(love_list, forcings) -> float:
    """``E_k = -sum_i sum_j F_i F_j Im(k_j at forcing i's mode))``.

    Matches ``tests/Consistency_test_Energy.m`` lines 246-261 exactly
    (including the cross-forcing terms). For 1-mode (uncoupled) spectra
    this mechanically reduces to the diagonal ``i == j`` terms, since
    ``lj``'s only mode is ``(n_f_j, m_f_j)`` and the three forcing modes
    are pairwise distinct -- see module docstring.
    """
    E_k = 0.0
    for fi in forcings:
        for j, fj in enumerate(forcings):
            lj = love_list[j]
            idx = np.where((np.asarray(lj.n) == fi.n) & (np.asarray(lj.m) == fi.m))[0]
            if len(idx) == 0:
                continue
            E_k -= float(fi.F) * float(fj.F) * complex(lj.k[idx[0]]).imag
    return E_k


def _normalization_prefactor(model: InteriorModel) -> float:
    r"""Derived Love-estimate -> direct-energy normalization factor ``C``.

    Full derivation (normalized/dimensionless "solver units" throughout,
    ``rheology.normalize``'s convention: length by ``R0_surface``, rigidity
    by ``mu0_surface``, density by ``rho0_surface``, time by the forcing
    period ``Td``, and gravitational constant replaced by
    ``Gg = G*(R0_surface[m])^2*rho0_surface^2/mu0_surface``):

    1. ``pylov3d.energy.global_dissipation``'s own docstring gives the
       *physical* total dissipation from a single degree-``n`` Love number
       as ``E_dot = -(2n+1)^2/(4 pi G) * omega * R^(2n-1) * Im(k_n) *
       |Phi_n|^2``, valid when the forcing potential amplitude is
       ``|Phi_n| = gs*R/(2n+1)`` -- the codebase's own reference tidal
       potential unit. This script's (and MATLAB's) forcing ``F_i`` values
       are dimensionless multipliers on exactly that reference unit
       (standard Kaula/eccentricity-tide decomposition convention), so the
       physical potential of component ``i`` is ``F_i * gs*R/(2n+1)``.
    2. Substituting and summing the cross terms
       ``sum_i sum_j F_i F_j Im(k_j at mode i)`` collapses the ``(2n+1)^2``
       against the two ``1/(2n+1)`` factors from ``|Phi_n|^2``, leaving
       ``E_dot_phys = [omega * R^5 * gs^2 / (4 pi G)] * E_k`` for ``n=2``,
       with ``E_k`` as defined by :func:`love_energy_estimate` (the minus
       sign already absorbed).
    3. Converting the bracket to normalized units: ``omega_norm = 2*pi``
       exactly (``Td`` is defined as one forcing period, so
       ``omega_phys * Td = 2*pi`` always); ``R_norm = 1`` at the surface;
       ``gs_phys = gs_norm * mu_ref/(rho_ref * R_ref[m])``; and eliminating
       ``G`` via ``Gg = G*R_ref[m]^2*rho_ref^2/mu_ref`` gives, after
       algebra, ``C = gs_norm(surface)^2 / (2 * Gg)`` -- independent of
       ``Td``, ``R_ref``, ``rho_ref``, ``mu_ref`` individually (they cancel),
       leaving only the two normalized-model quantities ``gs`` and ``Gg``.

    This has the same structural ``1/Gg`` dependence as MATLAB's
    ``2*pi*10/(4*pi*Gg)`` (the ``2*pi``/``4*pi`` here already cancelled to
    ``1/2``); the literal factor of 10 in the MATLAB constant was not
    independently re-derived (would require reading the out-of-scope
    MATLAB ``get_energy.m`` source) -- see module docstring for how this
    script reports the resulting (possibly incomplete) normalization
    honestly rather than papering over it.
    """
    Gg = float(model.Gg)
    gs_surface = float(model.gs[model.n_layers - 1])
    return (gs_surface ** 2) / (2.0 * Gg)


def _forcing_convergence_check(nrbase: int, mu_variable: dict, eta_variable: dict) -> dict:
    """Gate A4: scaling the lateral field by eps -> forcing-mode k -> uniform k.

    Solves the (2,0) forcing at the full field and at the field scaled by
    eps in {0.1, 0.01}, plus the uniform (no-lateral) reference, all at a
    single fixed ``nrbase`` (cheaper than repeating this at every ladder
    rung; the convergence-in-eps behaviour is not expected to depend on
    radial resolution). Reports the fitted log-log convergence order
    between the two eps values (same construction as
    ``scripts/moon_lateral_convergence.py::_c20_channel_exponent``).
    """
    model_raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(nrbase)

    model_uni, numerics_uni, res_uni = _solve_uniform(model_raw, [forcings[0]], numerics)
    k_uniform = complex(res_uni[0]["love"].k[0])

    shifts = {}
    k_at_eps = {}
    for eps in (1.0, 0.1, 0.01):
        mu_eps = {
            IO_ASTHENOSPHERE_LAYER_INDEX: [
                (n, m, eps * amp) for n, m, amp in mu_variable[IO_ASTHENOSPHERE_LAYER_INDEX]
            ]
        }
        eta_eps = {
            IO_ASTHENOSPHERE_LAYER_INDEX: [
                (n, m, eps * amp) for n, m, amp in eta_variable[IO_ASTHENOSPHERE_LAYER_INDEX]
            ]
        }
        _model2, _numerics2, couplings, _lateral, res_lat = _solve_lateral(
            model_raw, [forcings[0]], numerics, mu_eps, eta_eps,
        )
        love = res_lat[0]["love"]
        idx = np.where((love.n == 2) & (love.m == 0))[0][0]
        k_eps = complex(love.k[idx])
        k_at_eps[eps] = k_eps
        shifts[eps] = abs(k_eps - k_uniform)

    exponent = (
        math.log(shifts[0.01] / shifts[0.1]) / math.log(0.01 / 0.1)
        if shifts[0.1] > 0 and shifts[0.01] > 0 else float("nan")
    )
    return {
        "nrbase": nrbase,
        "k_uniform": k_uniform,
        "k_at_eps": k_at_eps,
        "shift_eps_1.0": shifts[1.0],
        "shift_eps_0.1": shifts[0.1],
        "shift_eps_0.01": shifts[0.01],
        "exponent": exponent,
    }


def _run_rung(nrbase: int, mu_variable: dict, eta_variable: dict) -> dict:
    t0 = time.perf_counter()
    model_raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(nrbase)

    model_uni, numerics_uni, res_uni = _solve_uniform(model_raw, forcings, numerics)
    model_lat, numerics_lat, couplings, lateral, res_lat = _solve_lateral(
        model_raw, forcings, numerics, mu_variable, eta_variable,
    )

    # --- Gate A1: finiteness ------------------------------------------------
    finite_uni = all(bool(np.all(np.isfinite(r["y_sol"]))) for r in res_uni)
    finite_lat = all(bool(np.all(np.isfinite(r["y_sol"]))) for r in res_lat)

    # --- Gate A2: lateral model excites non-forcing modes -------------------
    n_coupled_modes = int(len(couplings.n_s))
    excites_nonforcing = n_coupled_modes > 1

    # --- Gate A3: forcing-mode Im(k) sign ------------------------------------
    love20_uni = res_uni[0]["love"]
    k20_uni = complex(love20_uni.k[0])
    love20_lat = res_lat[0]["love"]
    idx20 = np.where((love20_lat.n == 2) & (love20_lat.m == 0))[0][0]
    k20_lat = complex(love20_lat.k[idx20])

    # --- Gate B: direct energy ------------------------------------------------
    # Sign convention (SIGN_FLIP_NOTE, see module docstring): get_energy /
    # get_energy_coupled compute dissipation[k] = Im(conj(stress) . strain).
    # For the leading-order deviatoric relation sigma ~ 2*muC*epsilon this
    # equals -Im(muC)*|epsilon|^2, i.e. it is the *negative* of the
    # standard time-averaged dissipated power Im(muC)*|epsilon|^2 for a
    # material with positive loss modulus. This model's own muC is
    # verified Im(muC) > 0 on every viscoelastic layer (standard lossy
    # convention), and get_energy's raw energy_integral is verified
    # negative here for a demonstrably lossy Io model at converged Nrbase
    # -- matching the derivation, not contradicting it. This is the same
    # "Im(k) < 0 signals dissipation, negate it" convention
    # pylov3d.energy.global_dissipation already applies explicitly; get_energy
    # apparently does not apply the equivalent negation for its own direct
    # integral. This script applies -1 here so both the direct and
    # Love-derived energies share one physical sign convention (positive =
    # dissipative); it does not modify pylov3d/energy.py itself.
    e_direct_uni_raw = 0.0
    for f, r in zip(forcings, res_uni):
        e = get_energy(r["y_sol"], r["r_grid"], r["Aprop_aux"], model_uni, f, numerics_uni)
        e_direct_uni_raw += (float(f.F) ** 2) * float(e.energy_integral[0])
    e_direct_uni = -e_direct_uni_raw

    y_solutions = [(r["y_sol"], r["r_grid"], r["Aprop_aux"]) for r in res_lat]
    e_coupled = get_energy_coupled(
        y_solutions, forcings, model_lat, numerics_lat,
        couplings.n_s, couplings.m_s, Nenergy=numerics_lat.Nenergy,
    )
    zero_idx = np.where((np.asarray(e_coupled.n) == 0) & (np.asarray(e_coupled.m) == 0))[0]
    e_direct_lat_raw = float(e_coupled.energy_integral[zero_idx[0]]) if len(zero_idx) else float("nan")
    e_direct_lat = -e_direct_lat_raw

    # --- Gate B: Love-number estimate -----------------------------------------
    love_uni_list = [r["love"] for r in res_uni]
    love_lat_list = [r["love"] for r in res_lat]
    E_k_uni = love_energy_estimate(love_uni_list, forcings)
    E_k_lat = love_energy_estimate(love_lat_list, forcings)

    C = _normalization_prefactor(model_uni)
    e_love_uni = C * E_k_uni
    e_love_lat = C * E_k_lat

    mismatch_uni = abs(e_direct_uni - e_love_uni) / abs(e_direct_uni) if e_direct_uni else float("nan")
    mismatch_lat = abs(e_direct_lat - e_love_lat) / abs(e_direct_lat) if e_direct_lat else float("nan")
    raw_ratio_uni = e_direct_uni / E_k_uni if E_k_uni else float("nan")
    raw_ratio_lat = e_direct_lat / E_k_lat if E_k_lat else float("nan")

    wall_s = time.perf_counter() - t0

    return {
        "nrbase": nrbase,
        "Nr": int(numerics_uni.Nr),
        "finite_uni": finite_uni,
        "finite_lat": finite_lat,
        "n_coupled_modes": n_coupled_modes,
        "excites_nonforcing": excites_nonforcing,
        "k20_uni": k20_uni,
        "k20_lat": k20_lat,
        "e_direct_uni": e_direct_uni,
        "e_direct_lat": e_direct_lat,
        "E_k_uni": E_k_uni,
        "E_k_lat": E_k_lat,
        "C_prefactor": C,
        "e_love_uni": e_love_uni,
        "e_love_lat": e_love_lat,
        "mismatch_uni": mismatch_uni,
        "mismatch_lat": mismatch_lat,
        "raw_ratio_uni": raw_ratio_uni,
        "raw_ratio_lat": raw_ratio_lat,
        "n_s_lat": np.asarray(couplings.n_s),
        "m_s_lat": np.asarray(couplings.m_s),
        "k_lat_by_forcing": [np.asarray(r["love"].k) for r in res_lat],
        "n_lat_by_forcing": [np.asarray(r["love"].n) for r in res_lat],
        "m_lat_by_forcing": [np.asarray(r["love"].m) for r in res_lat],
        "k_uni_by_forcing": [complex(r["love"].k[0]) for r in res_uni],
        "wall_s": wall_s,
    }


# ---------------------------------------------------------------------------
# Gate C prep: export the shared complex mu/eta coefficients for the .m script
# ---------------------------------------------------------------------------

_GATE_C_README = """\
TASK-046 Io lateral mu/eta export -- for machine B's native-MATLAB Gate C
anchor (scripts/io_energy_cross_check.m), run at Nrbase=50 per the spec.

Contents
--------
asthenosphere_layer_idx : 0-based Python layer index (2) of the Io
    asthenosphere layer these entries apply to; MATLAB
    Interior_Model(asthenosphere_layer_idx + 1) = Interior_Model(3).
mu_n, mu_m, mu_amp_real, mu_amp_imag : complex mu_variable entries
    (fractional delta-mu/mu0 SH coefficients), amp = amp_real + 1j*amp_imag.
eta_n, eta_m, eta_amp_real, eta_amp_imag : same for eta_variable
    (fractional delta-eta/eta0).

Basis convention warning (read before use in MATLAB)
---------------------------------------------------------------------------
These coefficients are in the pylov3d.rheology._sh_synthesis/_sh_analysis
basis (scipy.special.sph_harm_y: orthonormal, Condon-Shortley phase
included) -- NOT the pylov3d.mars_lateral / fully_normalized_legendre real
4pi-normalized basis used by the Mars/Moon *elastic* crust exports. See
pylov3d/io_lateral.py's module docstring for why: process_lateral_variations
routes viscoelastic layers (which the Io asthenosphere is) through
_sh_synthesis internally, so the input convention must match that basis,
not the elastic-branch one. If MATLAB's own get_rheology.m mu_latlon/
eta_latlon path expects a different convention (its own SPH_Tools/Legendre.m
recursion, likely matching the real 4pi convention instead -- see
Consistency_test_Energy.m lines 116-125, which pass raw lat/lon/z GRIDS,
not pre-converted SH coefficients, to Interior_Model(3).mu_latlon/
eta_latlon), then a native reproduction of the Io pattern in MATLAB should
most likely resynthesize the lat/lon grid directly from
pylov3d.io_lateral.io_mu_eta_grids() (or the pattern formulas in that
module) and hand it to MATLAB's own mu_latlon/eta_latlon struct path,
rather than converting these already-analyzed coefficients into a MATLAB
mu_variable matrix -- that conversion would need its own from-scratch
basis-matching derivation, which is explicitly out of scope for this
export (Gate C is B's MATLAB run; this file documents the ambiguity so B
does not silently assume a basis that has not been checked).

Model / forcing provenance
---------------------------------------------------------------------------
R0_km, rho0, mu0, Ks0, eta0 (core-to-surface, 4 layers): the
Consistency_test_Energy.m Io model (pylov3d.io_lateral.IO_R0_KM etc; Ks0
uses the MATLAB 200e12 Pa value, not the pylov3d/tests/test_energy.py
fixture's 200e16 -- see this export's script's module docstring for the
audit). forcing_n, forcing_m, forcing_F (3 rows): the (2,0)/(2,-2)/(2,2)
eccentricity-tide components. omega0, Td: Io's orbital frequency/period.
lmax_sh: SH truncation degree used for both mu and eta.
"""


def export_gate_c_fields(
    mu_variable: dict, eta_variable: dict, lmax_sh: int, path: Path = GATE_C_EXPORT,
) -> Path:
    from pylov3d.io_lateral import IO_ETA0, IO_KS0, IO_MU0, IO_R0_KM, IO_RHO0

    path.parent.mkdir(parents=True, exist_ok=True)
    mu_entries = mu_variable[IO_ASTHENOSPHERE_LAYER_INDEX]
    eta_entries = eta_variable[IO_ASTHENOSPHERE_LAYER_INDEX]
    mu_amp = np.array([e[2] for e in mu_entries], dtype=complex)
    eta_amp = np.array([e[2] for e in eta_entries], dtype=complex)

    np.savez(
        path,
        asthenosphere_layer_idx=IO_ASTHENOSPHERE_LAYER_INDEX,
        mu_n=np.array([e[0] for e in mu_entries], dtype=int),
        mu_m=np.array([e[1] for e in mu_entries], dtype=int),
        mu_amp_real=mu_amp.real, mu_amp_imag=mu_amp.imag,
        eta_n=np.array([e[0] for e in eta_entries], dtype=int),
        eta_m=np.array([e[1] for e in eta_entries], dtype=int),
        eta_amp_real=eta_amp.real, eta_amp_imag=eta_amp.imag,
        R0_km=np.asarray(IO_R0_KM), rho0=np.asarray(IO_RHO0),
        mu0=np.asarray(IO_MU0), Ks0=np.asarray(IO_KS0),
        eta0=np.array([float("nan") if v is None else v for v in IO_ETA0]),
        forcing_n=np.array([c[0] for c in IO_FORCING_COMPONENTS], dtype=int),
        forcing_m=np.array([c[1] for c in IO_FORCING_COMPONENTS], dtype=int),
        forcing_F=np.array([c[2] for c in IO_FORCING_COMPONENTS]),
        omega0=4.1086e-05, Td=IO_FORCING_TD, lmax_sh=lmax_sh,
        readme=_GATE_C_README,
    )
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ladder", action="store_true",
                         help="Use the full Nrbase ladder (adds 100, 200).")
    parser.add_argument("--nrbase-list", type=int, nargs="+", default=None)
    parser.add_argument("--convergence-nrbase", type=int, default=20)
    parser.add_argument("--lmax-sh", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gate-c-export", type=Path, default=GATE_C_EXPORT)
    args = parser.parse_args()

    if args.nrbase_list is not None:
        nrbase_list = tuple(args.nrbase_list)
    elif args.full_ladder:
        nrbase_list = FULL_NRBASE_LADDER
    else:
        nrbase_list = DEFAULT_NRBASE_LADDER

    t_start = time.perf_counter()

    print("[1/4] Building Io asthenosphere lateral mu/eta fields ...", flush=True)
    mu_variable, eta_variable, field_diag = io_mu_eta_variable(lmax_sh=args.lmax_sh)
    print(f"    mu entries: {len(mu_variable[IO_ASTHENOSPHERE_LAYER_INDEX])}"
          f"  eta entries: {len(eta_variable[IO_ASTHENOSPHERE_LAYER_INDEX])}")
    print("    mu degree spectrum (real 4pi, reporting basis):")
    for n, amp in sorted(field_diag["mu_degree_spectrum"].items()):
        print(f"      n={n:2d}  {amp:.6e}")
    print("    eta degree spectrum (real 4pi, reporting basis):")
    for n, amp in sorted(field_diag["eta_degree_spectrum"].items()):
        print(f"      n={n:2d}  {amp:.6e}")

    print("\n[2/4] Gate A4: eps-scaling convergence to the uniform k20 ...", flush=True)
    conv = _forcing_convergence_check(args.convergence_nrbase, mu_variable, eta_variable)
    print(f"    Nrbase={conv['nrbase']}  k_uniform={conv['k_uniform']!r}")
    for eps in (1.0, 0.1, 0.01):
        print(f"    eps={eps:<5} k={conv['k_at_eps'][eps]!r}  |shift|={conv[f'shift_eps_{eps}']:.6e}")
    print(f"    convergence exponent (eps 0.1 -> 0.01): {conv['exponent']:.4f}"
          f"  ({'first' if conv['exponent'] < 1.5 else 'second'}-order-like)")

    print(f"\n[3/4] Gate A/B ladder: Nrbase in {nrbase_list} ...", flush=True)
    ladder = []
    for nrbase in nrbase_list:
        print(f"  [solve] Nrbase={nrbase} ...", flush=True)
        row = _run_rung(nrbase, mu_variable, eta_variable)
        ladder.append(row)
        print(
            f"    Nr={row['Nr']} finite(uni,lat)=({row['finite_uni']},{row['finite_lat']}) "
            f"N_coupled={row['n_coupled_modes']} excites_nonforcing={row['excites_nonforcing']}\n"
            f"    k20_uni={row['k20_uni']!r}  k20_lat={row['k20_lat']!r}\n"
            f"    e_direct(uni,lat)=({row['e_direct_uni']:.6e}, {row['e_direct_lat']:.6e})\n"
            f"    E_k(uni,lat)=({row['E_k_uni']:.6e}, {row['E_k_lat']:.6e})  C={row['C_prefactor']:.6e}\n"
            f"    e_love(uni,lat)=({row['e_love_uni']:.6e}, {row['e_love_lat']:.6e})\n"
            f"    mismatch(uni,lat)=({row['mismatch_uni']:.6e}, {row['mismatch_lat']:.6e})\n"
            f"    raw_ratio(uni,lat)=({row['raw_ratio_uni']:.6e}, {row['raw_ratio_lat']:.6e})\n"
            f"    wall={row['wall_s']:.1f}s",
            flush=True,
        )

    print("\n[Gate B trend] mismatch vs Nrbase (must decrease for a pass):")
    print(f"{'Nrbase':>7} {'mismatch_uni':>14} {'mismatch_lat':>14} "
          f"{'raw_ratio_uni':>15} {'raw_ratio_lat':>15}")
    for row in ladder:
        print(f"{row['nrbase']:>7} {row['mismatch_uni']:14.6e} {row['mismatch_lat']:14.6e} "
              f"{row['raw_ratio_uni']:15.6e} {row['raw_ratio_lat']:15.6e}")
    mism_uni = [row["mismatch_uni"] for row in ladder]
    mism_lat = [row["mismatch_lat"] for row in ladder]
    shrinks_uni = all(b <= a for a, b in zip(mism_uni, mism_uni[1:]))
    shrinks_lat = all(b <= a for a, b in zip(mism_lat, mism_lat[1:]))
    print(f"  mismatch monotonically non-increasing: uniform={shrinks_uni}  lateral={shrinks_lat}")
    ratios_uni = [row["raw_ratio_uni"] for row in ladder]
    ratios_lat = [row["raw_ratio_lat"] for row in ladder]
    if len(ratios_uni) > 1:
        spread_uni = (max(ratios_uni) - min(ratios_uni)) / abs(np.mean(ratios_uni))
        spread_lat = (max(ratios_lat) - min(ratios_lat)) / abs(np.mean(ratios_lat))
        print(f"  raw_ratio fractional spread across ladder: uniform={spread_uni:.3e}  "
              f"lateral={spread_lat:.3e}  (small => Nrbase-independent normalization constant)")

    print(f"\n[4/4] Writing artifacts ...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        nrbase_list=np.asarray(nrbase_list, dtype=int),
        Nr_list=np.asarray([row["Nr"] for row in ladder], dtype=int),
        finite_uni=np.asarray([row["finite_uni"] for row in ladder]),
        finite_lat=np.asarray([row["finite_lat"] for row in ladder]),
        n_coupled_modes=np.asarray([row["n_coupled_modes"] for row in ladder], dtype=int),
        excites_nonforcing=np.asarray([row["excites_nonforcing"] for row in ladder]),
        k20_uni=np.asarray([row["k20_uni"] for row in ladder], dtype=complex),
        k20_lat=np.asarray([row["k20_lat"] for row in ladder], dtype=complex),
        e_direct_uni=np.asarray([row["e_direct_uni"] for row in ladder]),
        e_direct_lat=np.asarray([row["e_direct_lat"] for row in ladder]),
        E_k_uni=np.asarray([row["E_k_uni"] for row in ladder]),
        E_k_lat=np.asarray([row["E_k_lat"] for row in ladder]),
        C_prefactor=np.asarray([row["C_prefactor"] for row in ladder]),
        e_love_uni=np.asarray([row["e_love_uni"] for row in ladder]),
        e_love_lat=np.asarray([row["e_love_lat"] for row in ladder]),
        mismatch_uni=np.asarray(mism_uni),
        mismatch_lat=np.asarray(mism_lat),
        raw_ratio_uni=np.asarray(ratios_uni),
        raw_ratio_lat=np.asarray(ratios_lat),
        wall_s=np.asarray([row["wall_s"] for row in ladder]),
        conv_nrbase=conv["nrbase"],
        conv_k_uniform=conv["k_uniform"],
        conv_shift_eps_1_0=conv["shift_eps_1.0"],
        conv_shift_eps_0_1=conv["shift_eps_0.1"],
        conv_shift_eps_0_01=conv["shift_eps_0.01"],
        conv_exponent=conv["exponent"],
        mu_n=np.array([e[0] for e in mu_variable[IO_ASTHENOSPHERE_LAYER_INDEX]], dtype=int),
        mu_m=np.array([e[1] for e in mu_variable[IO_ASTHENOSPHERE_LAYER_INDEX]], dtype=int),
        mu_amp=np.array([e[2] for e in mu_variable[IO_ASTHENOSPHERE_LAYER_INDEX]], dtype=complex),
        eta_n=np.array([e[0] for e in eta_variable[IO_ASTHENOSPHERE_LAYER_INDEX]], dtype=int),
        eta_m=np.array([e[1] for e in eta_variable[IO_ASTHENOSPHERE_LAYER_INDEX]], dtype=int),
        eta_amp=np.array([e[2] for e in eta_variable[IO_ASTHENOSPHERE_LAYER_INDEX]], dtype=complex),
        mu_degree_spectrum_n=np.array(sorted(field_diag["mu_degree_spectrum"])),
        mu_degree_spectrum_amp=np.array(
            [field_diag["mu_degree_spectrum"][n] for n in sorted(field_diag["mu_degree_spectrum"])]
        ),
        eta_degree_spectrum_n=np.array(sorted(field_diag["eta_degree_spectrum"])),
        eta_degree_spectrum_amp=np.array(
            [field_diag["eta_degree_spectrum"][n] for n in sorted(field_diag["eta_degree_spectrum"])]
        ),
        lmax_sh=args.lmax_sh,
        total_wall_s=time.perf_counter() - t_start,
    )
    for i, row in enumerate(ladder):
        save[f"n_s_lat_rung{i}"] = row["n_s_lat"]
        save[f"m_s_lat_rung{i}"] = row["m_s_lat"]
        for j in range(3):
            save[f"k_lat_rung{i}_forcing{j}"] = row["k_lat_by_forcing"][j]
            save[f"n_lat_rung{i}_forcing{j}"] = row["n_lat_by_forcing"][j]
            save[f"m_lat_rung{i}_forcing{j}"] = row["m_lat_by_forcing"][j]
    np.savez(args.output, **save)

    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"  saved {args.output}")
    print(f"  SHA-256: {sha}")

    gate_c_path = export_gate_c_fields(mu_variable, eta_variable, args.lmax_sh, args.gate_c_export)
    sha_c = hashlib.sha256(gate_c_path.read_bytes()).hexdigest()
    print(f"  saved {gate_c_path} (for scripts/io_energy_cross_check.m)")
    print(f"  SHA-256: {sha_c}")

    print(f"\ntotal wall = {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
