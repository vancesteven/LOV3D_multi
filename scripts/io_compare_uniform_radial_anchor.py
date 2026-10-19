#!/usr/bin/env python3
"""Compare Python uniform Io radial fields with MATLAB TASK-046 anchor.

Run MATLAB first:

    /Applications/MATLAB_R2025b.app/bin/matlab -batch \
        "run('scripts/io_matlab_uniform_radial_anchor.m')"

Then run this script from the repository root. The comparison is deliberately
point-by-point and block-by-block so the remaining direct-energy discrepancy can
be localized to state propagation, GSH displacement recovery, stress recovery,
strain recovery, or only the final energy contraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.energy_fields import recover_coupled_fields
from pylov3d.energy_multibasis import get_energy_coupled_multibasis
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import build_io_forcings, build_io_model, io_default_numerics
from pylov3d.love import extract_love_numbers
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution

ANCHOR = REPO_ROOT / "data" / "tests" / "io" / "io_uniform_radial_anchor.mat"


def rel_l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    den = np.linalg.norm(b.ravel())
    return np.linalg.norm((a-b).ravel()) / max(den, np.finfo(float).tiny)


def max_scaled(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    scale = np.maximum(np.abs(b), 1e-14 * max(float(np.max(np.abs(b))), 1.0))
    return float(np.max(np.abs(a-b) / scale))


def python_matlab_style(raw_model, forcings, numerics, forcing):
    numerics, model = set_boundary_indices(numerics, raw_model)
    model = get_rheology(model, forcings)
    y, r, _Y, aprop = get_solution(model, forcing, numerics)
    u, stress, strain = recover_coupled_fields(
        y, r, aprop, model, np.asarray([forcing.n], dtype=int), numerics,
    )
    out = np.zeros((len(r), 24), dtype=np.complex128)
    out[:, 0] = r
    out[:, 1:9] = y
    out[:, 9:12] = u
    out[:, 12:18] = stress
    out[:, 18:24] = strain
    love = extract_love_numbers(y, model, forcing)
    idx = np.where((np.asarray(love.n)==forcing.n) & (np.asarray(love.m)==forcing.m))[0][0]
    return out, complex(love.k[idx]), numerics, model, (y, r, aprop)


def report_block(name, py, ml, sl):
    p = py[sl]
    m = ml[sl]
    print(f"{name:12s} relL2={rel_l2(p,m):.6e}  maxScaled={max_scaled(p,m):.6e}")


def main():
    if not ANCHOR.exists():
        raise SystemExit(
            f"missing {ANCHOR}\nRun scripts/io_matlab_uniform_radial_anchor.m in MATLAB first."
        )

    d = loadmat(ANCHOR, squeeze_me=True)
    ml = np.asarray(d["y_m0"])
    if ml.ndim == 1:
        ml = ml[None, :]

    raw = build_io_model()
    forcings = build_io_forcings()
    py, k_py, numerics, model, sol0 = python_matlab_style(
        raw, forcings, io_default_numerics(50), forcings[0]
    )

    if py.shape != ml.shape:
        print(f"shape mismatch: Python {py.shape}, MATLAB {ml.shape}")
        print(f"Python Nrlayer={np.asarray(numerics.Nrlayer)[:raw.n_layers]}")
        print(f"MATLAB Nrlayer={np.asarray(d['Nrlayer']).ravel()}")
        return 2

    k_ml = complex(np.asarray(d["k_forcing"]).ravel()[0])
    print("TASK-046 uniform radial-field comparison, Nrbase=50")
    print(f"shape: {py.shape}")
    print(f"max |dr|: {np.max(np.abs(py[:,0].real-ml[:,0].real)):.6e}")
    print(f"k Python: {k_py.real:+.12e}{k_py.imag:+.12e}i")
    print(f"k MATLAB: {k_ml.real:+.12e}{k_ml.imag:+.12e}i")
    print(f"k relerr: {abs(k_py-k_ml)/abs(k_ml):.6e}")
    print("\nfield blocks, all rows:")
    report_block("state U..dPhi", py, ml, (slice(None), slice(1,9)))
    report_block("u_GSH", py, ml, (slice(None), slice(9,12)))
    report_block("stress", py, ml, (slice(None), slice(12,18)))
    report_block("strain", py, ml, (slice(None), slice(18,24)))

    print("\nfield blocks, excluding outermost row:")
    report_block("state U..dPhi", py, ml, (slice(None,-1), slice(1,9)))
    report_block("u_GSH", py, ml, (slice(None,-1), slice(9,12)))
    report_block("stress", py, ml, (slice(None,-1), slice(12,18)))
    report_block("strain", py, ml, (slice(None,-1), slice(18,24)))

    print("\nper-component relL2, excluding outermost row:")
    stress_names = ["sig_nn0","sig_n,n-2,2","sig_n,n-1,2","sig_nn2","sig_n,n+1,2","sig_n,n+2,2"]
    strain_names = ["eps_nn0","eps_n,n-2,2","eps_n,n-1,2","eps_nn2","eps_n,n+1,2","eps_n,n+2,2"]
    for j, name in enumerate(stress_names):
        print(f"  {name:13s}: {rel_l2(py[:-1,12+j], ml[:-1,12+j]):.6e}")
    for j, name in enumerate(strain_names):
        print(f"  {name:13s}: {rel_l2(py[:-1,18+j], ml[:-1,18+j]):.6e}")

    # Rebuild all three uniform forcing solutions and compare the actual E00(r)
    # profile before radial integration. This is the decisive test for whether
    # the remaining discrepancy lives in the GSH angular contraction or only
    # in the final radial integral/sign convention.
    solutions = []
    n_s_list = []
    m_s_list = []
    for f in forcings:
        y, r, _Y, aprop = get_solution(model, f, numerics)
        solutions.append((y, r, aprop))
        n_s_list.append(np.asarray([f.n], dtype=int))
        m_s_list.append(np.asarray([f.m], dtype=int))
    e_py = get_energy_coupled_multibasis(
        solutions,
        forcings,
        model,
        numerics,
        n_s_list,
        m_s_list,
        Nenergy=numerics.Nenergy,
    )
    iz_py = np.where((np.asarray(e_py.n)==0) & (np.asarray(e_py.m)==0))[0]

    e_ml = np.real_if_close(np.asarray(d["energy_integral"]).ravel())
    n_ml = np.asarray(d["energy_n"]).ravel().astype(int)
    m_ml = np.asarray(d["energy_m"]).ravel().astype(int)
    iz_ml = np.where((n_ml==0) & (m_ml==0))[0]
    prof_ml_all = np.asarray(d["energy_profile"])
    if prof_ml_all.ndim == 1:
        prof_ml_all = prof_ml_all[:, None]

    if len(iz_py) and len(iz_ml):
        p_py = np.asarray(e_py.energy_profile[:, iz_py[0]], dtype=float)
        p_ml = np.real_if_close(prof_ml_all[:, iz_ml[0]]).astype(float)
        int_py = float(np.real_if_close(e_py.energy_integral[iz_py[0]]))
        int_ml = float(np.real_if_close(e_ml[iz_ml[0]]))
        print("\nenergy-contraction comparison:")
        print(f"Python raw E00 integral: {int_py:+.12e}")
        print(f"MATLAB raw E00 integral: {int_ml:+.12e}")
        print(f"profile relL2, all rows: {rel_l2(p_py,p_ml):.6e}")
        print(f"profile relL2, excluding outermost row: {rel_l2(p_py[:-1],p_ml[:-1]):.6e}")
        print(f"profile relL2 after global sign flip: {rel_l2(-p_py[:-1],p_ml[:-1]):.6e}")
        # Best scalar fit is diagnostic only. A value far from +/-1 indicates
        # a coupling/prefactor mismatch rather than a sign convention.
        den = float(np.vdot(p_py[:-1], p_py[:-1]).real)
        alpha = float(np.vdot(p_py[:-1], p_ml[:-1]).real / den) if den else np.nan
        print(f"best scalar alpha (MATLAB ~= alpha*Python): {alpha:+.12e}")
        print("largest |MATLAB-Python| E00 profile rows:")
        order = np.argsort(np.abs(p_ml-p_py))[::-1][:10]
        r = np.asarray(solutions[0][1])
        for k in order:
            print(
                f"  row {k:4d} r={r[k]:.8f} "
                f"Python={p_py[k]:+.8e} MATLAB={p_ml[k]:+.8e} "
                f"diff={p_py[k]-p_ml[k]:+.8e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())