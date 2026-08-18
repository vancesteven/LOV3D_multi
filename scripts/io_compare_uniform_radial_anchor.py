#!/usr/bin/env python3
"""Compare Python uniform Io radial fields with MATLAB TASK-046 anchor.

Run MATLAB first:

    /Applications/MATLAB_R2025b.app/bin/matlab -batch \
        "run('scripts/io_matlab_uniform_radial_anchor.m')"

Then run this script from the repository root.  The comparison is deliberately
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
    return out, complex(love.k[idx]), numerics


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
    py, k_py, numerics = python_matlab_style(
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

    # MATLAB intentionally leaves the outermost recovered stress/strain row
    # zero.  Excluding that endpoint separates field-recovery parity from this
    # known integration convention.
    print("\nfield blocks, excluding outermost row:")
    report_block("state U..dPhi", py, ml, (slice(None,-1), slice(1,9)))
    report_block("u_GSH", py, ml, (slice(None,-1), slice(9,12)))
    report_block("stress", py, ml, (slice(None,-1), slice(12,18)))
    report_block("strain", py, ml, (slice(None,-1), slice(18,24)))

    # Per stress/strain component, useful if one GSH tensor component carries
    # the remaining disagreement.
    print("\nper-component relL2, excluding outermost row:")
    stress_names = ["sig_nn0","sig_n,n-2,2","sig_n,n-1,2","sig_nn2","sig_n,n+1,2","sig_n,n+2,2"]
    strain_names = ["eps_nn0","eps_n,n-2,2","eps_n,n-1,2","eps_nn2","eps_n,n+1,2","eps_n,n+2,2"]
    for j, name in enumerate(stress_names):
        print(f"  {name:13s}: {rel_l2(py[:-1,12+j], ml[:-1,12+j]):.6e}")
    for j, name in enumerate(strain_names):
        print(f"  {name:13s}: {rel_l2(py[:-1,18+j], ml[:-1,18+j]):.6e}")

    e_ml = np.asarray(d["energy_integral"]).ravel()
    n_ml = np.asarray(d["energy_n"]).ravel().astype(int)
    m_ml = np.asarray(d["energy_m"]).ravel().astype(int)
    iz = np.where((n_ml==0) & (m_ml==0))[0]
    if len(iz):
        print(f"\nMATLAB direct E00: {float(e_ml[iz[0]]):.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
