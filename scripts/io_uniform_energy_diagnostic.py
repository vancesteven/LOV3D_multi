#!/usr/bin/env python3
"""Diagnose TASK-046 uniform direct-energy reconstruction against MATLAB.

The uniform Love solution is already validated against the MATLAB Gate-C
anchor, so this script isolates only the post-solve stress/strain and energy
reconstruction. It reports the radial contribution to the monopole energy and
repeats the contraction after imposing MATLAB get_solution.m's endpoint
convention: the outermost u_dot/stress/strain row is left zero.

Use ``--nrbase`` to build a radial-convergence ladder without changing any
other physics or sign conventions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pylov3d.energy_couplings import get_energy_couplings
from pylov3d.energy_fields import recover_coupled_fields
from pylov3d.grid import set_boundary_indices
from pylov3d.io_lateral import build_io_forcings, build_io_model, io_default_numerics
from pylov3d.rheology import get_rheology
from pylov3d.solver import get_solution


def contract_uniform(results, forcings, model, numerics, zero_surface=False):
    modes = sorted((f.n, f.m) for f in forcings)
    n_s = np.asarray([x[0] for x in modes], dtype=int)
    m_s = np.asarray([x[1] for x in modes], dtype=int)
    idx = {m: i for i, m in enumerate(modes)}
    nr = numerics.Nr
    r = results[0][1]

    stress = np.zeros((nr + 1, 6, len(modes)), dtype=complex)
    strain = np.zeros_like(stress)
    for f, (y, rj, aprop) in zip(forcings, results):
        _, sf, ef = recover_coupled_fields(
            y, rj, aprop, model, np.asarray([f.n]), numerics
        )
        k = idx[(f.n, f.m)]
        stress[:, :, k] += f.F * sf.reshape(nr + 1, 1, 6)[:, 0, :]
        strain[:, :, k] += f.F * ef.reshape(nr + 1, 1, 6)[:, 0, :]

    if zero_surface:
        stress[-1, :, :] = 0.0
        strain[-1, :, :] = 0.0

    reorder = [1, 2, 3, 4, 5, 0]
    sp = stress[:, reorder, :]
    ep = strain[:, reorder, :]
    sn = np.zeros_like(sp)
    en = np.zeros_like(ep)
    for i, (n, m) in enumerate(modes):
        j = idx[(n, -m)]
        sn[:, :, i] = np.conj(sp[:, :, j])
        en[:, :, i] = np.conj(ep[:, :, j])

    ec = get_energy_couplings(n_s, m_s, Nenergy=numerics.Nenergy)
    E = ec.EC
    zero = np.where((ec.n_en == 0) & (ec.m_en == 0))[0]
    if len(zero) != 1:
        raise RuntimeError("missing unique (0,0) energy mode")
    kz = int(zero[0])
    profile = np.zeros(nr + 1, dtype=float)
    offsets = np.asarray([-2, -1, 0, 1, 2, 0])
    nz = np.nonzero(E[:, :, kz, :, :])
    for i1, i2, i3, i4 in zip(*nz):
        n2a = int(n_s[i1] + offsets[i3])
        n2b = int(n_s[i2] + offsets[i4])
        phase1 = (-1) ** (n2a + int(n_s[i1]) - int(m_s[i1]))
        phase2 = (-1) ** (n2b + int(n_s[i2]) - int(m_s[i2]))
        c = E[i1, i2, kz, i3, i4]
        term = (
            1j * 2*np.pi * phase1 * sn[:, i3, i1] * ep[:, i4, i2] * c
            - 1j * 2*np.pi * phase2 * sp[:, i3, i1] * en[:, i4, i2] * c
        )
        profile += term.real

    rm = 0.5 * (r[:-1] + r[1:])
    shell = rm**2 * np.diff(r) * 0.5 * (profile[:-1] + profile[1:])
    integral = -float(np.sum(shell))
    return integral, profile, shell, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrbase", type=int, default=10)
    args = parser.parse_args()

    raw = build_io_model()
    forcings = build_io_forcings()
    numerics = io_default_numerics(args.nrbase)
    numerics, model = set_boundary_indices(numerics, raw)
    model = get_rheology(model, forcings)
    results = []
    for f in forcings:
        y, r, _Y, aprop = get_solution(model, f, numerics)
        results.append((y, r, aprop))

    e_full, prof, shell, r = contract_uniform(results, forcings, model, numerics, False)
    e_zero, _, shell_zero, _ = contract_uniform(results, forcings, model, numerics, True)

    matlab_target = 2.1668778416
    print(f"TASK-046 uniform energy diagnostic, Nrbase={args.nrbase}")
    print(f"r range: {r[0]:.8f} .. {r[-1]:.8f}")
    print(f"direct E, solver-consistent endpoint handling: {e_full:.12e}")
    print(f"direct E, MATLAB zero-surface convention: {e_zero:.12e}")
    print(f"|E_direct|: {abs(e_full):.12e}")
    print(f"MATLAB coefficient-path target at Nrbase=50: {matlab_target:.12e}")
    print(f"magnitude relerr vs coefficient-path target: {abs(abs(e_full)-matlab_target)/matlab_target:.6%}")
    order = np.argsort(np.abs(shell))[::-1][:10]
    print("largest |radial shell contributions| (solver-consistent):")
    for k in order:
        print(
            f"  shell {k:4d}: rmid={0.5*(r[k]+r[k+1]):.8f} "
            f"dE={-shell[k]:+.12e} dE_zeroSurf={-shell_zero[k]:+.12e}"
        )
    print(f"surface profile value: {prof[-1]:+.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())