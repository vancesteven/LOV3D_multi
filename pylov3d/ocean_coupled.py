# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Coupled-solver propagator for subsurface-ocean layers.

Inside an inviscid ocean the elastic system does not apply: displacements
and stresses are undefined and only the potential propagates, so the
8N×8N propagator is zero except the per-mode Poisson (Laplace) blocks.
Translates the ``ocean_flag==1`` branch of ``get_Aprop`` in MATLAB
``src/get_solution.m:1924-1935``.
"""

from __future__ import annotations

import math

import numpy as np

from .propagator import build_others


def build_aprop_ocean_coupled(r, n_s, Gg, gK=0.0):
    """Build the 8N×8N in-ocean propagator (Laplace-only Poisson blocks).

    MATLAB's degree-0 special case rewrites only the FIRST mode's first
    Poisson row (``Aprop(6N+1,...)``) because ``n_s`` always lists a
    degree-0 mode first when one exists; here the rewrite is applied to
    whichever mode has n==0, which reduces to MATLAB's behavior.

    ``gK`` (gravity at *r*) is only used by the degree-0 rows.
    """
    N = len(n_s)
    N6 = 6 * N
    N8 = 8 * N
    Ap = np.zeros((N8, N8), dtype=np.complex128)
    for k in range(N):
        n = int(n_s[k])
        _, _, _, _, _, _, _, A100, A101, A102, _, _ = build_others(n, 0.0, Gg)
        b = N6 + 2 * k
        Ap[b:b + 2, b:b + 2] = A100 + A101 / r + A102 / r**2
        if n == 0:
            # MATLAB get_solution.m:1928-1931 (Longman degree-0 row)
            Ap[b, :] = 0
            Ap[b, b] = 4 * math.pi * Gg / gK
    return Ap
