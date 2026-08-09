# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Physical constants and array size limits for pylov3d."""

import jax
jax.config.update("jax_enable_x64", True)


# Gravitational constant [N m^2 kg^{-2}]
G = 6.67430e-11

# Array dimension limits for JAX static shapes
MAX_LAYERS = 16
MAX_MODES = 64
MAX_NR_PER_LAYER = 512
MAX_NR_TOTAL = 2048

# State vector size per mode (U, V, W, R, S, T, phi, dphi/dr)
STATE_SIZE = 8

# Number of coupling tensor types
N_COUPLING_TYPES = 27
