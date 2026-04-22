"""Physical constants and array size limits for pylov3d."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

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
