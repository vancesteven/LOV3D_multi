"""pylov3d — GPU-accelerated tidal Love number computation.

Uses JAX for jit compilation, automatic differentiation, and vectorized
parallelism over forcings/frequencies via vmap.
"""

from ._version import __version__
from .constants import G, MAX_LAYERS, MAX_MODES, MAX_NR_TOTAL
from .types import (
    InteriorModel,
    Forcing,
    NumericsConfig,
    LoveSpectra,
    RadialSolution,
    EnergySpectra,
    make_interior_model,
    make_forcing,
    make_numerics,
)
from .grid import set_boundary_indices
from .rheology import get_rheology, normalize, compute_complex_rheology
from .propagator import build_aprop, compute_gravity
from .solver import get_solution
from .love import get_love, extract_love_numbers
from .energy import (
    get_energy,
    compute_stress_strain,
    build_A14_A15,
    global_dissipation,
)

__all__ = [
    "__version__",
    "G",
    "MAX_LAYERS",
    "MAX_MODES",
    "MAX_NR_TOTAL",
    "InteriorModel",
    "Forcing",
    "NumericsConfig",
    "LoveSpectra",
    "RadialSolution",
    "EnergySpectra",
    "make_interior_model",
    "make_forcing",
    "make_numerics",
    "set_boundary_indices",
    "get_rheology",
    "normalize",
    "compute_complex_rheology",
    "build_aprop",
    "compute_gravity",
    "get_solution",
    "get_love",
    "extract_love_numbers",
    "get_energy",
    "compute_stress_strain",
    "build_A14_A15",
    "global_dissipation",
]
