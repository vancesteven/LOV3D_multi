# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Import the neutral PlanetProfile/PlanetThrak radial ``.npz`` artifact.

The artifact schema is intentionally independent of either package's internal
classes. pylov3d uses only the fields required for radial tidal structure:
body radius, depth, density, bulk modulus and shear modulus. The first version
is elastic unless an explicit viscosity field is added to the artifact schema
later.

Because reducing a high-resolution PlanetProfile profile can change global
structure, this module also reports exact mass and spherical axial moment of
inertia for the piecewise-constant shells. Those diagnostics are kept separate
from the converter so a reduction cannot silently pass through the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import MAX_LAYERS
from .types import InteriorModel, make_interior_model


PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RadialArtifactShells:
    outer_radius_m: np.ndarray
    rho_kgm3: np.ndarray
    K_Pa: np.ndarray
    mu_Pa: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class RadialBulkDiagnostics:
    mass_kg: float
    axial_moi_kgm2: float
    cmr2: float
    mass_relative_error: float | None
    cmr2_error: float | None


def _read_metadata(data) -> dict[str, object]:
    meta = {}
    for key in data.files:
        if key.startswith("meta_"):
            meta[key[5:]] = np.asarray(data[key]).item()
    return meta


def load_radial_artifact_shells(
    path,
    *,
    body_radius_m: float | None = None,
    enforce_max_layers: bool = True,
) -> RadialArtifactShells:
    """Read and validate a neutral radial artifact as core-to-surface shells.

    ``enforce_max_layers=False`` is only for inspection/reduction workflows.
    Direct conversion to ``InteriorModel`` still enforces the JAX static layer
    limit and never silently decimates a high-resolution profile.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        version = int(np.asarray(data["schema_version"]).item())
        if version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported radial artifact schema_version={version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        meta = _read_metadata(data)
        if body_radius_m is None:
            if "body_radius_m" not in meta:
                raise ValueError(
                    "radial artifact requires meta_body_radius_m or an explicit body_radius_m"
                )
            body_radius_m = float(meta["body_radius_m"])
        if body_radius_m <= 0:
            raise ValueError("body_radius_m must be positive")

        required = ("depth_m", "density_kg_m3", "bulk_modulus_Pa", "shear_modulus_Pa")
        arrays = {}
        for name in required:
            if name not in data:
                raise ValueError(f"radial artifact is missing required field {name}")
            arrays[name] = np.asarray(data[name], dtype=float).reshape(-1)
        n = arrays["depth_m"].size
        if n < 2 or any(arr.size != n for arr in arrays.values()):
            raise ValueError("radial artifact fields must have equal length >= 2")
        if any(not np.all(np.isfinite(arr)) for arr in arrays.values()):
            raise ValueError("radial artifact tidal fields must be finite")

        r = float(body_radius_m) - arrays["depth_m"]
        if np.any(r <= 0) or np.any(r > float(body_radius_m) * (1.0 + 1e-12)):
            raise ValueError("depth_m values do not define positive shell outer radii")
        if np.any(arrays["density_kg_m3"] <= 0) or np.any(arrays["bulk_modulus_Pa"] <= 0):
            raise ValueError("density and bulk modulus must be positive")
        if np.any(arrays["shear_modulus_Pa"] < 0):
            raise ValueError("shear modulus must be non-negative")

        order = np.argsort(r)
        r = r[order]
        if np.any(np.diff(r) <= 0):
            raise ValueError("radial artifact shell radii must be unique")
        if enforce_max_layers and n > MAX_LAYERS:
            raise ValueError(
                f"radial artifact has {n} shells but pylov3d MAX_LAYERS={MAX_LAYERS}; "
                "inspect/reduce it explicitly before conversion"
            )
        return RadialArtifactShells(
            outer_radius_m=r,
            rho_kgm3=arrays["density_kg_m3"][order],
            K_Pa=arrays["bulk_modulus_Pa"][order],
            mu_Pa=arrays["shear_modulus_Pa"][order],
            metadata=meta,
        )


def radial_shell_bulk_diagnostics(shells: RadialArtifactShells) -> RadialBulkDiagnostics:
    """Compute exact M and C/MR^2 for piecewise-constant spherical shells."""
    r_out = np.asarray(shells.outer_radius_m, dtype=float)
    r_in = np.concatenate(([0.0], r_out[:-1]))
    rho = np.asarray(shells.rho_kgm3, dtype=float)
    mass = float((4.0 * np.pi / 3.0) * np.sum(rho * (r_out**3 - r_in**3)))
    C = float((8.0 * np.pi / 15.0) * np.sum(rho * (r_out**5 - r_in**5)))
    R = float(r_out[-1])
    cmr2 = C / (mass * R**2)

    mass_err = None
    if "body_mass_kg" in shells.metadata:
        reference = float(shells.metadata["body_mass_kg"])
        if reference <= 0:
            raise ValueError("meta_body_mass_kg must be positive")
        mass_err = (mass - reference) / reference
    cmr2_err = None
    if "cmr2" in shells.metadata:
        reference_cmr2 = float(shells.metadata["cmr2"])
        cmr2_err = cmr2 - reference_cmr2
    return RadialBulkDiagnostics(
        mass_kg=mass,
        axial_moi_kgm2=C,
        cmr2=cmr2,
        mass_relative_error=mass_err,
        cmr2_error=cmr2_err,
    )


def radial_artifact_to_interior_model(
    path,
    *,
    body_radius_m: float | None = None,
    fluid_mu_tol_Pa: float = 1.0,
) -> InteriorModel:
    """Convert a neutral radial artifact into an elastic pylov3d model."""
    if fluid_mu_tol_Pa < 0:
        raise ValueError("fluid_mu_tol_Pa must be non-negative")
    shells = load_radial_artifact_shells(path, body_radius_m=body_radius_m)
    n = shells.outer_radius_m.size
    ocean = (shells.mu_Pa <= fluid_mu_tol_Pa).astype(int)
    if n:
        ocean[0] = 0
    mu = shells.mu_Pa.copy()
    mu[ocean.astype(bool)] = 0.0
    return make_interior_model(
        R0_km=(shells.outer_radius_m / 1e3).tolist(),
        rho0=shells.rho_kgm3.tolist(),
        mu0=mu.tolist(),
        Ks0=shells.K_Pa.tolist(),
        eta0=[None] * n,
        ocean=ocean.tolist(),
    )
