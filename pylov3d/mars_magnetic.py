# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Proposal-facing Mars remanent-magnetization benchmarks.

This module deliberately stores *observational/geophysical requirements*, not
an inversion from field strength directly to present hydration.  Bultel et
al. (2025, JGR Planets, doi:10.1029/2023JE008111) estimated the minimum
remanent magnetization and magnetite abundance required for two reference
observations under an assumed 50 microtesla magnetizing field.

The values below are useful for a Task-1 plausibility calculation linking a
geochemical alteration model's predicted magnetite abundance to magnetic
observations while keeping source depth/thickness, paleofield strength and
later demagnetization as nuisance parameters.

All Bultel et al. abundances are lower bounds under their compact,
unidirectionally magnetized source model.  Titanomagnetite can require larger
abundances than pure magnetite, and the ancient martian magnetizing field is
poorly constrained.  Consequently this module must not be interpreted as a
direct hydration estimator.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MagneticBenchmark:
    label: str
    depth_top_km: float
    depth_bottom_km: float
    magnetization_A_m: float
    magnetite_wt_percent: float
    magnetite_vol_percent: float
    assumed_paleofield_uT: float = 50.0

    @property
    def thickness_km(self) -> float:
        return self.depth_bottom_km - self.depth_top_km


# Bultel et al. (2025), Table 1.
INSIGHT_BENCHMARKS: tuple[MagneticBenchmark, ...] = (
    MagneticBenchmark("InSight 0-8 km", 0.0, 8.0, 11.0, 0.91, 0.53),
    MagneticBenchmark("InSight 0-20 km", 0.0, 20.0, 5.9, 0.49, 0.29),
    MagneticBenchmark("InSight 0-39 km", 0.0, 39.0, 4.5, 0.37, 0.22),
    MagneticBenchmark("InSight 20-39 km", 20.0, 39.0, 14.0, 1.10, 0.67),
)

# Strongest orbital anomaly, magnetic layer buried beneath 20 km overburden.
ORBITAL_BENCHMARKS: tuple[MagneticBenchmark, ...] = (
    MagneticBenchmark("Orbital 5 km layer", 20.0, 25.0, 172.0, 14.0, 8.4),
    MagneticBenchmark("Orbital 10 km layer", 20.0, 30.0, 87.0, 7.1, 4.2),
    MagneticBenchmark("Orbital 20 km layer", 20.0, 40.0, 45.0, 3.7, 2.2),
    MagneticBenchmark("Orbital 40 km layer", 20.0, 60.0, 24.0, 2.0, 1.2),
)


def paleofield_scaled_required_magnetite(
    benchmark: MagneticBenchmark,
    paleofield_uT: float,
) -> float:
    """First-order inverse scaling of required magnetite with paleofield.

    This is a sensitivity scaling, not a full remanence acquisition model.
    It assumes remanent magnetization per unit magnetite scales linearly with
    the magnetizing field around Bultel et al.'s 50-uT reference case.
    """
    B = float(paleofield_uT)
    if B <= 0:
        raise ValueError("paleofield strength must be positive")
    return benchmark.magnetite_wt_percent * benchmark.assumed_paleofield_uT / B


def abundance_margin(
    predicted_magnetite_wt_percent: float,
    benchmark: MagneticBenchmark,
    *,
    paleofield_uT: float = 50.0,
) -> float:
    """Predicted / required magnetite abundance for a benchmark geometry."""
    predicted = float(predicted_magnetite_wt_percent)
    if predicted < 0:
        raise ValueError("predicted magnetite abundance must be non-negative")
    required = paleofield_scaled_required_magnetite(benchmark, paleofield_uT)
    return predicted / required
