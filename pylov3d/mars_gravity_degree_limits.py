# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0

"""Invert GMM-3 degree amplitudes into hydration-harmonic thickness scales.

The proposal-scale finite-shell gravity model is linear in the thickness
coefficient H_lm.  Once a trial signal has been placed in the GMM-3 normalized
coefficient convention, the thickness coefficient that would occupy a chosen
fraction ``f`` of the observed GMM-3 degree norm follows directly from
linearity:

    H_limit = H_trial * f * degree_norm / |signal_trial|.

This is not a statistical upper limit.  Different geological contributions can
cancel vectorially within a degree.  It is a transparent scale diagnostic for
asking when one hypothetical hydration harmonic becomes a substantial fraction
of the entire observed gravity power at that degree.
"""
from __future__ import annotations


def thickness_for_degree_fraction(
    trial_thickness_m: float,
    trial_signal_coefficient: float,
    degree_norm: float,
    target_fraction: float,
) -> float:
    """Return H_lm whose signal equals ``target_fraction`` of degree norm."""
    if trial_thickness_m <= 0:
        raise ValueError("trial_thickness_m must be positive")
    if degree_norm <= 0:
        raise ValueError("degree_norm must be positive")
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must lie in (0, 1]")
    amplitude = abs(float(trial_signal_coefficient))
    if amplitude == 0:
        return float("inf")
    return trial_thickness_m * target_fraction * degree_norm / amplitude


def thickness_for_degree_rms(
    trial_thickness_m: float,
    trial_signal_coefficient: float,
    degree_rms: float,
    target_multiple: float = 1.0,
) -> float:
    """Return H_lm whose signal equals a multiple of the observed mode RMS."""
    if trial_thickness_m <= 0:
        raise ValueError("trial_thickness_m must be positive")
    if degree_rms <= 0:
        raise ValueError("degree_rms must be positive")
    if target_multiple <= 0:
        raise ValueError("target_multiple must be positive")
    amplitude = abs(float(trial_signal_coefficient))
    if amplitude == 0:
        return float("inf")
    return trial_thickness_m * target_multiple * degree_rms / amplitude
