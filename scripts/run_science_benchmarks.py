# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D.

"""Run the compact, publication-facing pylov3d science validation suite.

This is intentionally narrower than the full unit/regression suite. It gathers
benchmarks that exercise qualitatively different physics and independent
validation paths into one reproducible command.

Examples
--------
Run the core suite (MATLAB LOV3D references + analytic + planetary cases)::

    python scripts/run_science_benchmarks.py

Also require the independent PyALMA3 elastic/Maxwell benchmarks::

    python scripts/run_science_benchmarks.py --with-pyalma3

Pass additional pytest arguments after ``--``::

    python scripts/run_science_benchmarks.py -- -vv
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Each entry is deliberately science-facing rather than a broad unit-test file.
# The selected cases jointly cover an analytic elastic limit, a laterally
# heterogeneous shell, a fluid layer, multilayer planetary structure,
# dissipation, independent viscoelastic validation, and the proposal-facing
# Mars hydration/seismic/magnetic/gravity/EM constraint interfaces.
CORE_BENCHMARKS = [
    # Analytic homogeneous elastic body / convention anchor.
    "pylov3d/tests/test_analytical.py",
    # Enceladus: 2-layer lateral-mu spectra against archived MATLAB LOV3D data.
    "pylov3d/tests/test_matlab_validation.py",
    # Moon: Weber multilayer model with fluid outer core and lateral-mu spectra
    # against MATLAB/Qin reference data.
    "pylov3d/tests/test_matlab_validation_ocean.py",
    # Mars: independent planetary-structure constraints and fitted k2/h2/l2.
    "pylov3d/tests/test_mars.py::TestMass",
    "pylov3d/tests/test_mars.py::TestMoI",
    "pylov3d/tests/test_mars.py::TestK2",
    "pylov3d/tests/test_mars.py::TestLoveNumberSanity",
    "pylov3d/tests/test_mars.py::TestDensityProfile",
    # Proposal-facing Mars multi-observable interfaces: InSight seismic
    # likelihood, dry-frame/Gassmann bounding physics, remanent-magnetization
    # plausibility anchors, self-consistent hydrated-solid density, finite-shell
    # gravity coefficients, GMM-3 parsing/normalization/background, and
    # connected-pore EM bounds.
    "pylov3d/tests/test_mars_seismic.py",
    "pylov3d/tests/test_mars_poroelastic.py",
    "pylov3d/tests/test_mars_magnetic.py",
    "pylov3d/tests/test_mars_joint_constraints.py",
    "pylov3d/tests/test_mars_gravity_sensitivity.py",
    "pylov3d/tests/test_mars_gravity_harmonics.py",
    "pylov3d/tests/test_mars_gravity_coefficients.py",
    "pylov3d/tests/test_mars_gmm3.py",
    "pylov3d/tests/test_mars_gravity_normalization.py",
    "pylov3d/tests/test_mars_gravity_background.py",
    "pylov3d/tests/test_mars_em_sensitivity.py",
    "pylov3d/tests/test_mars_em_profiles.py",
    # Dissipation sanity: elastic material must dissipate zero energy, while
    # the viscoelastic Io reference model must dissipate non-zero energy.
    "pylov3d/tests/test_energy.py::TestGetEnergy::test_elastic_zero_dissipation",
    "pylov3d/tests/test_energy.py::TestGetEnergy::test_io_nonzero_dissipation",
    # Sign and scaling of the independent Im(k)-based global heating formula.
    "pylov3d/tests/test_energy.py::TestGlobalDissipation",
    # TASK-046: native per-forcing closures, quadratic energy scaling, and
    # MATLAB-equivalent GSH energy-coupling grouping.
    "pylov3d/tests/test_energy_multibasis.py",
    "pylov3d/tests/test_energy_couplings_matlab_order.py",
    # TASK-046: authoritative six-mode viscoelastic Io spectrum and
    # [43,41,41] perturbation closures from the native MATLAB raw-grid path.
    "pylov3d/tests/test_io_rheology_spectrum_parity.py",
    # TASK-046 closure: literal MATLAB SPH transform convention and the
    # production fractional-grid viscoelastic-rheology processor. These pin
    # the native half-cell FFT phase and the six retained complex coefficients
    # against the independent MATLAB raw-grid anchor.
    "pylov3d/tests/test_matlab_sph.py",
    "pylov3d/tests/test_rheology_grid.py",
    "pylov3d/tests/test_io_raw_grid_coefficient_parity.py",
    # Slow but decisive publication gate: raw maps through nonlinear Maxwell
    # transform, coupled solver and both energy calculations at Nrbase=50.
    "pylov3d/tests/test_io_raw_grid_gate_c.py",
]

PYALMA3_BENCHMARK = "pylov3d/tests/test_benchmark_pyalma3.py"


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split script arguments from pytest arguments following ``--``."""
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own_args, pytest_args = _split_passthrough(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-pyalma3",
        action="store_true",
        help=(
            "Include the independent PyALMA3 elastic + Maxwell benchmark. "
            "This is a hard requirement when requested: the script fails if "
            "the alma package is unavailable. Install pylov3d[compat]/PyALMA3 first."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected benchmark node IDs without running pytest.",
    )
    args = parser.parse_args(own_args)

    selected = list(CORE_BENCHMARKS)
    if args.with_pyalma3:
        if importlib.util.find_spec("alma") is None:
            print(
                "ERROR: --with-pyalma3 requested, but the 'alma' package is not "
                "importable. Install pylov3d[compat]/PyALMA3 first.",
                file=sys.stderr,
            )
            return 2
        selected.append(PYALMA3_BENCHMARK)

    if args.list:
        print("pylov3d science benchmark suite")
        for item in selected:
            print(f"  {item}")
        return 0

    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "", *selected, *pytest_args]
    print("Science benchmark command:")
    print(" ".join(cmd))
    print()
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
