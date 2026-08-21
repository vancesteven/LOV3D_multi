# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D.

"""Run the compact, publication-facing pylov3d science validation suite.

This is intentionally narrower than the full unit/regression suite. It gathers
benchmarks that exercise qualitatively different physics and independent
validation paths into one reproducible command.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_BENCHMARKS = [
    "pylov3d/tests/test_analytical.py",
    "pylov3d/tests/test_matlab_validation.py",
    "pylov3d/tests/test_matlab_validation_ocean.py",
    "pylov3d/tests/test_mars.py::TestMass",
    "pylov3d/tests/test_mars.py::TestMoI",
    "pylov3d/tests/test_mars.py::TestK2",
    "pylov3d/tests/test_mars.py::TestLoveNumberSanity",
    "pylov3d/tests/test_mars.py::TestDensityProfile",
    "pylov3d/tests/test_mars_seismic.py",
    "pylov3d/tests/test_mars_poroelastic.py",
    "pylov3d/tests/test_mars_magnetic.py",
    "pylov3d/tests/test_mars_joint_constraints.py",
    "pylov3d/tests/test_mars_gravity_sensitivity.py",
    "pylov3d/tests/test_mars_gravity_harmonics.py",
    "pylov3d/tests/test_mars_gravity_coefficients.py",
    "pylov3d/tests/test_mars_alteration_gravity.py",
    "pylov3d/tests/test_mars_alteration_state.py",
    "pylov3d/tests/test_mars_gmm3.py",
    "pylov3d/tests/test_mars_gravity_normalization.py",
    "pylov3d/tests/test_mars_gravity_background.py",
    "pylov3d/tests/test_mars_em_sensitivity.py",
    "pylov3d/tests/test_mars_em_profiles.py",
    "pylov3d/tests/test_energy.py::TestGetEnergy::test_elastic_zero_dissipation",
    "pylov3d/tests/test_energy.py::TestGetEnergy::test_io_nonzero_dissipation",
    "pylov3d/tests/test_energy.py::TestGlobalDissipation",
    "pylov3d/tests/test_energy_multibasis.py",
    "pylov3d/tests/test_energy_couplings_matlab_order.py",
    "pylov3d/tests/test_io_rheology_spectrum_parity.py",
    "pylov3d/tests/test_matlab_sph.py",
    "pylov3d/tests/test_rheology_grid.py",
    "pylov3d/tests/test_io_raw_grid_coefficient_parity.py",
    "pylov3d/tests/test_io_raw_grid_gate_c.py",
]

PYALMA3_BENCHMARK = "pylov3d/tests/test_benchmark_pyalma3.py"


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
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
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(own_args)

    selected = list(CORE_BENCHMARKS)
    if args.with_pyalma3:
        if importlib.util.find_spec("alma") is None:
            print(
                "ERROR: --with-pyalma3 requested, but the 'alma' package is not importable.",
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
