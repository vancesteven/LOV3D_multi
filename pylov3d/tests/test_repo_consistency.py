# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Repository-level documentation consistency checks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent
README_PATH = REPO_ROOT / "README.md"

CLAIM_RE = re.compile(
    r"(?P<full>\d+) tests across (?P<files>\d+) test files "
    r"\((?P<fast>\d+) in the default fast lane,"
)
COLLECTED_RE = re.compile(r"(?P<selected>\d+)(?:/(?P<total>\d+))? tests collected")
TEST_TOLERANCE = 10


def _collect_count(*extra_args: str) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(TESTS_DIR),
        "-q",
        "--collect-only",
        *extra_args,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise AssertionError(
            f"pytest collection failed with exit code {result.returncode}:\n{output}"
        )
    matches = list(COLLECTED_RE.finditer(output))
    if not matches:
        raise AssertionError(f"could not parse pytest collection summary:\n{output}")
    return int(matches[-1].group("selected"))


def test_readme_validation_counts_match_live_collection():
    """Keep README counts useful without forcing edits for every new test.

    Test totals allow a +/-10-test tolerance so ordinary additions in an
    existing file do not create documentation churn; that still catches the
    large drift this guard was introduced to prevent.  The file count is
    exact because new test files are comparatively infrequent and the README
    makes an exact inventory claim.

    Counts come from fresh, independent pytest collection subprocesses rather
    than ``request.session.items``.  This costs under a second but remains
    correct when this file is invoked alone, where the current session would
    otherwise contain only this test.  Collection does not execute this test,
    so the subprocesses do not recurse.
    """
    readme = README_PATH.read_text(encoding="utf-8")
    match = CLAIM_RE.search(readme)
    assert match is not None, "README validation-count claim was not found"

    claimed_full = int(match.group("full"))
    claimed_files = int(match.group("files"))
    claimed_fast = int(match.group("fast"))
    actual_full = _collect_count("-m", "")
    actual_fast = _collect_count()
    actual_files = len(list(TESTS_DIR.glob("test_*.py")))

    counts_within_tolerance = (
        abs(claimed_full - actual_full) <= TEST_TOLERANCE
        and abs(claimed_fast - actual_fast) <= TEST_TOLERANCE
    )
    files_match = claimed_files == actual_files
    if not counts_within_tolerance or not files_match:
        replacement = CLAIM_RE.sub(
            f"{actual_full} tests across {actual_files} test files "
            f"({actual_fast} in the default fast lane,",
            readme,
            count=1,
        )
        replacement_sentence = next(
            line for line in replacement.splitlines() if "tests across" in line
        )
        raise AssertionError(
            "README validation counts are stale. Replace the sentence with:\n"
            f"{replacement_sentence}"
        )
