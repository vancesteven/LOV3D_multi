# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for direct Mars gravity and shape coefficient loaders."""

import gzip
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from pylov3d.sh_data import load_shadr, load_shape, truncate


MARS_DATA = Path(__file__).resolve().parents[2] / "data" / "mars"
GMM3_PATH = MARS_DATA / "gmm3_120_sha.tab"
TOPO_PATH = MARS_DATA / "MarsTopo719.shape.gz"


@pytest.fixture(scope="module")
def gravity_coeffs():
    return load_shadr(GMM3_PATH)


@pytest.fixture(scope="module")
def shape_coeffs():
    return load_shape(TOPO_PATH)


def test_gmm3_header_and_coefficients(gravity_coeffs):
    coeffs = gravity_coeffs
    assert coeffs["r0_m"] == pytest.approx(3.396e6)
    assert coeffs["gm"] == pytest.approx(4.2828372854187757e13, rel=1e-9)
    assert coeffs["lmax"] == 120
    assert coeffs["clm"].shape == (121, 121)
    assert coeffs["slm"].shape == (121, 121)
    assert coeffs["sigma_clm"].shape == (121, 121)
    assert coeffs["sigma_slm"].shape == (121, 121)
    assert coeffs["clm"][0, 0] == 1.0
    assert coeffs["clm"][2, 0] == pytest.approx(
        -8.750211323545289e-4, abs=1e-15,
    )
    np.testing.assert_array_equal(coeffs["slm"][:, 0], 0.0)
    assert np.all(coeffs["sigma_clm"] >= 0)
    assert np.all(coeffs["sigma_slm"] >= 0)
    assert np.any(coeffs["sigma_clm"] > 0)
    assert np.any(coeffs["sigma_slm"] > 0)


def test_mars_topo_header_and_degree_one_rows(shape_coeffs):
    assert shape_coeffs["lmax"] == 719
    assert shape_coeffs["clm"].shape == (720, 720)
    assert shape_coeffs["slm"].shape == (720, 720)
    assert shape_coeffs["clm"][0, 0] == pytest.approx(
        3.38950012207057e6, rel=1e-6,
    )

    # Independent three-line parse: C00, then both degree-1 records.
    with gzip.open(TOPO_PATH, "rt", encoding="ascii") as stream:
        rows = [
            [float(value) for value in stream.readline().split(",")]
            for _ in range(3)
        ]
    for degree, order, clm, slm in rows[1:]:
        degree, order = int(degree), int(order)
        assert degree == 1
        assert shape_coeffs["clm"][degree, order] == clm
        assert shape_coeffs["slm"][degree, order] == slm


def test_shape_gzip_and_plain_agree(shape_coeffs, tmp_path):
    plain_path = tmp_path / "MarsTopo719.shape"
    with gzip.open(TOPO_PATH, "rb") as source, plain_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    plain = load_shape(plain_path)
    assert plain["lmax"] == shape_coeffs["lmax"]
    np.testing.assert_array_equal(plain["clm"], shape_coeffs["clm"])
    np.testing.assert_array_equal(plain["slm"], shape_coeffs["slm"])


@pytest.mark.parametrize("fixture_name", ["gravity_coeffs", "shape_coeffs"])
def test_truncate_preserves_values_and_original(fixture_name, request):
    original = request.getfixturevalue(fixture_name)
    original_arrays = {
        key: value.copy()
        for key, value in original.items()
        if isinstance(value, np.ndarray)
    }
    shortened = truncate(original, 5)
    assert shortened["lmax"] == 5
    for key, expected in original_arrays.items():
        assert shortened[key].shape == (6, 6)
        np.testing.assert_array_equal(shortened[key], expected[:6, :6])
        np.testing.assert_array_equal(original[key], expected)
        assert not np.shares_memory(shortened[key], original[key])


@pytest.mark.parametrize(
    ("loader", "path", "array_names"),
    [
        (load_shadr, GMM3_PATH, ("clm", "slm", "sigma_clm", "sigma_slm")),
        (load_shape, TOPO_PATH, ("clm", "slm")),
    ],
)
def test_loading_is_deterministic(loader, path, array_names):
    first = loader(path)
    second = loader(path)
    assert first.keys() == second.keys()
    for name in array_names:
        np.testing.assert_array_equal(first[name], second[name])
    for name in first.keys() - set(array_names):
        assert first[name] == second[name]


def test_gmm3_j2_consistency(gravity_coeffs):
    j2 = -gravity_coeffs["clm"][2, 0] * math.sqrt(5.0)
    assert j2 == pytest.approx(1.9566e-3, abs=1e-6)
    # The bulk-model product uses a slightly different published gravity field.
    assert j2 == pytest.approx(1.9555e-3, abs=2e-6)


def test_gmm3_gm_sigma_matches_header(gravity_coeffs):
    """gm_sigma (TASK-012 review nit) must be the header's 3rd field,
    converted km^3/s^2 -> m^3/s^2 exactly like `gm` is, and must reflect
    that GM is a very precisely determined quantity."""
    with GMM3_PATH.open("rt", encoding="ascii") as stream:
        header = [float(value) for value in stream.readline().split(",")]
    assert gravity_coeffs["gm_sigma"] == pytest.approx(header[2] * 1e9, rel=1e-12)
    assert gravity_coeffs["gm_sigma"] > 0
    assert gravity_coeffs["gm_sigma"] < gravity_coeffs["gm"] * 1e-6


def test_load_shadr_rejects_explicit_l0_row(tmp_path):
    """C00=1.0 is only a safe convention to inject when l=0 is genuinely
    absent from the file; an explicit l=0 row must not be silently
    overwritten (TASK-012 review D-nit)."""
    header = "3396.0, 42828.372854187757, 1.62e-05, 1, 1, 1, 0.0, 0.0\n"
    rows = (
        "0, 0, 1.0, 0.0, 0.0, 0.0\n"
        "1, 0, 0.0, 0.0, 0.0, 0.0\n"
        "1, 1, 0.0, 0.0, 0.0, 0.0\n"
    )
    path = tmp_path / "bad_shadr.tab"
    path.write_text(header + rows)
    with pytest.raises(ValueError, match="degree-0"):
        load_shadr(path)


def test_load_shape_rejects_incomplete_triangle(tmp_path):
    """A file missing an (n, m) row inside its own inferred lmax must be
    rejected, not silently zero-filled (TASK-012 review D-nit: completeness
    check mirroring load_shadr's declared-lmax check)."""
    # lmax inferred as 2 (max degree present), but (2, 1) is missing.
    rows = (
        "0, 0, 1.0, 0.0\n"
        "1, 0, 0.0, 0.0\n"
        "1, 1, 0.0, 0.0\n"
        "2, 0, 0.0, 0.0\n"
        "2, 2, 0.0, 0.0\n"
    )
    path = tmp_path / "bad_shape.shape"
    path.write_text(rows)
    with pytest.raises(ValueError, match="incomplete"):
        load_shape(path)
