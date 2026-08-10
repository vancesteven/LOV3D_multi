# Copyright (c) 2026 pylov3d contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Part of pylov3d, a Python/JAX port of LOV3D
# (https://github.com/mroviranavarro/LOV3D_multi, Apache-2.0).
# See LICENSE and NOTICE at the repository root.

"""Tests for the committed GRAIL gravity and LOLA Moon shape products."""

import gzip
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from pylov3d.bodies import get_body
from pylov3d.constants import G
from pylov3d.moon import MOON
from pylov3d.sh_data import load_shadr, load_shape


MOON_DATA = Path(__file__).resolve().parents[2] / "data" / "moon"
GRGM900C_PATH = MOON_DATA / "grgm900c_120_sha.tab"
MOON_TOPO_PATH = MOON_DATA / "MoonTopo719.shape.gz"


@pytest.fixture(scope="module")
def moon_gravity_coeffs():
    return load_shadr(GRGM900C_PATH)


@pytest.fixture(scope="module")
def moon_shape_coeffs():
    return load_shape(MOON_TOPO_PATH)


def test_grgm900c_header_round_trip(moon_gravity_coeffs):
    coeffs = moon_gravity_coeffs
    assert coeffs["r0_m"] == pytest.approx(1738.0e3)
    assert coeffs["gm"] == pytest.approx(4902.799967088640e9, rel=1e-14)
    assert coeffs["lmax"] == 120
    assert coeffs["clm"].shape == (121, 121)
    assert coeffs["slm"].shape == (121, 121)
    assert coeffs["clm"][0, 0] == 1.0

    with GRGM900C_PATH.open("rt", encoding="ascii") as stream:
        header = [float(value) for value in stream.readline().split(",")]
    assert header[3:5] == [120.0, 120.0]
    assert header[5] == 1.0  # 4pi-normalized coefficients


def test_grgm900c_j2_consistency(moon_gravity_coeffs):
    j2 = -moon_gravity_coeffs["clm"][2, 0] * math.sqrt(5.0)
    assert j2 == pytest.approx(2.033e-4, rel=0.01)


def test_grgm900c_mass_matches_body_catalog(moon_gravity_coeffs):
    gravity_mass = moon_gravity_coeffs["gm"] / G
    assert gravity_mass == pytest.approx(get_body(31)["Mass"], rel=1e-5)


def test_grgm900c_has_complete_declared_triangle():
    with GRGM900C_PATH.open("rt", encoding="ascii") as stream:
        next(stream)
        rows = [line for line in stream if line.strip()]
    assert len(rows) == 7380
    assert all(int(row.split(",", 1)[0]) != 0 for row in rows)


def test_moon_topo_degree_zero_and_full_triangle(moon_shape_coeffs):
    coeffs = moon_shape_coeffs
    assert coeffs["lmax"] == 719
    assert coeffs["clm"].shape == (720, 720)
    assert coeffs["slm"].shape == (720, 720)
    assert coeffs["clm"][0, 0] == pytest.approx(1737151.19826508)
    assert coeffs["clm"][0, 0] == pytest.approx(MOON["R"], rel=1e-6)

    with gzip.open(MOON_TOPO_PATH, "rt", encoding="ascii") as stream:
        assert sum(1 for line in stream if line.strip()) == 259560


def test_moon_shape_gzip_and_plain_agree(moon_shape_coeffs, tmp_path):
    plain_path = tmp_path / "MoonTopo719.shape"
    with gzip.open(MOON_TOPO_PATH, "rb") as source, plain_path.open("wb") as target:
        shutil.copyfileobj(source, target)

    plain = load_shape(plain_path)
    assert plain["lmax"] == moon_shape_coeffs["lmax"]
    np.testing.assert_array_equal(plain["clm"], moon_shape_coeffs["clm"])
    np.testing.assert_array_equal(plain["slm"], moon_shape_coeffs["slm"])
